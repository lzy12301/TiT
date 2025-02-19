# Modified from Meta DiT

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT:   https://github.com/facebookresearch/DiT/tree/main
# GLIDE: https://github.com/openai/glide-text2im
# MAE:   https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from timm.models.vision_transformer import Mlp

from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.models.layers.blocks import (
    Attention,
    CrossAttention,
    CaptionEmbedder,
    FinalLayer,
    LabelEmbedder,
    PatchEmbed3D,
    TimestepEmbedder,
    TimestepEmbedderEDM,
    ParamEmbedderEDM,
    approx_gelu,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_layernorm,
    modulate,
)
from modules_irsde.module_util import Linear
from opensora.registry import MODELS
from opensora.utils.ckpt_utils import load_checkpoint


class DiTBlockCond(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.enable_flash_attn = enable_flash_attn
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attn=enable_flash_attn,
        )
        self.cross_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attn=enable_flash_attn,
        )
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.norm3 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True))

    def forward(self, x, c, y, return_qkv=False):
        shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        if not return_qkv:
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1, x, shift_msa, scale_msa))
            x = x + gate_mca.unsqueeze(1) * self.cross_attn(modulate(self.norm2, x, shift_mca, scale_mca), y)
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3, x, shift_mlp, scale_mlp))
            return x
        else:
            attn, qkv_attn = self.attn(modulate(self.norm1, x, shift_msa, scale_msa), return_qkv=True)
            x = x + gate_msa.unsqueeze(1) * attn
            cross_attn, qkv_cross_attn = self.cross_attn(modulate(self.norm2, x, shift_mca, scale_mca), y, return_qkv=True)
            x = x + gate_mca.unsqueeze(1) * cross_attn
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3, x, shift_mlp, scale_mlp))
            return x, (qkv_attn, qkv_cross_attn)


@MODELS.register_module()
class DiTCond(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=(16, 32, 32),
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
        condition="text",
        no_temporal_pos_emb=False,
        caption_channels=512,
        model_max_length=77,
        dtype=torch.float32,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_patches = num_patches
        self.num_temporal = input_size[0] // patch_size[0]
        self.num_spatial = num_patches // self.num_temporal
        self.num_heads = num_heads
        self.dtype = dtype
        self.use_text_encoder = not condition.startswith("label") and not condition.startswith('vec')
        if enable_flash_attn:
            assert dtype in [
                torch.float16,
                torch.bfloat16,
            ], f"Flash attention only supports float16 and bfloat16, but got {self.dtype}"
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        assert enable_sequence_parallelism is False, "Sequence parallelism is not supported in DiT"

        self.register_buffer("pos_embed_spatial", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, embed_dim=hidden_size)
        if not self.use_text_encoder:
            if condition.startswith('label'):
                num_classes = int(condition.split("_")[-1])
                self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            else:
                num_conddim = int(condition.split("_")[-1])
                # self.y_embedder = ParamEmbedderEDM(hidden_size, num_conddim, 0, class_dropout_prob)
                self.y_embedder = Linear(in_features=num_conddim, out_features=hidden_size, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(num_conddim), dropout_prob=class_dropout_prob)
        else:
            self.y_embedder = CaptionEmbedder(
                in_channels=caption_channels,
                hidden_size=hidden_size,
                uncond_prob=class_dropout_prob,
                act_layer=approx_gelu,
                token_num=1,  # pooled token
            )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlockCond(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    enable_flash_attn=enable_flash_attn,
                    enable_layernorm_kernel=enable_layernorm_kernel,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, np.prod(self.patch_size), self.out_channels)

        self.initialize_weights()
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel

    def get_spatial_pos_embed(self, input_size=None, patch_size=None):
        if input_size is None or patch_size is None:
            input_size = self.input_size
            patch_size = self.patch_size
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            input_size[1] // patch_size[1],
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def get_temporal_pos_embed(self, input_size=None, patch_size=None):
        if input_size is None or patch_size is None:
            input_size = self.input_size
            patch_size = self.patch_size
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            input_size[0] // patch_size[0],
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def unpatchify(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "n t h w r p q c -> n c t r h p w q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def forward(self, x, t, y, augment_labels=None):
        """
        Forward pass of DiT.
        x: (B, C, T, H, W) tensor of inputs
        t: (B,) tensor of diffusion timesteps
        y: list of text
        """
        # origin inputs should be float32, cast to specified dtype
        x = x.to(self.dtype)
        if self.use_text_encoder:
            y = y.to(self.dtype)

        # embedding
        x = self.x_embedder(x)  # (B, N, D)
        x = rearrange(x, "b (t s) d -> b t s d", t=self.num_temporal, s=self.num_spatial)
        x = x + self.pos_embed_spatial
        if not self.no_temporal_pos_emb:
            x = rearrange(x, "b t s d -> b s t d")
            x = x + self.pos_embed_temporal
            x = rearrange(x, "b s t d -> b (t s) d")
        else:
            x = rearrange(x, "b t s d -> b (t s) d")

        t = self.t_embedder(t[:, 0, 0, 0, 0], dtype=x.dtype)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        y = y.unsqueeze(1)      # (N, 1, D)
        if self.use_text_encoder:
            y = y.squeeze(1).squeeze(1)
        condition = t

        # blocks
        for _, block in enumerate(self.blocks):
            c = condition
            x = auto_grad_checkpoint(block, x, c, y)  # (B, N, D)

        # final process
        x = self.final_layer(x, condition)  # (B, N, num_patches * out_channels)
        x = self.unpatchify(x)  # (B, out_channels, T, H, W)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                if module.weight.requires_grad_:
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Zero-out text embedding layers:
        if self.use_text_encoder:
            nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
            nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)


@MODELS.register_module()
class DiTCondEDM(DiTCond):

    def __init__(
        self,
        input_size=(16, 32, 32),
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
        condition="text",
        no_temporal_pos_emb=False,
        caption_channels=512,
        model_max_length=77,
        dtype=torch.float32,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        sigma_data=0.5,
    ):
        super().__init__(input_size, 
                         in_channels, 
                         patch_size, 
                         hidden_size, 
                         depth, 
                         num_heads, 
                         mlp_ratio, 
                         class_dropout_prob, 
                         learn_sigma, 
                         condition, 
                         no_temporal_pos_emb, 
                         caption_channels, 
                         model_max_length, 
                         dtype, 
                         enable_flash_attn, 
                         enable_layernorm_kernel, 
                         enable_sequence_parallelism)
        self.c_skip = lambda sigma: sigma_data**2/(sigma**2+sigma_data**2)
        self.c_out = lambda sigma: sigma_data*sigma/(sigma**2+sigma_data**2).sqrt()
        self.c_in = lambda sigma: 1./(sigma**2+sigma_data**2).sqrt()
        self.c_noise = lambda sigma: torch.log(sigma+1e-6)/4

        self.random_or_learned_sinusoidal_cond = False
        self.learned_sinusoidal_dim = 16

        self.t_embedder = TimestepEmbedderEDM(self.hidden_size, 
                                              frequency_embedding_size=self.learned_sinusoidal_dim, 
                                              random_or_learned_sinusoidal_cond=self.random_or_learned_sinusoidal_cond)
    
    def forward(self, xt, t, y, augment_labels=None, return_feat=False):
        """
        Forward pass of DiT.
        x: (B, C, T, H, W) tensor of inputs
        t: (B,) tensor of diffusion timesteps
        y: list of text
        """
        # origin inputs should be float32, cast to specified dtype
        xt = xt.to(self.dtype)
        if self.use_text_encoder:
            y = y.to(self.dtype)

        # EDM scalar
        c_noise = self.c_noise(t)
        c_in = self.c_in(t)
        c_skip = self.c_skip(t)
        c_out = self.c_out(t)

        # embedding
        x = c_in * xt
        x = self.x_embedder(x)  # (B, N, D)
        x = rearrange(x, "b (t s) d -> b t s d", t=self.num_temporal, s=self.num_spatial)
        x = x + self.pos_embed_spatial
        if not self.no_temporal_pos_emb:
            x = rearrange(x, "b t s d -> b s t d")
            x = x + self.pos_embed_temporal
            x = rearrange(x, "b s t d -> b (t s) d")
        else:
            x = rearrange(x, "b t s d -> b (t s) d")

        t = self.t_embedder(c_noise[..., 0, 0, 0, 0], dtype=x.dtype)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        y = y.unsqueeze(1)  # (N, 1, D)
        if self.use_text_encoder:
            y = y.squeeze(1).squeeze(1)
        condition = t

        # blocks
        feats = []
        for _, block in enumerate(self.blocks):
            c = condition
            x = auto_grad_checkpoint(block, x, c, y)  # (B, N, D)
            feats.append(x)

        # final process
        x = self.final_layer(x, condition)  # (B, N, num_patches * out_channels)
        x = self.unpatchify(x)  # (B, out_channels, T, H, W)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        if return_feat:
            return c_skip * xt + c_out * x, feats
        else:
            return c_skip * xt + c_out * x
        

@MODELS.register_module()
class DiTFrameCAEDM(DiTCond):

    def __init__(
        self,
        input_size=(16, 32, 32),
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
        condition="text",
        no_temporal_pos_emb=False,
        caption_channels=512,
        model_max_length=77,
        dtype=torch.float32,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        sigma_data=0.5,
        num_cond_frame=2,
    ):
        super().__init__(input_size, 
                         in_channels, 
                         patch_size, 
                         hidden_size, 
                         depth, 
                         num_heads, 
                         mlp_ratio, 
                         class_dropout_prob, 
                         learn_sigma, 
                         condition, 
                         no_temporal_pos_emb, 
                         caption_channels, 
                         model_max_length, 
                         dtype, 
                         enable_flash_attn, 
                         enable_layernorm_kernel, 
                         enable_sequence_parallelism)
        self.c_skip = lambda sigma: sigma_data**2/(sigma**2+sigma_data**2)
        self.c_out = lambda sigma: sigma_data*sigma/(sigma**2+sigma_data**2).sqrt()
        self.c_in = lambda sigma: 1./(sigma**2+sigma_data**2).sqrt()
        self.c_noise = lambda sigma: torch.log(sigma+1e-6)/4

        self.random_or_learned_sinusoidal_cond = False
        self.learned_sinusoidal_dim = 16

        # for frame condition
        self.num_cond_frame = num_cond_frame
        self.input_size_framecond = (num_cond_frame, self.input_size[1], self.input_size[2])
        self.register_buffer(
            'pos_embed_spatial_framecond', self.get_spatial_pos_embed(input_size=self.input_size_framecond, patch_size=self.patch_size)
        )
        self.register_buffer(
            'pos_embed_temporal_framecond', self.get_temporal_pos_embed(input_size=self.input_size_framecond, patch_size=self.patch_size)
        )
        num_patches_framecond = np.prod([self.input_size_framecond[i] // patch_size[i] for i in range(3)])
        self.num_patches_framecond = num_patches_framecond
        self.num_temporal_framecond = self.input_size_framecond[0] // patch_size[0]
        self.num_spatial_framecond = num_patches_framecond // self.num_temporal_framecond

        self.t_embedder = TimestepEmbedderEDM(self.hidden_size, 
                                              frequency_embedding_size=self.learned_sinusoidal_dim, 
                                              random_or_learned_sinusoidal_cond=self.random_or_learned_sinusoidal_cond)
        self.frame_embedder = PatchEmbed3D(patch_size, in_channels, embed_dim=hidden_size)
        w = self.frame_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.frame_embedder.proj.bias, 0)
    
    def forward(self, xt, t, y, augment_labels=None, return_feat=False, out_from_feat=None):
        """
        Forward pass of DiT.
        x: (B, C, T, H, W) tensor of inputs
        t: (B,) tensor of diffusion timesteps
        y: list of text
        """
        # origin inputs should be float32, cast to specified dtype
        xt = xt.to(self.dtype)
        if self.use_text_encoder:
            y = y.to(self.dtype)

        # EDM scalar
        c_noise = self.c_noise(t)
        c_in = self.c_in(t)
        c_skip = self.c_skip(t)
        c_out = self.c_out(t)

        # embedding
        x = c_in * xt
        x = self.x_embedder(x)  # (B, N, D)
        x = rearrange(x, "b (t s) d -> b t s d", t=self.num_temporal, s=self.num_spatial)
        x = x + self.pos_embed_spatial
        if not self.no_temporal_pos_emb:
            x = rearrange(x, "b t s d -> b s t d")
            x = x + self.pos_embed_temporal
            x = rearrange(x, "b s t d -> b (t s) d")
        else:
            x = rearrange(x, "b t s d -> b (t s) d")

        t = self.t_embedder(c_noise[..., 0, 0, 0, 0], dtype=x.dtype)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        # y = y.unsqueeze(1)  # (N, 1, D)
        if self.use_text_encoder:
            y = y.squeeze(1).squeeze(1)
        condition = t + y

        # drop the frame condition with a prob. of 50%
        if self.training:
            drop_ids = torch.rand(augment_labels.shape[0]).cuda() < 0.5     
            drop_ids = drop_ids[:, None, None, None, None].repeat(1, *augment_labels.shape[1:])
            augment_labels = torch.where(drop_ids, torch.zeros_like(augment_labels), augment_labels)
        condition_f = self.frame_embedder(augment_labels)
        condition_f = rearrange(condition_f, "b (t s) d -> b t s d", t=self.num_temporal_framecond, s=self.num_spatial_framecond)
        condition_f = condition_f + self.pos_embed_spatial_framecond
        if not self.no_temporal_pos_emb:
            condition_f = rearrange(condition_f, "b t s d -> b s t d")
            condition_f = condition_f + self.pos_embed_temporal_framecond
            condition_f = rearrange(condition_f, "b s t d -> b (t s) d")
        else:
            condition_f = rearrange(condition_f, "b t s d -> b (t s) d")


        # blocks
        feats = []
        qkvs = []
        for _, block in enumerate(self.blocks):
            c = condition
            if not return_feat:
                x = auto_grad_checkpoint(block, x, c, condition_f)  # (B, N, D)
            else:
                x, qkv = auto_grad_checkpoint(block, x, c, condition_f, return_qkv=True)
                qkvs.append(qkv)
            feats.append(x)

        # final process
        x = self.final_layer(x, condition)  # (B, N, num_patches * out_channels)
        x = self.unpatchify(x)  # (B, out_channels, T, H, W)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        out = c_skip * xt + c_out * x

        # skip deep features
        if out_from_feat is not None:
            out_ = []
            for idx in out_from_feat:
                f_ = self.final_layer(feats[idx], condition)
                f_ = c_skip * xt + c_out * self.unpatchify(f_)
                out_.append(f_.to(torch.float32))
                if idx == len(feats) - 1:
                    continue
            out_.append(out)
        else:
            out_ = out

        if return_feat:
            return out_, (feats, condition, qkvs)
        else:
            return out_
        

@MODELS.register_module()
class DiTCondSSL(DiTCond):

    def __init__(
        self,
        input_size=(16, 32, 32),
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
        condition="text",
        no_temporal_pos_emb=False,
        caption_channels=512,
        model_max_length=77,
        dtype=torch.float32,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        sigma_data=0.5,
        dim_ssl=512,
    ):
        super().__init__(input_size, 
                         in_channels, 
                         patch_size, 
                         hidden_size, 
                         depth, 
                         num_heads, 
                         mlp_ratio, 
                         class_dropout_prob, 
                         learn_sigma, 
                         condition, 
                         no_temporal_pos_emb, 
                         caption_channels, 
                         model_max_length, 
                         dtype, 
                         enable_flash_attn, 
                         enable_layernorm_kernel, 
                         enable_sequence_parallelism)
        self.c_skip = lambda sigma: sigma_data**2/(sigma**2+sigma_data**2)
        self.c_out = lambda sigma: sigma_data*sigma/(sigma**2+sigma_data**2).sqrt()
        self.c_in = lambda sigma: 1./(sigma**2+sigma_data**2).sqrt()
        self.c_noise = lambda sigma: torch.log(sigma+1e-6)/4

        self.random_or_learned_sinusoidal_cond = False
        self.learned_sinusoidal_dim = 16
        self.feat_regressor = nn.ModuleList([
            nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.GELU(),
            ),
            nn.Linear(np.array([int(i//j) for i, j in zip(input_size, patch_size)]).prod(), dim_ssl)
        ])

        self.t_embedder = TimestepEmbedderEDM(self.hidden_size, 
                                              frequency_embedding_size=self.learned_sinusoidal_dim, 
                                              random_or_learned_sinusoidal_cond=self.random_or_learned_sinusoidal_cond)
    
    def forward(self, xt, t, y, augment_labels=None, return_feat=False):
        """
        Forward pass of DiT.
        x: (B, C, T, H, W) tensor of inputs
        t: (B,) tensor of diffusion timesteps
        y: list of text
        """
        # origin inputs should be float32, cast to specified dtype
        xt = xt.to(self.dtype)
        if self.use_text_encoder:
            y = y.to(self.dtype)

        # EDM scalar
        c_noise = self.c_noise(t)
        c_in = self.c_in(t)
        c_skip = self.c_skip(t)
        c_out = self.c_out(t)

        # embedding
        x = c_in * xt
        x = self.x_embedder(x)  # (B, N, D)
        x = rearrange(x, "b (t s) d -> b t s d", t=self.num_temporal, s=self.num_spatial)
        x = x + self.pos_embed_spatial
        if not self.no_temporal_pos_emb:
            x = rearrange(x, "b t s d -> b s t d")
            x = x + self.pos_embed_temporal
            x = rearrange(x, "b s t d -> b (t s) d")
        else:
            x = rearrange(x, "b t s d -> b (t s) d")

        t = self.t_embedder(c_noise[..., 0, 0, 0, 0], dtype=x.dtype)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        y = y.unsqueeze(1)  # (N, 1, D)
        if self.use_text_encoder:
            y = y.squeeze(1).squeeze(1)
        condition = t + y

        # blocks
        feats = []
        for _, block in enumerate(self.blocks):
            c = condition
            x = auto_grad_checkpoint(block, x, c, y)  # (B, N, D)
            feats.append(x)

        # final process
        x = self.final_layer(x, condition)  # (B, N, num_patches * out_channels)
        x = self.unpatchify(x)  # (B, out_channels, T, H, W)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        if return_feat:
            return c_skip * xt + c_out * x, feats
        else:
            return c_skip * xt + c_out * x
    
    def regress_ssl(self, feat):
        h = self.feat_regressor[0](feat).squeeze(-1)
        return self.feat_regressor[1](h)


# @MODELS.register_module("DiT-XL/2")
# def DiT_XL_2(from_pretrained=None, **kwargs):
#     model = DiTCond(
#         depth=28,
#         hidden_size=1152,
#         patch_size=(1, 2, 2),
#         num_heads=16,
#         **kwargs,
#     )
#     if from_pretrained is not None:
#         load_checkpoint(model, from_pretrained)
#     return model


# @MODELS.register_module("DiT-XL/2x2")
# def DiT_XL_2x2(from_pretrained=None, **kwargs):
#     model = DiTCond(
#         depth=28,
#         hidden_size=1152,
#         patch_size=(2, 2, 2),
#         num_heads=16,
#         **kwargs,
#     )
#     if from_pretrained is not None:
#         load_checkpoint(model, from_pretrained)
#     return model

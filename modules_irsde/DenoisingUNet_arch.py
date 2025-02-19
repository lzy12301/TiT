import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import functools

from .module_util import (
    SinusoidalPosEmb,
    ResidualBlock,
    Linear,
    RandomOrLearnedSinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv, default_conv3d, zero_module,
    ResBlock, Upsampler, TimestepEmbedSequential,
    LinearAttention, Attention,
    PreNorm, Residual)


class ConditionalUNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, ch_mul=(1, 2, 4, 8, 16, ), upscale=1, sigma_data=0.5, num_dim=2, num_conddim=0, class_dropout_prob=0.1):
        super().__init__()
        self.depth = depth = len(ch_mul)-1
        self.upscale = upscale # not used

        if num_dim == 2:
            default_conv_ = default_conv
        elif num_dim == 3:
            default_conv_ = default_conv3d

        block_class = functools.partial(ResBlock, conv=default_conv_, act=NonLinearity())

        self.init_conv = default_conv_(in_nc, nf, 7)
        
        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        # temporal embedding
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        # condition embedding
        from modules_irsde.module_util import Linear
        import numpy as np
        if num_conddim > 0:
            self.y_embedder = Linear(in_features=num_conddim, out_features=time_dim, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(num_conddim), dropout_prob=class_dropout_prob)
        else:
            self.y_embedder = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i in range(depth):
            # dim_in = nf * int(math.pow(2, i))
            # dim_out = nf * int(math.pow(2, i+1))
            dim_in = nf * ch_mul[i]
            dim_out = nf * ch_mul[i+1]
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in, num_dim=num_dim), num_dim=num_dim)),
                Downsample(dim_in, dim_out, num_dim=num_dim) if i != (depth-1) else default_conv_(dim_in, dim_out)
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out, num_dim=num_dim), num_dim=num_dim)),
                Upsample(dim_out, dim_in, num_dim=num_dim) if i!=0 else default_conv_(dim_out, dim_in)
            ]))

        # mid_dim = nf * int(math.pow(2, depth))
        mid_dim = nf * ch_mul[-1]
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim, num_dim=num_dim), num_dim=num_dim))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1) if num_dim==2 else nn.Conv3d(nf, out_nc, 3, 1, 1)

        self.c_skip = lambda sigma: sigma_data**2/(sigma**2+sigma_data**2)
        self.c_out = lambda sigma: sigma_data*sigma/(sigma**2+sigma_data**2).sqrt()
        self.c_in = lambda sigma: 1./(sigma**2+sigma_data**2).sqrt()
        self.c_noise = lambda sigma: torch.log(sigma)/4

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def check_image_size3d(self, x, d, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_d = (s - d % s) % s
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d), 'reflect')
        return x

    def forward(self, xt, time, labels=None, augment_labels=None):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)
        
        c_noise = self.c_noise(time)
        c_in = self.c_in(time)
        c_skip = self.c_skip(time)
        c_out = self.c_out(time)
        
        x = c_in * xt

        if len(x.shape) == 4:
            H, W = x.shape[2:]
            x = self.check_image_size(x, H, W)
        elif len(x.shape) == 5:
            D, H, W = x.shape[2:]
            x = self.check_image_size3d(x, D, H, W)

        x = self.init_conv(x)
        x_ = x.clone()

        if len(x.shape) == 4:
            t = self.time_mlp(c_noise[..., 0, 0, 0].to(xt.device))
        elif len(x.shape) == 5:
            t = self.time_mlp(c_noise[..., 0, 0, 0, 0].to(xt.device))

        if labels is not None and self.y_embedder is not None:
            t += self.y_embedder(labels, train=self.training)

        h = []

        for b1, b2, attn, downsample in self.downs:
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for b1, b2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)
            
            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W] if len(x.shape) == 4 else x[..., :D, :H, :W]
        
        return c_skip * xt + c_out * x
    

class ConditionalUNetOld(nn.Module):
    def __init__(self, in_nc, out_nc, nf, ch_mul=(1, 2, 4, 8, 16, ), upscale=1, sigma_data=0.5, label_dim=0, label_dropout=0):
        super().__init__()
        self.depth = depth = len(ch_mul)-1
        self.upscale = upscale # not used
        self.label_dropout = label_dropout

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_nc, nf, 7)
        
        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        from modules_irsde.module_util import Linear
        import numpy as np
        self.map_label = Linear(in_features=label_dim, out_features=time_dim, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i in range(depth):
            # dim_in = nf * int(math.pow(2, i))
            # dim_out = nf * int(math.pow(2, i+1))
            dim_in = nf * ch_mul[i]
            dim_out = nf * ch_mul[i+1]
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if i != (depth-1) else default_conv(dim_in, dim_out)
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if i!=0 else default_conv(dim_out, dim_in)
            ]))

        # mid_dim = nf * int(math.pow(2, depth))
        mid_dim = nf * ch_mul[-1]
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

        self.c_skip = lambda sigma: sigma_data**2/(sigma**2+sigma_data**2)
        self.c_out = lambda sigma: sigma_data*sigma/(sigma**2+sigma_data**2).sqrt()
        self.c_in = lambda sigma: 1./(sigma**2+sigma_data**2).sqrt()
        self.c_noise = lambda sigma: torch.log(sigma)/4

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, xt, time, labels=None, augment_labels=None):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)
        
        c_noise = self.c_noise(time)
        c_in = self.c_in(time)
        c_skip = self.c_skip(time)
        c_out = self.c_out(time)
        
        x = c_in * xt

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)

        x = self.init_conv(x)
        x_ = x.clone()

        t = self.time_mlp(c_noise[..., 0, 0, 0].to(xt.device))

        if self.map_label is not None:
            tmp = labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([xt.shape[0], 1], device=xt.device) >= self.label_dropout).to(tmp.dtype)
            t = t + self.map_label(tmp, self.training)

        h = []

        for b1, b2, attn, downsample in self.downs:
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for b1, b2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)
            
            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W]
        
        return c_skip * xt + c_out * x
    

class ConditionalUNetPalSB(nn.Module):
    def __init__(self, in_nc, out_nc, nf, ch_mul=(1, 2, 4, 8, 16, ), upscale=1, noise_levels=None, num_dim=2):
        super().__init__()
        self.depth = depth = len(ch_mul)-1
        self.upscale = upscale # not used

        if num_dim == 2:
            default_conv_ = default_conv
        elif num_dim == 3:
            default_conv_ = default_conv3d

        self.noise_levels = noise_levels

        block_class = functools.partial(ResBlock, conv=default_conv_, act=NonLinearity())

        self.init_conv = default_conv_(in_nc*2, nf, 7)
        
        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i in range(depth):
            # dim_in = nf * int(math.pow(2, i))
            # dim_out = nf * int(math.pow(2, i+1))
            dim_in = nf * ch_mul[i]
            dim_out = nf * ch_mul[i+1]
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in, num_dim=num_dim), num_dim=num_dim)),
                Downsample(dim_in, dim_out, num_dim=num_dim) if i != (depth-1) else default_conv_(dim_in, dim_out)
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out, num_dim=num_dim), num_dim=num_dim)),
                Upsample(dim_out, dim_in, num_dim=num_dim) if i!=0 else default_conv_(dim_out, dim_in)
            ]))

        # mid_dim = nf * int(math.pow(2, depth))
        mid_dim = nf * ch_mul[-1]
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim, num_dim=num_dim), num_dim=num_dim))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1) if num_dim==2 else nn.Conv3d(nf, out_nc, 3, 1, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def check_image_size3d(self, x, d, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_d = (s - d % s) % s
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d), 'reflect')
        return x

    def forward(self, xt, cond, time):

        if self.noise_levels is not None:
            time = self.noise_levels[time]
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)
        
        x = xt - cond
        x = torch.cat([x, cond], dim=1)

        if len(x.shape) == 4:
            H, W = x.shape[2:]
            x = self.check_image_size(x, H, W)
        elif len(x.shape) == 5:
            D, H, W = x.shape[2:]
            x = self.check_image_size3d(x, D, H, W)

        x = self.init_conv(x)
        x_ = x.clone()

        t = self.time_mlp(time.to(xt.device))

        h = []

        for b1, b2, attn, downsample in self.downs:
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for b1, b2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)
            
            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W]
        
        return x


class ControlledConditionalUNet(ConditionalUNet):
    def forward(self, xt, cond, time, control=None, only_mid_control=False):
        h = []
        with torch.no_grad():
            if self.noise_levels is not None:
                time = self.noise_levels[time]
            if isinstance(time, int) or isinstance(time, float):
                time = torch.tensor([time]).to(xt.device)
            
            x = xt - cond
            x = torch.cat([x, cond], dim=1)

            H, W = x.shape[2:]
            x = self.check_image_size(x, H, W)

            x = self.init_conv(x)
            x_ = x.clone()

            t = self.time_mlp(time.to(xt.device))

            for b1, b2, attn, downsample in self.downs:
                x = b1(x, t)
                h.append(x)

                x = b2(x, t)
                x = attn(x)
                h.append(x)

                x = downsample(x)

            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t)

        if control is not None:
            x += control.pop()

        for b1, b2, attn, upsample in self.ups:
            if only_mid_control or control is None:
                x = torch.cat([x, h.pop()], dim=1)
            else:
                x = torch.cat([x, h.pop() + control.pop()], dim=1)
            
            x = b1(x, t)
            
            if only_mid_control or control is None:
                x = torch.cat([x, h.pop()], dim=1)
            else:
                x = torch.cat([x, h.pop() + control.pop()], dim=1)
            
            x = b2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W]
        
        return x 


class UNet2dCond(nn.Module):

    def __init__(
            self, 
            in_channels=3, 
            out_channels=1, 
            init_features=32, 
            hidden_dim=128, 
            activation='gelu', 
            norm=False, 
            n_groups=1,
            use_scale_shift_norm=False,
            noise_levels=None,
            condition=None,
    ):
        super().__init__()

        self.noise_levels = noise_levels

        features = init_features
        block_args = dict(
            cond_channels=hidden_dim,
            activation=activation,
            norm=norm,
            n_groups=n_groups,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(init_features)
            fourier_dim = init_features
        self.t_embedder = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        if condition is not None:
            num_conddim = int(condition.split("_")[-1])
            self.add_cond_embedder = Linear(in_features=num_conddim, out_features=hidden_dim, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(num_conddim), dropout_prob=0)
        else:
            self.add_cond_embedder = lambda x: x

        self.encoder1 = ResidualBlock(in_channels, features, **block_args)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ResidualBlock(features, features * 2, **block_args)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ResidualBlock(features * 2, features * 4, **block_args)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = ResidualBlock(features * 4, features * 8, **block_args)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ResidualBlock(features * 8, features * 16, **block_args)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = ResidualBlock((features * 8) * 2, features * 8, **block_args)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = ResidualBlock((features * 4) * 2, features * 4, **block_args)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = ResidualBlock((features * 2) * 2, features * 2, **block_args)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = ResidualBlock(features * 2, features, **block_args)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, xt, cond, time, add_cond=None):
        if self.noise_levels is not None:
            time = self.noise_levels[time]
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)
        t = self.t_embedder(time)
        if add_cond is not None:
            t += self.add_cond_embedder(add_cond.to(xt.device), train=self.training)
        
        x = xt - cond
        x = torch.cat([x, cond], dim=1)

        enc1 = self.encoder1(x, t)
        enc2 = self.encoder2(self.pool1(enc1), t)
        enc3 = self.encoder3(self.pool2(enc2), t)
        enc4 = self.encoder4(self.pool3(enc3), t)

        bottleneck = self.bottleneck(self.pool4(enc4), t)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4, t)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3, t)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2, t)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1, t)
        return self.conv(dec1)
    

class UNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, ch_mul=(1, 2, 4, 8, 16, ), upscale=1, noise_levels=None):
        super().__init__()
        self.depth = depth = len(ch_mul)-1
        self.upscale = upscale # not used
        self.noise_levels = noise_levels

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_nc, nf, 7)
        
        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i in range(depth):
            # dim_in = nf * int(math.pow(2, i))
            # dim_out = nf * int(math.pow(2, i+1))
            dim_in = nf * ch_mul[i]
            dim_out = nf * ch_mul[i+1]
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if i != (depth-1) else default_conv(dim_in, dim_out)
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if i!=0 else default_conv(dim_out, dim_in)
            ]))

        # mid_dim = nf * int(math.pow(2, depth))
        mid_dim = nf * ch_mul[-1]
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def check_image_size3d(self, x, d, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_d = (s - d % s) % s
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d), 'reflect')
        return x

    def forward(self, xt, time=None):

        if time is None:
            time = torch.zeros((len(xt),)).long()
        if self.noise_levels is not None:
            time = self.noise_levels[time]
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)
        
        x = xt

        if len(x.shape) == 4:
            H, W = x.shape[2:]
            x = self.check_image_size(x, H, W)
        elif len(x.shape) == 5:
            D, H, W = x.shape[2:]
            x = self.check_image_size3d(x, D, H, W)

        x = self.init_conv(x)
        x_ = x.clone()

        t = self.time_mlp(time.to(xt.device))

        h = []

        for b1, b2, attn, downsample in self.downs:
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for b1, b2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)
            
            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W]
        
        return x


class ControlUNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, depth=4, upscale=1, noise_levels=None):
        super().__init__()
        self.depth = depth
        self.upscale = upscale # not used
        self.noise_levels = noise_levels

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_nc*2, nf, 7)
        
        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.zero_convs = nn.ModuleList([self.make_zero_conv(nf)])

        self.input_hint_block = TimestepEmbedSequential(
            nn.Conv2d(in_nc, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            zero_module(default_conv(32, nf, 3))
        )

        self.downs = nn.ModuleList([])

        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i+1))
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if i != (depth-1) else default_conv(dim_in, dim_out)
            ]))
            self.zero_convs.append(self.make_zero_conv(dim_in))
            self.zero_convs.append(self.make_zero_conv(dim_in))

        mid_dim = nf * int(math.pow(2, depth))
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.zero_convs.append(self.make_zero_conv(mid_dim))

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, xt, cond, time, hint):

        if self.noise_levels is not None:
            time = self.noise_levels[time]
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)
        t = self.time_mlp(time.to(xt.device))
        
        guided_hint = self.input_hint_block(hint, t)

        x = xt - cond
        x = torch.cat([x, cond], dim=1)

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)

        outs = []
        i_zc = 0
        x = self.init_conv(x)
        outs.append(self.zero_convs[i_zc](x + guided_hint, t))
        i_zc += 1
        x_ = x.clone()

        for b1, b2, attn, downsample in self.downs:
            x = b1(x, t)
            outs.append(self.zero_convs[i_zc](x, t))
            i_zc += 1

            x = b2(x, t)
            x = attn(x)
            outs.append(self.zero_convs[i_zc](x, t))
            i_zc += 1

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        outs.append(self.zero_convs[i_zc](x, t))
        
        return outs
    
    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(default_conv(channels, channels, 1, bias=True)))

from .models.dit import DiT, DiTEDM, DiTSSL, DiTCond, DiTCondEDM, DiTCondSSL, DiTCat, DiTCatEDM, DiTCatSSL, DiTFrameCAEDM
import torch


def get_conditional_dit(opt):
    if 'diffusion' in opt.model_name.lower():
        kwargs = dict(
            input_size=opt.input_size,
            in_channels=opt.in_channels,
            patch_size=opt.patch_size,
            hidden_size=opt.hidden_size,
            depth=opt.depth,
            num_heads=opt.num_heads,
            mlp_ratio=opt.mlp_ratio,
            learn_sigma=bool(opt.learn_sigma),
            dtype=getattr(torch, opt.dtype),
            enable_flash_attn=bool(opt.enable_flash_attn),
            enable_layernorm_kernel=bool(opt.enable_layernorm_kernel),
            sigma_data=opt.sigma_data,
            condition=opt.condition,
            class_dropout_prob=opt.class_dropout_prob if hasattr(opt, 'class_dropout_prob') else 0.1,
                    )
        if hasattr(opt, 'dim_ssl'):
            kwargs['dim_ssl'] = opt.dim_ssl
            return DiTCondSSL(**kwargs) if hasattr(opt, 'cross_attn') and opt.cross_attn else DiTCatSSL(**kwargs) if opt.condition.startswith('cat') else DiTSSL(**kwargs)
        elif hasattr(opt, 'frame_cond_strategy') and 'cross_attn' in opt.frame_cond_strategy:
            kwargs['num_cond_frame'] = int(opt.frame_cond_strategy.split('_')[-1])
            return DiTFrameCAEDM(**kwargs)
        else:
            return DiTCondEDM(**kwargs) if hasattr(opt, 'cross_attn') and opt.cross_attn else DiTCatEDM(**kwargs) if opt.condition.startswith('cat') else DiTEDM(**kwargs)
    else:
        raise ValueError('Unknown model name!')

class Identity(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, x):
        return x

from .DenoisingUNet_arch import ConditionalUNet, ControlledConditionalUNet, ControlUNet, ConditionalUNetOld, ConditionalUNetPalSB, UNet2dCond, UNet

import torch


def get_conditional_unet(opt):
    if 'diffusion' in opt.model_name.lower():
        kwargs = dict(
            in_nc=opt.in_channels,
            out_nc=opt.in_channels,
            nf=opt.nf,
            # depth=len(opt.ch_mult),
            ch_mul=opt.ch_mult,
            upscale=1,
            sigma_data=opt.sigma_data,
            num_conddim=0 if not hasattr(opt, 'num_conddim') else opt.num_conddim,
            class_dropout_prob=0 if not hasattr(opt, 'class_dropout_prob') else opt.class_dropout_prob,
                    )
        if hasattr(opt, 'num_dim'):
            kwargs['num_dim'] = opt.num_dim
        return ConditionalUNet(**kwargs)
    else:
        raise ValueError('Unknown model name!')
    

def get_conditional_unet_old(opt):
    if 'diffusion' in opt.model_name.lower():
        kwargs = dict(
            in_nc=opt.in_channels,
            out_nc=opt.in_channels,
            nf=opt.nf,
            # depth=len(opt.ch_mult),
            ch_mul=opt.ch_mult,
            upscale=1,
            sigma_data=opt.sigma_data,
            label_dim=opt.label_dim if hasattr(opt, 'label_dim') else 0,
            label_dropout=opt.label_dropout if hasattr(opt, 'label_dropout') else 0,
                    )
        return ConditionalUNetOld(**kwargs)
    else:
        raise ValueError('Unknown model name!')
    

def get_palsb_unet(opt):
    if 'diffusion' in opt.model_name.lower():
        noise_levels = torch.linspace(opt.t0, opt.T, opt.num_scales, device=opt.device) * opt.num_scales
        if 'trival' in opt.model_name.lower():
            kwargs = dict(
                in_channels=opt.in_channels*2,
                out_channels=opt.in_channels,
                init_features=opt.nf,
                activation='gelu',
                hidden_dim=opt.nf*4,
                norm=True,
                n_groups=1,
                use_scale_shift_norm=True,
                noise_levels=noise_levels,
                condition=opt.condition,
                    )
            return UNet2dCond(**kwargs)
        else:
            kwargs = dict(
                in_nc=opt.in_channels,
                out_nc=opt.in_channels,
                nf=opt.nf,
                # depth=len(opt.ch_mult),
                ch_mul=opt.ch_mult,
                upscale=1,
                noise_levels=noise_levels,
                num_dim=opt.num_dim if hasattr(opt, 'num_dim') else 2
                        )
            return ConditionalUNetPalSB(**kwargs)
    elif 'unet' in opt.model_name:
        noise_levels = torch.linspace(opt.t0, opt.T, opt.num_scales, device=opt.device) * opt.num_scales
        kwargs = dict(
            in_nc=opt.in_channels,
            out_nc=opt.in_channels,
            nf=opt.nf,
            # depth=len(opt.ch_mult),
            ch_mul=opt.ch_mult,
            upscale=1,
            noise_levels=noise_levels
                    )
        return UNet(**kwargs)
    else:
        raise ValueError('Unknown model name!')
    

def get_control_unet(opt):
    noise_levels = torch.linspace(opt.t0, opt.T, opt.num_scales, device=opt.device) * opt.num_scales
    kwargs = dict(
        in_nc=opt.in_channels,
        out_nc=opt.in_channels,
        nf=opt.nf,
        depth=len(opt.ch_mult),
        upscale=1,
        noise_levels=noise_levels
                  )
    return ControlledConditionalUNet(**kwargs), ControlUNet(**kwargs)

import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange
from svd_operator import H_functions, Inpainting2


def sampler_heun2nd(net, ema, steps, shape, device, w_cfg=-1, cond=None, eps=1e-3, verbose=False):
    z = torch.randn(shape).to(device)
    x = z * steps[0]
    ones = torch.ones(len(x), 1, 1, 1).float().to(device)
    paired_steps = tqdm(zip(steps[:-1], steps[1:])) if verbose else zip(steps[:-1], steps[1:])
    with ema.average_parameters():
        net.eval()
        with torch.no_grad():
            for i, (t, t_next) in enumerate(paired_steps):
                sigma = ones*t
                sigma_next = ones*t_next
                x0_tilde = (1+w_cfg)*net(x, sigma, cond) - w_cfg*net(x, sigma, torch.zeros_like(cond)) if cond is not None else net(x, sigma)
                d = 1./t * x - 1./t * x0_tilde
                x_next = x + (t_next-t) * d
                if t_next > eps:
                    x0_tilde_next = net(x_next, sigma_next, torch.zeros_like(cond)) if cond is not None else net(x_next, sigma_next)
                    d_ = 1./t_next * x_next - 1./t_next * x0_tilde_next
                    x = x + (t_next-t) * (d + d_)/2
        return x
    

def sampler_longseq(net, ema, steps, gamma, shape, y, H, coeff, coeff_consis, device, T_prime=16, overlap=2, eps=1e-3, verbose=False, label=None, w_cfg=-1, dtype=torch.float32, 
                    enc=None, dec=None, jacob='dps', consis_xt=True, **optim_para):
    # jacob: dps, pgdm or False
    nb = shape[0]
    nf = shape[2]
    nc = shape[1]
    ol = overlap
    b = int(np.ceil((T_prime-ol)/(nf-ol)))      # the number of samples that need to generate
    ns_real = b*(nf-ol)+ol       # exact number of steps generated
    shape_prime = [shape[0]*b, nc, nf, shape[-2], shape[-1]]
    dec = (lambda x: x) if dec is None else dec
    enc = (lambda x: x) if enc is None else enc
    loss_rec = dict(
        loss=[],
        loss_obs=[],
        loss_consis=[],
    )
    def update_loss(**losses):
        for k in losses.keys():
            loss_rec[k].append(losses[k])

    # Global buffer definition
    seq_buffer = None  # We will allocate it on-demand.

    def get_seq_buffer(nb, nc, ns_real, H, W, dtype, device):
        """
        Returns a global buffer of shape (nb, nc, ns_real, H, W).
        If the global buffer is None or the shape/dtype/device doesn't match,
        allocate a new buffer. Otherwise, reuse the same memory block.
        """
        nonlocal seq_buffer  # capture the variable from outer_function
        desired_shape = (nb, nc, ns_real, H, W)
        
        # Allocate if needed
        if (seq_buffer is None
            or seq_buffer.shape != desired_shape
            or seq_buffer.dtype != dtype
            or seq_buffer.device != device):
            
            seq_buffer = torch.empty(desired_shape, dtype=dtype, device=device)
        
        # Zero out only if you need all-zero initial values
        # seq_buffer.zero_()
        
        return seq_buffer

    def batch_to_seq(batch):
        """
        batch: (B*Parallel_B)*C*T*H*W
        
        We'll assume you have global nb, nc, ns_real, b, nf, ol, dtype, and device defined,
        or you pass them in. For simplicity, let's assume they're defined globally here.
        """

        # 1) Reshape
        #    from shape [(nb*b), C, T, H, W] to [nb, b, C, T, H, W]
        batch = rearrange(batch, '(b1 b2) c t h w -> b1 b2 c t h w', b1=nb)
        
        # 2) Get the pre-allocated seq buffer
        seq = get_seq_buffer(nb=nb, nc=nc, ns_real=ns_real, H=batch.shape[-2], W=batch.shape[-1],
                             dtype=batch.dtype, device=batch.device,)

        # 3) Loop to copy slices
        #    seq.shape = [nb, nc, ns_real, H, W]
        #    batch.shape = [nb, b, c, t, h, w]
        for i in range(b):
            i_inv = b - i - 1
            start = i_inv * (nf - ol)
            end   = start + nf
            seq[:, :, start:end] = batch[:, i_inv]

        return seq
    
    def cal_loss_consis(batch):
        batch = rearrange(batch, '(b1 b2) c t h w -> b1 b2 c t h w', b1=nb)
        return ((batch[:, :-1, :, -ol:].detach() - batch[:, 1:, :, :ol])**2).sum()

    z = torch.randn(shape_prime).to(device)     # (B*Parallel_B)*C*T*H*W
    x = z * steps[0]
    ones_shape = [len(x)] + [1]*(len(shape)-1)
    ones = torch.ones(ones_shape).to(dtype).to(device)
    paired_steps = tqdm(zip(steps[:-1], steps[1:])) if verbose else zip(steps[:-1], steps[1:])
    label_ = torch.cat([torch.zeros_like(label), label], dim=0) if w_cfg != -1 else torch.zeros_like(label)

    # optimizer_name = optim_para.pop('optimizer')
    # optimizer = getattr(torch.optim, optimizer_name)([x], **optim_para)
    with ema.average_parameters():
        net.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for i, (t, t_next) in enumerate(paired_steps):
                    sigma = ones*t
                    sigma_next = ones*t_next
                    n = torch.randn_like(x.detach())      # sample stochastic noise
                    if jacob:
                        with torch.enable_grad():
                            x_ = x.clone().requires_grad_(True)
                            if len(label) == len(label_):
                                x0_hat = net(x_, sigma, label)
                            else:
                                x0_hat_f, x0_hat_c = net(torch.cat([x_, x_]), torch.cat([sigma, sigma]), label_).split(len(x), dim=0)
                                x0_hat = (1+w_cfg)*x0_hat_c - w_cfg*x0_hat_f
                            x0_hat_dec = dec(x0_hat)
                            x0_hat_dec_seq = batch_to_seq(x0_hat_dec)
                            if isinstance(H, H_functions):
                                if jacob.lower() == 'pgdm':
                                    mat = H.H_pinv(y.reshape(len(y), -1)-H.H(x0_hat_dec_seq.reshape(len(x0_hat_dec_seq), -1)))      # PiGMD
                                    loss_obs = (mat.detach()*x0_hat_dec_seq.reshape(len(x0_hat_dec_seq), -1)).sum()      # PiGMD
                                else:
                                    loss_obs = ((y.reshape(len(y), -1)-H.H(x0_hat_dec_seq.reshape(len(x0_hat_dec_seq), -1)))**2).sum()
                            else:
                                loss_obs = ((y-H(x0_hat_dec_seq))**2).sum() + coeff_consis * loss_consis
                            loss_consis = cal_loss_consis(x_ if consis_xt else x0_hat_dec)
                            loss = loss_obs + coeff_consis * loss_consis
                            # denoise_mat = ((1-gamma**2)**0.5 * sigma_next/sigma - 1)*(x0_hat-x) - (gamma * sigma_next) * n
                            dx = torch.autograd.grad(loss, x_)[0].detach()
                        d = 1./t * x - 1./t * x0_hat                            
                        d = (sigma_next**2 * (1-gamma**2))**(1./2) * d
                        norm = torch.norm(dx.reshape(len(dx), -1), dim=-1).reshape(ones_shape)
                        # guidance = - dx*(np.sqrt(dx[0].numel())*sigma_next/norm)
                        # guidance = coeff * guidance + (1 - coeff) * (gamma * sigma_next) * n
                        # x = x0_hat + d + guidance
                        radius = np.sqrt(dx[0].numel()) * (gamma * sigma_next)
                        guidance = - radius * dx/norm
                        guidance = coeff * guidance + (1 - coeff) * (gamma * sigma_next) * n
                        norm_g = torch.norm(guidance.reshape(len(guidance), -1), dim=-1).reshape(ones_shape)
                        x = x0_hat + d + radius * guidance/norm_g
                    else:
                        if len(label) == len(label_):
                            x0_hat = net(x, sigma, label)
                        else:
                            x0_hat_f, x0_hat_c = net(torch.cat([x, x]), torch.cat([sigma, sigma]), label_).split(len(x), dim=0)
                            x0_hat = (1+w_cfg)*x0_hat_c - w_cfg*x0_hat_f
                        x0_hat_dec = dec(x0_hat)                        
                        with torch.enable_grad():
                            x0_hat_dec.requires_grad_(True)
                            x0_hat_dec_seq = batch_to_seq(x0_hat_dec)
                            if isinstance(H, H_functions):
                                loss_obs = ((y.reshape(len(y), -1)-H.H(x0_hat_dec_seq.reshape(len(x0_hat_dec_seq), -1)))**2).sum()
                            else:
                                loss_obs = ((y-H(x0_hat_dec_seq))**2).sum()
                            loss_consis = cal_loss_consis(x0_hat_dec)
                            loss = coeff * loss_obs + coeff_consis * loss_consis
                            dx = torch.autograd.grad(loss, x0_hat_dec)[0].detach()
                        x0_hat_dec = x0_hat_dec.detach() - dx
                        x0_hat = enc(x0_hat_dec)
                        d = 1./t * x - 1./t * x0_hat                            
                        d = (sigma_next**2 * (1-gamma**2))**(1./2) * d + (gamma * sigma_next) * n
                        x = x0_hat + d
                    update_loss(loss=loss.item(), loss_obs=loss_obs.item(), loss_consis=loss_consis.item())
                    
        x0_hat_seq = batch_to_seq(x0_hat)
        del seq_buffer
    return x0_hat_seq.detach(), x0_hat_dec_seq.detach(), loss_rec

def bound_func_periodic(field, num_grid=3):    
    flag = False
    if not isinstance(field, torch.Tensor):
        flag = True
        field = torch.from_numpy(field)
    if len(field.shape) == 3:
        pd = (num_grid, )*2
    elif len(field.shape) == 4:
        pd = (num_grid, )*4
    elif len(field.shape) == 5:
        pd = (num_grid, )*6
    field_ = torch.nn.functional.pad(field, pd, mode='circular')
    return field_.numpy() if flag else field_
    
def get_bound_func(bound_type='periodic', num_grid=3):
    if 'periodic' in bound_type.lower():
        return lambda x: bound_func_periodic(x, num_grid)
    elif 'dirichlet' in bound_type.lower():
        return
    elif 'neumann' in bound_type.lower():
        return
    elif 'robin' in bound_type.lower():
        return
    else:
        raise ValueError('Unknown type of boundary condition!')

import torch.nn.functional as F
def get_srdec_func(net, ema, model_opt, model_type='diffusion', device='cuda', **kwargs):
    # input of net: x (shape B (T C) H' W'), **kwargs
    # model_type: diffusion, e2e
    # input of dec_func: (B Parallel_B) C T H W
    if 'scalars' in kwargs.keys():
        scalar, scalar_inv, scalar_dec, scalar_inv_dec = kwargs['scalars']
    else:
        scalar = scalar_inv = scalar_dec = scalar_inv_dec = lambda x: x
    bound_func = get_bound_func(kwargs['bound_type'], num_grid=kwargs['num_bound_padding'])
    trunc_padding = model_opt.scale*kwargs['num_bound_padding']
    def dec_func(x_lr):
        T = x_lr.shape[2]
        x_lr = scalar_dec(scalar_inv(x_lr))
        x_lr = rearrange(x_lr, 'b c t h w -> b (t c) h w')
        x1 = F.interpolate(bound_func(x_lr), scale_factor=model_opt.scale, mode=model_opt.interp_method, align_corners=False)    # , align_corners=False
        if ema is not None:
            with ema.average_parameters():
                x0 = net(x1)
        else:
            x0 = net(x1)
        if kwargs['num_bound_padding'] > 0:
            x0 = x0[..., trunc_padding:-trunc_padding, trunc_padding:-trunc_padding]
        x0 = scalar(scalar_inv_dec(x0))
        return rearrange(x0, 'b (t c) h w -> b c t h w', t=T)
    return dec_func
    
def sampler_heun2nd_dps_featguide(net, ema, steps, shape, y, H, coeff, device, eps=1e-3, verbose=False, label=None, w_cfg=-1, feat_layer=-2, dtype=torch.float32):
    z = torch.randn(shape).to(dtype).to(device)
    x = z * steps[0]
    ones_shape = [len(x)] + [1]*(len(shape)-1)
    ones = torch.ones(ones_shape).to(dtype).to(device)
    paired_steps = tqdm(zip(steps[:-1], steps[1:])) if verbose else zip(steps[:-1], steps[1:])
    with ema.average_parameters():
        net.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for i, (t, t_next) in enumerate(paired_steps):
                    sigma = ones*t
                    sigma_next = ones*t_next

                    # guide the sampling with good-bad pair
                    outs = net(x, sigma, label, out_from_feat=[feat_layer])
                    x0_hat_bad, x0_hat_good = outs
                    x0_hat = (1+w_cfg)*x0_hat_good - w_cfg*x0_hat_bad

                    d = 1./t * x - 1./t * x0_hat.detach()
                    x_next = x + (t_next-t) * d
                    if t_next > eps:
                        with torch.enable_grad():
                            x_ = x_next.clone().requires_grad_(True)
                            outs = net(x_, sigma_next, label, out_from_feat=[feat_layer])
                            x0_hat_next_bad, x0_hat_next_good = outs
                            x0_hat_next = (1+w_cfg)*x0_hat_next_good - w_cfg*x0_hat_next_bad
                            if isinstance(H, H_functions):
                                loss = ((y.reshape(len(y), -1)-H.H(x0_hat_next.reshape(len(x0_hat_next), -1)))**2).sum()
                            else:
                                loss = ((y-H(x0_hat_next))**2).sum()
                            dx = torch.autograd.grad(loss, x_)[0].detach()
                        d_ = 1./t_next * x_next - 1./t_next * x0_hat_next.detach()
                        x = x + (t_next-t) * (d + d_)/2 - coeff * dx
        return x


def sample_ar(sampler, T_prime, T, m=2, dec=None, **sampler_args):
    B           =   int(np.ceil((T_prime-m) / (T - m)))     # number of samples for parallel generation
    T_prime     =   B * (T - m) + m                         # the total length might be a little longer than the given length
    dec = (lambda x: x) if dec is None else dec
    shape = sampler_args['shape']
    mask = torch.zeros(shape)
    mask[:, :, :m] = 1
    coeff_ar = sampler_args.pop('coeff_ar')
    H_seq = Inpainting2(shape[2], shape[-1], mask, device=sampler_args['device'])
    H_seq.set_indices(torch.tensor([0]))
    if 'H' not in sampler_args.keys():
        sampler_args['H'] = H_seq
        sampler_args['y'] = sampler_args['H'].H(sampler_args['y']).reshape(*shape)
    preds = []
    preds_coarse = []
    with torch.no_grad():
        for b in tqdm(range(B)):
            pred = sampler(**sampler_args)      # B C T H W            
            pred_dec = dec(pred)
            if b == 0:
                preds_coarse.append(pred.detach().cpu())
                preds.append(pred_dec.detach().cpu())
            else:
                preds_coarse.append(pred[:, :, m:].detach().cpu())
                preds.append(pred_dec[:, :, m:].detach().cpu())
            sampler_args['H'] = H_seq
            temp = torch.zeros_like(pred)
            temp[:, :, :m] = pred[:, :, -m:]
            sampler_args['y'] = temp
            if 'frame_cond' in sampler_args.keys():
                sampler_args['frame_cond'] = temp[:, :, :m]
            sampler_args['coeff'] = coeff_ar
    return torch.cat(preds_coarse, dim=2), torch.cat(preds, dim=2)
    

def sampler_heun2nd_stoch(net, ema, steps, shape, device, S_noise, S_churn, S_tmin, S_tmax, eps=1e-3, verbose=False, label=None, dtype=torch.float32):
    z = torch.randn(shape).to(dtype).to(device)
    x = z * steps[0]
    ones_shape = [len(x)] + [1]*(len(shape)-1)
    ones = torch.ones(ones_shape).to(dtype).to(device)
    
    paired_steps = tqdm(zip(steps[:-1], steps[1:])) if verbose else zip(steps[:-1], steps[1:])
    with ema.average_parameters():
        net.eval()
        with torch.no_grad():
            for i, (t, t_next) in enumerate(paired_steps):
                n = torch.randn_like(x)*S_noise      # sample stochastic noise
                gamma = min(S_churn/len(steps), 0.414) if S_tmin<=t<=S_tmax else 0
                t_ = t + gamma * t
                sigma = ones*t_
                sigma_next = ones*t_next
                x = x + (t_**2-t**2) * n
                d = 1./t_ * x - 1./t_ * net(x, sigma, label)
                x_next = x + (t_next-t_) * d
                if t_next > eps:
                    d_ = 1./t_next * x_next - 1./t_next * net(x_next, sigma_next, label)
                    x = x + (t_next-t_) * (d + d_)/2
        return x
    
def sampler_mix_dps(net, ema, steps, gamma, shape, y, H, coeff, device, eps=1e-3, verbose=False, label=None, w_cfg=-1, dtype=torch.float32):
    z = torch.randn(shape).to(device)
    x = z * steps[0]
    ones_shape = [len(x)] + [1]*(len(shape)-1)
    ones = torch.ones(ones_shape).to(dtype).to(device)
    paired_steps = tqdm(zip(steps[:-1], steps[1:])) if verbose else zip(steps[:-1], steps[1:])
    label_ = torch.cat([torch.zeros_like(label), label], dim=0) if w_cfg != -1 else torch.zeros_like(label)
    with ema.average_parameters():
        net.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for i, (t, t_next) in enumerate(paired_steps):
                    sigma = ones*t
                    sigma_next = ones*t_next
                    n = torch.randn_like(x)      # sample stochastic noise
                    with torch.enable_grad():
                        x_ = x.clone().requires_grad_(True)
                        if len(label) == len(label_):
                            x0_hat = net(x_, sigma, label)
                        else:
                            x0_hat_f, x0_hat_c = net(torch.cat([x_, x_]), torch.cat([sigma, sigma]), label_).split(len(x), dim=0)
                            x0_hat = (1+w_cfg)*x0_hat_c - w_cfg*x0_hat_f
                        if isinstance(H, H_functions):
                            loss = ((y.reshape(len(y), -1)-H.H(x0_hat.reshape(len(x0_hat), -1)))**2).sum()
                        else:
                            loss = ((y-H(x0_hat))**2).sum()
                        dx = torch.autograd.grad(loss, x_)[0].detach()
                    d = 1./t * x - 1./t * x0_hat                            
                    d = (sigma_next**2 * (1-gamma**2))**(1./2) * d
                    norm = torch.norm(dx.reshape(len(dx), -1), dim=-1).reshape(ones_shape)
                    # guidance = - dx*(np.sqrt(dx[0].numel())*sigma_next/norm)
                    # guidance = coeff * guidance + (1 - coeff) * (gamma * sigma_next) * n
                    # x = x0_hat + d + guidance
                    radius = np.sqrt(dx[0].numel()) * (gamma * sigma_next)
                    guidance = - radius * dx/norm
                    guidance = coeff * guidance + (1 - coeff) * (gamma * sigma_next) * n
                    norm_g = torch.norm(guidance.reshape(len(guidance), -1), dim=-1).reshape(ones_shape)
                    x = x0_hat + d + radius * guidance/norm_g
        return x

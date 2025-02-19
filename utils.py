import numpy as np
from scipy.interpolate import interp2d
# from scipy.integrate import simps
from scipy.ndimage import rotate
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import random
from einops import rearrange
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def log_stats(stats, writer, epoch):
    for k, v in stats.items():
        writer.add_scalar(k, v, epoch)


def kes_from_vorticity(w, dx, dy, num_bins=30):
    """
    Calculates the kinetic energy spectrum of a 2D velocity field.
    Args:
        w (ndarray): 2D array containing the y-component of the vorticity field.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
    Returns:
        k (ndarray): 1D array containing the wavenumber values.
        energy_spectrum (ndarray): 1D array containing the kinetic energy spectrum values.
    """
    # Calculate the wavenumber values
    w_hat = np.fft.fft2(w)
    Nx, Ny = w.shape
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k = np.sqrt(kx ** 2 + ky ** 2)
    # Calculate the kinetic energy spectrum
    energy_spectrum = 0.5 * (np.abs(w_hat) ** 2)
    energy_spectrum = energy_spectrum.flatten()
    k = k.flatten()
    # Bin the kinetic energy spectrum values by wavenumber
    # num_bins = int(np.ceil(np.max(k) / (2 * np.pi / dx)))
    num_bins = num_bins
    bin_edges = np.linspace(0, np.max(k), num_bins + 1)
    digitized = np.digitize(k, bin_edges)
    bin_means = np.zeros(num_bins)
    for i in range(num_bins):
        bin_means[i] = np.mean(energy_spectrum[digitized == i])
    return bin_edges[:-1], bin_means


def kes_plot(vors, dx, dy, start, end, num_bins=30, desc=None, is_plot=False, savefig=None):
    E_mean = [0 for _ in range(len(vors))]
    k, _ = kes_from_vorticity(vors[0][0], dx=dx, dy=dy, num_bins=num_bins)
    for t in range(start, end):
        for i, vor in enumerate(vors):
            _, E = kes_from_vorticity(vor[t], dx=dx, dy=dy, num_bins=num_bins)
            E_mean[i] += E
    total = end - start
    E_mean = [i/total for i in E_mean]
    E_mean_k5 = [i*k**5 for i in E_mean]
    if is_plot:
        fig1, ax1 = plt.subplots()
        for i in range(len(vors)):
            ax1.loglog(k, E_mean[i], label=desc[i])
        plt.xlabel('Wavenumber')
        plt.ylabel('Kinetic Energy Spectrum')
        plt.legend()
        plt.show()
        if savefig is not None:
            fig1.savefig('./results/energy_spectrum_time_series.pdf')
            fig1.savefig('./results/energy_spectrum_time_series.png')

        fig2, ax2 = plt.subplots()
        for i in range(len(vors)):
            ax2.loglog(k, E_mean_k5[i], label=desc[i])
        plt.xlabel('Wavenumber')
        plt.ylabel('Kinetic Energy Spectrum')
        plt.legend()
        plt.show()
        if savefig is not None:
            fig2.savefig('./results/energy_spectrumk5_time_series.pdf')
            fig2.savefig('./results/energy_spectrumk5_time_series.png')
    return k, E_mean, E_mean_k5


def nearest_interpolation_batched(batch_matrix):
    """
    Interpolates the zero entries of a batched matrix using nearest-neighbor interpolation.
    Each matrix in the batch has shape (H, W) and non-zero entries are considered known values.
    
    Parameters:
        batch_matrix (np.ndarray): A 3D numpy array of shape (B, H, W) where non-zeros are known values.
    
    Returns:
        np.ndarray: A new array with the zeros filled in for each matrix in the batch.
    """
    # Ensure input is a numpy array
    batch_matrix = np.array(batch_matrix)
    B, H, W = batch_matrix.shape
    
    # Output array to store the interpolated matrices
    interpolated_batch = np.empty_like(batch_matrix)
    
    # Process each matrix in the batch individually
    for i in range(B):
        matrix = batch_matrix[i]
        
        # Create a mask: True for unknown (zero) entries, False for known (non-zero)
        unknown_mask = (matrix == 0)
        
        # Compute the Euclidean distance transform.
        # For each pixel, this returns the indices of the nearest pixel where unknown_mask is False.
        # (i.e., the nearest known value)
        distance, indices = distance_transform_edt(unknown_mask, return_distances=True, return_indices=True)
        
        # Create a copy of the matrix to fill in interpolated values
        interpolated = matrix.copy()
        
        # Use the indices from the distance transform to assign the nearest known value
        interpolated[unknown_mask] = matrix[indices[0][unknown_mask], indices[1][unknown_mask]]
        
        # Save the interpolated matrix in the batch output
        interpolated_batch[i] = interpolated
    
    return interpolated_batch


import einops
import torch.nn as nn

def vorticity_to_velocity_spectral(vorticity, Lx=2*np.pi, Ly=2*np.pi):
    """
    Compute velocity components from vorticity using spectral methods.

    Parameters:
    - vorticity: numpy array of shape (B, H, W)
    - Lx, Ly: Domain size in x and y directions (default: 2*pi for periodicity)

    Returns:
    - velocity: numpy array of shape (B, 2, H, W) where velocity[:,0,:,:] = u and velocity[:,1,:,:] = v
    """
    B, H, W = vorticity.shape
    # Compute the wave numbers
    kx = np.fft.fftfreq(W, d=Lx/W) * 2 * np.pi  # Shape (W,)
    ky = np.fft.fftfreq(H, d=Ly/H) * 2 * np.pi  # Shape (H,)
    kx[0] = 1e-20  # To avoid division by zero
    ky[0] = 1e-20

    # Create 2D wave number grids
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')  # Shape (W, H)
    kx_grid = kx_grid.T  # Shape (H, W)
    ky_grid = ky_grid.T  # Shape (H, W)
    k_squared = kx_grid**2 + ky_grid**2  # Shape (H, W)
    k_squared[0, 0] = 1e-20  # Avoid division by zero at the zero frequency

    # Perform FFT on vorticity
    omega_hat = np.fft.fft2(vorticity, axes=(-2, -1))  # Shape (B, H, W)

    # Compute streamfunction in Fourier space
    psi_hat = omega_hat / (-k_squared)  # Δψ = -ω => ψ_hat = ω_hat / (-k_squared)
    psi_hat[:,0,0] = 0.0  # Set the mean of the streamfunction to zero

    # Compute velocity components in Fourier space
    u_hat = 1j * ky_grid * psi_hat  # u = dψ/dy
    v_hat = -1j * kx_grid * psi_hat  # v = -dψ/dx

    # Inverse FFT to get velocity in physical space
    u = np.fft.ifft2(u_hat, axes=(-2, -1)).real  # Shape (B, H, W)
    v = np.fft.ifft2(v_hat, axes=(-2, -1)).real  # Shape (B, H, W)

    # Stack velocity components
    velocity = np.stack((u, v), axis=1)  # Shape (B, 2, H, W)

    return velocity

def voriticity_residual(w, re=1000.0, dt=1/32, source=True, drag=True):
    # w [b t h w]
    batchsize = w.size(0)
    # w = w.clone()
    # w.requires_grad_(True)
    nx = w.size(2)
    ny = w.size(3)
    device = w.device

    w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(N, 1).repeat(1, N).reshape(1,1,N,N)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).\
        reshape(1, N).repeat(N, 1).reshape(1,1,N,N)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[..., 0, 0] = 1.0
    psi_h = w_h / lap

    u_h = 1j * k_y * psi_h
    v_h = -1j * k_x * psi_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    u = torch.fft.irfft2(u_h[..., :, :k_max + 1], dim=[2, 3])
    v = torch.fft.irfft2(v_h[..., :, :k_max + 1], dim=[2, 3])
    wx = torch.fft.irfft2(wx_h[..., :, :k_max + 1], dim=[2, 3])
    wy = torch.fft.irfft2(wy_h[..., :, :k_max + 1], dim=[2, 3])
    wlap = torch.fft.irfft2(wlap_h[..., :, :k_max + 1], dim=[2, 3])
    advection = u*wx + v*wy

    wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

    # establish forcing term
    x = torch.linspace(0, 2*np.pi, nx + 1, device=device)
    x = x[0:-1]
    X, Y = torch.meshgrid(x, x)
    f = -4*torch.cos(4*Y)

    residual = wt + (advection - (1.0 / re) * wlap)
    if drag:
        residual += 0.1*w[:, 1:-1]
    if source:
        residual -= f
    residual_loss = (residual**2).mean()
    # dw = torch.autograd.grad(residual_loss, w)[0]
    return residual, residual_loss


def cal_rmse(gt, pred, normalize=True, reduct='sum'):
    # reduct = 'sum' or 'mean' etc.
    lib_name = np if isinstance(gt[0], np.ndarray) else torch
    reduct_fn = getattr(lib_name, reduct)
    rmse = []
    for a, b in zip(gt, pred):
        if normalize:
            coeff = 1./lib_name.sqrt(reduct_fn(a**2))
        else:
            coeff = 1.
        rmse.append(coeff*lib_name.sqrt(reduct_fn((a-b)**2)))
    return np.array(rmse) if isinstance(a, np.ndarray) else torch.tensor(rmse)


def batch_nrmse(xs, ref, reduct=True, device='cpu', batch_size=128, verbose=True):
    errors = []
    xs = tqdm(xs) if verbose else xs
    ref = torch.from_numpy(ref).float() if isinstance(ref, np.ndarray) else ref.float()
    for x in xs:
        x_ = torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x.float()
        dataset = torch.utils.data.TensorDataset(x_, ref)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        e = []
        for d, r in dataloader:
            d = d.to(device)
            r = r.to(device)
            h = cal_rmse(r, d, normalize=True)
            e.append(h)
        e = torch.cat(e).detach().cpu().numpy()
        if reduct:
            e = e.mean()
        errors.append(e)
    return errors


def cal_correlation(gt, pred, standardize=True, reduct='sum'):
    # standardize: whether to substract mean value of input data
    lib_name = np if isinstance(gt[0], np.ndarray) else torch
    reduct_fn = getattr(lib_name, reduct)
    cossim = []
    for a, b in zip(gt, pred):
        if standardize:
            a_mean = lib_name.mean(a)
            b_mean = lib_name.mean(b)
        else:
            a_mean = 0.
            b_mean = 0.
        a_norm = lib_name.sqrt(reduct_fn(a**2))
        b_norm = lib_name.sqrt(reduct_fn(b**2))
        cossim.append(reduct_fn((a-a_mean).reshape(-1)*(b-b_mean).reshape(-1))/(a_norm*b_norm))
    return np.array(cossim) if isinstance(a, np.ndarray) else torch.tensor(cossim)


def batch_corr(xs, ref, reduct=True, device='cpu', batch_size=128, verbose=True):
    corr = []
    xs = tqdm(xs) if verbose else xs
    ref = torch.from_numpy(ref).float() if isinstance(ref, np.ndarray) else ref.float()
    for x in xs:
        x_ = torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x.float()
        dataset = torch.utils.data.TensorDataset(x_, ref)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # dataloader = tqdm(dataloader, desc='Calculating nER: ') if verbose else dataloader
        c = []
        for d, r in dataloader:
            d = d[0].to(device)
            r = r[0].to(device)
            h = cal_correlation(r, d, standardize=False, reduct='sum')
            c.append(h)
        c = torch.cat(c).detach().cpu().numpy()
        if reduct:
            c = c.mean()
        corr.append(c)
    return corr

def to_device(xs, device='cpu', dtype='float32'):
    torch_dtype = getattr(torch, dtype)
    if isinstance(xs, list) or isinstance(xs, tuple):
        return [torch.tensor(x).type(torch_dtype).to(device) for x in xs]
    else:
        return torch.tensor(xs).type(torch_dtype).to(device)
    
def to_ndarray(xs, ):
    if isinstance(xs, list) or isinstance(xs, tuple):
        return [x.detach().cpu().numpy() for x in xs]
    else:
        return xs.detach().cpu().numpy()

def mask_gen(input_shape, mask_ratio=0.5, seed=None):
    m = np.ones(input_shape)

    indices = [np.arange(i) for i in input_shape]
    I = np.meshgrid(*indices, indexing='ij')
    indices = np.array([index.reshape(-1) for index in I]).transpose(1, 0)
    num_pixel = len(indices)
    if seed is None:
        i_indices = np.random.choice(num_pixel, int(mask_ratio*num_pixel), replace=False)
    else:
        rng = np.random.RandomState(seed)
        i_indices = rng.choice(num_pixel, int(mask_ratio * num_pixel), replace=False)
    indices = indices[i_indices]
    m[tuple(indices.transpose(1, 0))] = 0
    m = m.astype(bool)
    return m

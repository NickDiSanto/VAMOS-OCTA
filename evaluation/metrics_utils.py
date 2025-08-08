import numpy as np
import torch
from scipy.ndimage import uniform_filter, sobel
import cv2


def compute_local_ncc(a, b, window_size):
    """Compute windowed NCC between two 2D arrays."""
    eps = 1e-8
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    a_mean = uniform_filter(a, window_size)
    b_mean = uniform_filter(b, window_size)

    a2_mean = uniform_filter(a * a, window_size)
    b2_mean = uniform_filter(b * b, window_size)
    ab_mean = uniform_filter(a * b, window_size)

    a_var = a2_mean - a_mean * a_mean
    b_var = b2_mean - b_mean * b_mean
    ab_cov = ab_mean - a_mean * b_mean

    denominator = np.sqrt(a_var * b_var) + eps
    local_ncc = ab_cov / denominator

    return local_ncc  # shape: same as input, values between -1 and 1


def compute_local_3d_ncc(vol1, vol2, window_size=(7, 7, 3)):
    """Compute local 3D NCC over a volume using a sliding window."""
    eps = 1e-8
    vol1 = vol1.astype(np.float32)
    vol2 = vol2.astype(np.float32)

    # Means
    mean1 = uniform_filter(vol1, window_size)
    mean2 = uniform_filter(vol2, window_size)

    # Variances and Covariance
    vol1_sq = uniform_filter(vol1 ** 2, window_size)
    vol2_sq = uniform_filter(vol2 ** 2, window_size)
    vol1_vol2 = uniform_filter(vol1 * vol2, window_size)

    var1 = vol1_sq - mean1 ** 2
    var2 = vol2_sq - mean2 ** 2
    cov12 = vol1_vol2 - mean1 * mean2

    denom = np.sqrt(var1 * var2) + eps
    ncc = cov12 / denom

    return ncc


def gradient_magnitude(slice_2d):
    dx = np.diff(slice_2d, axis=0, append=slice_2d[-1:, :])
    dy = np.diff(slice_2d, axis=1, append=slice_2d[:, -1:])
    return np.sqrt(dx**2 + dy**2)


def to_lpips_tensor(x):
    """Convert 2D [0,1] float32 np.ndarray to (1,3,H,W) tensor in [-1,1]"""
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0, 1)
    x = torch.tensor(x, dtype=torch.float32)
    x = x.unsqueeze(0).repeat(1, 3, 1, 1)  # (1,3,H,W)
    x = 2.0 * x - 1.0
    return x


def edge_strength(im):
    dx = sobel(im, axis=0)
    dy = sobel(im, axis=1)
    return np.sqrt(dx**2 + dy**2).sum()


def laplacian_blur_score(image):
    """Compute variance of the Laplacian — lower means blurrier."""
    image = image.astype(np.float32)
    lap = cv2.Laplacian(image, cv2.CV_32F)
    return lap.var()


def compute_psnr(gt, pred, data_range=1.0):
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((data_range ** 2) / mse)


def gradient_magnitude(img):
    gx = np.diff(img, axis=1, append=img[:, -1:])
    gy = np.diff(img, axis=0, append=img[-1:, :])
    return np.sqrt(gx**2 + gy**2)
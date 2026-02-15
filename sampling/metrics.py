"""Image quality metrics for FSampler research."""
import torch
import numpy as np
from PIL import Image


def compute_metrics(image1, image2):
    """
    Compute SSIM, RMSE, and MAE between two images.

    Args:
        image1: First image (torch.Tensor, numpy array, or PIL.Image)
        image2: Second image (torch.Tensor, numpy array, or PIL.Image)

    Returns:
        dict: {"ssim": float, "rmse": float, "mae": float}
    """
    # Convert both images to numpy arrays [H, W, C] in range [0, 1]
    img1_np = _to_numpy(image1)
    img2_np = _to_numpy(image2)

    # Ensure same shape
    if img1_np.shape != img2_np.shape:
        raise ValueError(f"Images must have same shape. Got {img1_np.shape} and {img2_np.shape}")

    # Compute metrics
    ssim = compute_ssim(img1_np, img2_np)
    rmse = compute_rmse(img1_np, img2_np)
    mae = compute_mae(img1_np, img2_np)

    return {
        "ssim": float(ssim),
        "rmse": float(rmse),
        "mae": float(mae)
    }


def _to_numpy(image):
    """Convert image to numpy array [H, W, C] in range [0, 1]."""
    if isinstance(image, torch.Tensor):
        # Handle torch tensor [B, H, W, C] or [H, W, C] or [B, C, H, W] or [C, H, W]
        img = image.detach().cpu()

        # Remove batch dimension if present
        if img.dim() == 4:
            img = img[0]

        # Convert channels-first to channels-last if needed
        if img.dim() == 3 and img.shape[0] in [1, 3, 4]:
            img = img.permute(1, 2, 0)

        img = img.numpy()
    elif isinstance(image, Image.Image):
        # PIL Image to numpy
        img = np.array(image).astype(np.float32) / 255.0
    elif isinstance(image, np.ndarray):
        img = image.astype(np.float32)
        # If in range [0, 255], normalize to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Ensure 3D array [H, W, C]
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    return img


def compute_mae(img1, img2):
    """Mean Absolute Error (average absolute pixel difference)."""
    return np.mean(np.abs(img1 - img2))


def compute_rmse(img1, img2):
    """Root Mean Squared Error (root-mean-square pixel difference)."""
    mse = np.mean((img1 - img2) ** 2)
    return np.sqrt(mse)


def compute_ssim(img1, img2, window_size=11, k1=0.01, k2=0.03):
    """
    Structural Similarity Index (SSIM).

    Reference implementation based on the original SSIM paper:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
    Image quality assessment: from error visibility to structural similarity.
    IEEE transactions on image processing, 13(4), 600-612.
    """
    # Convert to grayscale if color image
    if img1.ndim == 3 and img1.shape[2] > 1:
        # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        img1 = 0.299 * img1[:, :, 0] + 0.587 * img1[:, :, 1] + 0.114 * img1[:, :, 2]
        img2 = 0.299 * img2[:, :, 0] + 0.587 * img2[:, :, 1] + 0.114 * img2[:, :, 2]
    else:
        img1 = img1.squeeze()
        img2 = img2.squeeze()

    # Constants
    C1 = (k1 * 1.0) ** 2  # (k1*L)^2, L=1 for normalized images
    C2 = (k2 * 1.0) ** 2  # (k2*L)^2

    # Create Gaussian window
    window = _gaussian_window(window_size, 1.5)

    # Compute local means
    mu1 = _convolve(img1, window)
    mu2 = _convolve(img2, window)

    # Compute local variances and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _convolve(img1 ** 2, window) - mu1_sq
    sigma2_sq = _convolve(img2 ** 2, window) - mu2_sq
    sigma12 = _convolve(img1 * img2, window) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator

    # Return mean SSIM
    return np.mean(ssim_map)


def _gaussian_window(size, sigma):
    """Create a 2D Gaussian window."""
    coords = np.arange(size) - size // 2
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return np.outer(g, g)


def _convolve(image, window):
    """Apply 2D convolution with the given window."""
    from scipy.ndimage import convolve
    return convolve(image, window, mode='constant', cval=0.0)

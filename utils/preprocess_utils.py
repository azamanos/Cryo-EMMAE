import cv2
import torch
import numpy as np
from PIL import Image
from scipy.signal import gaussian
from utils.fft_utils import ht2_center,iht2_center
from utils.utils import normalize_array, create_2d_circular_mask

def micrograph_mean_and_sigma(micrograph, mask, device='cpu'):
    """
    Calculate the mean and standard deviation of a micrograph using FFT-based convolution.

    Parameters
    ----------
    micrograph : numpy.ndarray or torch.Tensor
        Input micrograph image.
    mask : numpy.ndarray or torch.Tensor
        Mask to apply for calculating statistics.
    device : str, optional
        Device to perform computation on, either 'cpu' or 'cuda'.

    Returns
    -------
    mean : numpy.ndarray
        Mean values of the micrograph.
    sigma : numpy.ndarray
        Standard deviation values of the micrograph.
    """
    if not isinstance(micrograph, torch.Tensor):
        micrograph = torch.from_numpy(micrograph)
    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(mask)
    micrograph, mask = micrograph.to(device), mask.to(device)

    fft_micrograph = torch.fft.fft2(micrograph)
    fft_mask = torch.fft.fft2(mask)
    conj_fft_mask = torch.conj(fft_mask)
    fft_micrograph_squared = torch.fft.fft2(micrograph ** 2)

    mean = torch.abs(torch.fft.fftshift(torch.fft.ifft2(fft_micrograph * conj_fft_mask))) / torch.sum(mask)
    variance = (torch.abs(torch.fft.fftshift(torch.fft.ifft2(fft_micrograph_squared * conj_fft_mask))) / torch.sum(mask)) - mean ** 2
    sigma = variance.sqrt()

    return np.array(mean.cpu()), np.array(sigma.cpu())

def relion_normalization(micrograph, normalization_filter, device='cpu'):
    """
    Apply Relion normalization to a micrograph.

    Parameters
    ----------
    micrograph : numpy.ndarray
        Input micrograph image.
    normalization_filter : numpy.ndarray
        Filter for normalization.
    device : str, optional
        Device to perform computation on, either 'cpu' or 'cuda'.

    Returns
    -------
    numpy.ndarray
        Normalized micrograph.
    """
    normalized_micrograph = normalize_array(micrograph)
    mean, sigma = micrograph_mean_and_sigma(normalized_micrograph, normalization_filter, device)
    return normalize_array((normalized_micrograph - mean) / sigma)


def gaussian_kernel(kernel_size=3):
    """
    Generate a Gaussian kernel.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the kernel, by default 3.

    Returns
    -------
    numpy.ndarray
        Gaussian kernel of shape (kernel_size, kernel_size).
    """
    # Create a 1D Gaussian distribution
    gauss_1d = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    # Create a 2D Gaussian kernel by outer product
    gauss_2d = np.dot(gauss_1d, gauss_1d.transpose())
    # Normalize the kernel so that the sum of all elements equals 1
    gauss_2d /= np.sum(gauss_2d)
    return gauss_2d


kernel = gaussian_kernel(kernel_size = 9)
kernel_torch = torch.from_numpy(kernel)


def wiener_filter_torch(image, kernel, K):
    """
    Apply Wiener filter to an image using Torch.

    Parameters
    ----------
    image : torch.Tensor
        Input image.
    kernel : torch.Tensor
        Convolution kernel.
    K : float
        Noise-to-signal power ratio.

    Returns
    -------
    torch.Tensor
        Filtered image.
    """
    kernel /= torch.sum(kernel)
    fft_image = torch.fft.fft2(image)
    fft_kernel = torch.fft.fft2(kernel, s=image.shape)
    wiener_kernel = torch.conj(fft_kernel) / (torch.abs(fft_kernel) ** 2 + K)
    filtered_image = torch.abs(torch.fft.ifft2(fft_image * wiener_kernel))
    return filtered_image

def transform(image):
    """
    Transform image to uint8 format after scaling its values to [0, 255].

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        Transformed image in uint8 format.
    """
    min_val, max_val = image.min(), image.max()
    scaled_image = (image - min_val) / (max_val - min_val) * 255
    return scaled_image.astype(np.uint8)

def clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        Image after CLAHE.
    """
    clahe_processor = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    equalized_image = clahe_processor.apply(transform(image))
    return equalized_image

def guided_filter(input_image, guidance_image, radius=20, epsilon=0.1):
    """
    Apply guided filter to an image.

    Parameters
    ----------
    input_image : numpy.ndarray
        Input image to be filtered.
    guidance_image : numpy.ndarray
        Guidance image for filtering.
    radius : int, optional
        Radius of the guided filter.
    epsilon : float, optional
        Regularization parameter to avoid division by zero.

    Returns
    -------
    numpy.ndarray
        Filtered image.
    """
    input_image = input_image.astype(np.float32) / 255.0
    guidance_image = guidance_image.astype(np.float32) / 255.0

    mean_guidance = cv2.boxFilter(guidance_image, -1, (radius, radius))
    mean_input = cv2.boxFilter(input_image, -1, (radius, radius))
    mean_guidance_input = cv2.boxFilter(guidance_image * input_image, -1, (radius, radius))
    covariance_guidance_input = mean_guidance_input - mean_guidance * mean_input
    mean_guidance_sq = cv2.boxFilter(guidance_image * guidance_image, -1, (radius, radius))
    variance_guidance = mean_guidance_sq - mean_guidance * mean_guidance

    a = covariance_guidance_input / (variance_guidance + epsilon)
    b = mean_input - a * mean_guidance
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    output_image = mean_a * guidance_image + mean_b

    return transform(output_image)

def denoise_jpg_image_reduced_torch(image):
    """
    Apply denoising steps to a JPEG image using Torch.

    Parameters
    ----------
    image : numpy.ndarray
        Input JPEG image.

    Returns
    -------
    numpy.ndarray
        Denoised image.
    """
    kernel = create_2d_circular_mask(image.shape, diameter=5)  # Example kernel creation
    weiner_filtered_image = np.array(wiener_filter_torch(torch.from_numpy(image), kernel_torch, K=30))
    clahe_image = clahe(weiner_filtered_image)
    guided_filter_image = guided_filter(clahe_image, weiner_filtered_image)
    return guided_filter_image

def relion_and_contrast_preprocess(micrograph, particle_diameter, output_path, image_name):
    """
    Apply Relion normalization and contrast enhancement to a micrograph, then save the result.

    Parameters
    ----------
    micrograph : numpy.ndarray
        Input micrograph image.
    particle_diameter : int
        Diameter of particles for creating the normalization filter.
    output_path : str
        Path to save the processed image.
    image_name : str
        Name of the processed image file.

    Returns
    -------
    None
    """
    padding_x, padding_y = (np.array(micrograph.shape) - particle_diameter) // 2
    normalization_filter = np.pad(create_2d_circular_mask((particle_diameter, particle_diameter), particle_diameter, True).astype(int),((padding_x, padding_x), (padding_y, padding_y)))
    normalized_micrograph = relion_normalization(micrograph, normalization_filter)
    enhanced_micrograph = denoise_jpg_image_reduced_torch(normalized_micrograph)
    resized_micrograph = Image.fromarray(normalize_array(np.array(enhanced_micrograph)) * 255).convert('L')
    resized_micrograph.save(f'{output_path}{image_name}.png')
    return

import cv2
import torch
import numpy as np
from utils.utils import normalize_array
from scipy.signal import gaussian

def micrograph_mean_and_sigma(x,mo,device='cpu'):
    if type(x) != torch.Tensor:
        x = torch.from_numpy(x)
    if type(mo) != torch.Tensor:
        mo = torch.from_numpy(mo)
    x, mo = x.to(device), mo.to(device)
    fx = torch.fft.fft2(x)
    fmo = torch.fft.fft2(mo)
    cfmo = torch.conj(fmo)
    fx_2 = torch.fft.fft2(x**2)
    mean = torch.abs(torch.fft.fftshift(torch.fft.ifft2(fx*cfmo)))/torch.sum(mo)
    sigma = ((torch.abs(torch.fft.fftshift(torch.fft.ifft2(fx_2*cfmo)))/torch.sum(mo))-mean**2)**0.5
    return np.array(mean.cpu()), np.array(sigma.cpu())

def relion_normalization(micrograph,normalization_filter,device='cpu'):
    m_norm = normalize_array(micrograph)
    mean, sigma = micrograph_mean_and_sigma(m_norm,normalization_filter,device)
    return normalize_array((m_norm-mean)/sigma)

############

def transform(image):
    i_min = image.min()
    i_max = image.max()

    image = ((image - i_min)/(i_max - i_min)) * 255
    return image.astype(np.uint8)

def gaussian_kernel(kernel_size = 3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h

def wiener_filter_torch(img, kernel, K):
    kernel /= torch.sum(kernel)
    dummy = torch.clone(img)
    dummy = torch.fft.fft2(dummy)
    kernel = torch.fft.fft2(kernel, s = img.shape)
    kernel = torch.conj(kernel) / (torch.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = torch.abs(torch.fft.ifft2(dummy))
    return dummy

def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))

    # Apply CLAHE to the image
    img_equalized = clahe.apply(transform(image))
    return img_equalized

def guided_filter(input_image, guidance_image, radius=20, epsilon=0.1):
    # Convert images to float32
    input_image = input_image.astype(np.float32) / 255.0
    guidance_image = guidance_image.astype(np.float32) / 255.0

    # Compute mean values of the guidance image and input image
    mean_guidance = cv2.boxFilter(guidance_image, -1, (radius, radius))
    mean_input = cv2.boxFilter(input_image, -1, (radius, radius))

    # Compute correlation and covariance of the guidance and input images
    mean_guidance_input = cv2.boxFilter(guidance_image * input_image, -1, (radius, radius))
    covariance_guidance_input = mean_guidance_input - mean_guidance * mean_input

    # Compute squared mean of the guidance image
    mean_guidance_sq = cv2.boxFilter(guidance_image * guidance_image, -1, (radius, radius))
    variance_guidance = mean_guidance_sq - mean_guidance * mean_guidance

    # Compute weights and mean of the weights
    a = covariance_guidance_input / (variance_guidance + epsilon)
    b = mean_input - a * mean_guidance
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    # Compute the filtered image
    output_image = mean_a * guidance_image + mean_b

    return transform(output_image)

kernel = gaussian_kernel(kernel_size = 3)
kernel_torch = torch.from_numpy(kernel)

def denoise_jpg_image_reduced_torch(image):
    #normalized_image = standard_scaler(np.array(image))
    #contrast_enhanced_image = contrast_enhancement(normalized_image)
    weiner_filtered_image = np.array(wiener_filter_torch(torch.from_numpy(image), kernel_torch, K = 30))
    clahe_image = clahe(weiner_filtered_image)
    guided_filter_image = guided_filter(clahe_image, weiner_filtered_image)
    return normalize_array(guided_filter_image)

def create_hamming_image(image_shape,radius):
    hamx = np.hamming(image_shape[0])[:,None] # 1D hamming
    hamy = np.hamming(image_shape[1])[:,None] # 1D hamming
    return np.sqrt(np.dot(hamx, hamy.T)) ** radius # expand to 2D hamming

def fft2_center(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=(-1, -2))), axes=(-1, -2))

def ht2_center(img):
    f = fft2_center(img)
    return f.real - f.imag

def iht2_center(img):
    img = fft2_center(img)
    img /= (img.shape[-1] * img.shape[-2])
    return img.real - img.imag

def low_pass_filter_image(image,radius):
    ham2d = create_hamming_image(image.shape,radius)
    #image to hartley space
    h_complex = ht2_center(image)
    #apply ham2d
    h_filtered = ham2d * h_complex
    return iht2_center(h_filtered)

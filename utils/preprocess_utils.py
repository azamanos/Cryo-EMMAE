import cv2
import torch
import struct
import numpy as np
from PIL import Image
from scipy.signal import gaussian
from utils.fft_utils import ht2_center,iht2_center
from utils.utils import normalize_array, create_2d_circular_mask

class MRC(object):
    """Load MRC file"""
    def __init__(self, filename):
        if filename.split('.')[-1] == 'gz':
            fin = gzip.open(filename, 'rb')
        else:
            fin = open(filename, 'rb')
        #with open(filename, 'rb') as fin:
        MRCdata = fin.read()
        self.nx = struct.unpack_from('<i', MRCdata, 0)[0]
        self.ny = struct.unpack_from('<i', MRCdata, 4)[0]
        self.nz = struct.unpack_from('<i', MRCdata, 8)[0]

        self.mode = struct.unpack_from('<i', MRCdata, 12)[0]
        #Starting point of sub image
        self.nxstart = struct.unpack_from('<i', MRCdata, 16)[0]
        self.nystart = struct.unpack_from('<i', MRCdata, 20)[0]
        self.nzstart = struct.unpack_from('<i', MRCdata, 24)[0]
        #Grid size in X, Y, and Z
        self.mx = struct.unpack_from('<i', MRCdata, 28)[0]
        self.my = struct.unpack_from('<i', MRCdata, 32)[0]
        self.mz = struct.unpack_from('<i', MRCdata, 36)[0]
        #Cell size; pixel spacing = xlen/mx, ylen/my, zlen/mz
        self.xlen = struct.unpack_from('<f', MRCdata, 40)[0]
        self.ylen = struct.unpack_from('<f', MRCdata, 44)[0]
        self.zlen = struct.unpack_from('<f', MRCdata, 48)[0]
        #self.voxel = round(self.xlen/self.mx, 3)
        #cell angles
        self.alpha = struct.unpack_from('<f', MRCdata, 52)[0]
        self.beta = struct.unpack_from('<f', MRCdata, 56)[0]
        self.gamma = struct.unpack_from('<f', MRCdata, 60)[0]
        #MAP C R S 	axis corresp to cols, rows, sections respectively (1,2,3 for X,Y,Z)
        self.mapc = struct.unpack_from('<i', MRCdata, 64)[0]
        self.mapr = struct.unpack_from('<i', MRCdata, 68)[0]
        self.maps = struct.unpack_from('<i', MRCdata, 72)[0]
        #DMIN, DMAX, DMEAN, RMS
        self.dmin = struct.unpack_from('<f', MRCdata, 76)[0]
        self.dmax = struct.unpack_from('<f', MRCdata, 80)[0]
        self.dmean = struct.unpack_from('<f', MRCdata, 84)[0]
        self.rms = struct.unpack_from('<f', MRCdata, 216)[0]
        #Origin of image
        self.xorg = struct.unpack_from('<f', MRCdata, 196)[0]
        self.yorg = struct.unpack_from('<f', MRCdata, 200)[0]
        self.zorg = struct.unpack_from('<f', MRCdata, 204)[0]

        ind = self.nx*self.ny*self.nz*4
        self.data = np.frombuffer(MRCdata[-ind:], dtype=np.dtype(np.float32)).reshape((self.nx,self.ny,self.nz),order='F')
        fin.close()
        #fin.seek(1024, os.SEEK_SET)
        #self.data = np.frombuffer(fin.read(), dtype=np.dtype(np.float32)).reshape((self.nx,self.ny,self.nz),order='F')
        #self.data = np.fromfile(file=fin, dtype=np.dtype(np.float32)).reshape((self.nx,self.ny,self.nz),order='F')

def write_mrc(rho, nxstart=0,nystart=0,nzstart=0, mapc=1,mapr=2,maps=3, xorg=0,yorg=0,zorg=0, alpha=90., beta=90., gamma=90., voxel_size=1.000, filename="map.mrc"):
    """Write an MRC formatted electron density map.
       See here: http://www2.mrc-lmb.cam.ac.uk/research/locally-developed-software/image-processing-software/#image
    """
    xs, ys, zs = rho.shape
    if type(voxel_size) is list or type(voxel_size) is tuple:
        a, b, c = xs*voxel_size[0], ys*voxel_size[1], zs*voxel_size[2]
    else:
        a, b, c = xs*voxel_size, ys*voxel_size, zs*voxel_size

    with open(filename, "wb") as fout:
        # NC, NR, NS, MODE = 2 (image : 32-bit reals)
        fout.write(struct.pack('<iiii', xs, ys, zs, 2))
        # NCSTART, NRSTART, NSSTART
        fout.write(struct.pack('<iii', nxstart, nystart, nzstart))
        # MX, MY, MZ
        fout.write(struct.pack('<iii', xs, ys, zs))
        # X length, Y, length, Z length
        fout.write(struct.pack('<fff', a, b, c))
        # Alpha, Beta, Gamma
        fout.write(struct.pack('<fff', alpha, beta, gamma))
        # MAPC, MAPR, MAPS
        fout.write(struct.pack('<iii', mapc, mapr, maps))
        # DMIN, DMAX, DMEAN
        fout.write(struct.pack('<fff', np.min(rho), np.max(rho), np.average(rho)))
        # ISPG, NSYMBT, mlLSKFLG
        fout.write(struct.pack('<iii', 1, 0, 0))
        # EXTRA
        fout.write(struct.pack('<'+'f'*12, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        for i in range(0, 12):
            fout.write(struct.pack('<f', 0.0))

        # XORIGIN, YORIGIN, ZORIGIN
        fout.write(struct.pack('<fff', xorg, yorg, zorg )) #nxstart*(a/xs), nystart*(b/ys), nzstart*(c/zs) ))
        # MAP
        fout.write('MAP '.encode())
        # MACHST (little endian)
        fout.write(struct.pack('<BBBB', 0x44, 0x41, 0x00, 0x00))
        # RMS (std)
        fout.write(struct.pack('<f', np.std(rho)))
        # NLABL
        fout.write(struct.pack('<i', 0))
        # LABEL(20,10) 10 80-character text labels
        for i in range(0, 800):
            fout.write(struct.pack('<B', 0x00))
        # Write out data
        s = struct.pack('=%sf' % rho.size, *rho.flatten('F'))
        fout.write(s)

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

def relion_and_contrast_preprocess(micrograph, particle_diameter, output_path, image_name, resize_shape):
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
    resize_shape : int
        Resize shape for micrograph
    Returns
    -------
    None
    """
    padding_x, padding_y = (np.array(micrograph.shape) - particle_diameter) // 2
    normalization_filter = np.pad(create_2d_circular_mask((particle_diameter, particle_diameter), particle_diameter, True).astype(int),((padding_x, padding_x), (padding_y, padding_y)))
    normalized_micrograph = relion_normalization(micrograph, normalization_filter)
    enhanced_micrograph = denoise_jpg_image_reduced_torch(normalized_micrograph)
    resized_micrograph = Image.fromarray(normalize_array(np.array(enhanced_micrograph)) * 255).convert('L')
    #Resize
    resized_micrograph = resized_micrograph.resize((resize_shape,resize_shape))
    resized_micrograph.save(f'{output_path}{image_name}.png')
    return

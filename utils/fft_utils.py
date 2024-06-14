import numpy as np


def fft2_center(img):
    """
    Perform centered 2D Fast Fourier Transform (FFT) on an image.

    Parameters
    ----------
    img : numpy.ndarray
        Input image array.

    Returns
    -------
    numpy.ndarray
        The centered 2D FFT of the input image.
    """
    # Shift zero-frequency component to the center, apply FFT, then shift back
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=(-1, -2))), axes=(-1, -2))

def ht2_center(img):
    """
    Compute the Hilbert transform of a 2D image using centered FFT.

    Parameters
    ----------
    img : numpy.ndarray
        Input image array.

    Returns
    -------
    numpy.ndarray
        The Hilbert transform of the input image.
    """
    # Compute centered 2D FFT
    f = fft2_center(img)
    # Return the real part minus the imaginary part of the transform
    return f.real - f.imag

def iht2_center(img):
    """
    Compute the inverse Hilbert transform of a 2D image using centered FFT.

    Parameters
    ----------
    img : numpy.ndarray
        Input image array.

    Returns
    -------
    numpy.ndarray
        The inverse Hilbert transform of the input image.
    """
    # Compute centered 2D FFT
    img = fft2_center(img)
    # Normalize by the number of pixels
    img /= (img.shape[-1] * img.shape[-2])
    # Return the real part minus the imaginary part of the transform
    return img.real - img.imag

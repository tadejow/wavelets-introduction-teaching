"""Noise, linear smoothing and median filtering of images (week 3).

Images follow the convention of :mod:`wavelets_lab.images`: ``float64``
arrays on the scale ``[0, 255]``.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: Integer 3x3 Gaussian mask from the lecture notes (normalized by 16).
GAUSS_3X3: FloatArray = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0

#: Integer 5x5 Gaussian mask from the lecture notes (normalized by 159).
GAUSS_5X5: FloatArray = (
    np.array(
        [
            [2, 4, 5, 4, 2],
            [4, 9, 12, 9, 4],
            [5, 12, 15, 12, 5],
            [4, 9, 12, 9, 4],
            [2, 4, 5, 4, 2],
        ]
    )
    / 159.0
)


def add_gaussian_noise(
    image: ArrayLike,
    sigma: float,
    rng: np.random.Generator | int | None = None,
) -> FloatArray:
    """Add white Gaussian noise with standard deviation ``sigma``.

    Parameters
    ----------
    image : array_like
        Input image.
    sigma : float
        Standard deviation of the noise, e.g. 20.
    rng : numpy.random.Generator or int, optional
        Random generator or seed, for reproducible examples.

    Returns
    -------
    numpy.ndarray
        Noisy image (not clipped).
    """
    array = np.asarray(image, dtype=np.float64)
    generator = np.random.default_rng(rng)
    return array + sigma * generator.standard_normal(array.shape)


def gaussian_kernel(size: int, sigma: float) -> FloatArray:
    """Return a normalized ``size x size`` Gaussian mask.

    Parameters
    ----------
    size : int
        Odd side length of the mask.
    sigma : float
        Standard deviation of the Gaussian in pixels.

    Returns
    -------
    numpy.ndarray
        Mask with entries summing to 1.

    Raises
    ------
    ValueError
        If ``size`` is not a positive odd integer or ``sigma`` is not positive.
    """
    if size <= 0 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    offsets = np.arange(size) - size // 2
    profile = np.exp(-(offsets**2) / (2.0 * sigma**2))
    kernel = np.outer(profile, profile)
    return np.asarray(kernel / kernel.sum(), dtype=np.float64)


def convolve2d(image: ArrayLike, kernel: ArrayLike) -> FloatArray:
    """Convolve an image with a small mask, mirroring the image at the edges.

    The output has the same shape as the input. For symmetric masks, such as
    the Gaussian masks, convolution and correlation coincide.

    Parameters
    ----------
    image : array_like
        Greyscale image.
    kernel : array_like
        Mask with odd side lengths.

    Returns
    -------
    numpy.ndarray
        Filtered image.

    Raises
    ------
    ValueError
        If the mask does not have odd side lengths.
    """
    array = np.asarray(image, dtype=np.float64)
    mask = np.asarray(kernel, dtype=np.float64)
    if mask.ndim != 2 or mask.shape[0] % 2 == 0 or mask.shape[1] % 2 == 0:
        raise ValueError("kernel must be a 2D mask with odd side lengths")
    pad = (mask.shape[0] // 2, mask.shape[1] // 2)
    padded = np.pad(array, ((pad[0], pad[0]), (pad[1], pad[1])), mode="symmetric")
    windows = sliding_window_view(padded, mask.shape)
    filtered = np.einsum("ijkl,kl->ij", windows, mask[::-1, ::-1])
    return np.asarray(filtered, dtype=np.float64)


def gaussian_filter(image: ArrayLike, size: int = 3) -> FloatArray:
    """Smooth an image with the 3x3 or 5x5 Gaussian mask of the notes.

    Parameters
    ----------
    image : array_like
        Greyscale image.
    size : {3, 5}, optional
        Mask size; 3 by default.

    Returns
    -------
    numpy.ndarray
        Smoothed image.

    Raises
    ------
    ValueError
        If ``size`` is neither 3 nor 5.
    """
    masks = {3: GAUSS_3X3, 5: GAUSS_5X5}
    if size not in masks:
        raise ValueError("size must be 3 or 5")
    return convolve2d(image, masks[size])


def median_filter(image: ArrayLike, size: int = 3) -> FloatArray:
    """Replace every pixel by the median of its ``size x size`` neighbourhood.

    Parameters
    ----------
    image : array_like
        Greyscale image.
    size : int, optional
        Odd side length of the neighbourhood; 3 by default.

    Returns
    -------
    numpy.ndarray
        Filtered image.

    Raises
    ------
    ValueError
        If ``size`` is not a positive odd integer.
    """
    if size <= 0 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")
    array = np.asarray(image, dtype=np.float64)
    pad = size // 2
    padded = np.pad(array, pad, mode="symmetric")
    windows = sliding_window_view(padded, (size, size))
    return np.median(windows, axis=(2, 3))


def two_level_image(
    shape: tuple[int, int] = (256, 256), low: float = 80.0, high: float = 180.0
) -> FloatArray:
    """Build a test image: a bright square on a dark background.

    Parameters
    ----------
    shape : tuple of int, optional
        Shape of the image.
    low : float, optional
        Background intensity; 80 by default.
    high : float, optional
        Intensity of the central square; 180 by default.

    Returns
    -------
    numpy.ndarray
        Two-level image with the square covering the middle half.
    """
    image = np.full(shape, low, dtype=np.float64)
    rows, cols = shape
    image[rows // 4 : 3 * rows // 4, cols // 4 : 3 * cols // 4] = high
    return image


def mse(reference: ArrayLike, image: ArrayLike) -> float:
    """Return the mean squared error between two images.

    Parameters
    ----------
    reference : array_like
        Reference image.
    image : array_like
        Compared image of the same shape.

    Returns
    -------
    float
        Mean of the squared differences.
    """
    difference = np.asarray(reference, dtype=np.float64) - np.asarray(
        image, dtype=np.float64
    )
    return float(np.mean(difference**2))


def psnr(reference: ArrayLike, image: ArrayLike, peak: float = 255.0) -> float:
    """Return the peak signal-to-noise ratio in decibels.

    Parameters
    ----------
    reference : array_like
        Reference image.
    image : array_like
        Compared image of the same shape.
    peak : float, optional
        Largest possible intensity; 255 by default.

    Returns
    -------
    float
        ``10 * log10(peak**2 / MSE)``; infinite for identical images.
    """
    error = mse(reference, image)
    if error == 0:
        return float("inf")
    return float(10.0 * np.log10(peak**2 / error))

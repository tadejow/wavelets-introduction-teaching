"""Gradient masks and the Canny edge detector (week 3).

The detector follows the steps of the lecture notes: Gaussian smoothing,
gradient estimation with 3x3 masks, quantization of the gradient direction
to four directions, non-maximum suppression and hysteresis thresholding.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from wavelets_lab.filters import convolve2d, gaussian_filter

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

#: Sobel mask estimating the derivative along columns (x direction).
SOBEL_X: FloatArray = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

#: Sobel mask estimating the derivative along rows (y direction).
SOBEL_Y: FloatArray = SOBEL_X.T.copy()

#: Pixel offsets ``(d_row, d_col)`` of the neighbours along each direction:
#: 0 degrees, 45 degrees, 90 degrees and 135 degrees.
_NEIGHBOURS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))


def gradient(image: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Estimate the image gradient with the Sobel masks.

    Parameters
    ----------
    image : array_like
        Greyscale image.

    Returns
    -------
    gx : numpy.ndarray
        Derivative along columns (left to right).
    gy : numpy.ndarray
        Derivative along rows (top to bottom).
    """
    # convolve2d flips the mask, so pass flipped masks to obtain correlation.
    gx = convolve2d(image, SOBEL_X[::-1, ::-1])
    gy = convolve2d(image, SOBEL_Y[::-1, ::-1])
    return gx, gy


def gradient_magnitude(gx: ArrayLike, gy: ArrayLike) -> FloatArray:
    """Return the Euclidean length of the gradient.

    Parameters
    ----------
    gx, gy : array_like
        Gradient components from :func:`gradient`.

    Returns
    -------
    numpy.ndarray
        ``sqrt(gx**2 + gy**2)``.
    """
    magnitude = np.hypot(np.asarray(gx, dtype=float), np.asarray(gy, dtype=float))
    return np.asarray(magnitude, dtype=np.float64)


def quantize_direction(gx: ArrayLike, gy: ArrayLike) -> NDArray[np.int_]:
    """Quantize the gradient direction to one of four directions.

    Rows grow downwards, so the angle is measured with ``-gy`` to keep the
    usual counter-clockwise orientation.

    Parameters
    ----------
    gx, gy : array_like
        Gradient components from :func:`gradient`.

    Returns
    -------
    numpy.ndarray
        Integers 0, 1, 2, 3 meaning 0, 45, 90 and 135 degrees.
    """
    angle = np.degrees(np.arctan2(-np.asarray(gy), np.asarray(gx))) % 180.0
    return np.asarray(np.rint(angle / 45.0).astype(int) % 4)


def non_maximum_suppression(magnitude: ArrayLike, direction: ArrayLike) -> FloatArray:
    """Keep only pixels that are local maxima along the gradient direction.

    Parameters
    ----------
    magnitude : array_like
        Gradient magnitude.
    direction : array_like
        Quantized directions from :func:`quantize_direction`.

    Returns
    -------
    numpy.ndarray
        Magnitude with non-maximal pixels set to zero.
    """
    mag = np.asarray(magnitude, dtype=np.float64)
    dirs = np.asarray(direction)
    padded = np.pad(mag, 1, mode="constant")
    rows, cols = mag.shape
    result = np.zeros_like(mag)
    for index, (d_row, d_col) in enumerate(_NEIGHBOURS):
        forward = padded[1 + d_row : 1 + d_row + rows, 1 + d_col : 1 + d_col + cols]
        backward = padded[1 - d_row : 1 - d_row + rows, 1 - d_col : 1 - d_col + cols]
        keep = (dirs == index) & (mag >= forward) & (mag >= backward)
        result[keep] = mag[keep]
    return result


def hysteresis_threshold(image: ArrayLike, low: float, high: float) -> BoolArray:
    """Two-threshold hysteresis: grow strong edges along weak ones.

    Pixels above ``high`` are edges. Pixels above ``low`` are edges when they
    are connected (8-neighbourhood) to a pixel above ``high``.

    Parameters
    ----------
    image : array_like
        Thinned gradient magnitude.
    low : float
        Lower threshold.
    high : float
        Upper threshold.

    Returns
    -------
    numpy.ndarray
        Boolean edge map.

    Raises
    ------
    ValueError
        If ``low > high``.
    """
    if low > high:
        raise ValueError("low must not exceed high")
    values = np.asarray(image, dtype=np.float64)
    weak = values > low
    labels, count = ndimage.label(weak, structure=np.ones((3, 3)))
    if count == 0:
        return np.zeros(values.shape, dtype=bool)
    strong_labels = np.unique(labels[values > high])
    strong_labels = strong_labels[strong_labels > 0]
    return np.isin(labels, strong_labels)


def canny(
    image: ArrayLike,
    low: float = 20.0,
    high: float = 50.0,
    smoothing: int = 5,
) -> BoolArray:
    """Detect edges with the Canny algorithm.

    Parameters
    ----------
    image : array_like
        Greyscale image on the scale ``[0, 255]``.
    low : float, optional
        Lower hysteresis threshold for the gradient magnitude.
    high : float, optional
        Upper hysteresis threshold for the gradient magnitude.
    smoothing : {3, 5}, optional
        Size of the Gaussian mask used before differentiation; 5 by default.

    Returns
    -------
    numpy.ndarray
        Boolean edge map.
    """
    smoothed = gaussian_filter(image, size=smoothing)
    gx, gy = gradient(smoothed)
    thin = non_maximum_suppression(
        gradient_magnitude(gx, gy), quantize_direction(gx, gy)
    )
    return hysteresis_threshold(thin, low, high)

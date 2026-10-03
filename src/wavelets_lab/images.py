"""Images as arrays and basic pixel operations (weeks 1-2).

Greyscale images are represented as two-dimensional ``float64`` arrays with
intensities on the scale ``[0, 255]``, the same scale as in the lecture notes,
so thresholds such as ``100`` or offsets such as ``25`` keep their meaning.
Colour images are ``(rows, cols, 3)`` arrays on the same scale.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from skimage import data

FloatArray = NDArray[np.float64]

MAX_INTENSITY = 255.0

# ITU-R BT.601 (JPEG) conversion matrix from RGB to YCbCr.
_RGB_TO_YCBCR = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ]
)
_YCBCR_OFFSET = np.array([0.0, 128.0, 128.0])


def load_test_image(name: str = "camera") -> FloatArray:
    """Load a licence-free test image from :mod:`skimage.data`.

    Parameters
    ----------
    name : str, optional
        Name of a function in :mod:`skimage.data`, e.g. ``"camera"``,
        ``"astronaut"`` or ``"coins"``. Default is ``"camera"``.

    Returns
    -------
    numpy.ndarray
        Image as a ``float64`` array on the scale ``[0, 255]``.

    Raises
    ------
    ValueError
        If ``skimage.data`` has no image with the given name.
    """
    loader = getattr(data, name, None)
    if loader is None or not callable(loader):
        raise ValueError(f"skimage.data has no test image called {name!r}")
    return to_float(loader())


def load_image(path: str | Path) -> FloatArray:
    """Read an image file (BMP, GIF, TIFF, PNG, JPEG, ...).

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        Image as a ``float64`` array on the scale ``[0, 255]``. An alpha
        channel, if present, is dropped.
    """
    image = np.asarray(iio.imread(path))
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return to_float(image)


def save_image(path: str | Path, image: ArrayLike) -> None:
    """Write an image to a file after clipping it to ``[0, 255]``.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file; the format is chosen from its extension.
    image : array_like
        Image on the scale ``[0, 255]``.
    """
    iio.imwrite(path, to_uint8(image))


def to_float(image: ArrayLike) -> FloatArray:
    """Convert an image to ``float64`` on the scale ``[0, 255]``.

    Integer images are converted directly. Floating-point images with all
    values in ``[0, 1]`` are assumed to use the unit scale and are multiplied
    by 255; other floating-point images are only cast.

    Parameters
    ----------
    image : array_like
        Input image.

    Returns
    -------
    numpy.ndarray
        Image as ``float64``.
    """
    array = np.asarray(image)
    if array.dtype == np.bool_:
        return array.astype(np.float64) * MAX_INTENSITY
    if np.issubdtype(array.dtype, np.floating) and array.size > 0:
        if array.min() >= 0.0 and array.max() <= 1.0:
            return array.astype(np.float64) * MAX_INTENSITY
    return array.astype(np.float64)


def to_uint8(image: ArrayLike) -> NDArray[np.uint8]:
    """Round an image and clip it to ``[0, 255]`` as ``uint8``.

    Parameters
    ----------
    image : array_like
        Image on the scale ``[0, 255]``.

    Returns
    -------
    numpy.ndarray
        Image as ``uint8``.
    """
    return np.clip(np.rint(np.asarray(image, dtype=np.float64)), 0, 255).astype(
        np.uint8
    )


def rgb_to_ycbcr(image: ArrayLike) -> FloatArray:
    """Convert an RGB image to the YCbCr colour space (ITU-R BT.601).

    Parameters
    ----------
    image : array_like
        RGB image of shape ``(rows, cols, 3)`` on the scale ``[0, 255]``.

    Returns
    -------
    numpy.ndarray
        YCbCr image of the same shape; ``Y`` is the luminance.
    """
    rgb = _check_colour(image)
    return np.asarray(rgb @ _RGB_TO_YCBCR.T + _YCBCR_OFFSET, dtype=np.float64)


def ycbcr_to_rgb(image: ArrayLike) -> FloatArray:
    """Convert a YCbCr image back to RGB.

    Parameters
    ----------
    image : array_like
        YCbCr image of shape ``(rows, cols, 3)``.

    Returns
    -------
    numpy.ndarray
        RGB image of the same shape (not clipped).
    """
    ycbcr = _check_colour(image)
    rgb = (ycbcr - _YCBCR_OFFSET) @ np.linalg.inv(_RGB_TO_YCBCR).T
    return np.asarray(rgb, dtype=np.float64)


def rgb_to_grey(image: ArrayLike) -> FloatArray:
    """Return the luminance ``Y`` of an RGB image.

    Parameters
    ----------
    image : array_like
        RGB image of shape ``(rows, cols, 3)``.

    Returns
    -------
    numpy.ndarray
        Greyscale image of shape ``(rows, cols)``.
    """
    return rgb_to_ycbcr(image)[:, :, 0]


def normalize(image: ArrayLike) -> FloatArray:
    """Rescale an image linearly to the unit interval ``[0, 1]``.

    Parameters
    ----------
    image : array_like
        Input image.

    Returns
    -------
    numpy.ndarray
        Image with minimum 0 and maximum 1; a constant image maps to zeros.
    """
    array = np.asarray(image, dtype=np.float64)
    low, high = array.min(), array.max()
    if high == low:
        return np.zeros_like(array)
    return np.asarray((array - low) / (high - low), dtype=np.float64)


def show_image(
    image: ArrayLike,
    title: str | None = None,
    ax: Axes | None = None,
    scaled: bool = False,
    xlabel: str = "column [pixel]",
    ylabel: str = "row [pixel]",
) -> Axes:
    """Display an image with a 256-level grey colour map and labelled axes.

    Parameters
    ----------
    image : array_like
        Greyscale or RGB image on the scale ``[0, 255]``.
    title : str, optional
        Title of the plot.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created when omitted.
    scaled : bool, optional
        If ``True``, stretch the greyscale range to the data (like Matlab's
        ``imagesc``); otherwise map ``[0, 255]`` to black and white.
    xlabel : str, optional
        Label of the horizontal axis; ``"column [pixel]"`` by default.
    ylabel : str, optional
        Label of the vertical axis; ``"row [pixel]"`` by default.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the image.
    """
    array = np.asarray(image, dtype=np.float64)
    if ax is None:
        _, ax = plt.subplots()
    if array.ndim == 3:
        ax.imshow(to_uint8(array))
    else:
        limits = (None, None) if scaled else (0.0, MAX_INTENSITY)
        ax.imshow(
            array,
            cmap=plt.get_cmap("gray", 256),
            vmin=limits[0],
            vmax=limits[1],
        )
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def scale_intensity(image: ArrayLike, factor: float) -> FloatArray:
    """Multiply all intensities by ``factor`` and clip to ``[0, 255]``.

    A factor above 1 brightens the image and a factor below 1 darkens it.

    Parameters
    ----------
    image : array_like
        Input image.
    factor : float
        Multiplicative factor, e.g. ``1.2`` or ``1 / 1.2``.

    Returns
    -------
    numpy.ndarray
        Rescaled image.
    """
    return _clip(np.asarray(image, dtype=np.float64) * factor)


def affine_intensity(image: ArrayLike, gain: float, offset: float) -> FloatArray:
    """Apply ``offset + gain * image`` and clip to ``[0, 255]``.

    For example ``gain=0.8, offset=25`` reduces the contrast.

    Parameters
    ----------
    image : array_like
        Input image.
    gain : float
        Multiplicative coefficient.
    offset : float
        Additive coefficient.

    Returns
    -------
    numpy.ndarray
        Transformed image.
    """
    return _clip(offset + gain * np.asarray(image, dtype=np.float64))


def stretch_contrast(image: ArrayLike) -> FloatArray:
    """Stretch the intensity range of an image to the full ``[0, 255]``.

    Parameters
    ----------
    image : array_like
        Input image.

    Returns
    -------
    numpy.ndarray
        Image with minimum 0 and maximum 255.
    """
    return normalize(image) * MAX_INTENSITY


def gamma_correction(image: ArrayLike, gamma: float) -> FloatArray:
    """Apply the gamma correction ``255 * (image / 255) ** gamma``.

    Parameters
    ----------
    image : array_like
        Input image on the scale ``[0, 255]``.
    gamma : float
        Positive exponent; ``gamma < 1`` brightens and ``gamma > 1`` darkens.

    Returns
    -------
    numpy.ndarray
        Corrected image.

    Raises
    ------
    ValueError
        If ``gamma`` is not positive.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    unit = _clip(np.asarray(image, dtype=np.float64)) / MAX_INTENSITY
    return MAX_INTENSITY * unit**gamma


def binarize(image: ArrayLike, threshold: float) -> FloatArray:
    """Map intensities above ``threshold`` to 255 and the rest to 0.

    Parameters
    ----------
    image : array_like
        Input image.
    threshold : float
        Threshold on the scale ``[0, 255]``, e.g. 50, 100, 150 or 200.

    Returns
    -------
    numpy.ndarray
        Black-and-white image.
    """
    return np.where(np.asarray(image) > threshold, MAX_INTENSITY, 0.0)


def mirror_extend(image: ArrayLike, axis: int = 1) -> FloatArray:
    """Append the mirror image of ``image`` along ``axis``.

    A 512x512 image extended along columns becomes 512x1024.

    Parameters
    ----------
    image : array_like
        Input image.
    axis : int, optional
        Axis along which to mirror; 1 (columns) by default.

    Returns
    -------
    numpy.ndarray
        Image twice as large along ``axis``.
    """
    array = np.asarray(image, dtype=np.float64)
    return np.concatenate([array, np.flip(array, axis=axis)], axis=axis)


def rectangle_mask(
    shape: tuple[int, int],
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    ramp: int = 0,
) -> FloatArray:
    """Build a crop mask equal to 1 inside a rectangle and 0 far outside it.

    Parameters
    ----------
    shape : tuple of int
        Shape ``(rows, cols)`` of the mask.
    top_left : tuple of int
        Row and column of the first pixel inside the rectangle.
    bottom_right : tuple of int
        Row and column one past the last pixel inside the rectangle.
    ramp : int, optional
        Width in pixels of a linear transition from 1 to 0 outside the
        rectangle. ``0`` (default) gives a hard mask.

    Returns
    -------
    numpy.ndarray
        Mask with values in ``[0, 1]``.
    """
    rows = np.arange(shape[0])[:, np.newaxis]
    cols = np.arange(shape[1])[np.newaxis, :]
    row_dist = np.maximum(
        np.maximum(top_left[0] - rows, rows - (bottom_right[0] - 1)), 0
    )
    col_dist = np.maximum(
        np.maximum(top_left[1] - cols, cols - (bottom_right[1] - 1)), 0
    )
    distance = np.maximum(row_dist, col_dist).astype(np.float64)
    if ramp <= 0:
        return np.asarray(distance == 0, dtype=np.float64)
    return np.clip(1.0 - distance / ramp, 0.0, 1.0)


def apply_mask(
    image: ArrayLike, mask: ArrayLike, background: float = MAX_INTENSITY
) -> FloatArray:
    """Blend an image with a constant background using a mask.

    Parameters
    ----------
    image : array_like
        Greyscale image.
    mask : array_like
        Mask with values in ``[0, 1]``; 1 keeps the image.
    background : float, optional
        Background intensity; white (255) by default.

    Returns
    -------
    numpy.ndarray
        ``mask * image + (1 - mask) * background``.
    """
    weights = np.asarray(mask, dtype=np.float64)
    return weights * np.asarray(image, dtype=np.float64) + (1.0 - weights) * (
        background
    )


def _clip(image: FloatArray) -> FloatArray:
    return np.clip(image, 0.0, MAX_INTENSITY)


def _check_colour(image: ArrayLike) -> FloatArray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected a colour image of shape (rows, cols, 3)")
    return array

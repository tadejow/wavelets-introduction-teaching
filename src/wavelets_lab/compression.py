"""Thresholding and wavelet image compression (week 13).

The compression ratio follows the lecture notes: the percentage of wavelet
coefficients set to zero among all coefficients (equal to the number of
pixels for an orthogonal transform).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from wavelets_lab.dwt import decompose_image, reconstruct_image

FloatArray = NDArray[np.float64]
Coefficients2D = list[FloatArray | tuple[FloatArray, FloatArray, FloatArray]]


def hard_threshold(x: ArrayLike, threshold: float) -> FloatArray:
    """Set to zero every value with ``|x| < threshold``.

    Parameters
    ----------
    x : array_like
        Coefficients.
    threshold : float
        Non-negative threshold.

    Returns
    -------
    numpy.ndarray
        Thresholded coefficients; equal to ``pywt.threshold(x, t, "hard")``.
    """
    values = np.asarray(x, dtype=np.float64)
    return np.where(np.abs(values) >= threshold, values, 0.0)


def soft_threshold(x: ArrayLike, threshold: float) -> FloatArray:
    """Shrink every value towards zero by ``threshold``.

    Parameters
    ----------
    x : array_like
        Coefficients.
    threshold : float
        Non-negative threshold.

    Returns
    -------
    numpy.ndarray
        ``sign(x) * max(|x| - threshold, 0)``; equal to
        ``pywt.threshold(x, t, "soft")``.
    """
    values = np.asarray(x, dtype=np.float64)
    return np.asarray(
        np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0),
        dtype=np.float64,
    )


def threshold_details(
    coefficients: Coefficients2D, threshold: float, mode: str = "hard"
) -> Coefficients2D:
    """Threshold the detail subbands of a 2D DWT, keeping the approximation.

    Parameters
    ----------
    coefficients : list
        Output of :func:`wavelets_lab.dwt.decompose_image`.
    threshold : float
        Non-negative threshold.
    mode : {"hard", "soft"}, optional
        Thresholding rule.

    Returns
    -------
    list
        Thresholded coefficients in the same format.

    Raises
    ------
    ValueError
        If ``mode`` is neither ``"hard"`` nor ``"soft"``.
    """
    rules = {"hard": hard_threshold, "soft": soft_threshold}
    if mode not in rules:
        raise ValueError("mode must be 'hard' or 'soft'")
    rule = rules[mode]
    result: Coefficients2D = [np.asarray(coefficients[0], dtype=np.float64)]
    for bands in coefficients[1:]:
        h_band, v_band, d_band = bands
        result.append(
            (rule(h_band, threshold), rule(v_band, threshold), rule(d_band, threshold))
        )
    return result


def _details(coefficients: Coefficients2D) -> FloatArray:
    return np.concatenate(
        [np.ravel(band) for bands in coefficients[1:] for band in bands]
    )


def zero_fraction(coefficients: Coefficients2D) -> float:
    """Return the fraction of all coefficients that are exactly zero.

    Parameters
    ----------
    coefficients : list
        Coefficients of a 2D DWT.

    Returns
    -------
    float
        Number of zeros divided by the number of coefficients.
    """
    approximation = np.ravel(coefficients[0])
    details = _details(coefficients)
    zeros = np.count_nonzero(approximation == 0) + np.count_nonzero(details == 0)
    return float(zeros / (approximation.size + details.size))


def threshold_for_ratio(coefficients: Coefficients2D, ratio: float) -> float:
    """Find the threshold that zeroes the given fraction of all coefficients.

    Only detail coefficients are thresholded, so ratios larger than the share
    of details cannot be reached exactly.

    Parameters
    ----------
    coefficients : list
        Coefficients of a 2D DWT.
    ratio : float
        Target fraction of zeroed coefficients, e.g. 0.9, 0.95 or 0.98.

    Returns
    -------
    float
        Threshold on the absolute value of the detail coefficients.

    Raises
    ------
    ValueError
        If ``ratio`` is not in ``[0, 1)``.
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio must be in [0, 1)")
    details = np.sort(np.abs(_details(coefficients)))
    total = details.size + np.size(coefficients[0])
    count = min(int(np.ceil(ratio * total)), details.size)
    if count == 0:
        return 0.0
    return float(details[count - 1])


def compress_image(
    image: ArrayLike,
    ratio: float,
    wavelet: str = "db2",
    level: int = 4,
    mode: str = "hard",
    rescale: bool = False,
) -> tuple[FloatArray, float]:
    """Compress an image by thresholding its 2D DWT and reconstruct it.

    Parameters
    ----------
    image : array_like
        Greyscale image with sides divisible by ``2**level``.
    ratio : float
        Target fraction of zeroed coefficients.
    wavelet : str, optional
        Wavelet name.
    level : int, optional
        Number of decomposition levels.
    mode : {"hard", "soft"}, optional
        Thresholding rule.
    rescale : bool, optional
        After soft thresholding, multiply the details by ``M / (M - eps)``,
        where ``M`` is the largest absolute detail, to restore their range.

    Returns
    -------
    reconstruction : numpy.ndarray
        Decompressed image.
    achieved_ratio : float
        Fraction of zero coefficients actually obtained.
    """
    coefficients = decompose_image(image, wavelet, level)
    threshold = threshold_for_ratio(coefficients, ratio)
    compressed = threshold_details(coefficients, threshold, mode)
    if rescale and mode == "soft":
        largest = float(np.max(np.abs(_details(coefficients))))
        if largest > threshold:
            factor = largest / (largest - threshold)
            compressed = (
                [compressed[0]]
                + [
                    tuple(np.asarray(band) * factor for band in bands)  # type: ignore[misc]
                    for bands in compressed[1:]
                ]
            )
    achieved = zero_fraction(compressed)
    return reconstruct_image(compressed, wavelet), achieved

"""Wavelet denoising of signals and images (week 14)."""

from __future__ import annotations

import numpy as np
import pywt
from numpy.typing import ArrayLike, NDArray

from wavelets_lab.compression import hard_threshold, soft_threshold

FloatArray = NDArray[np.float64]

#: Median of ``|Z|`` for a standard normal ``Z``, used by the MAD estimator.
MAD_CONSTANT = 0.6744897501960817


def doppler(n: int = 1024) -> FloatArray:
    """Return the Donoho-Johnstone Doppler test signal on ``[0, 1)``.

    Parameters
    ----------
    n : int, optional
        Number of samples.

    Returns
    -------
    numpy.ndarray
        ``sqrt(t (1 - t)) sin(2.1 pi / (t + 0.05))``.
    """
    t = np.arange(n) / n
    return np.asarray(
        np.sqrt(t * (1.0 - t)) * np.sin(2.1 * np.pi / (t + 0.05)), dtype=np.float64
    )


def heavisine(n: int = 1024) -> FloatArray:
    """Return the Donoho-Johnstone HeaviSine test signal on ``[0, 1)``.

    Parameters
    ----------
    n : int, optional
        Number of samples.

    Returns
    -------
    numpy.ndarray
        ``4 sin(4 pi t) - sign(t - 0.3) - sign(0.72 - t)``.
    """
    t = np.arange(n) / n
    return np.asarray(
        4.0 * np.sin(4.0 * np.pi * t) - np.sign(t - 0.3) - np.sign(0.72 - t),
        dtype=np.float64,
    )


def estimate_sigma(detail: ArrayLike) -> float:
    """Estimate the noise level from the finest detail coefficients.

    Parameters
    ----------
    detail : array_like
        Finest-scale detail coefficients.

    Returns
    -------
    float
        ``median(|d|) / 0.6745``.
    """
    return float(np.median(np.abs(np.asarray(detail))) / MAD_CONSTANT)


def universal_threshold(sigma: float, n: int) -> float:
    """Return the VisuShrink threshold ``sigma sqrt(2 log n)``.

    Parameters
    ----------
    sigma : float
        Noise standard deviation.
    n : int
        Number of samples.

    Returns
    -------
    float
        The universal threshold.
    """
    return float(sigma * np.sqrt(2.0 * np.log(n)))


def sure_threshold(detail: ArrayLike, sigma: float) -> float:
    """Return the soft threshold minimizing Stein's unbiased risk estimate.

    Parameters
    ----------
    detail : array_like
        Detail coefficients of one level.
    sigma : float
        Noise standard deviation.

    Returns
    -------
    float
        Threshold among ``|d_i|`` (and 0) with the smallest SURE value,
        capped by the universal threshold.
    """
    values = np.abs(np.ravel(np.asarray(detail, dtype=np.float64)))
    n = values.size
    if n == 0 or sigma <= 0:
        return 0.0
    scaled = np.sort(values / sigma) ** 2
    candidates = np.concatenate([[0.0], scaled])
    cumulative = np.concatenate([[0.0], np.cumsum(scaled)])
    counts = np.arange(n + 1)
    risks = n - 2.0 * counts + cumulative + (n - counts) * candidates
    best = float(np.sqrt(candidates[int(np.argmin(risks))]) * sigma)
    return min(best, universal_threshold(sigma, n))


def denoise_signal(
    x: ArrayLike,
    wavelet: str = "db4",
    level: int = 5,
    method: str = "visu",
    mode: str = "soft",
) -> FloatArray:
    """Remove white Gaussian noise from a signal by thresholding its DWT.

    Parameters
    ----------
    x : array_like
        Noisy signal with length divisible by ``2**level``.
    wavelet : str, optional
        Wavelet name.
    level : int, optional
        Number of decomposition levels.
    method : {"visu", "sure"}, optional
        One universal threshold, or a SURE threshold for every level.
    mode : {"hard", "soft"}, optional
        Thresholding rule.

    Returns
    -------
    numpy.ndarray
        Denoised signal.

    Raises
    ------
    ValueError
        If ``method`` or ``mode`` is unknown.
    """
    if method not in ("visu", "sure"):
        raise ValueError("method must be 'visu' or 'sure'")
    rule = {"hard": hard_threshold, "soft": soft_threshold}.get(mode)
    if rule is None:
        raise ValueError("mode must be 'hard' or 'soft'")
    signal = np.asarray(x, dtype=np.float64)
    coefficients = pywt.wavedec(signal, wavelet, mode="periodization", level=level)
    sigma = estimate_sigma(coefficients[-1])
    cleaned = [coefficients[0]]
    for detail in coefficients[1:]:
        if method == "visu":
            threshold = universal_threshold(sigma, signal.size)
        else:
            threshold = sure_threshold(detail, sigma)
        cleaned.append(rule(detail, threshold))
    return np.asarray(
        pywt.waverec(cleaned, wavelet, mode="periodization"), dtype=np.float64
    )


def denoise_image(
    image: ArrayLike, wavelet: str = "db4", level: int = 3, mode: str = "soft"
) -> FloatArray:
    """Remove white Gaussian noise from an image with VisuShrink.

    Parameters
    ----------
    image : array_like
        Noisy greyscale image with sides divisible by ``2**level``.
    wavelet : str, optional
        Wavelet name.
    level : int, optional
        Number of decomposition levels.
    mode : {"hard", "soft"}, optional
        Thresholding rule.

    Returns
    -------
    numpy.ndarray
        Denoised image.

    Raises
    ------
    ValueError
        If ``mode`` is unknown.
    """
    rule = {"hard": hard_threshold, "soft": soft_threshold}.get(mode)
    if rule is None:
        raise ValueError("mode must be 'hard' or 'soft'")
    array = np.asarray(image, dtype=np.float64)
    coefficients = pywt.wavedec2(array, wavelet, mode="periodization", level=level)
    sigma = estimate_sigma(coefficients[-1][2])
    threshold = universal_threshold(sigma, array.size)
    cleaned: list[object] = [coefficients[0]]
    for bands in coefficients[1:]:
        cleaned.append(tuple(rule(band, threshold) for band in bands))
    return np.asarray(
        pywt.waverec2(cleaned, wavelet, mode="periodization"), dtype=np.float64
    )

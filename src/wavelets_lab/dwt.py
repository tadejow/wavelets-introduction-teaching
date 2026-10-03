"""Mallat algorithm in 1D and the 2D discrete wavelet transform (weeks 10, 12).

Signals are treated as periodic, which makes the transform orthogonal and
matches ``mode="periodization"`` of PyWavelets.
"""

from __future__ import annotations

import numpy as np
import pywt
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _decomposition_filters(wavelet: str) -> tuple[FloatArray, FloatArray]:
    filters = pywt.Wavelet(wavelet)
    return (
        np.asarray(filters.dec_lo, dtype=np.float64),
        np.asarray(filters.dec_hi, dtype=np.float64),
    )


def _analysis_indices(n: int, length: int) -> NDArray[np.int_]:
    # Entry [k, m] is the sample index multiplied by filter tap m in output k.
    k = np.arange(n // 2)[:, np.newaxis]
    m = np.arange(length)[np.newaxis, :]
    return np.asarray((2 * k + length // 2 - m) % n)


def analysis_step(x: ArrayLike, wavelet: str = "db2") -> tuple[FloatArray, FloatArray]:
    """Perform one step of the Mallat algorithm: filter and downsample.

    ``a[k] = sum_m h[m] x[2k + L/2 - m]`` and the same with ``g`` for the
    details, with indices taken modulo ``N``.

    Parameters
    ----------
    x : array_like
        Signal of even length ``N``.
    wavelet : str, optional
        Name of an orthogonal PyWavelets wavelet.

    Returns
    -------
    approximation : numpy.ndarray
        Coarse coefficients of length ``N / 2``.
    detail : numpy.ndarray
        Detail coefficients of length ``N / 2``.

    Raises
    ------
    ValueError
        If the length of the signal is odd.
    """
    signal = np.asarray(x, dtype=np.float64)
    if signal.size % 2:
        raise ValueError("the signal length must be even")
    low, high = _decomposition_filters(wavelet)
    samples = signal[_analysis_indices(signal.size, low.size)]
    return samples @ low, samples @ high


def synthesis_step(
    approximation: ArrayLike, detail: ArrayLike, wavelet: str = "db2"
) -> FloatArray:
    """Invert :func:`analysis_step`: upsample, filter and add.

    The transform is orthogonal, so synthesis is the adjoint of analysis.

    Parameters
    ----------
    approximation : array_like
        Coarse coefficients.
    detail : array_like
        Detail coefficients of the same length.
    wavelet : str, optional
        Name of the wavelet used for the analysis.

    Returns
    -------
    numpy.ndarray
        Reconstructed signal of twice the length.
    """
    coarse = np.asarray(approximation, dtype=np.float64)
    fine = np.asarray(detail, dtype=np.float64)
    low, high = _decomposition_filters(wavelet)
    n = 2 * coarse.size
    indices = _analysis_indices(n, low.size)
    signal = np.zeros(n)
    contributions = coarse[:, np.newaxis] * low + fine[:, np.newaxis] * high
    np.add.at(signal, indices, contributions)
    return signal


def decompose(x: ArrayLike, wavelet: str = "db2", level: int = 3) -> list[FloatArray]:
    """Compute a multilevel DWT by repeating :func:`analysis_step`.

    Parameters
    ----------
    x : array_like
        Signal whose length is divisible by ``2**level``.
    wavelet : str, optional
        Wavelet name.
    level : int, optional
        Number of decomposition levels.

    Returns
    -------
    list of numpy.ndarray
        ``[a_J, d_J, ..., d_1]``, the same order as :func:`pywt.wavedec`.
    """
    approximation = np.asarray(x, dtype=np.float64)
    details: list[FloatArray] = []
    for _ in range(level):
        approximation, detail = analysis_step(approximation, wavelet)
        details.append(detail)
    return [approximation, *details[::-1]]


def reconstruct(coefficients: list[FloatArray], wavelet: str = "db2") -> FloatArray:
    """Invert :func:`decompose`.

    Parameters
    ----------
    coefficients : list of numpy.ndarray
        ``[a_J, d_J, ..., d_1]``.
    wavelet : str, optional
        Wavelet name.

    Returns
    -------
    numpy.ndarray
        Reconstructed signal.
    """
    signal = np.asarray(coefficients[0], dtype=np.float64)
    for detail in coefficients[1:]:
        signal = synthesis_step(signal, detail, wavelet)
    return signal


def decompose_image(
    image: ArrayLike, wavelet: str = "db2", level: int = 3
) -> list[FloatArray | tuple[FloatArray, FloatArray, FloatArray]]:
    """Compute the 2D DWT of an image with :func:`pywt.wavedec2`.

    Parameters
    ----------
    image : array_like
        Greyscale image whose sides are divisible by ``2**level``.
    wavelet : str, optional
        Wavelet name, e.g. ``"db1"``, ``"db3"`` or ``"db5"``.
    level : int, optional
        Number of levels.

    Returns
    -------
    list
        ``[cA_J, (cH_J, cV_J, cD_J), ..., (cH_1, cV_1, cD_1)]``.
    """
    coefficients = pywt.wavedec2(
        np.asarray(image, dtype=np.float64), wavelet, mode="periodization", level=level
    )
    return list(coefficients)


def reconstruct_image(
    coefficients: list[FloatArray | tuple[FloatArray, FloatArray, FloatArray]],
    wavelet: str = "db2",
) -> FloatArray:
    """Invert :func:`decompose_image`.

    Parameters
    ----------
    coefficients : list
        Coefficients in the format of :func:`decompose_image`.
    wavelet : str, optional
        Wavelet name.

    Returns
    -------
    numpy.ndarray
        Reconstructed image.
    """
    return np.asarray(
        pywt.waverec2(coefficients, wavelet, mode="periodization"), dtype=np.float64
    )


def coefficient_image(
    coefficients: list[FloatArray | tuple[FloatArray, FloatArray, FloatArray]],
) -> FloatArray:
    """Arrange 2D DWT coefficients into one image for display.

    Each subband is rescaled to ``[0, 255]`` separately, so that the details
    remain visible next to the much larger approximation.

    Parameters
    ----------
    coefficients : list
        Coefficients in the format of :func:`decompose_image`.

    Returns
    -------
    numpy.ndarray
        Image of the same size as the transformed image.
    """

    def rescale(band: ArrayLike) -> FloatArray:
        values = np.abs(np.asarray(band, dtype=np.float64))
        top = values.max()
        return values * (255.0 / top) if top > 0 else values

    scaled: list[FloatArray | tuple[FloatArray, FloatArray, FloatArray]] = [
        rescale(coefficients[0])
    ]
    for bands in coefficients[1:]:
        h_band, v_band, d_band = bands
        scaled.append((rescale(h_band), rescale(v_band), rescale(d_band)))
    array, _ = pywt.coeffs_to_array(scaled)
    return np.asarray(array, dtype=np.float64)

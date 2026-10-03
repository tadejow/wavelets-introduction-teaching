"""Multiscale analysis of financial time series (week 15 project).

Following chapter 7 of the lecture notes, a series is split into additive
components living on different scales (for daily prices: years, months and
days), and two series are compared scale by scale.
"""

from __future__ import annotations

import numpy as np
import pywt
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def synthetic_prices(
    n: int = 2048, rng: np.random.Generator | int | None = None
) -> FloatArray:
    """Generate a daily price series with a trend, cycles and noise.

    Parameters
    ----------
    n : int, optional
        Number of days.
    rng : numpy.random.Generator or int, optional
        Random generator or seed.

    Returns
    -------
    numpy.ndarray
        Positive prices starting near 100.
    """
    generator = np.random.default_rng(rng)
    days = np.arange(n)
    log_price = (
        np.log(100.0)
        + 0.3 * np.sin(2.0 * np.pi * days / 500.0)
        + 0.05 * np.sin(2.0 * np.pi * days / 30.0)
        + np.cumsum(0.01 * generator.standard_normal(n))
    )
    return np.asarray(np.exp(log_price), dtype=np.float64)


def multiscale_components(
    x: ArrayLike, wavelet: str = "db4", level: int = 6
) -> list[FloatArray]:
    """Split a series into components with one wavelet level each.

    Each component is reconstructed from a single band of the DWT with all
    other bands set to zero, so the components add up to the series.

    Parameters
    ----------
    x : array_like
        Series with length divisible by ``2**level``.
    wavelet : str, optional
        Wavelet name.
    level : int, optional
        Number of levels.

    Returns
    -------
    list of numpy.ndarray
        ``[smooth, detail_level, ..., detail_1]``: the smooth trend first,
        then details from coarse to fine.
    """
    signal = np.asarray(x, dtype=np.float64)
    coefficients = pywt.wavedec(signal, wavelet, mode="periodization", level=level)
    components = []
    for index in range(len(coefficients)):
        selected = [
            band if position == index else np.zeros_like(band)
            for position, band in enumerate(coefficients)
        ]
        components.append(
            np.asarray(
                pywt.waverec(selected, wavelet, mode="periodization"), dtype=np.float64
            )
        )
    return components


def group_scales(
    components: list[FloatArray], groups: dict[str, list[int]]
) -> dict[str, FloatArray]:
    """Add components into named groups of scales.

    Parameters
    ----------
    components : list of numpy.ndarray
        Output of :func:`multiscale_components`.
    groups : dict
        Maps a name to the indices of the components it contains, e.g.
        ``{"large": [0, 1], "medium": [2, 3], "small": [4, 5, 6]}``.

    Returns
    -------
    dict
        Maps each name to the sum of its components.
    """
    return {
        name: np.sum([components[i] for i in indices], axis=0)
        for name, indices in groups.items()
    }


def scale_correlations(
    x: ArrayLike, y: ArrayLike, wavelet: str = "db4", level: int = 6
) -> FloatArray:
    """Correlate two series separately on every scale.

    Parameters
    ----------
    x, y : array_like
        Series of the same length, e.g. log returns of two assets.
    wavelet : str, optional
        Wavelet name.
    level : int, optional
        Number of levels.

    Returns
    -------
    numpy.ndarray
        Correlation coefficients of the components, in the order of
        :func:`multiscale_components`.
    """
    first = multiscale_components(x, wavelet, level)
    second = multiscale_components(y, wavelet, level)
    return np.array(
        [float(np.corrcoef(a, b)[0, 1]) for a, b in zip(first, second, strict=True)]
    )

"""Convolution, sampling, aliasing and quantization of signals (week 7)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def convolve(x: ArrayLike, h: ArrayLike) -> FloatArray:
    """Compute the full linear convolution of two finite signals.

    Parameters
    ----------
    x : array_like
        Signal of length ``N``.
    h : array_like
        Filter of length ``M``.

    Returns
    -------
    numpy.ndarray
        ``y[n] = sum_k h[k] x[n - k]`` of length ``N + M - 1``; equal to
        :func:`numpy.convolve`.
    """
    signal = np.asarray(x, dtype=np.float64)
    taps = np.asarray(h, dtype=np.float64)
    result = np.zeros(signal.size + taps.size - 1)
    for shift, tap in enumerate(taps):
        result[shift : shift + signal.size] += tap * signal
    return result


def circular_convolve(x: ArrayLike, h: ArrayLike) -> FloatArray:
    """Compute the circular convolution of two signals of equal length.

    By the convolution theorem it equals ``ifft(fft(x) * fft(h))``.

    Parameters
    ----------
    x, h : array_like
        Signals of the same length ``N``.

    Returns
    -------
    numpy.ndarray
        ``y[n] = sum_k h[k] x[(n - k) mod N]``.

    Raises
    ------
    ValueError
        If the signals have different lengths.
    """
    signal = np.asarray(x, dtype=np.float64)
    taps = np.asarray(h, dtype=np.float64)
    if signal.shape != taps.shape:
        raise ValueError("x and h must have the same length")
    return np.asarray(
        sum(tap * np.roll(signal, shift) for shift, tap in enumerate(taps))
    )


def sample(
    f: Callable[[FloatArray], ArrayLike], rate: float, duration: float
) -> tuple[FloatArray, FloatArray]:
    """Sample a function uniformly on ``[0, duration)``.

    Parameters
    ----------
    f : callable
        Vectorized function of time.
    rate : float
        Sampling rate in samples per unit of time.
    duration : float
        Length of the time interval.

    Returns
    -------
    t : numpy.ndarray
        Sampling instants ``n / rate``.
    samples : numpy.ndarray
        Values ``f(t)``.
    """
    t = np.arange(int(np.floor(duration * rate))) / rate
    return t, np.asarray(f(t), dtype=np.float64)


def sinc_interpolate(samples: ArrayLike, rate: float, t: ArrayLike) -> FloatArray:
    """Reconstruct a signal from its samples with the Shannon formula.

    ``f(t) = sum_n f(n T) sinc((t - n T) / T)`` with ``T = 1 / rate``. The
    reconstruction is exact for band-limited signals sampled above the
    Nyquist rate, up to truncation of the series.

    Parameters
    ----------
    samples : array_like
        Samples ``f(n T)``, ``n = 0, 1, ...``.
    rate : float
        Sampling rate ``1 / T``.
    t : array_like
        Points at which to evaluate the reconstruction.

    Returns
    -------
    numpy.ndarray
        Reconstructed values.
    """
    values = np.asarray(samples, dtype=np.float64)
    points = np.asarray(t, dtype=np.float64)
    nodes = np.arange(values.size) / rate
    return np.asarray(np.sinc(rate * np.subtract.outer(points, nodes)) @ values)


def aliased_frequency(frequency: float, rate: float) -> float:
    """Return the apparent frequency of a sampled sinusoid.

    A sinusoid of frequency ``f`` sampled at rate ``fs`` is indistinguishable
    from one of frequency ``|f - k fs|`` in ``[0, fs / 2]``.

    Parameters
    ----------
    frequency : float
        True frequency.
    rate : float
        Sampling rate.

    Returns
    -------
    float
        Apparent frequency in ``[0, rate / 2]``.
    """
    folded = abs(frequency) % rate
    return float(min(folded, rate - folded))


def quantize(
    x: ArrayLike, n_bits: int, value_range: tuple[float, float] = (-1.0, 1.0)
) -> FloatArray:
    """Quantize a signal uniformly with ``2 ** n_bits`` levels.

    Parameters
    ----------
    x : array_like
        Signal.
    n_bits : int
        Number of bits per sample.
    value_range : tuple of float, optional
        Interval ``(low, high)`` covered by the levels; values outside are
        clipped.

    Returns
    -------
    numpy.ndarray
        Signal rounded to the centres of the quantization cells.

    Raises
    ------
    ValueError
        If ``n_bits`` is smaller than 1 or the range is empty.
    """
    low, high = value_range
    if n_bits < 1:
        raise ValueError("n_bits must be at least 1")
    if high <= low:
        raise ValueError("value_range must satisfy low < high")
    levels = 2**n_bits
    cell = (high - low) / levels
    index = np.floor((np.asarray(x, dtype=np.float64) - low) / cell)
    index = np.clip(index, 0, levels - 1)
    return np.asarray(low + (index + 0.5) * cell, dtype=np.float64)


def sqnr(signal: ArrayLike, quantized: ArrayLike) -> float:
    """Return the signal-to-quantization-noise ratio in decibels.

    Parameters
    ----------
    signal : array_like
        Original signal.
    quantized : array_like
        Quantized signal.

    Returns
    -------
    float
        ``10 * log10(power(signal) / power(signal - quantized))``.
    """
    original = np.asarray(signal, dtype=np.float64)
    noise = original - np.asarray(quantized, dtype=np.float64)
    noise_power = np.mean(noise**2)
    if noise_power == 0:
        return float("inf")
    return float(10.0 * np.log10(np.mean(original**2) / noise_power))

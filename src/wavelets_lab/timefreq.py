"""Time-frequency analysis: STFT (Gabor transform) and CWT (week 8)."""

from __future__ import annotations

import numpy as np
import pywt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def chirp(t: ArrayLike, f0: float, f1: float) -> FloatArray:
    """Return a linear chirp whose frequency grows from ``f0`` to ``f1``.

    Parameters
    ----------
    t : array_like
        Increasing time instants; the sweep spans ``t[0]`` to ``t[-1]``.
    f0 : float
        Instantaneous frequency at ``t[0]``.
    f1 : float
        Instantaneous frequency at ``t[-1]``.

    Returns
    -------
    numpy.ndarray
        ``cos(2 pi (f0 s + (f1 - f0) s**2 / (2 T)))`` with ``s = t - t[0]``.
    """
    times = np.asarray(t, dtype=np.float64)
    shifted = times - times[0]
    duration = shifted[-1]
    phase = f0 * shifted + (f1 - f0) * shifted**2 / (2.0 * duration)
    return np.asarray(np.cos(2.0 * np.pi * phase), dtype=np.float64)


def gaussian_window(length: int, sigma: float | None = None) -> FloatArray:
    """Return a Gaussian (Gabor) window.

    Parameters
    ----------
    length : int
        Number of samples.
    sigma : float, optional
        Standard deviation in samples; ``length / 6`` by default.

    Returns
    -------
    numpy.ndarray
        Window with maximum 1 at its centre.
    """
    width = length / 6.0 if sigma is None else sigma
    offsets = np.arange(length) - (length - 1) / 2.0
    return np.asarray(np.exp(-(offsets**2) / (2.0 * width**2)), dtype=np.float64)


def stft(
    x: ArrayLike, window: ArrayLike, hop: int, rate: float = 1.0
) -> tuple[FloatArray, FloatArray, ComplexArray]:
    """Compute the short-time Fourier transform of a real signal.

    The signal is cut into windowed segments starting every ``hop`` samples,
    and each segment is transformed with :func:`numpy.fft.rfft`.

    Parameters
    ----------
    x : array_like
        Real one-dimensional signal.
    window : array_like
        Analysis window, e.g. from :func:`gaussian_window`.
    hop : int
        Shift between consecutive segments in samples.
    rate : float, optional
        Sampling rate, used to label the axes.

    Returns
    -------
    frequencies : numpy.ndarray
        Frequencies of the rows.
    times : numpy.ndarray
        Centres of the segments.
    coefficients : numpy.ndarray
        Complex array of shape ``(len(frequencies), len(times))``.

    Raises
    ------
    ValueError
        If the window is longer than the signal or ``hop`` is not positive.
    """
    signal = np.asarray(x, dtype=np.float64)
    taper = np.asarray(window, dtype=np.float64)
    if taper.size > signal.size:
        raise ValueError("the window must not be longer than the signal")
    if hop <= 0:
        raise ValueError("hop must be positive")
    starts = np.arange(0, signal.size - taper.size + 1, hop)
    segments = np.stack([signal[s : s + taper.size] * taper for s in starts])
    coefficients = np.fft.rfft(segments, axis=1).T
    frequencies = np.asarray(np.fft.rfftfreq(taper.size, d=1.0 / rate), dtype=float)
    times = np.asarray((starts + (taper.size - 1) / 2.0) / rate, dtype=float)
    return frequencies, times, np.asarray(coefficients, dtype=np.complex128)


def cwt(
    x: ArrayLike,
    scales: ArrayLike,
    wavelet: str = "morl",
    rate: float = 1.0,
) -> tuple[NDArray[np.inexact], FloatArray]:
    """Compute the continuous wavelet transform with :func:`pywt.cwt`.

    Parameters
    ----------
    x : array_like
        One-dimensional signal.
    scales : array_like
        Positive scales.
    wavelet : str, optional
        Name of a continuous wavelet, e.g. ``"morl"`` or ``"mexh"``.
    rate : float, optional
        Sampling rate, used to convert scales into frequencies.

    Returns
    -------
    coefficients : numpy.ndarray
        Array of shape ``(len(scales), len(x))``.
    frequencies : numpy.ndarray
        Pseudo-frequency corresponding to each scale.
    """
    coefficients, frequencies = pywt.cwt(
        np.asarray(x, dtype=np.float64),
        np.asarray(scales, dtype=np.float64),
        wavelet,
        sampling_period=1.0 / rate,
    )
    return np.asarray(coefficients), np.asarray(frequencies, dtype=np.float64)


def scale_for_frequency(
    frequencies: ArrayLike, wavelet: str = "morl", rate: float = 1.0
) -> FloatArray:
    """Return the scales whose pseudo-frequencies are ``frequencies``.

    Parameters
    ----------
    frequencies : array_like
        Positive frequencies.
    wavelet : str, optional
        Name of a continuous wavelet.
    rate : float, optional
        Sampling rate.

    Returns
    -------
    numpy.ndarray
        ``central_frequency * rate / frequencies``.
    """
    centre = pywt.central_frequency(wavelet)
    return np.asarray(centre * rate / np.asarray(frequencies, dtype=np.float64))


def power(coefficients: ArrayLike) -> FloatArray:
    """Return the squared modulus, i.e. a spectrogram or a scalogram.

    Parameters
    ----------
    coefficients : array_like
        STFT or CWT coefficients.

    Returns
    -------
    numpy.ndarray
        ``abs(coefficients) ** 2``.
    """
    return np.asarray(np.abs(np.asarray(coefficients)) ** 2, dtype=np.float64)


def plot_time_frequency(
    ax: Axes,
    times: ArrayLike,
    frequencies: ArrayLike,
    values: ArrayLike,
    title: str | None = None,
) -> Axes:
    """Draw a spectrogram or scalogram as a colour map.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    times : array_like
        Time coordinates of the columns.
    frequencies : array_like
        Frequency coordinates of the rows.
    values : array_like
        Array of shape ``(len(frequencies), len(times))``.
    title : str, optional
        Title of the plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    ax.pcolormesh(
        np.asarray(times), np.asarray(frequencies), np.asarray(values), shading="auto"
    )
    ax.set_xlabel("time")
    ax.set_ylabel("frequency")
    if title is not None:
        ax.set_title(title)
    return ax

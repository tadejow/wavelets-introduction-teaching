"""Tests for :mod:`wavelets_lab.timefreq`."""

import numpy as np
import pytest

from wavelets_lab import timefreq


def test_chirp_starts_at_one() -> None:
    """The chirp is a cosine with zero initial phase."""
    t = np.linspace(0.0, 1.0, 100)
    signal = timefreq.chirp(t, 5.0, 20.0)
    assert signal[0] == pytest.approx(1.0)
    assert np.abs(signal).max() <= 1.0


def test_gaussian_window_is_symmetric() -> None:
    """The Gabor window peaks in the middle and is symmetric."""
    window = timefreq.gaussian_window(65)
    assert window[32] == pytest.approx(1.0)
    np.testing.assert_allclose(window, window[::-1])


def test_stft_locates_sinusoid() -> None:
    """The spectrogram of a pure tone peaks at its frequency."""
    rate = 1000.0
    t = np.arange(2048) / rate
    window = timefreq.gaussian_window(256)
    freqs, times, coeffs = timefreq.stft(
        np.sin(2 * np.pi * 125.0 * t), window, 64, rate
    )
    assert coeffs.shape == (freqs.size, times.size)
    peaks = freqs[np.argmax(timefreq.power(coeffs), axis=0)]
    np.testing.assert_allclose(peaks, 125.0, atol=rate / 256)
    with pytest.raises(ValueError):
        timefreq.stft(np.ones(10), window, 64)
    with pytest.raises(ValueError):
        timefreq.stft(np.ones(300), window, 0)


def test_stft_follows_chirp() -> None:
    """The dominant frequency of a chirp increases in time."""
    rate = 1000.0
    t = np.arange(4096) / rate
    freqs, _, coeffs = timefreq.stft(
        timefreq.chirp(t, 50.0, 400.0), timefreq.gaussian_window(256), 128, rate
    )
    peaks = freqs[np.argmax(timefreq.power(coeffs), axis=0)]
    assert peaks[-1] > peaks[0] + 250.0


def test_cwt_frequency_of_scales() -> None:
    """Scales chosen for given frequencies give those pseudo-frequencies."""
    rate = 100.0
    targets = np.array([5.0, 10.0, 20.0])
    scales = timefreq.scale_for_frequency(targets, "morl", rate)
    t = np.arange(1000) / rate
    coeffs, freqs = timefreq.cwt(np.sin(2 * np.pi * 10.0 * t), scales, "morl", rate)
    np.testing.assert_allclose(freqs, targets)
    energy = timefreq.power(coeffs).sum(axis=1)
    assert np.argmax(energy) == 1

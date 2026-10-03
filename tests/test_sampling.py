"""Tests for :mod:`wavelets_lab.sampling`."""

import numpy as np
import pytest

from wavelets_lab import sampling


def test_convolve_matches_numpy() -> None:
    """Own linear convolution agrees with :func:`numpy.convolve`."""
    rng = np.random.default_rng(0)
    x, h = rng.standard_normal(20), rng.standard_normal(5)
    np.testing.assert_allclose(sampling.convolve(x, h), np.convolve(x, h))


def test_circular_convolution_theorem() -> None:
    """Circular convolution is a product in the Fourier domain."""
    rng = np.random.default_rng(1)
    x, h = rng.standard_normal(16), rng.standard_normal(16)
    expected = np.fft.ifft(np.fft.fft(x) * np.fft.fft(h)).real
    np.testing.assert_allclose(sampling.circular_convolve(x, h), expected)
    with pytest.raises(ValueError):
        sampling.circular_convolve(x, h[:8])


def test_sinc_interpolation_recovers_band_limited_signal() -> None:
    """A slow sinusoid is reconstructed well inside the sampled interval."""

    def signal(t: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * 3.0 * t)

    _, samples = sampling.sample(signal, rate=20.0, duration=20.0)
    t = np.linspace(8.0, 12.0, 300)
    reconstruction = sampling.sinc_interpolate(samples, 20.0, t)
    np.testing.assert_allclose(reconstruction, signal(t), atol=0.02)


def test_aliased_frequency() -> None:
    """Frequencies above Nyquist fold back into ``[0, fs / 2]``."""
    assert sampling.aliased_frequency(3.0, 10.0) == 3.0
    assert sampling.aliased_frequency(7.0, 10.0) == 3.0
    assert sampling.aliased_frequency(13.0, 10.0) == 3.0


def test_quantize_levels_and_sqnr() -> None:
    """Each extra bit adds about 6 dB of SQNR."""
    x = np.sin(np.linspace(0, 2 * np.pi, 10000))
    q8 = sampling.quantize(x, 8)
    assert np.unique(q8).size <= 256
    gain = sampling.sqnr(x, q8) - sampling.sqnr(x, sampling.quantize(x, 7))
    assert gain == pytest.approx(6.02, abs=0.3)
    assert sampling.sqnr(x, x) == float("inf")
    with pytest.raises(ValueError):
        sampling.quantize(x, 0)

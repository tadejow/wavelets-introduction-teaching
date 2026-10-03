"""Tests for :mod:`wavelets_lab.fourier`."""

import numpy as np
import pytest
from scipy import fft as sp_fft

from wavelets_lab import fourier


def test_square_wave_coefficients() -> None:
    """The square wave has ``c_n = 2 / (i pi n)`` for odd ``n``."""
    coefficients = fourier.fourier_coefficients(fourier.square_wave, 5)
    for n in (1, 3, 5):
        assert coefficients[5 + n] == pytest.approx(2 / (1j * np.pi * n), abs=1e-3)
    assert coefficients[5 + 2] == pytest.approx(0.0, abs=1e-3)
    with pytest.raises(ValueError):
        fourier.fourier_coefficients(fourier.square_wave, 10, n_samples=20)


def test_partial_sum_of_trigonometric_polynomial() -> None:
    """A trigonometric polynomial equals its own partial sum."""
    t = np.linspace(0, 2 * np.pi, 50)
    coefficients = fourier.fourier_coefficients(lambda s: np.cos(2 * s), 3)
    np.testing.assert_allclose(
        fourier.fourier_partial_sum(coefficients, t).real, np.cos(2 * t), atol=1e-12
    )


def test_gibbs_overshoot() -> None:
    """Partial sums of the square wave overshoot by about 9 percent of the jump."""
    t = np.linspace(0.0, 0.5, 2001)
    coefficients = fourier.fourier_coefficients(fourier.square_wave, 101, 8192)
    peak = fourier.fourier_partial_sum(coefficients, t).real.max()
    assert peak == pytest.approx(1.179, abs=0.01)


@pytest.mark.parametrize("n", [1, 2, 8, 64])
def test_dft_and_fft_match_numpy(n: int) -> None:
    """Own DFT and radix-2 FFT agree with :func:`numpy.fft.fft`."""
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    np.testing.assert_allclose(fourier.dft(x), np.fft.fft(x), atol=1e-10)
    np.testing.assert_allclose(fourier.fft_radix2(x), np.fft.fft(x), atol=1e-10)
    np.testing.assert_allclose(fourier.idft(fourier.dft(x)), x, atol=1e-10)


def test_fft_rejects_bad_length() -> None:
    """The radix-2 FFT needs a power-of-two length."""
    with pytest.raises(ValueError):
        fourier.fft_radix2(np.ones(6))


def test_dct_matches_scipy() -> None:
    """Own DCT-II agrees with the orthonormal SciPy DCT."""
    x = np.random.default_rng(0).standard_normal(16)
    np.testing.assert_allclose(fourier.dct2(x), sp_fft.dct(x, norm="ortho"))


def test_magnitude_phase_roundtrip() -> None:
    """Magnitude and phase together rebuild the image."""
    image = np.random.default_rng(0).uniform(0, 255, (16, 16))
    magnitude, phase = fourier.magnitude_phase(image)
    np.testing.assert_allclose(fourier.from_magnitude_phase(magnitude, phase), image)
    assert fourier.log_magnitude(magnitude).max() < magnitude.max()


def test_band_code() -> None:
    """Band coding keeps the expected fraction and the mean intensity."""
    image = np.random.default_rng(0).uniform(0, 255, (512, 512))
    reconstruction, kept = fourier.band_code(image, 56)
    assert kept == pytest.approx(12544 / 512**2)
    assert reconstruction.mean() == pytest.approx(image.mean())
    stats = fourier.compression_stats(kept)
    assert stats["ratio"] == pytest.approx(20.9, abs=0.1)
    assert stats["bpp"] == pytest.approx(0.38, abs=0.01)
    with pytest.raises(ValueError):
        fourier.compression_stats(0.0)

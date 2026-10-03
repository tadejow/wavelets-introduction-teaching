"""Tests for :mod:`wavelets_lab.denoising`."""

import numpy as np
import pytest

from wavelets_lab import denoising, filters, images


def test_estimate_sigma_of_white_noise() -> None:
    """The MAD estimator recovers the deviation of Gaussian noise."""
    noise = 3.0 * np.random.default_rng(0).standard_normal(100000)
    assert denoising.estimate_sigma(noise) == pytest.approx(3.0, rel=0.02)


def test_universal_threshold() -> None:
    """VisuShrink uses ``sigma sqrt(2 log n)``."""
    assert denoising.universal_threshold(1.0, 1024) == pytest.approx(
        np.sqrt(2 * np.log(1024))
    )


def test_sure_threshold_bounds() -> None:
    """The SURE threshold is between zero and the universal threshold."""
    detail = np.random.default_rng(1).standard_normal(512)
    threshold = denoising.sure_threshold(detail, 1.0)
    assert 0.0 <= threshold <= denoising.universal_threshold(1.0, 512)
    assert denoising.sure_threshold(np.array([]), 1.0) == 0.0


@pytest.mark.parametrize("method", ["visu", "sure"])
def test_denoise_signal_reduces_error(method: str) -> None:
    """Denoising the noisy Doppler signal reduces its mean squared error."""
    clean = denoising.doppler(2048)
    noisy = clean + 0.05 * np.random.default_rng(2).standard_normal(clean.size)
    result = denoising.denoise_signal(noisy, "db4", 5, method=method)
    assert np.mean((result - clean) ** 2) < 0.5 * np.mean((noisy - clean) ** 2)


def test_denoise_signal_invalid_options() -> None:
    """Unknown methods and modes are rejected."""
    with pytest.raises(ValueError):
        denoising.denoise_signal(np.ones(64), level=2, method="bayes")
    with pytest.raises(ValueError):
        denoising.denoise_signal(np.ones(64), level=2, mode="median")
    with pytest.raises(ValueError):
        denoising.denoise_image(np.ones((16, 16)), level=2, mode="median")


def test_denoise_image_improves_psnr() -> None:
    """Wavelet denoising improves the PSNR of a noisy photograph."""
    clean = images.load_test_image("camera")
    noisy = filters.add_gaussian_noise(clean, 20.0, rng=3)
    result = denoising.denoise_image(noisy, "db4", 3)
    assert filters.psnr(clean, result) > filters.psnr(clean, noisy) + 3.0


def test_heavisine_has_jumps() -> None:
    """HeaviSine has jumps of size 2 at 0.3 and 0.72."""
    signal = denoising.heavisine(1000)
    assert abs(signal[301] - signal[299]) > 1.5
    assert abs(signal[721] - signal[719]) > 1.5

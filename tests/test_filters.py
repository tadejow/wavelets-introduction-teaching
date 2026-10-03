"""Tests for :mod:`wavelets_lab.filters`."""

import numpy as np
import pytest
from scipy import ndimage

from wavelets_lab import filters


def test_masks_are_normalized() -> None:
    """The Gaussian masks of the notes sum to one."""
    assert filters.GAUSS_3X3.sum() == pytest.approx(1.0)
    assert filters.GAUSS_5X5.sum() == pytest.approx(1.0)
    assert filters.gaussian_kernel(7, 1.5).sum() == pytest.approx(1.0)


def test_gaussian_kernel_invalid() -> None:
    """Even sizes and non-positive sigmas are rejected."""
    with pytest.raises(ValueError):
        filters.gaussian_kernel(4, 1.0)
    with pytest.raises(ValueError):
        filters.gaussian_kernel(3, 0.0)


def test_convolve2d_matches_scipy() -> None:
    """Own convolution agrees with ``scipy.ndimage.convolve``."""
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 255, (20, 30))
    kernel = rng.uniform(size=(3, 5))
    expected = ndimage.convolve(image, kernel, mode="reflect")
    np.testing.assert_allclose(filters.convolve2d(image, kernel), expected)


def test_gaussian_filter_preserves_constant() -> None:
    """Smoothing a constant image changes nothing."""
    image = np.full((16, 16), 77.0)
    np.testing.assert_allclose(filters.gaussian_filter(image, 5), image)
    with pytest.raises(ValueError):
        filters.gaussian_filter(image, 4)


def test_median_filter_matches_scipy() -> None:
    """Own median filter agrees with ``scipy.ndimage.median_filter``."""
    rng = np.random.default_rng(1)
    image = rng.uniform(0, 255, (15, 15))
    expected = ndimage.median_filter(image, size=3, mode="reflect")
    np.testing.assert_allclose(filters.median_filter(image, 3), expected)


def test_median_removes_impulse() -> None:
    """A single outlier is removed by the median filter."""
    image = np.zeros((9, 9))
    image[4, 4] = 255.0
    np.testing.assert_array_equal(filters.median_filter(image), 0.0)


def test_noise_is_reproducible() -> None:
    """The same seed gives the same noise with the right deviation."""
    image = np.zeros((200, 200))
    first = filters.add_gaussian_noise(image, 20.0, rng=3)
    second = filters.add_gaussian_noise(image, 20.0, rng=3)
    np.testing.assert_array_equal(first, second)
    assert first.std() == pytest.approx(20.0, rel=0.05)


def test_denoising_improves_psnr() -> None:
    """Both filters improve the PSNR of the noisy two-level image."""
    clean = filters.two_level_image()
    noisy = filters.add_gaussian_noise(clean, 5.0, rng=0)
    base = filters.psnr(clean, noisy)
    assert filters.psnr(clean, filters.gaussian_filter(noisy)) > base
    assert filters.psnr(clean, filters.median_filter(noisy)) > base
    assert filters.psnr(clean, clean) == float("inf")

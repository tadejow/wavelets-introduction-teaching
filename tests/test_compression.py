"""Tests for :mod:`wavelets_lab.compression`."""

import numpy as np
import pytest
import pywt

from wavelets_lab import compression, dwt, images


def test_thresholds_match_pywt() -> None:
    """Hard and soft thresholding agree with ``pywt.threshold``."""
    x = np.linspace(-3.0, 3.0, 61)
    np.testing.assert_allclose(
        compression.hard_threshold(x, 1.0), pywt.threshold(x, 1.0, "hard")
    )
    np.testing.assert_allclose(
        compression.soft_threshold(x, 1.0), pywt.threshold(x, 1.0, "soft")
    )


def test_threshold_details_keeps_approximation() -> None:
    """Only details are thresholded."""
    image = np.random.default_rng(0).uniform(0, 255, (32, 32))
    coefficients = dwt.decompose_image(image, "haar", 2)
    result = compression.threshold_details(coefficients, 1e9)
    np.testing.assert_array_equal(result[0], coefficients[0])
    assert all(not np.any(band) for bands in result[1:] for band in bands)
    with pytest.raises(ValueError):
        compression.threshold_details(coefficients, 1.0, "median")


@pytest.mark.parametrize("ratio", [0.9, 0.95, 0.98])
def test_compress_image_reaches_ratio(ratio: float) -> None:
    """The requested fraction of zeros is reached on a real photograph."""
    image = images.load_test_image("camera")
    reconstruction, achieved = compression.compress_image(image, ratio, "db3", 4)
    assert reconstruction.shape == image.shape
    assert achieved == pytest.approx(ratio, abs=0.005)
    assert np.sqrt(np.mean((reconstruction - image) ** 2)) < 25.0


def test_compression_error_grows_with_ratio() -> None:
    """Keeping fewer coefficients gives a larger error."""
    image = images.load_test_image("camera")
    errors = [
        np.mean((compression.compress_image(image, r, "db2", 4)[0] - image) ** 2)
        for r in (0.5, 0.9, 0.98)
    ]
    assert errors[0] < errors[1] < errors[2]


def test_soft_rescale_changes_result() -> None:
    """Rescaling after soft thresholding brings details closer to the original."""
    image = images.load_test_image("camera")
    plain, _ = compression.compress_image(image, 0.9, "db2", 3, mode="soft")
    rescaled, _ = compression.compress_image(
        image, 0.9, "db2", 3, mode="soft", rescale=True
    )
    assert not np.allclose(plain, rescaled)
    with pytest.raises(ValueError):
        compression.threshold_for_ratio(dwt.decompose_image(image, "db2", 3), 1.0)

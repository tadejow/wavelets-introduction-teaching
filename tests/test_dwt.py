"""Tests for :mod:`wavelets_lab.dwt`."""

import numpy as np
import pytest
import pywt

from wavelets_lab import dwt


@pytest.mark.parametrize("wavelet", ["haar", "db2", "db3", "sym4"])
def test_analysis_step_matches_pywt(wavelet: str) -> None:
    """One Mallat step equals ``pywt.dwt`` with periodization."""
    x = np.random.default_rng(1).standard_normal(32)
    approximation, detail = dwt.analysis_step(x, wavelet)
    expected = pywt.dwt(x, wavelet, mode="periodization")
    np.testing.assert_allclose(approximation, expected[0])
    np.testing.assert_allclose(detail, expected[1])
    np.testing.assert_allclose(dwt.synthesis_step(approximation, detail, wavelet), x)


def test_analysis_step_rejects_odd_length() -> None:
    """Odd signal lengths are rejected."""
    with pytest.raises(ValueError):
        dwt.analysis_step(np.ones(7))


def test_multilevel_matches_wavedec() -> None:
    """The multilevel transform equals ``pywt.wavedec`` and is invertible."""
    x = np.random.default_rng(2).standard_normal(64)
    coefficients = dwt.decompose(x, "db2", 3)
    expected = pywt.wavedec(x, "db2", mode="periodization", level=3)
    for ours, theirs in zip(coefficients, expected, strict=True):
        np.testing.assert_allclose(ours, theirs)
    np.testing.assert_allclose(dwt.reconstruct(coefficients, "db2"), x)


def test_transform_preserves_energy() -> None:
    """The periodic DWT is orthogonal."""
    x = np.random.default_rng(3).standard_normal(128)
    energy = sum(float(np.sum(c**2)) for c in dwt.decompose(x, "db3", 4))
    assert energy == pytest.approx(float(np.sum(x**2)))


def test_image_roundtrip_and_display() -> None:
    """The 2D DWT is invertible and its display has the image size."""
    image = np.random.default_rng(4).uniform(0, 255, (64, 64))
    coefficients = dwt.decompose_image(image, "db3", 3)
    np.testing.assert_allclose(dwt.reconstruct_image(coefficients, "db3"), image)
    display = dwt.coefficient_image(coefficients)
    assert display.shape == image.shape
    assert display.max() == pytest.approx(255.0)

"""Tests for :mod:`wavelets_lab.transforms`."""

import numpy as np
import pytest

from wavelets_lab import transforms


def test_radon_roundtrip() -> None:
    """Filtered back-projection approximately inverts the Radon transform."""
    size = 64
    rows, cols = np.mgrid[:size, :size] - size / 2
    image = ((rows / 12) ** 2 + (cols / 20) ** 2 <= 1).astype(float)
    sinogram = transforms.radon(image)
    assert sinogram.shape[1] == 180
    reconstruction = transforms.inverse_radon(sinogram)
    assert np.mean(np.abs(reconstruction - image)) < 0.1


def test_hough_finds_line() -> None:
    """A diagonal line gives the strongest vote at 45 degrees."""
    edges = np.eye(50, dtype=bool)[::-1]
    accumulator, angles, rhos = transforms.hough_lines(edges)
    assert accumulator.max() >= 49
    ((theta, rho),) = transforms.strongest_lines(accumulator, angles, rhos, 1)
    assert np.rad2deg(theta) == pytest.approx(45.0, abs=1.0)
    assert rho == pytest.approx(49 / np.sqrt(2), abs=1.0)


def test_zak_roundtrip_and_values() -> None:
    """The Zak transform is invertible and sums the right samples."""
    x = np.random.default_rng(0).standard_normal(24)
    z = transforms.zak(x, period=6)
    assert z.shape == (6, 4)
    np.testing.assert_allclose(z[:, 0], x.reshape(4, 6).sum(axis=0))
    np.testing.assert_allclose(transforms.inverse_zak(z).real, x)
    with pytest.raises(ValueError):
        transforms.zak(x, period=5)

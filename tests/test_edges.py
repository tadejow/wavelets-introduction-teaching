"""Tests for :mod:`wavelets_lab.edges`."""

import numpy as np
import pytest

from wavelets_lab import edges, filters


def test_gradient_of_ramp() -> None:
    """A horizontal ramp has a constant x derivative and no y derivative."""
    image = np.tile(np.arange(10.0), (10, 1))
    gx, gy = edges.gradient(image)
    np.testing.assert_allclose(gx[1:-1, 1:-1], 8.0)
    np.testing.assert_allclose(gy[1:-1, 1:-1], 0.0)


def test_quantize_direction() -> None:
    """Directions 0, 45, 90 and 135 degrees map to 0, 1, 2, 3."""
    gx = np.array([1.0, 1.0, 0.0, -1.0])
    gy = np.array([0.0, -1.0, -1.0, -1.0])
    np.testing.assert_array_equal(edges.quantize_direction(gx, gy), [0, 1, 2, 3])


def test_non_maximum_suppression_thins() -> None:
    """Only the ridge of a blurred vertical edge survives."""
    magnitude = np.tile(np.array([0.0, 1.0, 3.0, 1.0, 0.0]), (5, 1))
    direction = np.zeros((5, 5), dtype=int)
    thin = edges.non_maximum_suppression(magnitude, direction)
    np.testing.assert_array_equal(thin[:, 2], 3.0)
    assert thin[:, [0, 1, 3, 4]].sum() == 0.0


def test_hysteresis_keeps_connected_weak_pixels() -> None:
    """Weak pixels survive only when connected to a strong one."""
    values = np.array([[0, 30, 60, 30, 0, 30]], dtype=float)
    np.testing.assert_array_equal(
        edges.hysteresis_threshold(values, 20, 50), [[0, 1, 1, 1, 0, 0]]
    )
    with pytest.raises(ValueError):
        edges.hysteresis_threshold(values, 60, 50)


def test_canny_finds_square_outline() -> None:
    """The edges of the two-level image lie on the square's border."""
    image = filters.two_level_image((64, 64))
    edge_map = edges.canny(image)
    rows, cols = np.nonzero(edge_map)
    assert edge_map.any()
    on_border = (
        (np.abs(rows - 16) <= 1)
        | (np.abs(rows - 47) <= 1)
        | (np.abs(cols - 16) <= 1)
        | (np.abs(cols - 47) <= 1)
    )
    assert on_border.all()
    assert not edges.canny(np.full((32, 32), 100.0)).any()

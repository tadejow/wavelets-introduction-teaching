"""Tests for :mod:`wavelets_lab.hilbert`."""

import numpy as np
import pytest

from wavelets_lab import hilbert


def test_inner_product_conjugates_second_argument() -> None:
    """``<f, g>`` is linear in ``f`` and conjugate-linear in ``g``."""
    f = np.array([1.0, 1j])
    g = np.array([1j, 1.0])
    assert hilbert.inner_product(f, g) == pytest.approx(0.0)
    assert hilbert.inner_product(1j * f, f) == pytest.approx(2j)
    assert hilbert.norm([3.0, 4.0]) == pytest.approx(5.0)


def test_gram_schmidt_orthonormal_and_same_span() -> None:
    """Gram-Schmidt gives an orthonormal family with the same span."""
    grid = np.linspace(0.0, 1.0, 201)
    step = grid[1] - grid[0]
    family = hilbert.hat_functions(grid, np.linspace(0.0, 1.0, 6))
    basis = hilbert.gram_schmidt(family, step)
    assert basis.shape == family.shape
    assert hilbert.is_orthonormal(basis, step)
    assert not hilbert.is_orthonormal(family, step)
    for row in family:
        np.testing.assert_allclose(hilbert.project(row, basis, step), row, atol=1e-10)


def test_gram_schmidt_skips_dependent() -> None:
    """Linearly dependent vectors are dropped."""
    family = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert hilbert.gram_schmidt(family).shape == (2, 3)


def test_expansion_in_riesz_basis_is_exact() -> None:
    """A function in the span is recovered from its Riesz-basis coefficients."""
    grid = np.linspace(0.0, 1.0, 101)
    family = hilbert.hat_functions(grid, np.linspace(0.0, 1.0, 5))
    coefficients = np.array([1.0, -2.0, 0.5, 3.0, 0.0])
    f = hilbert.synthesize(coefficients, family)
    np.testing.assert_allclose(
        hilbert.expansion_coefficients(f, family), coefficients, atol=1e-12
    )


def test_projection_residual_is_orthogonal() -> None:
    """``f - Pf`` is orthogonal to every element of the family."""
    rng = np.random.default_rng(0)
    family = rng.standard_normal((3, 10))
    f = rng.standard_normal(10)
    residual = f - hilbert.project(f, family)
    np.testing.assert_allclose(family @ residual, 0.0, atol=1e-10)


def test_riesz_bounds() -> None:
    """An orthonormal basis has bounds 1, a dependent family has ``A = 0``."""
    lower, upper = hilbert.riesz_bounds(np.eye(4))
    assert lower == pytest.approx(1.0) and upper == pytest.approx(1.0)
    lower, _ = hilbert.riesz_bounds(np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert lower == pytest.approx(0.0, abs=1e-12)

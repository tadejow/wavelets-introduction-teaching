"""Hilbert-space tools on sampled functions (week 4).

Functions on an interval are represented by their samples on a uniform grid,
and the inner product of :math:`L^2` is approximated by the Riemann sum
``h * sum(f * conj(g))`` with the grid step ``h``. A family of functions is a
two-dimensional array whose rows are the sampled functions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Array = NDArray[np.inexact]


def inner_product(f: ArrayLike, g: ArrayLike, step: float = 1.0) -> complex:
    r"""Approximate the :math:`L^2` inner product of two sampled functions.

    Parameters
    ----------
    f, g : array_like
        Samples of the two functions on the same grid.
    step : float, optional
        Grid step ``h``; 1 gives the inner product of :math:`\ell^2`.

    Returns
    -------
    complex
        ``step * sum(f * conj(g))``.
    """
    return complex(step * np.vdot(np.asarray(g), np.asarray(f)))


def norm(f: ArrayLike, step: float = 1.0) -> float:
    """Return the norm induced by :func:`inner_product`.

    Parameters
    ----------
    f : array_like
        Samples of a function.
    step : float, optional
        Grid step.

    Returns
    -------
    float
        ``sqrt(<f, f>)``.
    """
    return float(np.sqrt(inner_product(f, f, step).real))


def gram_matrix(family: ArrayLike, step: float = 1.0) -> Array:
    """Return the Gram matrix ``G[i, j] = <e_j, e_i>`` of a family.

    Parameters
    ----------
    family : array_like
        Two-dimensional array with one sampled function per row.
    step : float, optional
        Grid step.

    Returns
    -------
    numpy.ndarray
        Hermitian positive semidefinite matrix of shape ``(n, n)``.
    """
    rows = np.atleast_2d(np.asarray(family))
    return np.asarray(step * (rows.conj() @ rows.T))


def gram_schmidt(
    family: ArrayLike, step: float = 1.0, tolerance: float = 1e-10
) -> Array:
    """Orthonormalize a family with the modified Gram-Schmidt process.

    Functions that are (numerically) linear combinations of the previous
    ones are skipped.

    Parameters
    ----------
    family : array_like
        Two-dimensional array with one sampled function per row.
    step : float, optional
        Grid step.
    tolerance : float, optional
        Norm below which a residual is treated as zero.

    Returns
    -------
    numpy.ndarray
        Orthonormal family spanning the same space, one function per row.
    """
    rows = np.atleast_2d(np.asarray(family))
    dtype = np.result_type(rows.dtype, np.float64)
    basis: list[Array] = []
    for vector in rows.astype(dtype):
        residual = vector.copy()
        for element in basis:
            residual = residual - inner_product(residual, element, step) * element
        length = norm(residual, step)
        if length > tolerance:
            basis.append(residual / length)
    if not basis:
        return np.zeros((0, rows.shape[1]), dtype=dtype)
    return np.asarray(np.vstack(basis))


def is_orthonormal(family: ArrayLike, step: float = 1.0, atol: float = 1e-8) -> bool:
    """Check whether a family is orthonormal.

    Parameters
    ----------
    family : array_like
        Two-dimensional array with one sampled function per row.
    step : float, optional
        Grid step.
    atol : float, optional
        Absolute tolerance for comparing the Gram matrix with the identity.

    Returns
    -------
    bool
        ``True`` if the Gram matrix is the identity up to ``atol``.
    """
    gram = gram_matrix(family, step)
    return bool(np.allclose(gram, np.eye(gram.shape[0]), atol=atol))


def expansion_coefficients(f: ArrayLike, family: ArrayLike, step: float = 1.0) -> Array:
    """Return the coefficients of the orthogonal projection of ``f``.

    The coefficients ``c`` minimize ``||f - sum_k c_k e_k||`` and solve the
    normal equations ``G c = (<f, e_i>)_i``. For an orthonormal family they
    are simply the inner products ``<f, e_k>``; for a Riesz basis they are the
    inner products with the dual basis.

    Parameters
    ----------
    f : array_like
        Samples of the expanded function.
    family : array_like
        Linearly independent family, one sampled function per row.
    step : float, optional
        Grid step.

    Returns
    -------
    numpy.ndarray
        Coefficients ``c_k``.
    """
    rows = np.atleast_2d(np.asarray(family))
    right_side = step * (rows.conj() @ np.asarray(f))
    return np.asarray(np.linalg.solve(gram_matrix(rows, step), right_side))


def synthesize(coefficients: ArrayLike, family: ArrayLike) -> Array:
    """Return the linear combination ``sum_k c_k e_k``.

    Parameters
    ----------
    coefficients : array_like
        Coefficients ``c_k``.
    family : array_like
        Family of functions, one per row.

    Returns
    -------
    numpy.ndarray
        Samples of the combination.
    """
    return np.asarray(np.asarray(coefficients) @ np.atleast_2d(np.asarray(family)))


def project(f: ArrayLike, family: ArrayLike, step: float = 1.0) -> Array:
    """Project ``f`` orthogonally onto the span of a family.

    Parameters
    ----------
    f : array_like
        Samples of the projected function.
    family : array_like
        Linearly independent family, one sampled function per row.
    step : float, optional
        Grid step.

    Returns
    -------
    numpy.ndarray
        Samples of the orthogonal projection.
    """
    return synthesize(expansion_coefficients(f, family, step), family)


def riesz_bounds(family: ArrayLike, step: float = 1.0) -> tuple[float, float]:
    """Return the optimal Riesz bounds ``A`` and ``B`` of a finite family.

    They satisfy ``A ||c||^2 <= ||sum_k c_k e_k||^2 <= B ||c||^2`` and are the
    smallest and largest eigenvalues of the Gram matrix.

    Parameters
    ----------
    family : array_like
        Family of functions, one per row.
    step : float, optional
        Grid step.

    Returns
    -------
    tuple of float
        The bounds ``(A, B)``; ``A > 0`` exactly for independent families.
    """
    eigenvalues = np.linalg.eigvalsh(gram_matrix(family, step))
    return float(eigenvalues[0]), float(eigenvalues[-1])


def hat_functions(grid: ArrayLike, knots: ArrayLike) -> FloatArray:
    """Sample the piecewise linear "hat" splines on a grid.

    The ``k``-th function equals 1 at ``knots[k]``, 0 at the other knots and
    is linear in between (order-1 splines). Hat functions form a Riesz basis
    that is not orthogonal, the example used for Gram-Schmidt in the notes.

    Parameters
    ----------
    grid : array_like
        Points at which the functions are sampled.
    knots : array_like
        Increasing knots.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(knots), len(grid))``.
    """
    points = np.asarray(grid, dtype=np.float64)
    nodes = np.asarray(knots, dtype=np.float64)
    identity = np.eye(nodes.size)
    return np.vstack([np.interp(points, nodes, row) for row in identity])

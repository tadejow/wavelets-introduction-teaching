"""Other transforms: Radon, Hough and Zak (week 15)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skimage import transform

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def radon(image: ArrayLike, angles: ArrayLike | None = None) -> FloatArray:
    """Compute the Radon transform (sinogram) of an image.

    Parameters
    ----------
    image : array_like
        Greyscale image, ideally zero outside the inscribed circle.
    angles : array_like, optional
        Projection angles in degrees; ``0, 1, ..., 179`` by default.

    Returns
    -------
    numpy.ndarray
        Sinogram with one column per angle.
    """
    theta = np.arange(180.0) if angles is None else np.asarray(angles, dtype=float)
    return np.asarray(
        transform.radon(np.asarray(image, dtype=np.float64), theta=theta),
        dtype=np.float64,
    )


def inverse_radon(sinogram: ArrayLike, angles: ArrayLike | None = None) -> FloatArray:
    """Reconstruct an image by filtered back-projection.

    Parameters
    ----------
    sinogram : array_like
        Output of :func:`radon`.
    angles : array_like, optional
        Projection angles in degrees, as passed to :func:`radon`.

    Returns
    -------
    numpy.ndarray
        Reconstructed image.
    """
    values = np.asarray(sinogram, dtype=np.float64)
    theta = (
        np.arange(float(values.shape[1]))
        if angles is None
        else np.asarray(angles, dtype=float)
    )
    return np.asarray(transform.iradon(values, theta=theta), dtype=np.float64)


def hough_lines(
    edges: ArrayLike, n_angles: int = 180
) -> tuple[NDArray[np.int_], FloatArray, FloatArray]:
    """Accumulate the Hough transform of a binary edge map.

    Every edge pixel ``(x, y)`` votes for all lines
    ``rho = x cos(theta) + y sin(theta)`` passing through it.

    Parameters
    ----------
    edges : array_like
        Boolean edge map.
    n_angles : int, optional
        Number of angles in ``[-90, 90)`` degrees.

    Returns
    -------
    accumulator : numpy.ndarray
        Vote counts of shape ``(len(rhos), n_angles)``.
    angles : numpy.ndarray
        Angles ``theta`` in radians.
    rhos : numpy.ndarray
        Distances ``rho`` in pixels.
    """
    mask = np.asarray(edges, dtype=bool)
    angles = np.deg2rad(np.linspace(-90.0, 90.0, n_angles, endpoint=False))
    diagonal = int(np.ceil(np.hypot(*mask.shape)))
    rhos = np.arange(-diagonal, diagonal + 1, dtype=np.float64)
    accumulator = np.zeros((rhos.size, angles.size), dtype=int)
    rows, cols = np.nonzero(mask)
    votes = np.rint(
        np.outer(cols, np.cos(angles)) + np.outer(rows, np.sin(angles))
    ).astype(int)
    for index in range(angles.size):
        np.add.at(accumulator[:, index], votes[:, index] + diagonal, 1)
    return accumulator, angles, rhos


def strongest_lines(
    accumulator: ArrayLike, angles: ArrayLike, rhos: ArrayLike, count: int = 1
) -> list[tuple[float, float]]:
    """Return the parameters of the lines with the most votes.

    Parameters
    ----------
    accumulator : array_like
        Output of :func:`hough_lines`.
    angles : array_like
        Angles of the accumulator columns.
    rhos : array_like
        Distances of the accumulator rows.
    count : int, optional
        Number of lines.

    Returns
    -------
    list of tuple of float
        Pairs ``(theta, rho)``, strongest first.
    """
    votes = np.asarray(accumulator)
    order = np.argsort(votes, axis=None)[::-1][:count]
    rho_idx, angle_idx = np.unravel_index(order, votes.shape)
    theta = np.asarray(angles, dtype=np.float64)
    rho = np.asarray(rhos, dtype=np.float64)
    return [
        (float(theta[a]), float(rho[r]))
        for r, a in zip(rho_idx, angle_idx, strict=True)
    ]


def zak(x: ArrayLike, period: int) -> ComplexArray:
    """Compute the discrete Zak transform.

    For a signal of length ``N = L M`` with ``L = period``,
    ``Z[n, k] = sum_{l < M} x[n + l L] exp(-2 pi i k l / M)``.

    Parameters
    ----------
    x : array_like
        Signal whose length is divisible by ``period``.
    period : int
        Time step ``L``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(L, M)``.

    Raises
    ------
    ValueError
        If the length of ``x`` is not divisible by ``period``.
    """
    signal = np.asarray(x, dtype=np.complex128)
    if signal.size % period:
        raise ValueError("the signal length must be divisible by period")
    blocks = signal.reshape(-1, period).T
    return np.asarray(np.fft.fft(blocks, axis=1), dtype=np.complex128)


def inverse_zak(coefficients: ArrayLike) -> ComplexArray:
    """Invert :func:`zak`.

    Parameters
    ----------
    coefficients : array_like
        Zak transform of shape ``(L, M)``.

    Returns
    -------
    numpy.ndarray
        Signal of length ``L M``.
    """
    blocks = np.fft.ifft(np.asarray(coefficients, dtype=np.complex128), axis=1)
    return np.asarray(blocks.T.reshape(-1), dtype=np.complex128)

"""Multiresolution analysis: filters, Daubechies wavelets, cascade algorithm.

Weeks 9 and 11. A filter is the sequence ``h[n]`` of the scaling equation
``phi(x) = sqrt(2) sum_n h[n] phi(2x - n)``, normalized so that
``sum_n h[n] = sqrt(2)``; this is the ``rec_lo`` filter of PyWavelets.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pywt
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

SQRT2 = float(np.sqrt(2.0))


def haar_filter() -> FloatArray:
    """Return the low-pass filter of the Haar MRA.

    Returns
    -------
    numpy.ndarray
        ``[1, 1] / sqrt(2)``.
    """
    return np.array([1.0, 1.0]) / SQRT2


def quadrature_mirror(h: ArrayLike) -> FloatArray:
    """Return the high-pass filter ``g[n] = (-1)**n h[L - 1 - n]``.

    Parameters
    ----------
    h : array_like
        Low-pass filter of even length ``L``.

    Returns
    -------
    numpy.ndarray
        Wavelet filter of the same length; equal to ``rec_hi`` in PyWavelets.
    """
    low = np.asarray(h, dtype=np.float64)
    signs = (-1.0) ** np.arange(low.size)
    return np.asarray(signs * low[::-1], dtype=np.float64)


def is_orthonormal_filter(h: ArrayLike, atol: float = 1e-10) -> bool:
    """Check the conditions of an orthonormal MRA filter.

    The filter must satisfy ``sum_n h[n] = sqrt(2)`` and
    ``sum_n h[n] h[n + 2k] = delta_k`` for every integer ``k``.

    Parameters
    ----------
    h : array_like
        Low-pass filter.
    atol : float, optional
        Absolute tolerance.

    Returns
    -------
    bool
        ``True`` if both conditions hold.
    """
    low = np.asarray(h, dtype=np.float64)
    autocorrelation = np.correlate(low, low, mode="full")[low.size - 1 :: 2]
    expected = np.zeros_like(autocorrelation)
    expected[0] = 1.0
    return bool(
        np.isclose(low.sum(), SQRT2, atol=atol)
        and np.allclose(autocorrelation, expected, atol=atol)
    )


def frequency_response(h: ArrayLike, omega: ArrayLike) -> NDArray[np.complex128]:
    """Evaluate ``m0(omega) = (1 / sqrt 2) sum_n h[n] exp(-i n omega)``.

    Parameters
    ----------
    h : array_like
        Filter.
    omega : array_like
        Frequencies.

    Returns
    -------
    numpy.ndarray
        Values of ``m0``; ``m0(0) = 1`` for a low-pass MRA filter.
    """
    taps = np.asarray(h, dtype=np.float64)
    points = np.asarray(omega, dtype=np.float64)
    exponent = np.exp(-1j * np.multiply.outer(points, np.arange(taps.size)))
    return np.asarray(exponent @ taps / SQRT2, dtype=np.complex128)


def daubechies_filter(p: int) -> FloatArray:
    """Construct the Daubechies filter with ``p`` vanishing moments.

    ``|m0|**2 = cos(w/2)**(2p) P(sin(w/2)**2)`` with
    ``P(y) = sum_{k<p} C(p-1+k, k) y**k``. The roots of ``P`` written in
    ``z = exp(-i w)`` come in pairs ``r, 1/r``; keeping those inside the unit
    circle (spectral factorization) gives the minimum-phase filter.

    Parameters
    ----------
    p : int
        Number of vanishing moments; the filter has length ``2 p``.

    Returns
    -------
    numpy.ndarray
        Filter equal to ``pywt.Wavelet(f"db{p}").rec_lo``.

    Raises
    ------
    ValueError
        If ``p < 1``.
    """
    if p < 1:
        raise ValueError("p must be at least 1")
    # y = sin^2(w/2) = (2 - z - 1/z) / 4. The polynomial z^(p-1) P(y) in z has
    # degree 2(p-1); the term y^k occupies the powers p-1-k, ..., p-1+k.
    poly = np.zeros(2 * p - 1)
    for k in range(p):
        term = np.array([float(comb(p - 1 + k, k))])
        for _ in range(k):
            term = np.convolve(term, [-0.25, 0.5, -0.25])
        poly += np.pad(term, p - 1 - k)
    roots = np.roots(poly) if poly.size > 1 else np.array([])
    inside = roots[np.abs(roots) < 1.0]
    taps: NDArray[np.complex128] = np.array([1.0 + 0.0j])
    for _ in range(p):
        taps = np.convolve(taps, [1.0, 1.0])
    for root in inside:
        taps = np.convolve(taps, [1.0, -root])
    real = np.real(taps)
    return np.asarray(real * SQRT2 / real.sum(), dtype=np.float64)


def cascade(
    h: ArrayLike, iterations: int = 8
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Approximate the scaling function and the wavelet by the cascade algorithm.

    Starting from a unit impulse, the iteration
    ``phi_{j+1}(x) = sqrt(2) sum_n h[n] phi_j(2x - n)`` is evaluated on the
    dyadic grid ``x = m / 2**iterations``.

    Parameters
    ----------
    h : array_like
        Low-pass filter of length ``L``; the support is ``[0, L - 1]``.
    iterations : int, optional
        Number of refinement steps.

    Returns
    -------
    x : numpy.ndarray
        Grid points.
    phi : numpy.ndarray
        Scaling function on the grid.
    psi : numpy.ndarray
        Wavelet ``psi(x) = sqrt(2) sum_n g[n] phi(2x - n)`` on the grid.
    """
    low = np.asarray(h, dtype=np.float64)
    high = quadrature_mirror(low)
    values = np.array([1.0])
    for _ in range(iterations):
        upsampled = np.zeros(2 * values.size - 1)
        upsampled[::2] = values
        values = SQRT2 * np.convolve(upsampled, low)
    step = 2**iterations
    # The impulse sequence approximates phi(m / step) directly; psi(m / 2 step)
    # combines samples of phi shifted by multiples of step.
    phi = values
    high_upsampled = np.zeros((high.size - 1) * step + 1)
    high_upsampled[::step] = high
    psi = SQRT2 * np.convolve(phi, high_upsampled)
    x_phi = np.arange(phi.size) / step
    x_psi = np.arange(psi.size) / (2 * step)
    return x_phi, phi, np.interp(x_phi, x_psi, psi)


def wavelet_filters(name: str) -> tuple[FloatArray, FloatArray]:
    """Return the reconstruction filters of a PyWavelets wavelet.

    Parameters
    ----------
    name : str
        Wavelet name, e.g. ``"haar"``, ``"db3"`` or ``"sym4"``.

    Returns
    -------
    h : numpy.ndarray
        Low-pass filter ``rec_lo``.
    g : numpy.ndarray
        High-pass filter ``rec_hi``.
    """
    wavelet = pywt.Wavelet(name)
    return (
        np.asarray(wavelet.rec_lo, dtype=np.float64),
        np.asarray(wavelet.rec_hi, dtype=np.float64),
    )


def packet_energies(
    x: ArrayLike, wavelet: str = "db2", level: int = 3
) -> dict[str, float]:
    """Return the energy of every node of a wavelet packet decomposition.

    Parameters
    ----------
    x : array_like
        One-dimensional signal.
    wavelet : str, optional
        Wavelet name.
    level : int, optional
        Depth of the packet tree.

    Returns
    -------
    dict
        Maps node paths in frequency order (``"aaa"``, ``"aad"``, ...) to the
        sum of squared coefficients.
    """
    tree = pywt.WaveletPacket(
        np.asarray(x, dtype=np.float64), wavelet, mode="periodization", maxlevel=level
    )
    return {
        node.path: float(np.sum(np.asarray(node.data) ** 2))
        for node in tree.get_level(level, order="freq")
    }

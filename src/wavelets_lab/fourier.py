"""Fourier series, discrete Fourier transforms and Fourier coding of images.

Weeks 5 and 6. The DFT uses the convention of :mod:`numpy.fft`,
``X[k] = sum_n x[n] exp(-2 pi i k n / N)``, so the own implementations can be
compared with the library directly.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def square_wave(t: ArrayLike) -> FloatArray:
    r"""Return the :math:`2\pi`-periodic square wave ``sign(sin t)``.

    Parameters
    ----------
    t : array_like
        Points.

    Returns
    -------
    numpy.ndarray
        Values in ``{-1, 0, 1}``.
    """
    return np.sign(np.sin(np.asarray(t, dtype=np.float64)))


def fourier_coefficients(
    f: Callable[[FloatArray], ArrayLike], n_max: int, n_samples: int = 4096
) -> ComplexArray:
    r"""Approximate the Fourier coefficients of a :math:`2\pi`-periodic function.

    ``c_n = (1 / 2 pi) * integral_0^{2 pi} f(t) exp(-i n t) dt`` is computed
    with the rectangle rule on ``n_samples`` points.

    Parameters
    ----------
    f : callable
        Vectorized function of one variable.
    n_max : int
        Largest index; coefficients ``c_{-n_max}, ..., c_{n_max}`` are returned.
    n_samples : int, optional
        Number of quadrature points; must exceed ``2 * n_max``.

    Returns
    -------
    numpy.ndarray
        Array of length ``2 * n_max + 1``; entry ``n_max + n`` is ``c_n``.

    Raises
    ------
    ValueError
        If ``n_samples <= 2 * n_max``.
    """
    if n_samples <= 2 * n_max:
        raise ValueError("n_samples must exceed 2 * n_max")
    t = 2.0 * np.pi * np.arange(n_samples) / n_samples
    spectrum = np.fft.fft(np.asarray(f(t), dtype=np.complex128)) / n_samples
    indices = np.arange(-n_max, n_max + 1)
    return np.asarray(spectrum[indices % n_samples])


def fourier_partial_sum(coefficients: ArrayLike, t: ArrayLike) -> ComplexArray:
    """Evaluate the partial sum ``sum_{|n| <= N} c_n exp(i n t)``.

    Parameters
    ----------
    coefficients : array_like
        Coefficients ordered as returned by :func:`fourier_coefficients`.
    t : array_like
        Points.

    Returns
    -------
    numpy.ndarray
        Values of the partial sum (complex; take ``.real`` for real ``f``).
    """
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    n_max = (coeffs.size - 1) // 2
    indices = np.arange(-n_max, n_max + 1)
    points = np.asarray(t, dtype=np.float64)
    return np.asarray(np.exp(1j * np.multiply.outer(points, indices)) @ coeffs)


def dft(x: ArrayLike) -> ComplexArray:
    """Compute the DFT directly from the definition in ``O(N^2)`` operations.

    Parameters
    ----------
    x : array_like
        One-dimensional signal.

    Returns
    -------
    numpy.ndarray
        ``X[k] = sum_n x[n] exp(-2 pi i k n / N)``.
    """
    signal = np.asarray(x, dtype=np.complex128)
    n = signal.size
    indices = np.arange(n)
    matrix = np.exp(-2j * np.pi * np.outer(indices, indices) / n)
    return np.asarray(matrix @ signal)


def idft(spectrum: ArrayLike) -> ComplexArray:
    """Invert :func:`dft` directly from the definition.

    Parameters
    ----------
    spectrum : array_like
        DFT coefficients.

    Returns
    -------
    numpy.ndarray
        ``x[n] = (1 / N) sum_k X[k] exp(2 pi i k n / N)``.
    """
    values = np.asarray(spectrum, dtype=np.complex128)
    return np.asarray(np.conj(dft(np.conj(values))) / values.size)


def fft_radix2(x: ArrayLike) -> ComplexArray:
    """Compute the DFT with the recursive Danielson-Lanczos algorithm.

    The signal is split into even and odd samples,
    ``X[k] = E[k] + w^k O[k]`` and ``X[k + N/2] = E[k] - w^k O[k]``
    with ``w = exp(-2 pi i / N)``, which needs ``O(N log N)`` operations.

    Parameters
    ----------
    x : array_like
        One-dimensional signal whose length is a power of two.

    Returns
    -------
    numpy.ndarray
        DFT of ``x``.

    Raises
    ------
    ValueError
        If the length of ``x`` is not a power of two.
    """
    signal = np.asarray(x, dtype=np.complex128)
    n = signal.size
    if n == 0 or n & (n - 1):
        raise ValueError("the signal length must be a power of two")
    if n == 1:
        return np.asarray(signal.copy())
    even = fft_radix2(signal[0::2])
    odd = fft_radix2(signal[1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(n // 2) / n) * odd
    return np.concatenate([even + twiddle, even - twiddle])


def dct2(x: ArrayLike) -> FloatArray:
    """Compute the orthonormal DCT-II of a signal from the definition.

    Parameters
    ----------
    x : array_like
        Real one-dimensional signal of length ``N``.

    Returns
    -------
    numpy.ndarray
        Coefficients ``X[k] = a_k sum_n x[n] cos(pi k (2n + 1) / (2N))`` with
        ``a_0 = sqrt(1/N)`` and ``a_k = sqrt(2/N)``; equal to
        ``scipy.fft.dct(x, norm="ortho")``.
    """
    signal = np.asarray(x, dtype=np.float64)
    n = signal.size
    indices = np.arange(n)
    matrix = np.cos(np.pi * np.outer(indices, 2 * indices + 1) / (2 * n))
    scale = np.full(n, np.sqrt(2.0 / n))
    scale[0] = np.sqrt(1.0 / n)
    return np.asarray(scale * (matrix @ signal))


def magnitude_phase(image: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return the centred magnitude and phase of the 2D DFT of an image.

    Parameters
    ----------
    image : array_like
        Greyscale image.

    Returns
    -------
    magnitude : numpy.ndarray
        ``abs(F)`` with the zero frequency moved to the centre
        (:func:`numpy.fft.fftshift`).
    phase : numpy.ndarray
        ``angle(F)``, centred in the same way.
    """
    spectrum = np.fft.fftshift(np.fft.fft2(np.asarray(image, dtype=np.float64)))
    return np.abs(spectrum), np.angle(spectrum)


def log_magnitude(magnitude: ArrayLike) -> FloatArray:
    """Compress a magnitude spectrum for display with ``log(1 + |F|)``.

    Parameters
    ----------
    magnitude : array_like
        Magnitude of a spectrum.

    Returns
    -------
    numpy.ndarray
        Logarithmically scaled magnitude.
    """
    return np.log1p(np.asarray(magnitude, dtype=np.float64))


def from_magnitude_phase(magnitude: ArrayLike, phase: ArrayLike) -> FloatArray:
    """Rebuild an image from a centred magnitude and phase.

    Combining the magnitude of one image with the phase of another shows that
    the phase carries most of the structure.

    Parameters
    ----------
    magnitude : array_like
        Centred magnitude, as returned by :func:`magnitude_phase`.
    phase : array_like
        Centred phase.

    Returns
    -------
    numpy.ndarray
        Real part of the inverse 2D DFT.
    """
    spectrum = np.asarray(magnitude) * np.exp(1j * np.asarray(phase))
    return np.asarray(np.fft.ifft2(np.fft.ifftshift(spectrum)).real)


def lowpass_mask(shape: tuple[int, int], half_width: int) -> NDArray[np.bool_]:
    """Square low-pass mask for a centred spectrum.

    Parameters
    ----------
    shape : tuple of int
        Shape of the spectrum.
    half_width : int
        Number of kept frequencies on each side of zero along each axis.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` on a ``(2 * half_width) ** 2`` square around
        the centre.
    """
    mask = np.zeros(shape, dtype=bool)
    centre = (shape[0] // 2, shape[1] // 2)
    mask[
        centre[0] - half_width : centre[0] + half_width,
        centre[1] - half_width : centre[1] + half_width,
    ] = True
    return mask


def band_code(image: ArrayLike, half_width: int) -> tuple[FloatArray, float]:
    """Keep only low frequencies of an image and reconstruct it.

    Parameters
    ----------
    image : array_like
        Greyscale image.
    half_width : int
        Half side of the kept square of frequencies, see :func:`lowpass_mask`.

    Returns
    -------
    reconstruction : numpy.ndarray
        Image rebuilt from the kept coefficients.
    kept_fraction : float
        Fraction of the DFT coefficients that were kept.
    """
    array = np.asarray(image, dtype=np.float64)
    spectrum = np.fft.fftshift(np.fft.fft2(array))
    mask = lowpass_mask(array.shape, half_width)
    reconstruction = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
    return np.asarray(reconstruction), float(mask.mean())


def compression_stats(
    kept_fraction: float, bits_per_pixel: float = 8.0
) -> dict[str, float]:
    """Summarize the compression achieved by keeping a fraction of coefficients.

    Parameters
    ----------
    kept_fraction : float
        Fraction of the coefficients that are stored, in ``(0, 1]``.
    bits_per_pixel : float, optional
        Bits per stored coefficient; 8 by default.

    Returns
    -------
    dict
        ``"ratio"`` (compression ratio) and ``"bpp"`` (bits per pixel).

    Raises
    ------
    ValueError
        If ``kept_fraction`` is not in ``(0, 1]``.
    """
    if not 0.0 < kept_fraction <= 1.0:
        raise ValueError("kept_fraction must be in (0, 1]")
    return {
        "ratio": 1.0 / kept_fraction,
        "bpp": bits_per_pixel * kept_fraction,
    }

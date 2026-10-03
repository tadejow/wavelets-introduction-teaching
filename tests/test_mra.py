"""Tests for :mod:`wavelets_lab.mra`."""

import numpy as np
import pytest
import pywt

from wavelets_lab import mra


def test_haar_filter() -> None:
    """The Haar filter is orthonormal and its wavelet filter is ``[1, -1]``."""
    h = mra.haar_filter()
    assert mra.is_orthonormal_filter(h)
    np.testing.assert_allclose(mra.quadrature_mirror(h), np.array([1, -1]) / np.sqrt(2))


@pytest.mark.parametrize("p", [1, 2, 3, 5, 8])
def test_daubechies_matches_pywt(p: int) -> None:
    """Own spectral factorization reproduces the PyWavelets filters."""
    h = mra.daubechies_filter(p)
    h_ref, g_ref = mra.wavelet_filters(f"db{p}")
    np.testing.assert_allclose(h, h_ref, atol=1e-8)
    np.testing.assert_allclose(mra.quadrature_mirror(h), g_ref, atol=1e-8)
    assert mra.is_orthonormal_filter(h)


def test_daubechies_invalid() -> None:
    """At least one vanishing moment is required."""
    with pytest.raises(ValueError):
        mra.daubechies_filter(0)


def test_frequency_response_low_pass() -> None:
    """``m0(0) = 1``, ``m0(pi) = 0`` and ``|m0|^2 + |m0(w + pi)|^2 = 1``."""
    h = mra.daubechies_filter(3)
    omega = np.linspace(0.0, np.pi, 50)
    m0 = mra.frequency_response(h, omega)
    shifted = mra.frequency_response(h, omega + np.pi)
    assert m0[0] == pytest.approx(1.0)
    assert abs(m0[-1]) == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(np.abs(m0) ** 2 + np.abs(shifted) ** 2, 1.0)


def test_cascade_matches_pywt() -> None:
    """The cascade algorithm agrees with ``pywt.Wavelet.wavefun``."""
    x, phi, psi = mra.cascade(mra.daubechies_filter(4), iterations=8)
    phi_ref, psi_ref, x_ref = pywt.Wavelet("db4").wavefun(level=8)
    np.testing.assert_allclose(np.interp(x_ref, x, phi), phi_ref, atol=0.03)
    np.testing.assert_allclose(np.interp(x_ref, x, psi), psi_ref, atol=0.03)
    step = x[1] - x[0]
    assert np.sum(phi) * step == pytest.approx(1.0, abs=1e-6)
    assert np.sum(psi) * step == pytest.approx(0.0, abs=1e-2)


def test_packet_energies_preserve_energy() -> None:
    """Orthogonal packets keep the total energy of the signal."""
    x = np.random.default_rng(0).standard_normal(256)
    energies = mra.packet_energies(x, "db2", 3)
    assert len(energies) == 8
    assert sum(energies.values()) == pytest.approx(float(np.sum(x**2)))

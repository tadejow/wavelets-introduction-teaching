"""Tests for :mod:`wavelets_lab.timeseries`."""

import numpy as np

from wavelets_lab import timeseries


def test_synthetic_prices_are_positive() -> None:
    """Synthetic prices are positive and reproducible."""
    first = timeseries.synthetic_prices(512, rng=0)
    np.testing.assert_array_equal(first, timeseries.synthetic_prices(512, rng=0))
    assert first.min() > 0.0


def test_components_add_up() -> None:
    """The multiscale components sum to the original series."""
    x = np.log(timeseries.synthetic_prices(1024, rng=1))
    components = timeseries.multiscale_components(x, "db4", 5)
    assert len(components) == 6
    np.testing.assert_allclose(np.sum(components, axis=0), x)
    groups = timeseries.group_scales(
        components, {"large": [0, 1], "medium": [2, 3], "small": [4, 5]}
    )
    np.testing.assert_allclose(sum(groups.values()), x)


def test_scale_correlations() -> None:
    """A series is perfectly correlated with itself on every scale."""
    x = np.diff(np.log(timeseries.synthetic_prices(1025, rng=2)))
    correlations = timeseries.scale_correlations(x, 2.0 * x + 1.0, "db2", 4)
    np.testing.assert_allclose(correlations[1:], 1.0)
    assert correlations.size == 5
    noise = np.random.default_rng(3).standard_normal(1024)
    independent = timeseries.scale_correlations(x, noise, "db2", 4)
    assert np.all(np.abs(independent[1:]) < 0.5)

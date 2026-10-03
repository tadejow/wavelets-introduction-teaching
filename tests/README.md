# Tests

Test suite for the `wavelets_lab` package, run with `pytest` from the
repository root:

```bash
pytest
```

## Rules

- One test file per module: `test_<module>.py` tests
  `src/wavelets_lab/<module>.py` (e.g. `test_dwt.py` for `dwt.py`).
- Where possible, own implementations are checked against reference
  libraries (`numpy.fft`, `scipy`, `pywt`, `skimage`).
- Tests follow the same conventions as the package: PEP 8, type hints,
  NumPy-style docstrings.

## Contents

| File              | Module tested                |
|-------------------|------------------------------|
| `test_images.py`  | `wavelets_lab.images`        |
| `test_filters.py` | `wavelets_lab.filters`       |
| `test_edges.py`   | `wavelets_lab.edges`         |
| `test_hilbert.py` | `wavelets_lab.hilbert`       |
| `test_fourier.py` | `wavelets_lab.fourier`       |
| `test_sampling.py`| `wavelets_lab.sampling`      |

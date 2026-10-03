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

No tests yet; they will be added together with the modules.

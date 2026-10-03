# Introduction to Applications of Wavelet Analysis — Lab Course

Laboratory materials for the course *Wstęp do zastosowań analizy falkowej*
(Introduction to Applications of Wavelet Analysis), Institute of Mathematics,
University of Wrocław. The course runs for 15 weeks with one 2-hour lab per
week and is intended for MSc and final-year BSc students of mathematics.

The labs are taught as live-coding sessions in Python. The original syllabus
assumes Matlab and the Rice Wavelet Toolbox; here they are replaced with
NumPy/SciPy and [PyWavelets](https://pywavelets.readthedocs.io/).

## Repository layout

| Path                 | Contents                                                        |
|----------------------|-----------------------------------------------------------------|
| `src/wavelets_lab/`  | Python package with all lab functions (one module per topic).   |
| `notebooks/`         | Demo Jupyter notebooks, one per lab, calling `wavelets_lab`.    |
| `tests/`             | `pytest` test suite, one test file per module.                  |
| `docs/`              | Syllabus, week-by-week schedule and other course documents.     |

Every directory contains a `README.md` describing its contents.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows,
`source .venv/bin/activate` on Linux/macOS), then install the package in
editable mode together with the development tools:

```bash
pip install -e ".[dev]"
```

Start Jupyter to work with the notebooks:

```bash
jupyter lab
```

## Conventions

- All code, docstrings and notebooks are written in English.
- Functions live in `.py` modules under `src/wavelets_lab/`; notebooks only
  import and call them, so that the same code is tested and reused.
- Code follows PEP 8 and is formatted and linted with `ruff`
  (`ruff format .` and `ruff check .`).
- All functions have full type hints (checked with `mypy`) and NumPy-style
  docstrings.
- Every module has a matching test file in `tests/`; run the suite with
  `pytest`.
- Test images come from `skimage.data` (e.g. `camera`) instead of Lena.

## Schedule

See [`docs/schedule.md`](docs/schedule.md) for the week-by-week plan.

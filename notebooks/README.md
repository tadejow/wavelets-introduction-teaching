# Notebooks

Demo Jupyter notebooks used during the live-coding labs, one per lab.

## Naming

Notebooks are named `NN_topic.ipynb`, where `NN` is the two-digit week number
and `topic` is a short snake_case description, e.g. `01_images.ipynb`,
`10_mallat_1d.ipynb`.

## Rules

- Notebooks contain no function definitions: they import and call functions
  from the `wavelets_lab` package (`src/wavelets_lab/`).
- All text and code are in English.
- Every plot has labelled X and Y axes (with units where they exist).
  `images.show_image` labels image axes in pixels by default and accepts
  `xlabel`/`ylabel` for other quantities (frequencies, angles, …).
- Every heading is followed by three paragraphs: **What we do.**,
  **What to expect.** and **How to read the code.**
- Every notebook ends with a section *Reference: scripts and functions used
  in this notebook*: a table of the library functions called directly, and for
  every `wavelets_lab` function used (including the ones it calls internally)
  its module file, purpose, input, output and a step-by-step recipe detailed
  enough to rewrite the function without seeing its source.
- Clear outputs before committing, unless a figure is part of the material.

## Contents

One notebook per week of [`docs/schedule.md`](../docs/schedule.md):

| Notebook                               | Week | Topic                                                               |
|----------------------------------------|------|---------------------------------------------------------------------|
| `01_images.ipynb`                      | 1    | Images as arrays, RGB↔YCbCr, display                                |
| `02_image_operations.ipynb`            | 2    | Brightness, contrast, gamma, binarization, crop masks               |
| `03_noise_filters_edges.ipynb`         | 3    | Gaussian noise, Gaussian and median filters, Canny                  |
| `04_hilbert_space.ipynb`               | 4    | Hat functions, Gram–Schmidt, Riesz bounds, projections              |
| `05_fourier_series_fft.ipynb`          | 5    | Fourier series, Gibbs phenomenon, DFT vs FFT, DCT                   |
| `06_fourier_images.ipynb`              | 6    | 2D FFT: magnitude and phase, non-locality, band coding              |
| `07_sampling.ipynb`                    | 7    | Convolution theorem, Shannon reconstruction, aliasing, quantization |
| `08_time_frequency.ipynb`              | 8    | STFT with a Gaussian window, CWT scalogram                          |
| `09_multiresolution.ipynb`             | 9    | Haar MRA, QMF filters, frequency responses                          |
| `10_mallat_1d.ipynb`                   | 10   | Mallat algorithm in 1D vs PyWavelets                                |
| `11_daubechies_cascade.ipynb`          | 11   | Daubechies filters, cascade algorithm, wavelet packets              |
| `12_dwt_images.ipynb`                  | 12   | 2D DWT of images, levels and filters                                |
| `13_compression.ipynb`                 | 13   | Hard/soft thresholding, 90/95/98% compression                       |
| `14_denoising.ipynb`                   | 14   | VisuShrink and SURE in 1D, image denoising                          |
| `15_other_transforms_timeseries.ipynb` | 15   | Radon, Hough, Zak; time-series project                              |

Run them from the repository root after `pip install -e ".[dev]"`.

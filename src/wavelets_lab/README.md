# `wavelets_lab` package

Python package with the functions used in the labs. Each topic has its own
module; notebooks in `notebooks/` only import and call these functions.

| Module          | Status  | Contents                                                                 | Week(s) |
|-----------------|---------|--------------------------------------------------------------------------|---------|
| `__init__.py`   | done    | Package docstring and `__version__`.                                     | —       |
| `images`        | done    | Image I/O, images as arrays, RGB↔YCbCr, basic pixel operations.          | 1–2     |
| `filters`       | done    | Gaussian noise, Gaussian and median filters.                             | 3       |
| `edges`         | done    | Gradient masks and the Canny edge detector.                              | 3       |
| `hilbert`       | done    | Projections, Gram–Schmidt, Riesz bases, expansions in a basis.           | 4       |
| `fourier`       | done    | Fourier series, DFT, own FFT, DCT, 2D FFT and band coding of images.     | 5–6     |
| `sampling`      | done    | Convolution, Shannon sampling, aliasing, quantization.                   | 7       |
| `timefreq`      | done    | STFT/Gabor transform, continuous wavelet transform and scalograms.       | 8       |
| `mra`           | done    | Haar MRA, QMF filters, Daubechies wavelets, cascade algorithm, packets.  | 9, 11   |
| `dwt`           | done    | Mallat algorithm in 1D and the 2D discrete wavelet transform.            | 10, 12  |
| `compression`   | done    | Hard/soft thresholding and wavelet image compression.                    | 13      |
| `denoising`     | done    | Wavelet denoising (VisuShrink, SURE) in 1D and 2D.                       | 14      |
| `transforms`    | done    | Other transforms: Radon, Hough, Zak.                                     | 15      |
| `timeseries`    | done    | Multiscale analysis of financial time series (project).                  | 15      |

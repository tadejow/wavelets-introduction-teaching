# `wavelets_lab` package

Python package with the functions used in the labs. Each topic has its own
module; notebooks in `notebooks/` only import and call these functions.

| Module          | Status  | Contents                                                                 | Week(s) |
|-----------------|---------|--------------------------------------------------------------------------|---------|
| `__init__.py`   | done    | Package docstring and `__version__`.                                     | —       |
| `images`        | planned | Image I/O, images as arrays, RGB↔YCbCr, basic pixel operations.          | 1–2     |
| `filters`       | planned | Gaussian noise, Gaussian and median filters.                             | 3       |
| `edges`         | planned | Gradient masks and the Canny edge detector.                              | 3       |
| `hilbert`       | planned | Projections, Gram–Schmidt, Riesz bases, expansions in a basis.           | 4       |
| `fourier`       | planned | Fourier series, DFT, own FFT, DCT, 2D FFT and band coding of images.     | 5–6     |
| `sampling`      | planned | Convolution, Shannon sampling, aliasing, quantization.                   | 7       |
| `timefreq`      | planned | STFT/Gabor transform, continuous wavelet transform and scalograms.       | 8       |
| `mra`           | planned | Haar MRA, QMF filters, Daubechies wavelets, cascade algorithm, packets.  | 9, 11   |
| `dwt`           | planned | Mallat algorithm in 1D and the 2D discrete wavelet transform.            | 10, 12  |
| `compression`   | planned | Hard/soft thresholding and wavelet image compression.                    | 13      |
| `denoising`     | planned | Wavelet denoising (VisuShrink, SURE) in 1D and 2D.                       | 14      |
| `transforms`    | planned | Other transforms: Radon, Hough, Zak.                                     | 15      |
| `timeseries`    | planned | Multiscale analysis of financial time series (project).                  | 15      |

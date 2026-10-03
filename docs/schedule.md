# Lab schedule

15 weeks, one 2-hour lab per week. The last column names the `wavelets_lab`
module developed in that lab.

| Week | Topic                                                                                   | Module(s)                |
|------|-----------------------------------------------------------------------------------------|--------------------------|
| 1    | Environment, NumPy/matplotlib, an image as an array, image formats, RGB→YCbCr           | `images`                 |
| 2    | Basic image operations: brightness, contrast, gamma, binarization, mirroring, crop masks | `images`                 |
| 3    | Gaussian noise, Gaussian 3×3/5×5 and median filters; Canny edge detection               | `filters`, `edges`       |
| 4    | Hilbert space numerically: projection, Gram–Schmidt, Riesz bases, expansion in a basis  | `hilbert`                |
| 5    | Fourier series, Gibbs phenomenon, DFT, own FFT (Danielson–Lanczos) vs `numpy.fft`, DCT  | `fourier`                |
| 6    | 2D FFT of images: magnitude/phase, `fftshift`, non-locality, band coding                | `fourier`                |
| 7    | Convolution and filters, Shannon sampling theorem, aliasing, quantization               | `sampling`               |
| 8    | STFT/Gabor transform, CWT and scalogram (`pywt.cwt`)                                    | `timefreq`               |
| 9    | Multiresolution analysis: Haar, m0/m1 QMF filters, `pywt.Wavelet`                       | `mra`                    |
| 10   | Mallat algorithm in 1D: own analysis/synthesis step vs `pywt.dwt`/`pywt.wavedec`        | `dwt`                    |
| 11   | Daubechies wavelets, cascade algorithm (`wavefun`), wavelet packets                     | `mra`                    |
| 12   | 2D DWT of images (`wavedec2`/`waverec2`), decomposition levels, db1/db3/db5             | `dwt`                    |
| 13   | Wavelet compression: hard/soft thresholding, 90/95/98% compression ratios              | `compression`            |
| 14   | Wavelet denoising in 1D/2D (VisuShrink, SURE) vs Gaussian/median filters                | `denoising`              |
| 15   | Other transforms: Radon, Hough, Zak; project: time series across scales                 | `transforms`, `timeseries` |

## Notes

- The syllabus assumes Matlab and the Rice Wavelet Toolbox; the labs use
  Python and PyWavelets instead. The RWT filter `daubcqf(N)` corresponds to
  the PyWavelets wavelet `db{N/2}` (e.g. `daubcqf(6)` ≈ `db3`).
- The Lena test image is replaced with `skimage.data.camera`.

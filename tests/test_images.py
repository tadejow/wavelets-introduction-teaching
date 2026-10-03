"""Tests for :mod:`wavelets_lab.images`."""

from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from wavelets_lab import images


def test_load_test_image_scale() -> None:
    """The camera image is a 512x512 float array on [0, 255]."""
    image = images.load_test_image("camera")
    assert image.shape == (512, 512)
    assert image.dtype == np.float64
    assert 0.0 <= image.min() and image.max() <= 255.0


def test_load_test_image_unknown() -> None:
    """An unknown image name raises ``ValueError``."""
    with pytest.raises(ValueError):
        images.load_test_image("no_such_image")


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Saving to PNG and reading back preserves an integer image."""
    image = np.arange(64, dtype=np.float64).reshape(8, 8) * 4
    path = tmp_path / "image.png"
    images.save_image(path, image)
    np.testing.assert_array_equal(images.load_image(path), image)


def test_to_float_unit_scale() -> None:
    """Float images on [0, 1] are rescaled to [0, 255]."""
    np.testing.assert_allclose(
        images.to_float(np.array([0.0, 0.5, 1.0])), [0, 127.5, 255]
    )


def test_ycbcr_roundtrip() -> None:
    """RGB -> YCbCr -> RGB is the identity."""
    rgb = images.load_test_image("astronaut")
    np.testing.assert_allclose(
        images.ycbcr_to_rgb(images.rgb_to_ycbcr(rgb)), rgb, atol=1e-8
    )


def test_grey_of_white_is_white() -> None:
    """The luminance of a white pixel is 255."""
    white = np.full((2, 2, 3), 255.0)
    np.testing.assert_allclose(images.rgb_to_grey(white), 255.0)


def test_normalize_and_stretch() -> None:
    """Normalization maps to [0, 1] and stretching to [0, 255]."""
    image = np.array([[10.0, 20.0], [30.0, 50.0]])
    assert images.normalize(image).min() == 0.0
    assert images.normalize(image).max() == 1.0
    assert images.stretch_contrast(image).max() == 255.0
    np.testing.assert_array_equal(images.normalize(np.ones((2, 2))), 0.0)


def test_intensity_operations_clip() -> None:
    """Brightening, offsets and gamma stay within [0, 255]."""
    image = np.array([0.0, 100.0, 250.0])
    np.testing.assert_allclose(images.scale_intensity(image, 1.2), [0, 120, 255])
    np.testing.assert_allclose(images.affine_intensity(image, 0.8, 25), [25, 105, 225])
    np.testing.assert_allclose(images.gamma_correction(image, 1.0), image)
    with pytest.raises(ValueError):
        images.gamma_correction(image, 0.0)


def test_binarize() -> None:
    """Binarization maps values above the threshold to 255."""
    np.testing.assert_array_equal(images.binarize([10, 100, 200], 100), [0, 0, 255])


def test_mirror_extend_shape() -> None:
    """Mirroring along columns doubles the width."""
    image = np.arange(6.0).reshape(2, 3)
    extended = images.mirror_extend(image)
    assert extended.shape == (2, 6)
    np.testing.assert_array_equal(extended[:, 3:], image[:, ::-1])


def test_show_image_labels_axes() -> None:
    """Displayed images have labelled axes, by default in pixels."""
    ax = Figure().subplots()
    images.show_image(np.zeros((4, 4)), "zeros", ax=ax)
    assert ax.get_xlabel() == "column [pixel]"
    assert ax.get_ylabel() == "row [pixel]"
    images.show_image(np.zeros((4, 4)), ax=ax, xlabel="angle", ylabel="offset")
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("angle", "offset")


def test_rectangle_mask() -> None:
    """Hard masks are 0/1 and smooth masks decay linearly."""
    hard = images.rectangle_mask((10, 10), (2, 2), (8, 8))
    assert hard.sum() == 36
    smooth = images.rectangle_mask((10, 10), (4, 4), (6, 6), ramp=2)
    assert smooth[4, 4] == 1.0
    assert smooth[4, 3] == 0.5
    assert smooth[0, 0] == 0.0
    masked = images.apply_mask(np.zeros((10, 10)), hard)
    assert masked[0, 0] == 255.0 and masked[5, 5] == 0.0

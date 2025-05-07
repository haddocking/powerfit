from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from powerfit_em.volume import Volume


@pytest.fixture
def example_volume() -> Volume:
    array = np.zeros((3, 4, 5))
    array[0, 0, 0] = 1.1
    array[1, 2, 3] = 2.2
    array[2, 3, 4] = 3.3
    return Volume(array)


@pytest.fixture
def example_nrc_file(tmp_path: Path, example_volume: Volume) -> Path:
    fn = tmp_path / "example_volume.mrc"
    example_volume.tofile(fn)
    return fn


def test_to_file(example_nrc_file: Path):
    assert example_nrc_file.exists()
    assert example_nrc_file.stat().st_size > 0


def test_shape(example_volume: Volume):
    assert example_volume.shape == (3, 4, 5)


def test_dimensions(example_volume: Volume):
    npt.assert_array_equal(example_volume.dimensions, (5.0, 4.0, 3.0))


def test_get_start(example_volume: Volume):
    npt.assert_array_equal(example_volume.start, (0, 0, 0))


def test_fromfile(example_nrc_file: Path):
    volume = Volume.fromfile(str(example_nrc_file))

    assert volume.shape == (3, 4, 5)
    npt.assert_array_equal(volume.start, (0, 0, 0))
    npt.assert_array_equal(volume.dimensions, (5.0, 4.0, 3.0))
    expected = np.zeros((3, 4, 5))
    expected[0, 0, 0] = 1.1
    expected[1, 2, 3] = 2.2
    expected[2, 3, 4] = 3.3
    npt.assert_array_almost_equal(volume.array, expected)

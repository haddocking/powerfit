"""Golden-fixture regression tests: rust powerfitrs vs the old C/Cython extension.

Fixtures are generated with `gen_golden.sh` and are a `.npz` (numpy zipped arrays)
with the following structure:

keys: ['grid', 'rotmat', 'radius', 'nearest', 'out']
  grid:    (9,9,9) float32   <- input
  rotmat:  (3,3)   float32   <- input
  radius:  scalar  int64     <- input
  nearest: scalar  bool      <- input
  out:     (9,9,9) float32   <- expected output (from old Cython run)

So the fixtures are "golden data" with inputs and expected output, which can be
used directly here to checks if  the rust conversion is numerically 1:1 with the code it replaced
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from powerfit_em.powerfitrs import blur_points, dilate_points, rotate_grid3d

BASELINE_DIR = Path(__file__).parent / "fixtures" / "cython_baseline"


def _cases(prefix: str) -> list[Path]:
    return sorted(BASELINE_DIR.glob(f"{prefix}__*.npz"))


ROTATE_GRID3D_CASES = _cases("rotate_grid3d")
BLUR_POINTS_CASES = _cases("blur_points")
DILATE_POINTS_CASES = _cases("dilate_points")


# adding a class here to make the fixture structure more obvious
class Fixture(NamedTuple):
    data: np.lib.npyio.NpzFile  # input data
    expected: np.ndarray  # array containing the expected data
    results_placeholder: np.ndarray  # empty array that will contain the results


def _load(path: Path) -> Fixture:
    """Load a fixture and pre-allocate the results placeholder the rust fn writes into."""
    data = np.load(path)
    expected = data["out"]
    results_placeholder = np.zeros_like(expected)
    return Fixture(data, expected, results_placeholder)


@pytest.mark.parametrize("path", ROTATE_GRID3D_CASES, ids=lambda p: p.stem)
def test_rotate_grid3d_matches_cython_baseline(path: Path):
    data, expected, results_placeholder = _load(path)

    rotate_grid3d(data["grid"], data["rotmat"], int(data["radius"]), results_placeholder, bool(data["nearest"]))

    # `rotate_grid3d` will change `results_placeholder` in place, so the line below is just for readability
    observed = results_placeholder

    assert np.allclose(observed, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("path", BLUR_POINTS_CASES, ids=lambda p: p.stem)
def test_blur_points_matches_cython_baseline(path: Path):
    data, expected, results_placeholder = _load(path)

    blur_points(
        data["points"], data["param"], float(data["sigma_or_radius"]), results_placeholder, bool(data["wraparound"])
    )

    # `blur_points` will change `results_placeholder` in place, so line below is just for readability
    observed = results_placeholder

    assert np.allclose(observed, expected, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("path", DILATE_POINTS_CASES, ids=lambda p: p.stem)
def test_dilate_points_matches_cython_baseline(path: Path):
    data, expected, results_placeholder = _load(path)

    dilate_points(data["points"], data["param"], results_placeholder, bool(data["wraparound"]))

    # `dilate_points` will change `results_placeholder` in place, so the line below is just for readability
    observed = results_placeholder

    assert np.array_equal(observed, expected)


# Since this tests uses fixtures and loading, make sure that its actually loading something
def test_fixtures_exist():
    """Guard against silently running zero cases if the fixtures dir is empty/missing."""
    assert len(ROTATE_GRID3D_CASES) >= 5
    assert len(BLUR_POINTS_CASES) >= 3
    assert len(DILATE_POINTS_CASES) >= 3

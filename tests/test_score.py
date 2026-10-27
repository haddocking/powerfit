"""Direct pose scoring, reusable target preparation, and CSV output."""

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import laplace as laplace_filter

from powerfit_em import Structure, Volume, structure_to_shape_like
from powerfit_em.correlators.cpu import CPUCorrelator
from powerfit_em.score import (
    main,
    score_pose,
    setup_structure_with_laplace,
    setup_target_with_laplace,
    write_solution,
)


def _binary_mask(distance2: np.ndarray) -> np.ndarray:
    return (distance2 < 16).astype(float)


def _core_weighted_mask(distance2: np.ndarray) -> np.ndarray:
    return _binary_mask(distance2) * np.where(distance2 < 4, 2, 1)


@pytest.mark.parametrize("laplace, filter_density", [(False, np.copy), (True, partial(laplace_filter, mode="wrap"))])
@pytest.mark.parametrize("make_mask", [_binary_mask, _core_weighted_mask])
@pytest.mark.parametrize("sign", [-1, 1])
def test_score_pose_matches_correlator(
    tmp_path: Path,
    laplace: bool,
    filter_density: Callable[[np.ndarray], np.ndarray],
    make_mask: Callable[[np.ndarray], np.ndarray],
    sign: int,
):
    rng = np.random.default_rng(92)
    shape = (24, 24, 24)
    z, y, x = np.indices(shape)
    distance2 = sum(np.minimum(a, 24 - a) ** 2 for a in (x, y, z))
    mask = make_mask(distance2)
    structure = rng.random(shape)
    target = 2 + sign * structure + rng.normal(0, 0.1, shape)
    rotations = np.eye(3, dtype=np.float32)[None]
    correlator = CPUCorrelator(target, structure, rotations, mask, laplace=laplace)
    correlator.compute_rotation(0, rotations[0])
    target_file = tmp_path / "target.mrc"
    # Compare an interior pose: search uses periodic filtering, whereas fixed-pose
    # scoring treats density beyond the target boundary as zero.
    Volume(np.fft.fftshift(target)).tofile(target_file)
    with target_file.open("rb") as stream:
        prepared = setup_target_with_laplace(
            target_volume=stream, resolution=8, no_resampling=True, no_trimming=True, laplace=laplace
        )

    solution = score_pose(
        target=prepared,
        structure=Volume(np.fft.fftshift(filter_density(structure))),
        mask=Volume(np.fft.fftshift(mask)),
        z_sigma=0.25,
    )

    # lcc_scan retains negative scores, whereas lcc retains only improvements over zero.
    # FFT scoring uses float32 and subtraction of nearly equal second moments.
    assert solution[0] == pytest.approx(correlator.lcc_scan[0, 0, 0], abs=2e-5)


def test_score_returns_fisher_scores():
    structure = Volume(np.array([[[0.0, 1.0, 0.0, -1.0]]]))
    target = Volume(np.array([[[1.0, 1.0, -1.0, -1.0]]]))
    mask = Volume(np.ones(structure.shape))

    solution = score_pose(target=target, structure=structure, mask=mask, z_sigma=0.25)

    assert solution == pytest.approx([0.70710678, 0.88137359, 3.52549435])


def test_setup_target_laplace_filters_density(tmp_path: Path):
    target = Volume(np.array([[[1.0, 0.0, 0.0]]]))
    target_file = tmp_path / "target.mrc"
    target.tofile(target_file)

    with target_file.open("rb") as stream:
        prepared = setup_target_with_laplace(target_volume=stream, resolution=8, no_resampling=True, no_trimming=True)

    np.testing.assert_array_equal(prepared.array, [[[-6, 1, 0]]])


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("face", [0, -1])
def test_boundary_pose_score_ignores_opposite_face(tmp_path: Path, axis: int, face: int):
    rng = np.random.default_rng(94)
    density = rng.random((12, 12, 12))
    origin = [0, 0, 0]
    origin[2 - axis] = -1 if face == 0 else 9
    structure = Volume(rng.random((4, 4, 4)), origin=origin)
    mask = Volume(np.ones(structure.shape), origin=origin)
    target_file = tmp_path / "target.mrc"

    def score_target() -> list[float]:
        Volume(density).tofile(target_file)
        with target_file.open("rb") as stream:
            target = setup_target_with_laplace(target_volume=stream, resolution=8, no_resampling=True, no_trimming=True)
        return score_pose(target=target, structure=structure, mask=mask, z_sigma=0.25)

    original_score = score_target()
    opposite_face = [slice(None)] * 3
    opposite_face[axis] = -1 if face == 0 else 0
    density[tuple(opposite_face)] += 100

    assert score_target() == pytest.approx(original_score, abs=1e-12)


@pytest.fixture
def target_file(tmp_path: Path) -> Path:
    target = Volume(np.random.default_rng(93).random((16, 18, 20)), voxelspacing=1.5, origin=[3, -6, 9])
    path = tmp_path / "target.mrc"
    target.tofile(path)
    return path


def _volume_state(volume: Volume) -> dict[str, object]:
    return {
        "density": volume.array.tolist(),
        "origin": list(volume.origin),
        "voxelspacing": volume.voxelspacing,
    }


def test_setup_target_without_filtering_preserves_density(target_file: Path):
    target = Volume.fromfile(str(target_file))

    with target_file.open("rb") as stream:
        prepared = setup_target_with_laplace(
            target_volume=stream, resolution=8, no_resampling=True, no_trimming=True, laplace=False
        )

    assert _volume_state(prepared) == _volume_state(target)


@pytest.mark.parametrize("laplace", [False, True])
def test_scoring_preserves_prepared_target(target_file: Path, laplace: bool):
    with target_file.open("rb") as stream:
        prepared = setup_target_with_laplace(
            target_volume=stream, resolution=8, no_resampling=True, no_trimming=True, laplace=laplace
        )
    snapshot = _volume_state(prepared)
    structure = Volume(np.random.default_rng(95).random(prepared.shape), prepared.voxelspacing, prepared.origin)
    mask = Volume(np.ones(structure.shape), prepared.voxelspacing, prepared.origin)

    score_pose(target=prepared, structure=structure, mask=mask, z_sigma=0.25)

    assert _volume_state(prepared) == snapshot


def test_no_overlap_does_not_wrap():
    target = Volume(np.ones((16, 16, 16)))
    model = Volume(np.arange(64, dtype=float).reshape(4, 4, 4), origin=(100, 200, -100))
    mask = Volume(np.ones(model.shape), origin=model.origin)

    solution = score_pose(target=target, structure=model, mask=mask, z_sigma=0.2)

    assert solution == [0, 0, 0]


@pytest.mark.parametrize("constant", ["target", "structure"])
def test_constant_density_has_clear_error(constant: str):
    volumes = {
        "target": Volume(np.arange(64, dtype=float).reshape(4, 4, 4)),
        "structure": Volume(np.arange(64, dtype=float).reshape(4, 4, 4)),
    }
    volumes[constant] = Volume(np.ones((4, 4, 4)))
    mask = Volume(np.ones((4, 4, 4)))

    with pytest.raises(ValueError, match=f"{constant} with zero density variance"):
        score_pose(**volumes, mask=mask, z_sigma=0.25)


@pytest.mark.parametrize("delimiter", [",", "\t"], ids=["csv", "tsv"])
def test_write_solution_writes_expected_file(tmp_path: Path, delimiter: str):
    solution = [0.12349, -0.98765, 2.34567]
    out = tmp_path / "scores.txt"
    expected = (
        "rank,cc,Fish-z,rel-z,x,y,z,a11,a12,a13,a21,a22,a23,a31,a32,a33\n"
        "1,0.123,-0.988,2.346,0.000,0.000,0.000,1.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,1.000\n"
    ).replace(",", delimiter)

    write_solution(solution, out=out, delimiter=delimiter)

    assert out.read_text() == expected


@pytest.fixture
def score_inputs(tmp_path: Path) -> tuple[Path, Path]:
    pdb = tmp_path / "pose.pdb"
    pdb.write_text(
        "ATOM      1  N   MET A   1      25.250  30.500  35.750  1.00 20.00           N  \n"
        "ATOM      2  CA  MET A   1      28.000  28.000  37.000  1.00 20.00           C  \n"
        "ATOM      3  C   MET A   1      26.000  32.000  38.000  1.00 20.00           C  \n"
        "ATOM      4  O   MET A   1      24.000  29.000  39.000  1.00 20.00           O  \n"
    )
    structure = Structure.fromfile(str(pdb))
    target = Volume(np.zeros((64, 60, 56)))
    target.array = structure_to_shape_like(target, structure.coor, resolution=8, weights=structure.atomnumber).array
    target.array += np.random.default_rng(123).random(target.shape) * 0.0001
    target_file = tmp_path / "target.mrc"
    target.tofile(target_file)
    return target_file, pdb


@pytest.mark.parametrize(
    "laplace, expected_density",
    [
        pytest.param(False, [7.382150017, 9.819834959, 10.505247146, 9.654443750, 8.186025145], id="unfiltered"),
        pytest.param(True, [-3.442947706, -5.006438421, -4.483174977, -2.981073614, -2.250094814], id="laplace"),
    ],
)
def test_setup_structure_with_laplace(score_inputs: tuple[Path, Path], laplace: bool, expected_density: list[float]):
    target_file, pdb = score_inputs
    target = Volume.fromfile(str(target_file))

    with pdb.open("rb") as stream:
        structure, mask, z_sigma = setup_structure_with_laplace(
            structure=stream, chain=None, target=target, resolution=8, core_weighted=True, laplace=laplace
        )

    assert {
        "density": structure.array[11, 10, 8:13].tolist(),
        "mask": mask.array[11, 10, 8:13].tolist(),
        "z_sigma": z_sigma,
    } == {
        "density": pytest.approx(expected_density),
        "mask": [4, 5, 5, 6, 6],
        "z_sigma": 0.4,
    }


def test_main_with_defaults(score_inputs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]):
    target, pdb = score_inputs
    expected_csv = (
        "rank,cc,Fish-z,rel-z,x,y,z,a11,a12,a13,a21,a22,a23,a31,a32,a33\n"
        "1,0.907,1.510,3.699,0.000,0.000,0.000,1.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,1.000\n"
    )
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level

    try:
        main([str(target), "8", str(pdb)])
    finally:
        # main configures the root logger; restore pytest's logging.
        root.handlers[:] = handlers
        root.setLevel(level)

    assert capsys.readouterr().out == expected_csv

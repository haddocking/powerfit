"""Score actual atomic coordinates for MD comparisons with shared preprocessing."""

import csv
import io
import subprocess
from shutil import which

import numpy as np
import pytest
from scipy.ndimage import laplace as laplace_filter

from powerfit_em import Structure, Volume, determine_core_indices, structure_to_shape_like
from powerfit_em.correlators.cpu import CPUCorrelator
from powerfit_em.powerfit import make_parser as make_fit_parser
from powerfit_em.powerfit import setup_target
from powerfit_em.score import make_parser as make_score_parser
from powerfit_em.score import score_pose


def _binary_mask(distance2):
    return (distance2 < 16).astype(float)


def _core_weighted_mask(distance2):
    return _binary_mask(distance2) * np.where(distance2 < 4, 2, 1)


@pytest.mark.parametrize("laplace", [False, True])
@pytest.mark.parametrize("make_mask", [_binary_mask, _core_weighted_mask])
@pytest.mark.parametrize("sign", [-1, 1])
def test_score_pose_matches_correlator(laplace, make_mask, sign):
    rng = np.random.default_rng(92)
    shape = (24, 24, 24)
    z, y, x = np.indices(shape)
    distance2 = sum(np.minimum(a, 24 - a) ** 2 for a in (x, y, z))
    mask = make_mask(distance2)
    template = rng.random(shape)
    target = 2 + sign * template + rng.normal(0, 0.1, shape)
    rotations = np.eye(3, dtype=np.float32)[None]
    correlator = CPUCorrelator(target, template, rotations, mask, laplace=laplace)
    correlator.compute_rotation(0, rotations[0])

    solution = score_pose(
        target=Volume(target), template=Volume(template), mask=Volume(mask), laplace=laplace, z_sigma=0.25
    )

    # lcc_scan retains negative scores, whereas lcc retains only improvements over zero.
    # FFT scoring uses float32 and subtraction of nearly equal second moments.
    assert solution[0] == pytest.approx(correlator.lcc_scan[0, 0, 0], abs=2e-5)
    assert solution[1] == pytest.approx(np.arctanh(solution[0]))
    assert solution[2] == pytest.approx(solution[1] / 0.25)


@pytest.mark.parametrize("core_weighted", [False, True])
@pytest.mark.parametrize("delimiter", [",", "\t"])
@pytest.mark.parametrize("output", ["stdout", "file", "dash"])
def test_cli_scores_actual_coordinates_before_and_after_deformation(tmp_path, core_weighted, delimiter, output):
    pdb = tmp_path / "pre-md.pdb"
    pdb.write_text(
        "ATOM      1  N   MET A   1      39.250   7.500  64.750  1.00 20.00           N  \n"
        "ATOM      2  CA  MET A   1      42.000   5.000  66.000  1.00 20.00           C  \n"
        "ATOM      3  C   MET A   1      40.000   9.000  67.000  1.00 20.00           C  \n"
        "ATOM      4  O   MET A   1      38.000   6.000  68.000  1.00 20.00           O  \n"
    )
    structure = Structure.fromfile(str(pdb))
    # An odd-sized map with coarse spacing catches accidental resampling/FFT padding.
    target = Volume(np.zeros((47, 45, 43)), voxelspacing=1.5, origin=(21, -12, 43))
    model = structure_to_shape_like(target, structure.coor, resolution=13, weights=structure.atomnumber)
    target.array = model.array + np.random.default_rng(123).random(target.shape) * 0.01
    target_file = tmp_path / "target.mrc"
    target.tofile(target_file)
    scores = []
    for name, displacement in [("pre-md", 0), ("post-md", 3.125)]:
        pose = structure.duplicate()
        # Internal deformation and a fractional-grid displacement must both be scored.
        pose.data["x"][0] += displacement
        pose_file = tmp_path / f"{name}.pdb"
        pose.tofile(str(pose_file))
        original_bytes = pose_file.read_bytes()
        out_file = tmp_path / f"{name}.csv"
        files_before = set(tmp_path.iterdir())
        args = [which("powerfit-score"), str(target_file), "13", str(pose_file), "-nr", "-nt", "-nl"]
        if output == "file":
            args += ["--out", str(out_file)]
        elif output == "dash":
            args += ["--out", "-"]
        if not core_weighted:
            args += ["-ncw"]
        if delimiter != ",":
            args += ["--delimiter", delimiter]
        completed = subprocess.run(args, check=True, capture_output=True, text=True, timeout=30, cwd=tmp_path)
        text = out_file.read_text() if output == "file" else completed.stdout
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        assert len(rows) == 2
        assert rows[0][:4] == ["rank", "cc", "Fish-z", "rel-z"]
        row = np.asarray(rows[1], dtype=float)
        assert set(tmp_path.iterdir()) - files_before == ({out_file} if output == "file" else set())
        if output == "file":
            assert completed.stdout == ""
        weights = structure_to_shape_like(target, pose.coor, resolution=13, shape="mask").array
        if core_weighted:
            weights = determine_core_indices(weights)
        model_density = structure_to_shape_like(target, pose.coor, resolution=13, weights=pose.atomnumber).array
        support = weights != 0
        expected = np.corrcoef((model_density * weights)[support], (target.array * weights)[support])[0, 1]
        assert row[1] == pytest.approx(expected, abs=0.0005)
        assert row[2] == pytest.approx(np.arctanh(expected), abs=0.0005)
        np.testing.assert_array_equal(row[4:7], 0)
        np.testing.assert_array_equal(row[7:].reshape(3, 3), np.eye(3))
        assert pose_file.read_bytes() == original_bytes
        assert "Laplace filtering=False" in completed.stderr
        scores.append(row[1])
    assert scores[0] > scores[1]


def test_score_defaults_match_powerfit(tmp_path):
    target_file = tmp_path / "target.mrc"
    structure_file = tmp_path / "structure.pdb"
    target_file.touch()
    structure_file.touch()
    values = [str(target_file), "13", str(structure_file)]
    for options in [[], ["-nl", "-ncw", "-nr", "-nt", "-rr", "3", "-tc", "0.2"]]:
        score_args = make_score_parser().parse_args(values + options)
        fit_args = make_fit_parser().parse_args(values + options)
        with score_args.target, score_args.structure, fit_args.target, fit_args.template:
            for field in [
                "no_laplace",
                "no_core_weighted",
                "no_resampling",
                "resampling_rate",
                "no_trimming",
                "trimming_cutoff",
            ]:
                assert getattr(score_args, field) == getattr(fit_args, field)


def test_cli_applies_default_preprocessing(tmp_path):
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
    result = subprocess.run(
        [which("powerfit-score"), str(target_file), "8", str(pdb)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "Laplace filtering=True; resampling=True (rate 2); trimming=True" in result.stderr
    assert "Shape after resampling" in result.stderr
    assert "Shape after trimming" in result.stderr
    with target_file.open("rb") as stream:
        processed = setup_target(stream, 8, False, 2, False, None)
    assert processed.voxelspacing == 2
    assert np.prod(processed.shape) < np.prod(target.shape)
    model = structure_to_shape_like(processed, structure.coor, resolution=8, weights=structure.atomnumber).array
    mask = determine_core_indices(structure_to_shape_like(processed, structure.coor, resolution=8, shape="mask").array)
    support = mask != 0
    expected = np.corrcoef(
        (laplace_filter(model, mode="wrap") * mask)[support],
        (laplace_filter(processed.array, mode="wrap") * mask)[support],
    )[0, 1]
    row = np.asarray(list(csv.reader(io.StringIO(result.stdout)))[1], dtype=float)
    assert row[1] == pytest.approx(expected, abs=0.0005)


@pytest.mark.parametrize(
    "cli_args",
    [
        ["--gpu"],
        ["--cpu-backend", "rust"],
        ["--cpu-backend", "hybrid"],
        ["--angle", "20"],
        ["--nproc", "2"],
        ["--batch-size", "0"],
        ["--progressbar"],
        ["--num", "3"],
        ["--report"],
        ["--reference-template", "model.pdb"],
        ["-d", "scores"],
        ["--directory", "scores"],
        ["-out", "scores.csv"],
    ],
)
def test_score_command_rejects_search_and_directory_arguments(tmp_path, cli_args):
    target_file = tmp_path / "target.mrc"
    model_file = tmp_path / "model.pdb"
    target_file.touch()
    model_file.touch()
    files_before = set(tmp_path.iterdir())
    completed = subprocess.run(
        [
            which("powerfit-score"),
            str(target_file),
            "6",
            str(model_file),
            *cli_args,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert "unrecognized arguments:" in completed.stderr
    assert set(tmp_path.iterdir()) == files_before


def test_fitting_cli_preserves_existing_arguments(tmp_path):
    target_file = tmp_path / "target.mrc"
    model_file = tmp_path / "model.pdb"
    target_file.touch()
    model_file.touch()
    args = make_fit_parser().parse_args(
        [
            str(target_file),
            "6",
            str(model_file),
            "--angle",
            "20",
            "--gpu",
            "cuda:0",
            "--nproc",
            "2",
            "--cpu-backend",
            "py",
            "--batch-size",
            "0",
            "--num",
            "3",
            "--report",
            "--delimiter",
            "\t",
            "-nl",
            "-ncw",
            "-nr",
            "-nt",
        ]
    )
    with args.target, args.template:
        assert args.angle == 20
        assert args.gpu == "cuda:0"
        assert args.nproc == 2
        assert args.cpu_backend == "py"
        assert args.batch_size == 0
        assert args.num == 3
        assert args.report
        assert args.delimiter == "\t"
        assert args.no_laplace
        assert args.no_core_weighted
        assert args.no_resampling
        assert args.no_trimming


def test_no_overlap_does_not_wrap():
    target = Volume(np.ones((16, 16, 16)))
    model = Volume(np.arange(64, dtype=float).reshape(4, 4, 4), origin=(100, 200, -100))
    mask = Volume(np.ones(model.shape), origin=model.origin)
    assert score_pose(target=target, template=model, mask=mask, laplace=False, z_sigma=0.2)[:3] == [0, 0, 0]


def test_constant_density_has_clear_error():
    model = Volume(np.arange(64, dtype=float).reshape(4, 4, 4))
    with pytest.raises(ValueError, match="zero density variance"):
        score_pose(
            target=Volume(np.ones(model.shape)),
            template=model,
            mask=Volume(np.ones(model.shape)),
            laplace=False,
            z_sigma=0.2,
        )

"""Regression tests for PowerFit fitting results."""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from tqdm.rich import tqdm as rich_tqdm

from powerfit_em import powerfit

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# The fixtures below where copied from https://github.com/haddocking/powerfit-tutorial


@pytest.fixture(scope="session")
def ribosome_map() -> Path:
    return FIXTURES_DIR / "ribosome-KsgA.map"


@pytest.fixture(scope="session")
def ksga_pdb() -> Path:
    return FIXTURES_DIR / "KsgA.pdb"


@pytest.fixture(scope="session")
def baseline_solutions() -> Path:
    return FIXTURES_DIR / "solutions.out"


def test_powerfit_solutions_match_baseline(
    ribosome_map: Path, ksga_pdb: Path, tmp_path: Path, powerfit_args: list[str], baseline_solutions: Path
):
    """Test that solutions.out matches baseline across execution profiles.

    The test accepts optional runtime arguments via --powerfit (for example, --gpu, --nproc 6)
    and verifies that all profiles produce numerically identical results (within rounding
    tolerance) by comparing against a single cached baseline.
    """
    args_list = [
        str(ribosome_map),
        "13",
        str(ksga_pdb),
        "-a",
        "20",
        "-d",
        str(tmp_path),
        "--delimiter",
        ",",
    ]
    args_list.extend(powerfit_args)

    args = powerfit.make_parser().parse_args(args_list)
    progress = partial(rich_tqdm, desc="Processing rotations", unit="rot") if args.progressbar else None

    powerfit.powerfit(
        target_volume=args.target,
        resolution=args.resolution,
        template_structure=args.template,
        angle=args.angle,
        laplace=not args.no_laplace,
        core_weighted=not args.no_core_weighted,
        no_resampling=args.no_resampling,
        resampling_rate=args.resampling_rate,
        no_trimming=args.no_trimming,
        trimming_cutoff=args.trimming_cutoff,
        chain=args.chain,
        directory=str(tmp_path),
        num=args.num,
        gpu=args.gpu,
        nproc=args.nproc,
        delimiter=args.delimiter,
        progress=progress,
    )

    generated_file = tmp_path / "solutions.out"
    print(generated_file)
    baseline_file = baseline_solutions

    assert generated_file.exists(), f"Generated solutions.out not found at {generated_file}"
    assert baseline_file.exists(), (
        f"Baseline solutions.out fixture not found at {baseline_file}. "
        f"See CONTRIBUTING.md under 'Baseline fixture maintenance' to create initial baseline."
    )

    baseline_df = pd.read_csv(baseline_file, skipinitialspace=True)
    generated_df = pd.read_csv(generated_file, skipinitialspace=True)

    # Some GPU backends can differ by a very small number of watershed peaks near
    # threshold boundaries. Keep this guard tight so substantive extraction changes fail.
    row_count_tolerance = 2
    row_count_delta = abs(len(baseline_df) - len(generated_df))
    assert row_count_delta <= row_count_tolerance, (
        "Unexpected number of extracted solutions. "
        f"Baseline has {len(baseline_df)} rows, generated has {len(generated_df)} rows "
        f"(delta={row_count_delta}, tolerance={row_count_tolerance})."
    )

    # Sort both DataFrames by spatial position before comparing.
    # GPU and CPU may compute slightly different LCC values for the same voxel position,
    # which can flip the rank of near-identical solutions. Sorting by (x, y, z) — which are
    # derived from integer voxel indices and are therefore bit-exact across hardware — aligns
    # each row with its counterpart regardless of LCC-based rank differences.
    sort_cols = ["x", "y", "z"]
    baseline_sorted = baseline_df.sort_values(by=sort_cols).reset_index(drop=True)
    generated_sorted = generated_df.sort_values(by=sort_cols).reset_index(drop=True)

    # Exclude 'rank' from comparison: ranks may legitimately differ between GPU and CPU when
    # two solutions have equal LCC at output precision but the hardware resolves them differently.
    compare_cols = [c for c in baseline_sorted.columns if c != "rank"]

    # Compare rows matched by spatial identity. This avoids failing on tiny backend-specific
    # differences in watershed peak count while still validating nearly all rows.
    merged = baseline_sorted[compare_cols].merge(
        generated_sorted[compare_cols], on=sort_cols, how="inner", suffixes=("_baseline", "_generated")
    )

    unmatched_rows = len(baseline_sorted) - len(merged)
    max_unmatched_rows = max(2, int(np.ceil(len(baseline_sorted) * 0.0075)))
    matched_fraction = len(merged) / len(baseline_sorted)
    assert unmatched_rows <= max_unmatched_rows, (
        "Too many unmatched spatial solutions between baseline and generated output. "
        f"Matched {len(merged)} / {len(baseline_sorted)} rows ({matched_fraction:.3%}); "
        f"unmatched={unmatched_rows}, allowed={max_unmatched_rows}."
    )

    value_cols = [c for c in compare_cols if c not in sort_cols]
    baseline_matched = merged[[f"{c}_baseline" for c in value_cols]].copy()
    generated_matched = merged[[f"{c}_generated" for c in value_cols]].copy()
    baseline_matched.columns = value_cols
    generated_matched.columns = value_cols

    # rel-z and Fish-z amplify small cc differences, so allow slightly larger absolute tolerance.
    close_matrix = np.ones((len(merged), len(value_cols)), dtype=bool)
    for i, col in enumerate(value_cols):
        atol = 1e-2 if col in {"Fish-z", "rel-z"} else 1e-3
        close_matrix[:, i] = np.isclose(
            baseline_matched[col].to_numpy(),
            generated_matched[col].to_numpy(),
            rtol=1e-3,
            atol=atol,
        )

    failing_cols = [col for i, col in enumerate(value_cols) if not close_matrix[:, i].all()]
    assert not failing_cols, f"Columns exceeded numeric tolerances after spatial matching: {failing_cols}"

    assert_frame_equal(
        baseline_matched,
        generated_matched,
        check_exact=False,
        rtol=1e-3,
        atol=1e-2,
    )

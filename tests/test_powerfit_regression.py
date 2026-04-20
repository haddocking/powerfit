"""Regression tests for PowerFit fitting results."""

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from powerfit_em import powerfit


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
        "--no-progressbar",
    ]
    args_list.extend(powerfit_args)

    parsed_args = powerfit.make_parser().parse_args(args_list)

    powerfit.powerfit(
        target_volume=parsed_args.target,
        resolution=parsed_args.resolution,
        template_structure=parsed_args.template,
        angle=parsed_args.angle,
        laplace=not parsed_args.no_laplace,
        core_weighted=not parsed_args.no_core_weighted,
        no_resampling=parsed_args.no_resampling,
        resampling_rate=parsed_args.resampling_rate,
        no_trimming=parsed_args.no_trimming,
        trimming_cutoff=parsed_args.trimming_cutoff,
        chain=parsed_args.chain,
        directory=str(tmp_path),
        num=parsed_args.num,
        gpu=parsed_args.gpu,
        nproc=parsed_args.nproc,
        delimiter=parsed_args.delimiter,
        progress=None,  # Disable progress output for tests
    )

    generated_file = tmp_path / "solutions.out"
    baseline_file = baseline_solutions

    assert generated_file.exists(), f"Generated solutions.out not found at {generated_file}"
    assert baseline_file.exists(), (
        f"Baseline solutions.out fixture not found at {baseline_file}. "
        f"See CONTRIBUTING.md under 'Baseline fixture maintenance' to create initial baseline."
    )

    baseline_df = pd.read_csv(baseline_file, skipinitialspace=True)
    generated_df = pd.read_csv(generated_file, skipinitialspace=True)

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

    assert_frame_equal(
        baseline_sorted[compare_cols],
        generated_sorted[compare_cols],
        check_exact=False,
        rtol=1e-3,
        atol=1e-3,
    )

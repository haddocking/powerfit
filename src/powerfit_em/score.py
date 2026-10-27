"""Score a non-rotated, non-translated structure with the powerfit-score command."""

import csv
import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType, RawDescriptionHelpFormatter
from contextlib import nullcontext
from pathlib import Path
from typing import BinaryIO

import numpy as np
from scipy.ndimage import affine_transform
from scipy.ndimage import laplace as laplace_filter

from powerfit_em import Volume, __version__
from powerfit_em.correlators.shared import get_normalization_factor, normalize_template
from powerfit_em.powerfit import add_scoring_arguments2parser, setup_target, setup_template_structure

logger = logging.getLogger(__name__)


def setup_target_with_laplace(
    target_volume: BinaryIO,
    resolution: float,
    no_resampling: bool = False,
    resampling_rate: float = 2,
    no_trimming: bool = False,
    trimming_cutoff: float | None = None,
    laplace: bool = True,
) -> Volume:
    """Load and preprocess a target, optionally applying Laplace filtering once.

    Reuse the returned volume with score_pose for many structures.
    Laplace filtering treats density beyond the target map as zero.

    Args:
        target_volume: Open density map file accepted by setup_target.
        resolution: Map resolution in angstrom.
        no_resampling: Whether to skip resampling.
        resampling_rate: Number of samples per resolution element.
        no_trimming: Whether to skip trimming.
        trimming_cutoff: Density cutoff for trimming. Defaults to 10% of the maximum.
        laplace: Whether to Laplace-filter the preprocessed target density.

    Returns:
        Preprocessed target ready for repeated scoring.

    Example:
        Score many structures against the same map with the CLI's default settings::

            from powerfit_em.score import (
                setup_target_with_laplace, setup_structure_with_laplace, score_pose
            )

            resolution = 3.7
            with open("target.mrc", "rb") as stream:
                target = setup_target_with_laplace(target_volume=stream, resolution=resolution)

            scores = {}
            for filename in ["fit_1.pdb", "refined_1.pdb.gz"]:
                with open(filename, "rb") as stream:
                    structure, mask, z_sigma = setup_structure_with_laplace(
                        structure=stream, chain=None, target=target,
                        resolution=resolution, core_weighted=True
                    )
                scores[filename] = score_pose(
                    target=target, structure=structure, mask=mask, z_sigma=z_sigma
                )
    """
    target = setup_target(
        target_volume=target_volume,
        resolution=resolution,
        no_resampling=no_resampling,
        resampling_rate=resampling_rate,
        no_trimming=no_trimming,
        trimming_cutoff=trimming_cutoff,
    )
    target.array = np.asarray(target.array, dtype=np.float64)
    if laplace:
        target.array = laplace_filter(target.array, mode="constant", cval=0.0)
    return target


def setup_structure_with_laplace(
    structure: BinaryIO,
    chain: str | None,
    target: Volume,
    resolution: float,
    core_weighted: bool,
    laplace: bool = True,
) -> tuple[Volume, Volume, float]:
    """Build a structure at its input coordinates and optionally Laplace-filter it.

    Args:
        structure: Open atomic structure file accepted by setup_template_structure.
        chain: Comma-separated chain IDs to select, or None for all chains.
        target: Preprocessed target defining the voxel lattice.
        resolution: Map resolution in angstrom.
        core_weighted: Whether to use a core-weighted mask.
        laplace: Whether to Laplace-filter the structure density. Must match the
            setting used in setup_target_with_laplace.

    Returns:
        Structure density, mask, and Fisher z standard deviation ready for score_pose.
    """
    _, structure, mask, z_sigma = setup_template_structure(
        template_structure=structure,
        chain=chain,
        target=target,
        resolution=resolution,
        core_weighted=core_weighted,
        center=False,
    )
    model = np.asarray(structure.array, dtype=np.float64)
    if laplace:
        model = laplace_filter(model, mode="wrap")
    structure.array = model
    return structure, mask, z_sigma


def score_pose(*, target: Volume, structure: Volume, mask: Volume, z_sigma: float) -> list[float]:
    """Score a structure against a target returned by setup_target_with_laplace.

    Prepare the structure with setup_structure_with_laplace using the
    same Laplace setting as the target. Each structure must use the target's voxel
    lattice, though its shape and origin may differ. Rebuild the structure, mask,
    and Fisher sigma for each structure, as shown in setup_target_with_laplace's example.

    Reuse the correlators' mask support and core-weighted normalization, evaluating
    the normalized dot product directly instead of through FFTs. Negative scores
    are retained without density cutoff or best-score selection. Density outside
    the target map is zero. Input volumes are not modified.

    Args:
        target: Target density returned by setup_target_with_laplace.
        structure: Prepared structure density on the target's voxel lattice.
        mask: Mask weights on the structure's grid, optionally core-weighted.
        z_sigma: Fisher z standard deviation for this structure and resolution.

    Returns:
        cc, Fisher z, and relative z.

    Raises:
        ValueError: The structure has zero density variance inside its mask, or
            the target has nonzero constant density inside the structure mask.
    """
    model = np.asarray(structure.array, dtype=np.float64)
    weights = np.asarray(mask.array, dtype=np.float64)
    # Structure and target share a voxel lattice. Target density outside the
    # supplied map is zero, rather than wrapping to the opposite map boundary.
    offset = np.rint((np.asarray(structure.origin) - target.origin) / target.voxelspacing)[::-1]
    density = affine_transform(
        target.array, np.eye(3), offset=offset, output_shape=model.shape, order=0, mode="constant", prefilter=False
    )
    norm_factor = float(get_normalization_factor(weights))
    support = weights != 0
    model_values = (model * weights)[support]
    target_values = (density * weights)[support]
    if model_values.std() == 0:
        raise ValueError("Cannot score a structure with zero density variance inside its mask.")
    if not target_values.any():
        cc = 0.0
    elif target_values.std() == 0:
        raise ValueError("Cannot score a target with zero density variance inside the structure mask.")
    else:
        normalized_model = normalize_template(model, weights)
        # normalize_template includes the second mask multiplication used by
        # the FFT correlators' numerator. Center target values only for their
        # variance, avoiding cancellation in the equivalent second-moment sum.
        numerator = np.dot(normalized_model[support], density[support])
        cc = float(np.clip(numerator / (norm_factor * target_values.std()), -1.0, 1.0))
    with np.errstate(divide="ignore"):
        fishers_z = float(np.arctanh(cc))
    return [cc, fishers_z, fishers_z / z_sigma]


def write_solution(solution: list[float], out: Path | None = None, delimiter: str = ",") -> None:
    """Write a score solution as a header and one data row.

    Add rank 1, zero translation, and an identity rotation to the scores.

    Args:
        solution: cc, Fisher z, and relative z returned by score_pose.
        out: Output file path. Defaults to stdout; Path("-") also selects stdout.
        delimiter: Separator between fields. Defaults to a comma.
    """
    is_stdout = out is None or out == Path("-")
    output = nullcontext(sys.stdout) if is_stdout else out.open("w", newline="")
    with output as stream:
        writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
        writer.writerow(
            [
                "rank",
                "cc",
                "Fish-z",
                "rel-z",
                "x",
                "y",
                "z",
                "a11",
                "a12",
                "a13",
                "a21",
                "a22",
                "a23",
                "a31",
                "a32",
                "a33",
            ]
        )
        values = [
            1,
            *(f"{value:.3f}" for value in solution),
            "0.000",
            "0.000",
            "0.000",
            "1.000",
            "0.000",
            "0.000",
            "0.000",
            "1.000",
            "0.000",
            "0.000",
            "0.000",
            "1.000",
        ]
        writer.writerow(values)


class HelpFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    """Preserve description formatting and display argument defaults in help."""


def make_parser():
    """Create the command-line parser for scoring an existing pose.

    Returns:
        Parser configured with input files, scoring settings, and output options.
    """
    parser = ArgumentParser(
        prog="powerfit-score",
        description=(
            "Score the supplied atomic coordinates against a density map, no rotations or translations are done."
        ),
        epilog=(
            "Scores can differ from powerfit search results because powerfit rotates\n"
            "precomputed density and mask grids, while powerfit-score rebuilds them\n"
            "from the supplied atomic coordinates. Grid interpolation and mask\n"
            "discretization can therefore change the score, even for an exported fit.\n"
            "With Laplace filtering enabled, scores can also differ when the pose's\n"
            "mask reaches a target map boundary: powerfit uses periodic target\n"
            "filtering, while powerfit-score treats density outside the target as\n"
            "zero. This filtering difference does not affect interior poses."
        ),
        formatter_class=HelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("target", type=FileType("rb"), help="Input density map in CCP4 or MRC format.")
    parser.add_argument("resolution", type=float, help="Map resolution in angstrom.")
    parser.add_argument(
        "structure",
        type=FileType("rb"),
        help="Structure at its current coordinates: PDB or mmCIF, optionally gzip-compressed.",
    )
    add_scoring_arguments2parser(parser)
    parser.add_argument("-c", "--chain", help="Chain IDs to score, separated by commas; default is all chains.")
    parser.add_argument(
        "--out", type=Path, help="Write the score CSV to this file; default is stdout. Use - for stdout."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument("--delimiter", default=",", help="Output delimiter; default is CSV (comma-separated).")
    return parser


def main(argv: list[str] | None = None):
    """Score the supplied pose and write its scores using command-line options.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            the process command-line arguments.
    """
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, stream=sys.stderr, format="%(message)s", force=True)
    logger.info(
        "Scoring settings: Laplace filtering=%s; resampling=%s (rate %s); trimming=%s (cutoff %s).",
        not args.no_laplace,
        not args.no_resampling,
        args.resampling_rate,
        not args.no_trimming,
        args.trimming_cutoff if args.trimming_cutoff is not None else "10% of maximum",
    )
    with args.target:
        target = setup_target_with_laplace(
            target_volume=args.target,
            resolution=args.resolution,
            no_resampling=args.no_resampling,
            resampling_rate=args.resampling_rate,
            no_trimming=args.no_trimming,
            trimming_cutoff=args.trimming_cutoff,
            laplace=not args.no_laplace,
        )
    logger.info(
        "Map shape: %s; voxel spacing: %s; resolution: %s A", target.shape, target.voxelspacing, args.resolution
    )
    with args.structure:
        structure, mask, z_sigma = setup_structure_with_laplace(
            structure=args.structure,
            chain=args.chain,
            target=target,
            resolution=args.resolution,
            core_weighted=not args.no_core_weighted,
            laplace=not args.no_laplace,
        )
    logger.info("Scoring the supplied coordinates without alignment or pose optimization; outside-map density is zero.")
    solution = score_pose(target=target, structure=structure, mask=mask, z_sigma=z_sigma)
    write_solution(solution, out=args.out, delimiter=args.delimiter)

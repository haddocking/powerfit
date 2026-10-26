"""Score a model at its input coordinates with the powerfit-score command."""

import csv
import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType, RawDescriptionHelpFormatter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform
from scipy.ndimage import laplace as laplace_filter

from powerfit_em import Volume, __version__
from powerfit_em.correlators.shared import get_normalization_factor, normalize_template
from powerfit_em.powerfit import add_scoring_arguments2parser, setup_target, setup_template_structure

logger = logging.getLogger(__name__)


def score_pose(*, target: Volume, template: Volume, mask: Volume, laplace: bool, z_sigma: float) -> list[float]:
    """Calculate local cross-correlation at the template's grid coordinates.

    Reuse the correlators' mask support and core-weighted template normalization,
    evaluating their normalized dot product directly instead of through FFTs.
    Retain negative scores without the search's density cutoff or best-score selection.
    """
    density = np.asarray(target.array, dtype=np.float64)
    model = np.asarray(template.array, dtype=np.float64)
    weights = np.asarray(mask.array, dtype=np.float64)
    if laplace:
        density = laplace_filter(density, mode="wrap")
        model = laplace_filter(model, mode="wrap")
    # Template and target share a voxel lattice. Target density outside the
    # supplied map is zero, rather than wrapping to the opposite map boundary.
    offset = np.rint((np.asarray(template.origin) - target.origin) / target.voxelspacing)[::-1]
    density = affine_transform(
        density, np.eye(3), offset=offset, output_shape=model.shape, order=0, mode="constant", prefilter=False
    )
    norm_factor = float(get_normalization_factor(weights))
    support = weights != 0
    model_values = (model * weights)[support]
    target_values = (density * weights)[support]
    if model_values.std() == 0:
        raise ValueError("Cannot score a template with zero density variance inside its mask.")
    if not target_values.any():
        cc = 0.0
    elif target_values.std() == 0:
        raise ValueError("Cannot score a target with zero density variance inside the template mask.")
    else:
        normalized_model = normalize_template(model, weights)
        # normalize_template includes the second mask multiplication used by
        # the FFT correlators' numerator. Center target values only for their
        # variance, avoiding cancellation in the equivalent second-moment sum.
        numerator = np.dot(normalized_model[support], density[support])
        cc = float(np.clip(numerator / (norm_factor * target_values.std()), -1.0, 1.0))
    with np.errstate(divide="ignore"):
        fishers_z = float(np.arctanh(cc))
    return [cc, fishers_z, fishers_z / z_sigma, 0.0, 0.0, 0.0, *np.eye(3).ravel()]


class HelpFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    pass


def make_parser():
    parser = ArgumentParser(
        prog="powerfit-score",
        description=(
            "Score the supplied atomic coordinates against a density map, no rotations or translations are done."
        ),
        epilog=(
            "Scores can differ from powerfit search results because powerfit rotates\n"
            "precomputed density and mask grids, while powerfit-score rebuilds them\n"
            "from the supplied atomic coordinates. Grid interpolation and mask\n"
            "discretization can therefore change the score, even for an exported fit."
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


def main():
    args = make_parser().parse_args()
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
        target = setup_target(
            args.target,
            args.resolution,
            args.no_resampling,
            args.resampling_rate,
            args.no_trimming,
            args.trimming_cutoff,
        )
    logger.info(
        "Map shape: %s; voxel spacing: %s; resolution: %s A", target.shape, target.voxelspacing, args.resolution
    )
    with args.structure:
        _, template, mask, z_sigma = setup_template_structure(
            args.structure, args.chain, target, args.resolution, not args.no_core_weighted, center=False
        )
    logger.info("Scoring the supplied coordinates without alignment or pose optimization; outside-map density is zero.")
    solution = score_pose(target=target, template=template, mask=mask, laplace=not args.no_laplace, z_sigma=z_sigma)
    output = nullcontext(sys.stdout) if args.out is None or str(args.out) == "-" else args.out.open("w", newline="")
    with output as stream:
        writer = csv.writer(stream, delimiter=args.delimiter, lineterminator="\n")
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
        writer.writerow([1, *[f"{value:.3f}" for value in solution]])

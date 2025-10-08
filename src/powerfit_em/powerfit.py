#! ../env/bin/python


from functools import partial
from os.path import splitext, join, abspath
from pathlib import Path
from time import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, BooleanOptionalAction, FileType
import logging
from typing import BinaryIO
import warnings

import numpy as np
from rich.logging import RichHandler
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm as rich_tqdm
from tqdm.auto import tqdm

from powerfit_em import (
    __version__,
    Volume,
    Structure,
    structure_to_shape_like,
    proportional_orientations,
    quat_to_rotmat,
    determine_core_indices,
)
from powerfit_em.powerfitter import PowerFitter
from powerfit_em.analyzer import Analyzer
from powerfit_em.helpers import write_fits_to_pdb, fisher_sigma
from powerfit_em.report import generate_report
from powerfit_em.volume import extend, nearest_multiple2357, trim, resample

try:
    import pyopencl as cl
    OPENCL = True
except:
    OPENCL = False

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


def make_parser():
    """Create the command-line argument parser."""
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Positional arguments
    p.add_argument(
        "target",
        type=FileType("rb"),
        help="Target density map to fit the model in. "
        "Data should either be in CCP4 or MRC format",
    )
    p.add_argument("resolution", type=float, help="Resolution of map in angstrom")
    p.add_argument(
        "template",
        type=FileType("rb"),
        help="Atomic model to be fitted in the density. "
        "Format should either be PDB or mmCIF. "
        "Valid extensions are .pdb, .ent, .cif, .pdb.gz, .ent.gz, .cif.gz",
    )

    # Optional arguments and flags
    p.add_argument(
        "-a",
        "--angle",
        dest="angle",
        type=float,
        default=10,
        metavar="<float>",
        help="Rotational sampling density in degree. Increasing "
        "this number by a factor of 2 results in approximately "
        "8 times more rotations sampled.",
    )
    # Scoring flags
    p.add_argument(
        "-nl",
        "--no-laplace",
        dest="no_laplace",
        action="store_true",
        help="Do not use the Laplace pre-filter density data.",
    )
    p.add_argument(
        "-ncw",
        "--no-core-weighted",
        dest="no_core_weighted",
        action="store_true",
        help="Do not use core-weighted local cross-correlation score.",
    )
    # Resampling
    p.add_argument(
        "-nr",
        "--no-resampling",
        dest="no_resampling",
        action="store_true",
        help="Do not resample the density map.",
    )
    p.add_argument(
        "-rr",
        "--resampling-rate",
        dest="resampling_rate",
        type=float,
        default=2,
        metavar="<float>",
        help="Resampling rate compared to Nyquist.",
    )
    # Trimming related
    p.add_argument(
        "-nt",
        "--no-trimming",
        dest="no_trimming",
        action="store_true",
        help="Do not trim the density map.",
    )
    p.add_argument(
        "-tc",
        "--trimming-cutoff",
        dest="trimming_cutoff",
        type=float,
        default=None,
        metavar="<float>",
        help="Intensity cutoff to which the map will be trimmed. "
        "Default is 10 percent of maximum intensity.",
    )
    # Selection parameter
    p.add_argument(
        "-c",
        "--chain",
        dest="chain",
        type=str,
        default=None,
        metavar="<char>",
        help=(
            "The chain IDs of the structure to be fitted. "
            "Multiple chains can be selected using a comma separated list, i.e. -c A,B,C. "
            "Default is the whole structure."
        ),
    )
    # Output parameters
    p.add_argument(
        "-d",
        "--directory",
        dest="directory",
        type=abspath,
        default=".",
        metavar="<dir>",
        help="Directory where the results are stored.",
    )
    p.add_argument(
        "-n",
        "--num",
        dest="num",
        type=int,
        default=10,
        metavar="<int>",
        help="Number of models written to file. This number "
        "will be capped if less solutions are found as requested.",
    )
    # Computational resources parameters
    p.add_argument(
        "-g",
        "--gpu",
        dest="gpu",
        nargs="?",
        const="0:0",
        default=None,
        metavar="[<platform>:<device>]",
        help="Off-load the intensive calculations to the GPU. Optionally specify platform and device as <platform>:<device> (e.g., --gpu 0:3). If not specified, uses first device in first platform. If omitted, does not use GPU.",
    )
    p.add_argument(
        "-p",
        "--nproc",
        dest="nproc",
        type=int,
        default=1,
        metavar="<int>",
        help="Number of processors used during search. "
        "The number will be capped at the total number "
        "of available processors on your machine.",
    )
    p.add_argument(
        "--log-level",
        dest="log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    p.add_argument(
        "--delimiter",
        dest="delimiter",
        type=str,
        default=None,
        metavar="<str>",
        help="Delimiter used in the 'solutions.out' file. For example use ',' or '\\t'. Defaults to fixed width.",
    )
    p.add_argument(
        "--progressbar",
        dest="progressbar",
        action=BooleanOptionalAction,
        default=True,
        help="Show a progress bar during the search. Disabling the progressbar will improve performance.",
    )
    p.add_argument(
        "--report",
        dest="report",
        action="store_true",
        help="Generate a html report with Mol* 3D viewer of the fitted models."
    )

    return p

def parse_args():
    """Parse command-line options."""
    p = make_parser()
    args = p.parse_args()

    return args


def get_filetype_template(fname):
    """Determine the file type from the file extension."""
    ext = splitext(fname)[1][1:]
    if ext in ["pdb", "ent"]:
        ft = "pdb"
    elif ext in ["map", "ccp4"]:
        ft = "map"
    else:
        msg = "Filetype of file {:} is not recognized.".format(fname)
        raise IOError(msg)
    return ft

def configure_logging(log_file, log_level= "INFO"):
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    # Write log messages to a file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logging.root.addHandler(file_handler)
    
    # Write to console with rich formatting
    console_handler = RichHandler(show_time=False, show_path=False, show_level=False)
    console_handler.setLevel(log_level)
    logging.root.addHandler(console_handler)
    logger.setLevel(log_level)


def main():
    args = parse_args()
    
    Path(args.directory).mkdir(exist_ok=True)
    configure_logging(join(args.directory, "powerfit.log"), args.log_level)
    
    progress = partial(
        rich_tqdm, desc="Processing rotations", unit="rot"
    ) if args.progressbar else None

    powerfit(
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
        directory=args.directory,
        num=args.num,
        gpu=args.gpu,
        nproc=args.nproc,
        delimiter=args.delimiter,
        progress=progress
    )
    if args.report:
        # Report shows all options that affect the fitting
        options = {
            'resolution': args.resolution,
            'angle': args.angle,
            'no laplace': args.no_laplace,
            'no core weighted': args.no_core_weighted,
            'no resampling': args.no_resampling,
            'resampling rate': args.resampling_rate,
            'no trimming': args.no_trimming,
            'trimming cutoff': args.trimming_cutoff,
        }
        generate_report(args.directory, args.target.name, args.num, args.delimiter, options=options)


def get_gpu_queue(gpu: str) -> "cl.CommandQueue":
    """Request an OpenCL Queue."""
    if not OPENCL:
        msg = "Running on GPU requires the pyopencl package, however importing pyopencl failed."
        raise ValueError(msg)
    # TODO allow to omit platform, so gpu='4' runs 5th device on first platform
    if ':' in gpu:
        platform_idx, device_idx = map(int, gpu.split(':'))
    else:
        platform_idx, device_idx = 0, 0
    platforms = cl.get_platforms()
    if platform_idx >= len(platforms):
        raise RuntimeError(f"Requested OpenCL platform {platform_idx} not found.")
    platform = platforms[platform_idx]
    devices = platform.get_devices()
    if device_idx >= len(devices):
        raise RuntimeError(f"Requested OpenCL device {device_idx} not found on platform {platform_idx}.")
    context = cl.Context(devices=[devices[device_idx]])
    return cl.CommandQueue(context, device=devices[device_idx])


def setup_target(
    target_volume: BinaryIO,
    resolution: float,
    no_resampling: bool,
    resampling_rate: float,
    no_trimming: bool,
    trimming_cutoff: float | None,
) -> Volume:
    """Load and preprocess the target density."""
    logger.info("Target file read from: {:s}".format(abspath(target_volume.name)))
    target = Volume.fromfile(target_volume)
    logger.info("Target resolution: {:.2f}".format(resolution))
    logger.info(("Initial shape of density:" + " {:d}" * 3).format(*target.shape))

    # Resample target density if requested
    if not no_resampling:
        factor = 2 * resampling_rate * target.voxelspacing / resolution
        if factor < 0.9:
            target = resample(target, factor)
            logger.info(("Shape after resampling:" + " {:d}" * 3).format(*target.shape))

    # Trim target density if requested
    if not no_trimming:
        if trimming_cutoff is None:
            trimming_cutoff = target.array.max() / 10
        target = trim(target, trimming_cutoff)
        logger.info(("Shape after trimming:" + " {:d}" * 3).format(*target.shape))

    # Extend the density to a multiple of 2, 3, 5, and 7 for clFFT
    extended_shape = [nearest_multiple2357(n) for n in target.shape]
    target = extend(target, extended_shape)
    logger.info(("Shape after extending:" + " {:d}" * 3).format(*target.shape))
    return target


def setup_template_structure(
    template_structure: BinaryIO,
    chain: str | None,
    target: Volume,
    resolution: float,
    core_weighted: bool
) -> tuple[Structure, Volume, Volume, float]:
    """Load structure, setup template and mask, precompute Fisher sigma for scoring."""
    # Read in structure or high-resolution map
    logger.info("Template file read from: {:s}".format(abspath(template_structure.name)))
    structure = Structure.fromfile(template_structure)
    if chain is not None:
        logger.info("Selecting chains: " + chain)
        structure = structure.select("chain", chain.split(","))
    if structure.data.size == 0:
        raise ValueError("No atoms were selected.")

    # Move structure to origin of density
    structure.translate(target.origin - structure.coor.mean(axis=1))
    template = structure_to_shape_like(
        target,
        structure.coor,
        resolution=resolution,
        weights=structure.atomnumber,
        shape="vol",
    )
    mask = structure_to_shape_like(
        target, structure.coor, resolution=resolution, shape="mask"
    )

    # Apply core-weighted mask if requested
    if core_weighted:
        logger.info("Calculating core-weighted mask.")
        mask.array = determine_core_indices(mask.array)

    # calculate the molecular volume of the structure
    mv = (
        structure_to_shape_like(
            target,
            structure.coor,
            resolution=resolution,
            radii=structure.rvdw,
            shape="mask",
        ).array.sum()
        * target.voxelspacing**3
    )
    z_sigma = fisher_sigma(mv, resolution)
    return structure, template, mask, z_sigma


def setup_rotational_matrix(
    angle: float        
) -> np.ndarray:
    logger.info("Reading in rotations.")
    q, _, degree = proportional_orientations(angle)
    rotmat = quat_to_rotmat(q)
    logger.info("Requested rotational sampling density: {:.2f}".format(angle))
    logger.info("Real rotational sampling density: {:}".format(degree))
    return rotmat


def powerfit(
    target_volume: BinaryIO,
    resolution: float,
    template_structure: BinaryIO,
    angle: float=10,
    laplace: bool=False,
    core_weighted: bool=False,
    no_resampling: bool=False,
    resampling_rate: float=2,
    no_trimming: bool=False,
    trimming_cutoff: float | None=None,
    chain: str | None =None,
    directory: str='.',
    num: int=10,
    gpu: str | None = None, 
    nproc: int=1,
    delimiter: str | None = None,
    progress: partial[tqdm] | None = tqdm,
):
    time0 = time()
    Path(directory).mkdir(exist_ok=True)

    # Get GPU queue if requested
    queue = None
    if gpu:
        queue = get_gpu_queue(gpu)

    target = setup_target(target_volume, resolution, no_resampling, resampling_rate, no_trimming, trimming_cutoff)
    structure, template, mask, z_sigma = setup_template_structure(template_structure, chain, target, resolution, core_weighted)
    rotmat = setup_rotational_matrix(angle)

    pf = PowerFitter(target, rotmat, template, mask, queue, nproc, laplace=laplace)
    if gpu:
        logger.info("Using GPU-accelerated search.")
    else:
        logger.info("Requested number of processors: {:d}".format(nproc))

    logger.info("Starting search")

    time1 = time()
    pf.scan(progress=progress)
    dtime = time() - time1
    if dtime < 10:
        logger.info("Time for search: {:.3f} s".format(dtime))
    else:
        logger.info("Time for search: {:.0f}m {:.0f}s".format(*divmod(dtime, 60)))
    logger.info("Analyzing results")

    analyzer = Analyzer(
        pf.lcc,
        rotmat,
        pf.rot,
        voxelspacing=target.voxelspacing,
        origin=target.origin,
        z_sigma=z_sigma,
    )

    logger.info("Writing solutions to file.")
    Volume(pf.lcc, target.voxelspacing, target.origin).tofile(
        join(directory, "lcc.mrc")
    )
    analyzer.tofile(join(directory, "solutions.out"), delimiter=delimiter)

    logger.info("Writing PDBs to file.")
    n = min(num, len(analyzer.solutions))
    write_fits_to_pdb(
        structure, analyzer.solutions[:n], basename=join(directory, "fit")
    )

    logger.info("Total time: {:.0f}m {:.0f}s".format(*divmod(time() - time0, 60)))


def powerfit_many(
    target_volume: Path,
    resolution: float,
    template_structures: list[Path],
    angle: float = 10,
    laplace: bool = False,
    core_weighted: bool=False,
    no_resampling: bool=False,
    resampling_rate: float=2,
    no_trimming: bool=False,
    trimming_cutoff: float | None=None,
    gpu: str | None = None, 
    reuse: bool = True,
    nproc: int=1,
) -> list[list[list[float]]]:
    """Run powerfit on multiple templates, returning the solution table for each.

    For a slight efficiency boost, and to avoid continuously creating many new OpenCL
    queues, the queues are reused. This can be disabled by setting reuse=False
    Outer list is same order as template_structures. Middle list is ordered on cc score. Inner list is the data for a solution.  Each solution has the following columns: cc Fish-z rel-z x y z a11 a12 a13 a21 a22 a23 a31 a32 a33.
    """
    time0 = time()

    # Get GPU queue if requested
    queue = None
    if gpu:
        queue = get_gpu_queue(gpu)

    with target_volume.open("rb") as f:
        target = setup_target(
            f,
            resolution,
            no_resampling,
            resampling_rate,
            no_trimming,
            trimming_cutoff,
        )

    template_vars: list[tuple[Structure, Volume, Volume, float]] = []
    for template_structure in template_structures:
        with template_structure.open("r") as f:
            template_vars.append(setup_template_structure(
                f, None, target, resolution, core_weighted
            ))
    rotmat = setup_rotational_matrix(angle)

    if gpu:
        logger.info("Using GPU-accelerated search.")
    else:
        logger.info(f"Requested number of processors: {nproc}.")

    logger.info(f"Starting search, analysing {len(template_structures)} structures.")

    time1 = time()
    results: list[tuple[np.ndarray, np.ndarray]] = []
    pf: PowerFitter | None = None
    for i in range(len(template_vars)):
        _, template, mask, z_sigma = template_vars[i]
        if pf is None or not reuse:
            pf = PowerFitter(
                target, rotmat, template, mask, queue, nproc, laplace=laplace
            )
        elif not gpu and nproc > 1:  # Can't reuse w/ multi-cpu search
            pf = PowerFitter(
                target, rotmat, template, mask, queue, nproc, laplace=laplace
            )
        else:
            pf.set_template(template, mask)

        pf.scan(progress=None)
        results.append((pf.lcc, pf.rot))

    dtime = time() - time1
    if dtime < 10:
        logger.info("Time for searches: {:.3f} s".format(dtime))
    else:
        logger.info("Time for searches: {:.0f}m {:.0f}s".format(*divmod(dtime, 60)))

    logger.info("Analyzing results")

    analysis_results: list[Analyzer] = []
    for result, template_var, template in zip(
        results, template_vars, template_structures, strict=True
    ):
        lcc, rot = result
        _, _, _, z_sigma = template_var

        analysis = Analyzer(
                lcc,
                rotmat,
                rot,
                voxelspacing=target.voxelspacing,
                origin=target.origin,
                z_sigma=z_sigma,
        )
        analysis_results.append(analysis)

    logger.info("Total time: {:.0f}m {:.0f}s".format(*divmod(time() - time0, 60)))
    return [r.solutions for r in analysis_results]

if __name__ == "__main__":
    main()

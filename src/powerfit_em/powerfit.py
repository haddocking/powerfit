#! ../env/bin/python


from functools import partial
from os.path import splitext, join, abspath
from time import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, BooleanOptionalAction, FileType
import logging
from typing import BinaryIO, TextIO
import warnings

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
from powerfit_em.helpers import mkdir_p, write_fits_to_pdb, fisher_sigma
from powerfit_em.volume import extend, nearest_multiple2357, trim, resample

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
        type=FileType("r"),
        help="Atomic model to be fitted in the density. "
        "Format should either be PDB or mmCIF",
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
        "-l",
        "--laplace",
        dest="laplace",
        action="store_true",
        help="Use the Laplace pre-filter density data. "
        "Can be combined "
        "with the core-weighted local cross-correlation.",
    )
    p.add_argument(
        "-cw",
        "--core-weighted",
        dest="core_weighted",
        action="store_true",
        help="Use core-weighted local cross-correlation score. "
        "Can be combined with the Laplace pre-filter.",
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
        help="Show a progress bar during the search.",
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
    
    mkdir_p(args.directory)
    configure_logging(join(args.directory, "powerfit.log"), args.log_level)
    
    progress = partial(rich_tqdm, desc="Processing rotations", unit="rot", disable=not args.progressbar)

    powerfit(
        target_volume=args.target,
        resolution=args.resolution,
        template_structure=args.template,
        angle=args.angle,
        laplace=args.laplace,
        core_weighted=args.core_weighted,
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


def powerfit(target_volume: BinaryIO,
             resolution: float,
             template_structure: TextIO,
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
             gpu: str | None =None, 
             nproc: int=1,
             delimiter: str = None,
             progress: partial[tqdm] = tqdm
             ):
    time0 = time()
    mkdir_p(directory)

    # Get GPU queue if requested
    queues = None
    if gpu:
        import pyopencl as cl
        # TODO allow to omit platform, so gpu='4' runs 5th device on first platform
        if isinstance(gpu, str) and ':' in gpu:
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
        queues = [cl.CommandQueue(context, device=devices[device_idx])]

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

    # Read in the rotations to sample
    logger.info("Reading in rotations.")
    q, w, degree = proportional_orientations(angle)
    rotmat = quat_to_rotmat(q)
    logger.info("Requested rotational sampling density: {:.2f}".format(angle))
    logger.info("Real rotational sampling density: {:}".format(degree))

    # Apply core-weighted mask if requested
    if core_weighted:
        logger.info("Calculating core-weighted mask.")
        mask.array = determine_core_indices(mask.array)

    pf = PowerFitter(target, laplace=laplace)
    pf._rotations = rotmat
    pf._template = template
    pf._mask = mask
    pf._nproc = nproc
    pf.directory = directory
    pf._queues = queues
    if gpu:
        logger.info("Using GPU-accelerated search.")
    else:
        logger.info("Requested number of processors: {:d}".format(nproc))
    logger.info("Starting search")
    time1 = time()
    pf.scan(progress=progress)
    logger.info("Time for search: {:.0f}m {:.0f}s".format(*divmod(time() - time1, 60)))

    logger.info("Analyzing results")
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
    analyzer = Analyzer(
        pf._lcc,
        rotmat,
        pf._rot,
        voxelspacing=target.voxelspacing,
        origin=target.origin,
        z_sigma=z_sigma,
    )

    logger.info("Writing solutions to file.")
    Volume(pf._lcc, target.voxelspacing, target.origin).tofile(
        join(directory, "lcc.mrc")
    )
    analyzer.tofile(join(directory, "solutions.out"), delimiter=delimiter)

    logger.info("Writing PDBs to file.")
    n = min(num, len(analyzer.solutions))
    write_fits_to_pdb(
        structure, analyzer.solutions[:n], basename=join(directory, "fit")
    )

    logger.info("Total time: {:.0f}m {:.0f}s".format(*divmod(time() - time0, 60)))


if __name__ == "__main__":
    main()

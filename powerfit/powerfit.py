#! ../env/bin/python
from __future__ import absolute_import, division

from __future__ import print_function
from os.path import splitext
from pathlib import Path
from sys import stdout, argv
from time import time
from argparse import ArgumentParser
import logging

from powerfit import (
    Volume,
    structure_to_shape_like,
    proportional_orientations,
    quat_to_rotmat,
    determine_core_indices,
)
from powerfit.powerfitter import PowerFitter
from powerfit.analyzer import Analyzer
from powerfit.helpers import mkdir_p, write_fits_to_pdb, fisher_sigma
from powerfit.volume import (
    ChainRestrictions,
    extend,
    nearest_multiple2357,
    trim,
    resample,
    xyz_fixed_transform

)
from powerfit.structure import Structure


def parse_args():
    """Parse command-line options."""

    p = ArgumentParser()

    # Positional arguments
    p.add_argument(
        "target",
        type=Path,
        help="Target density map to fit the model in. "
        "Data should either be in CCP4 or MRC format",
    )
    p.add_argument("resolution", type=float, help="Resolution of map in angstrom")
    p.add_argument(
        "template",
        type=Path,
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

    p.add_argument(
        "-bf",
        "--b-factor_weighted",
        dest="bfac",
        action="store_true",
        help="Uses b-factor information when creating the simulated map file  "
        "False by default",
    )

    p.add_argument(
        "-f",
        "--xyz_fixed",
        dest="xyz_fixed",
        type=Path,
        help="Runs Powerfit with a fixed model. "
        "Format should either be PDB or mmCIF",
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
    # TODO: reimplement this when I fix selection
    """p.add_argument('-c', '--chain', dest='chain', type=str, default=None,
            metavar='<char>',
            help=('The chain IDs of the structure to be fitted. '
                  'Multiple chains can be selected using a comma separated list, i.e. -c A,B,C. '
                  'Default is the whole structure.'),
                 )"""
    # Output parameters
    p.add_argument(
        "-d",
        "--directory",
        dest="directory",
        type=Path,
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
        action="store_true",
        help="Off-load the intensive calculations to the GPU. ",
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


def write(line):
    """Write line to stdout and logfile."""
    if stdout.isatty():
        print(line)
    logging.info(line)


def main(
    target: Path or Volume,
    resolution: float,
    structure: Path or Structure,
    directory=Path("."),
    nproc: int = 1,
    num: int = 10,
    xyz_fixed: Path or Structure = None,
    gpu: bool = False,
    no_resampling: bool = False,
    no_trimming: bool = False,
    bfac: bool = False,
    core_weighted: bool = False,
    laplace: bool = False,
    resampling_rate: float = 2.0,
    angle: float = 10.0,
    trimming_cutoff=None,
    return_instances: bool = False,
    return_files=True,
):

    time0 = time()
    mkdir_p(directory)
    # Configure logging file
    logging.basicConfig(
        filename=str(directory.joinpath("powerfit.log")),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    logging.info(" ".join(argv))

    # Get GPU queue if requested
    queues = None
    if gpu:
        import pyopencl as cl

        p = cl.get_platforms()[0]
        devs = p.get_devices()
        context = cl.Context(devices=devs)
        # For clFFT each queue should have its own Context
        queues = [cl.CommandQueue(context, device=dev) for dev in devs]

    if isinstance(target, Path):
        target = Volume.fromfile(str(target))

    # Need to cacluate threshold
    target.resolution = resolution
    target.calc_threshold()

    write("Target file read from: {:s}".format(target.filename))

    write("Target resolution: {:.2f}".format(resolution))

    write(("Initial shape of density:" + " {:d}" * 3).format(*target.shape))
    # Resample target density if requested
    if not no_resampling:
        factor = 2 * resampling_rate * target.voxelspacing / resolution
        if factor < 0.9:
            target = resample(target, factor)
            write(("Shape after resampling:" + " {:d}" * 3).format(*target.shape))
    # Trim target density if requested
    if not no_trimming:
        if trimming_cutoff is None:
            trimming_cutoff = target.grid.max() / 10
        target = trim(target, trimming_cutoff)
        write(("Shape after trimming:" + " {:d}" * 3).format(*target.shape))
    # Extend the density to a multiple of 2, 3, 5, and 7 for clFFT
    extended_shape = [nearest_multiple2357(n) for n in target.shape]
    target = extend(target, extended_shape)
    write(("Shape after extending:" + " {:d}" * 3).format(*target.shape))

    # Read in structure or high-resolution map

    if isinstance(structure, Path):
        structure = Structure.fromfile(str(structure.resolve()))

    write("Template file read from: {:s}".format(structure.filename))
    if xyz_fixed:
        if isinstance(xyz_fixed, Path):
            xyz_fixed_structure = Structure.fromfile(str(xyz_fixed.resolve()))
        write("Fixed model file read from: {:s}".format(xyz_fixed_structure.filename))

    # TODO: add this back in to at some point
    # if args.chain is not None:
    #     write('Selecting chains: ' + args.chain)
    #     structure = structure.select('chain', args.chain.split(','))

    # TODO: Figure out what size means, maybe the size of the dict
    # if structure.data.size == 0:
    #     raise ValueError("No atoms were selected.")

    # Move structure to origin of density
    # structure.translate(target.origin - structure.centre_of_mass)
    if bfac: 
        weights = 0.4/structure.bfacs
    else:
        weights = structure.atomnumber


    structure.translate(target.origin - structure.coor.mean(axis=1))

   
    template = structure_to_shape_like(
        target, 
        structure.coor, 
        resolution=resolution, 
        weights=weights,
        shape='vol'
    )


    mask = structure_to_shape_like(
          target, 
          structure.coor, 
          resolution=resolution, 
          shape='mask'
          )

    # template.resolution = resolution
    # template.calc_threshold(simulated=True)

    if xyz_fixed:
        cr = ChainRestrictions(
            volin = template,
            pdbin = structure,
            xyzfixed = xyz_fixed_structure,
        )

        template = cr.run()
        #     target, 
        #     xyz_fixed_structure.coor, 
        #     resolution=resolution, 
        #     weights=weights,
        #     shape='vol'
        # )

        # fixed_vol_mask = structure_to_shape_like(
        #     target, 
        #     xyz_fixed_structure.coor, 
        #     resolution=resolution, 
        #     shape='mask'
        # )


        # template = xyz_fixed_transform(
        #     target, 
        #     fixed_vol,
        #     fixed_vol_mask,
        #     )

        template.tofile(str(directory.joinpath("fixed_vol.mrc"))) # temp
        # template.calc_threshold()

    # Read in the rotations to sample
    write("Reading in rotations.")
    q, w, degree = proportional_orientations(angle)
    rotmat = quat_to_rotmat(q)
    write("Requested rotational sampling density: {:.2f}".format(angle))
    write("Real rotational sampling density: {:}".format(degree))

    # Apply core-weighted mask if requested
    if core_weighted:
        write("Calculating core-weighted mask.")
        mask.grid = determine_core_indices(mask.grid)

    pf = PowerFitter(target, laplace=laplace)
    pf._rotations = rotmat
    pf._template = template
    pf._mask = mask
    pf._nproc = nproc
    pf.directory = directory
    pf._queues = queues
    if gpu:
        write("Using GPU-accelerated search.")
    else:
        write("Requested number of processors: {:d}".format(nproc))
    write("Starting search")
    time1 = time()
    pf.scan()
    write("Time for search: {:.0f}m {:.0f}s".format(*divmod(time() - time1, 60)))

    write("Analyzing results")
    # calculate the molecular volume of the structure

    mv = structure_to_shape_like(
          target, structure.coor, resolution=resolution, radii=structure.rvdw, shape='mask'
          ).grid.sum() * target.voxelspacing ** 3

    z_sigma = fisher_sigma(mv, resolution)
    analyzer = Analyzer(
        pf._lcc,
        rotmat,
        pf._rot,
        voxelspacing=target.voxelspacing,
        origin=target.origin,
        z_sigma=z_sigma,
    )

    lccvol = Volume.fromdata(pf._lcc, target.voxelspacing, target.origin)
    lccvol.filename = "lcc.mrc"

    if return_files:
        write("Writing solutions to file.")
        lccvol.tofile(str(directory.joinpath("lcc.mrc")))
        analyzer.tofile(str(directory.joinpath("solutions.out")))
        write("Writing PDBs to file.")

    n = min(num, len(analyzer.solutions))
    if xyz_fixed:
        fixed = xyz_fixed_structure
    else:
        fixed = False

    out_fits = write_fits_to_pdb(
        structure,
        analyzer.solutions,
        n,
        basename=str(directory.joinpath("fit")),
        xyz_fixed=fixed,
        return_instances=return_instances,
        return_files=return_files,
    )

    if return_instances:
        to_return = {
            "fitted_models": out_fits,
            "lcc": lccvol,
            "analyzer": analyzer,
        }
    elif return_files:
        to_return = {
            "fitted_models": [out.filename for out in out_fits],
            "lcc": lccvol.filename,
            "analyzer": "solutions.out",
        }

    # Not needed
    else:
        to_return = None

    write("Total time: {:.0f}m {:.0f}s".format(*divmod(time() - time0, 60)))

    return to_return


def run():
    args = parse_args()

    main(
        args.target,
        args.resolution,
        args.template,
        directory=args.directory,
        nproc=args.nproc,
        num=args.num,
        xyz_fixed=args.xyz_fixed,
        gpu=args.gpu,
        no_resampling=args.no_resampling,
        no_trimming=args.no_trimming,
        bfac=args.bfac,
        core_weighted=args.core_weighted,
        laplace=args.laplace,
        resampling_rate=args.resampling_rate,
        angle=args.angle,
        trimming_cutoff=args.trimming_cutoff,
    )


if __name__ == "__main__":
    run()

#! ../env/bin/python


from os.path import splitext, join, abspath
from os import makedirs
from sys import stdout, argv
from time import time
from argparse import ArgumentParser, FileType
import logging

from powerfit import (
      Volume, Structure, structure_to_shape_like, proportional_orientations,
      quat_to_rotmat, determine_core_indices
      )
from powerfit.powerfitter import PowerFitter
from powerfit.analyzer import Analyzer
from powerfit.helpers import mkdir_p, write_fits_to_pdb, fisher_sigma
from powerfit.volume import extend, nearest_multiple2357, trim, resample


def parse_args():
    """Parse command-line options."""

    p = ArgumentParser()

    # Positional arguments
    p.add_argument('target', type=FileType('rb'),
            help='Target density map to fit the model in. '
                 'Data should either be in CCP4 or MRC format')
    p.add_argument('resolution', type=float,
            help='Resolution of map in angstrom')
    p.add_argument('template', type=file,
            help='Atomic model to be fitted in the density. '
                 'Format should either be PDB or mmCIF')

    # Optional arguments and flags
    p.add_argument('-a', '--angle', dest='angle', type=float, default=10,
            metavar='<float>',
            help='Rotational sampling density in degree. Increasing '
                 'this number by a factor of 2 results in approximately '
                 '8 times more rotations sampled.')
    # Scoring flags
    p.add_argument('-l', '--laplace', dest='laplace', action='store_true',
            help='Use the Laplace pre-filter density data. '
                 'Can be combined '
                 'with the core-weighted local cross-correlation.')
    p.add_argument('-cw', '--core-weighted', dest='core_weighted', action='store_true',
            help='Use core-weighted local cross-correlation score. '
                 'Can be combined with the Laplace pre-filter.')
    # Resampling
    p.add_argument('-nr', '--no-resampling', dest='no_resampling', action='store_true',
            help='Do not resample the density map.')
    p.add_argument('-rr', '--resampling-rate', dest='resampling_rate',
            type=float, default=2, metavar='<float>',
            help='Resampling rate compared to Nyquist.')
    # Trimming related
    p.add_argument('-nt', '--no-trimming', dest='no_trimming', action='store_true',
            help='Do not trim the density map.')
    p.add_argument('-tc', '--trimming-cutoff', dest='trimming_cutoff',
            type=float, default=None, metavar='<float>',
            help='Intensity cutoff to which the map will be trimmed. '
                 'Default is 10 percent of maximum intensity.')
    # Selection parameter
    p.add_argument('-c', '--chain', dest='chain', type=str, default=None,
            metavar='<char>',
            help=('The chain IDs of the structure to be fitted. '
                  'Multiple chains can be selected using a comma separated list, i.e. -c A,B,C. '
                  'Default is the whole structure.'),
                 )
    # Output parameters
    p.add_argument('-d', '--directory', dest='directory', type=abspath, default='.',
            metavar='<dir>',
            help='Directory where the results are stored.')
    p.add_argument('-n', '--num', dest='num', type=int, default=10,
            metavar='<int>',
            help='Number of models written to file. This number '
                 'will be capped if less solutions are found as requested.')
    # Computational resources parameters
    p.add_argument('-g', '--gpu', dest='gpu', action='store_true',
            help='Off-load the intensive calculations to the GPU. ')
    p.add_argument('-p', '--nproc', dest='nproc', type=int, default=1,
            metavar='<int>',
            help='Number of processors used during search. '
                 'The number will be capped at the total number '
                 'of available processors on your machine.')

    args = p.parse_args()

    return args


def get_filetype_template(fname):
    """Determine the file type from the file extension."""
    ext = splitext(fname)[1][1:]
    if ext in ['pdb', 'ent']:
        ft = 'pdb'
    elif ext in ['map', 'ccp4']:
        ft = 'map'
    else:
        msg = 'Filetype of file {:} is not recognized.'.format(fname)
        raise IOError(msg)
    return ft


def write(line):
    """Write line to stdout and logfile."""
    if stdout.isatty():
        print(line)
    logging.info(line)


def main():

    time0 = time()
    args = parse_args()
    mkdir_p(args.directory)
    # Configure logging file
    logging.basicConfig(filename=join(args.directory, 'powerfit.log'), 
            level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(' '.join(argv))

    # Get GPU queue if requested
    queues = None
    if args.gpu:
        import pyopencl as cl
        p = cl.get_platforms()[0]
        devs = p.get_devices()
        context = cl.Context(devices=devs)
        # For clFFT each queue should have its own Context
        queues = [cl.CommandQueue(context, device=dev) for dev in devs]

    write('Target file read from: {:s}'.format(abspath(args.target.name)))
    target = Volume.fromfile(args.target)
    write('Target resolution: {:.2f}'.format(args.resolution))
    resolution = args.resolution
    write(('Initial shape of density:' + ' {:d}'*3).format(*target.shape))
    # Resample target density if requested
    if not args.no_resampling:
        factor = 2 * args.resampling_rate * target.voxelspacing / resolution
        if factor < .9:
            target = resample(target, factor)
            write(('Shape after resampling:' + ' {:d}'*3).format(*target.shape))
    # Trim target density if requested
    if not args.no_trimming:
        if args.trimming_cutoff is None:
            args.trimming_cutoff = target.array.max() / 10
        target = trim(target, args.trimming_cutoff)
        write(('Shape after trimming:' + ' {:d}'*3).format(*target.shape))
    # Extend the density to a multiple of 2, 3, 5, and 7 for clFFT
    extended_shape = [nearest_multiple2357(n) for n in target.shape]
    target = extend(target, extended_shape)
    write(('Shape after extending:' + ' {:d}'*3).format(*target.shape))

    # Read in structure or high-resolution map
    write('Template file read from: {:s}'.format(abspath(args.template.name)))
    structure = Structure.fromfile(args.template)
    if args.chain is not None:
        write('Selecting chains: ' + args.chain)
        structure = structure.select('chain', args.chain.split(','))
    if structure.data.size == 0:
        raise ValueError("No atoms were selected.")

    # Move structure to origin of density
    structure.translate(target.origin - structure.coor.mean(axis=1))
    template = structure_to_shape_like(
          target, structure.coor, resolution=resolution,
          weights=structure.atomnumber, shape='vol'
          )
    mask = structure_to_shape_like(
          target, structure.coor, resolution=resolution, shape='mask'
          )

    # Read in the rotations to sample
    write('Reading in rotations.')
    q, w, degree = proportional_orientations(args.angle)
    rotmat = quat_to_rotmat(q)
    write('Requested rotational sampling density: {:.2f}'.format(args.angle))
    write('Real rotational sampling density: {:}'.format(degree))

    # Apply core-weighted mask if requested
    if args.core_weighted:
        write('Calculating core-weighted mask.')
        mask.array = determine_core_indices(mask.array)

    pf = PowerFitter(target, laplace=args.laplace)
    pf._rotations = rotmat
    pf._template = template
    pf._mask = mask
    pf._nproc = args.nproc
    pf.directory = args.directory
    pf._queues = queues
    if args.gpu:
        write('Using GPU-accelerated search.')
    else:
        write('Requested number of processors: {:d}'.format(args.nproc))
    write('Starting search')
    time1 = time()
    pf.scan()
    write('Time for search: {:.0f}m {:.0f}s'.format(*divmod(time() - time1, 60)))

    write('Analyzing results')
    # calculate the molecular volume of the structure
    mv = structure_to_shape_like(
          target, structure.coor, resolution=resolution, radii=structure.rvdw, shape='mask'
          ).array.sum() * target.voxelspacing ** 3
    z_sigma = fisher_sigma(mv, resolution)
    analyzer = Analyzer(
            pf._lcc, rotmat, pf._rot, voxelspacing=target.voxelspacing,
            origin=target.origin, z_sigma=z_sigma
            )

    write('Writing solutions to file.')
    Volume(pf._lcc, target.voxelspacing, target.origin).tofile(join(args.directory, 'lcc.mrc'))
    analyzer.tofile(join(args.directory, 'solutions.out'))

    write('Writing PDBs to file.')
    n = min(args.num, len(analyzer.solutions))
    write_fits_to_pdb(structure, analyzer.solutions[:n],
            basename=join(args.directory, 'fit'))

    write('Total time: {:.0f}m {:.0f}s'.format(*divmod(time() - time0, 60)))


if __name__ == '__main__':
    main()

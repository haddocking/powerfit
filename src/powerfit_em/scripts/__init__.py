
from argparse import ArgumentParser, FileType
from os.path import splitext

from powerfit_em.volume import Volume, trim, resample, lower_resolution

EM2EM_DESCRIPTION = "Convert, trim, and resample cryo-EM data."


def parse_em2em():
    p = ArgumentParser(description=EM2EM_DESCRIPTION)
    p.add_argument('infile', type=str,
            help="Input cryo-EM density file.")
    p.add_argument("outfile", type=str,
            help="Output cryo-EM density file.")

    p.add_argument('-f1', '--format-input', default=None,
            dest='input_format', metavar="<str>",
            help="Format of input file.")
    p.add_argument('-f2', '--format-output', default=None,
            dest='output_format', metavar="<str>",
            help="Format of output file.")
    p.add_argument("-t", "--trim", action='store_true', dest="trim",
            help="Trim the density.")
    p.add_argument('-tc', "--trim-cutoff", type=float, default=None,
            dest="trim_cutoff",
            help="Cutoff value for trimming.")
    p.add_argument("-r", "--resample", default=None, type=float,
            dest="resample", metavar='<float>',
            help="Resample the density to a specified voxel spacing in angstrom.")

    args = p.parse_args()
    return args


def em2em():
    args = parse_em2em()

    print('Reading input file ...')
    v = Volume.fromfile(args.infile, args.input_format)

    if args.resample is not None:
        print('Resampling ...')
        v = resample(v, v.voxelspacing / args.resample)

    if args.trim:
        print('Trimming ...')
        if args.trim_cutoff is None:
            args.trim_cutoff = 0.1 * v.array.max()
        v = trim(v, args.trim_cutoff)

    print('Writing to file ...')
    v.tofile(args.outfile, args.output_format)


def parse_image_pyramid():
    p = ArgumentParser()

    p.add_argument('map', type=FileType('rb'), 
            help='Initial density data.')
    p.add_argument('resolution', type=float, 
            help='Resolution of initial data.')
    p.add_argument('target_resolutions', nargs='+', type=float,
            help='The target resolutions of the resulting image-pyramid.')

    p.add_argument('-rr', '--resampling-rate', dest='resampling_rate',
            type=float, default=2, metavar='<float>',
            help=('Nyquist resampling rate. Default is 2 x '
            'Nyquist, i.e. resulting voxelspacing is '
            '1/4th of the resolution.'),
            )
    p.add_argument('-b', '--base-name', dest='base_name', type=str, default=None,
            metavar='<string>',
            help='Base name of the resulting maps. Default is original mapfile name.')
    args = p.parse_args()

    # some error checking
    if args.resolution <= 0:
        raise ValueError('Resolution should be bigger than 0.')
    if args.resampling_rate < 1:
        raise ValueError('Resampling rate should be bigger than 1 times Nyquist.')
    for resolution in args.target_resolutions:
        if resolution < args.resolution:
            raise ValueError('Target resolution of image-pyramid should be '
                'lower than original data.')
    if args.base_name is None:
        args.base_name = splitext(args.map.name)[0] 

    return args


def image_pyramid():
    args = parse_image_pyramid()

    # create image-pyramid
    vol = Volume.fromfile(args.map)
    fname = args.base_name + '_{:.0f}.mrc'
    for resolution in args.target_resolutions:
        vol2 = lower_resolution(vol, args.resolution, resolution)
        new_voxelspacing = resolution / (2 * args.resampling_rate)
        factor = vol.voxelspacing / new_voxelspacing 
        vol2 = resample(vol2, factor, order=1)
        vol2.tofile(fname.format(resolution))


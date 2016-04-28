#! /usr/bin/python
# -*- coding: utf-8

from __future__ import absolute_import, division
from os.path import splitext
from argparse import ArgumentParser
from powerfit.volume import Volume, lower_resolution, resample

def parse_args():
    p = ArgumentParser()

    p.add_argument('map', type=file, help='Initial density data.')
    p.add_argument('resolution', type=float, help='Resolution of initial data.')
    p.add_argument('-rr', '--resampling-rate', dest='resampling_rate',
        type=float, default=2, help='Nyquist resampling rate. Default is 2 x '
        'Nyquist, i.e. resulting voxelspacing is '
        '1/4th of the resolution.')
    p.add_argument('-b', '--base-name', dest='base_name', type=str, default=None,
            help='Base name of the resulting maps. Default is original mapfile name.')
    p.add_argument('target_resolutions', nargs='+', type=float, 
            help='The target resolutions of the resulting image-pyramid.')
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


def main():
    args = parse_args()

    # create image-pyramid
    vol = Volume.fromfile(args.map)
    fname = args.base_name + '_{:.0f}.mrc'
    for resolution in args.target_resolutions:
        vol2 = lower_resolution(vol, args.resolution, resolution)
        new_voxelspacing = resolution / (2 * args.resampling_rate)
        factor = vol.voxelspacing / new_voxelspacing 
        vol2 = resample(vol, factor, order=1)
        vol2.tofile(fname.format(resolution))


if __name__ == '__main__':
    main()

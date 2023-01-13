from __future__ import absolute_import, division
import os
import errno
from math import sqrt, pi

import numpy as np
from scipy.ndimage import binary_erosion
try:
    import pyopencl as cl
except ImportError:
    pass

from . import volume


def mkdir_p(path):
    if not path.exists():
        path.mkdir()
    else:
        pass


def determine_core_indices(mask):
    """Calculate the core indices of a shape"""

    core_indices = np.zeros(mask.shape)
    eroded_mask = mask > 0
    while eroded_mask.sum() > 0:
        core_indices += eroded_mask
        eroded_mask = binary_erosion(eroded_mask)
    return core_indices


def get_queue(platformid=0, deviceid=0):
    try:
        platform = cl.get_platforms()[platformid]
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        context = cl.Context(devices=devices)
        queue = cl.CommandQueue(context, device=devices[deviceid])
    except:
        queue = None

    return queue


def fisher_sigma(mv, fsc):
    return 1 / sqrt(mv / fsc - 3)


def write_fits_to_pdb(
    structure,
    solutions, 
    basename='fit', 
    xyz_fixed = False,
    return_files = True,
    return_instances = False
    ):

    if not return_files and not return_instances:
        raise AssertionError('Either return_files or return_instances must be specified.')

    translated_structure = structure.duplicate()
    center = translated_structure.coor.mean(axis=1)
    translated_structure.translate(-center)

    # output list a list of structure instances
    if return_instances:    output_list = []
    for n, sol in enumerate(solutions, start=1):
        out = translated_structure.duplicate()
        rot = np.asarray([float(x) for x in sol[6:]]).reshape(3, 3)
        trans = sol[3:6]
        out.rotate(rot)
        out.translate(trans)
        if xyz_fixed:        out.combine(xyz_fixed)
        if return_files:     out.tofile(basename + '_{:d}.pdb'.format(n))
        if return_instances: output_list.append(out)

    if return_instances:     return output_list



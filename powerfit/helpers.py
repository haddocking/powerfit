
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
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


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
        devices = platform.get_devices()
        context = cl.Context(devices=devices)
        queue = cl.CommandQueue(context, device=devices[deviceid])
    except Exception as e:
        raise e
        queue = None

    return queue


def fisher_sigma(mv, fsc):
    return 1 / sqrt(mv / fsc - 3)


def write_fits_to_pdb(structure, solutions, basename='fit'):
    translated_structure = structure.duplicate()
    center = translated_structure.coor.mean(axis=1)
    translated_structure.translate(-center)
    for n, sol in enumerate(solutions, start=1):
        out = translated_structure.duplicate()
        rot = np.asarray([float(x) for x in sol[6:]]).reshape(3, 3)
        trans = sol[3:6]
        out.rotate(rot)
        out.translate(trans)
        out.tofile(basename + '_{:d}.pdb'.format(n))

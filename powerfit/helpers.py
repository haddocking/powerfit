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


# Change this up so that an
def write_fits_to_pdb(
    structure,
    solutions,
    limit,
    basename="fit",
    xyz_fixed=False,
    return_files=True,
    return_instances=False,
):

    if not return_files and not return_instances:
        raise AssertionError(
            "Either return_files or return_instances must be specified."
        )

    translated_structure = structure.duplicate()
    center = translated_structure.coor.mean(axis=1)
    translated_structure.translate(-center)

    # output list a list of structure instances
    output_list = []
    n = 0
    for sol in solutions:
        if n == limit:
            break
        out = translated_structure.duplicate()
        rot = np.asarray([float(x) for x in sol[6:]]).reshape(3, 3)
        trans = sol[3:6]
        out.rotate(rot)
        out.translate(trans)
        out.filename = basename + "_{:d}.pdb".format(n + 1)
        if xyz_fixed:
            out.combine(xyz_fixed)
            # if not quick_structure_overlap(out, xyz_fixed):
            #     out.combine(xyz_fixed)
            # else:
            #     continue

        if return_files:
            out.tofile()

        n += 1

        output_list.append(out)

    return output_list


def quick_structure_overlap(structure1, structure2):

    (p1min_x, p1min_y, p1min_z), (p1max_x, p1max_y, p1max_z) = (
        np.min(structure1.coor, axis=1).astype(int),
        np.max(structure1.coor, axis=1).astype(int),
    )
    points = np.stack([structure2.coor[0], structure2.coor[1], structure2.coor[2]], 1)

    new_idx = (
        (points[:, 0] <= p1max_x)
        & (points[:, 0] >= p1min_x)
        & (points[:, 1] <= p1max_y)
        & (points[:, 1] >= p1min_y)
        & (points[:, 2] <= p1max_z)
        & (points[:, 2] >= p1min_z)
    )

    points = points[new_idx]

    if len(points) <= 10:  # picked arbitrarily
        return False
    else:
        return True

from __future__ import absolute_import, division

from math import sqrt
from scipy.spatial import cKDTree


from enum import Enum
import logging
import os
import sys


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

            if quick_structure_overlap(out, xyz_fixed):
                out.combine(xyz_fixed)
            else:
                continue

        if return_files:
            out.tofile()

        n += 1

        output_list.append(out)

    return output_list


def quick_structure_overlap(structure1, structure2):

    # Get the coordinates of the two structures
    coords1 = np.asarray([structure1.coor[0], structure1.coor[1], structure1.coor[2]]).T
    coords2 = np.asarray([structure2.coor[0], structure2.coor[1], structure2.coor[2]]).T

    # Define a minimum number of points for intersection
    min_overlap = min(len(coords1), len(coords2)) * 0.05

    # Build a KDTree for the second structure
    tree = cKDTree(coords2)

    # Define a distance threshold for point cloud intersection
    distance_threshold = 1

    # Check for nearby points between the two models
    num_intersecting_points = np.sum(tree.query(coords1, k=1)[0] < distance_threshold)

    # Check if the models intersecth
    if num_intersecting_points < min_overlap:
        return True
    else:
        return False

class LogColors(Enum):
    """Color container for log messages"""
    CRITICAL = 31
    DEBUG = 34
    DEFAULT = 0
    ERROR = 31
    WARNING = 33


class LogColorFormatter(logging.Formatter):
    """Formatter for log messages"""

    def format(self, record):
        if record.levelname in LogColors.__members__:
            prefix = '\033[1;{}m'.format(LogColors[record.levelname].value)
            postfix = '\033[{}m'.format(LogColors["DEFAULT"].value)
            record.msg = os.linesep.join([prefix + msg + postfix for msg in str(record.msg).splitlines()])
        return logging.Formatter.format(self, record)

def setup_logging(level="INFO", logfile=None, debugfile=None):
    # Author Adam Simpkin, Taken from SliceNDice code:
    # REF

    """Set up logging to the console for the root logger.

    :param level: str, optional
        The console logging level to be used [default: info]
        To change, use one of
            [ notset | info | debug | warning | error | critical ]
    :param logfile: str, optional
        The path to a log file containing INFO level output
    :param debugfile: str, optional
        The path to a log file containing all output
    :return: :obj:`~logging.Logger`
        Instance of a :obj:`~logging.Logger`
    """

    # Reset any Handlers or Filters already in the logger to start from scratch
    # https://stackoverflow.com/a/16966965
    map(logging.getLogger().removeHandler, logging.getLogger().handlers[:])
    map(logging.getLogger().removeFilter, logging.getLogger().filters[:])
    logging.getLogger().handlers = []
    logging_levels = {
        "notset": logging.NOTSET,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    # Create logger and default settings
    logging.getLogger().setLevel(level)

    # create console handler with a higher log level
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging_levels.get(level, logging.INFO))
    ch.setFormatter(LogColorFormatter("%(message)s"))
    logging.getLogger().addHandler(ch)

    # create file handler which logs only info messages
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(fh)
    # create file handler which logs even debug messages
    if debugfile:
        fh = logging.FileHandler(debugfile)
        fh.setLevel(logging.NOTSET)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(fh)

    logging.getLogger().debug("Console logger level: %s", logging_levels.get(level, logging.INFO))
    logging.getLogger().debug("File logger level: %s", logging.INFO)
    logging.getLogger().debug("File debug logger level: %s", logging.NOTSET)

    return logging.getLogger()
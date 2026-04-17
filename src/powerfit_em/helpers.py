import logging
import sys
from functools import cache
from importlib.util import find_spec
from math import sqrt

import numpy as np
from scipy.ndimage import binary_erosion

logger = logging.getLogger(__name__)


def pyfftw_available() -> bool:
    return find_spec("pyfftw") is not None


def opencl_available() -> bool:
    return find_spec("pyopencl") is not None


@cache
def cuda_available() -> bool:
    """Check if CUDA support is available.

    By checking for the presence of the `cupy` and `pyvkfft.cuda` packages.
    If `cupy` is found but `pyvkfft.cuda` is not, log a warning and return False.
    """
    try:
        return find_spec("cupy") is not None and find_spec("pyvkfft.cuda") is not None
    except ValueError:
        return "cupy" in sys.modules
    except OSError:
        logger.warning("pyvkfft CUDA backend is not available. CUDA support will be disabled.")
        return False


def determine_core_indices(mask):
    """Calculate the core indices of a shape"""

    core_indices = np.zeros(mask.shape)
    eroded_mask = mask > 0
    while eroded_mask.sum() > 0:
        core_indices += eroded_mask
        eroded_mask = binary_erosion(eroded_mask)
    return core_indices


def fisher_sigma(mv, fsc):
    return 1 / sqrt(mv / fsc - 3)


def write_fits_to_pdb(structure, solutions, basename="fit"):
    translated_structure = structure.duplicate()
    center = translated_structure.coor.mean(axis=1)
    translated_structure.translate(-center)
    for n, sol in enumerate(solutions, start=1):
        out = translated_structure.duplicate()
        rot = np.asarray([float(x) for x in sol[6:]]).reshape(3, 3)
        trans = sol[3:6]
        out.rotate(rot)
        out.translate(trans)
        out.tofile(basename + f"_{n:d}.pdb")

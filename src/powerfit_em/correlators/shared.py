"""Shared functionality between GPU and CPU correlators."""

import numpy as np


def get_lcc_mask(target: np.ndarray) -> np.ndarray:
    """Compute the local cross correlation (LCC) mask.
    
    Note that the mask is equal to all target voxels where the values
    exceed 5% of the maximum voxel value. Only these voxels are used for
    computing the LCC in the `calc_lcc_and_take_best` kernel function.
    """
    return (target > target.max() * 0.05)

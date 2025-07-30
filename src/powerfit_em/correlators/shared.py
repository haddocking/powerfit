"""Shared functionality between GPU and CPU correlators."""

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from pyopencl.array import Array as ClArray
    from pyopencl import Image


type Array = "ClArray" | np.ndarray

f32 = np.float32
i32 = np.int32

@dataclass
class Vars[T: Array, I: "Image" | np.ndarray]:
    """Non-complex GPU arrays."""
    target: T
    template: I
    mask: I
    lcc_mask: T
    target2: T
    rot_template: T
    rot_mask: T
    rot_mask2: T
    gcc: T
    ave: T
    ave2: T
    lcc: T
    rot: T


@dataclass
class VarsFT[T: Array]:
    """Fourier transformed (complex) arrays."""
    target: T
    target2: T
    template: T
    mask: T
    mask2: T
    ave: T
    ave2: T
    lcc: T
    gcc: T


def get_lcc_mask(target: np.ndarray) -> np.ndarray:
    """Compute the local cross correlation (LCC) mask.
    
    Note that the mask is equal to all target voxels where the values
    exceed 5% of the maximum voxel value. Only these voxels are used for
    computing the LCC in the `calc_lcc_and_take_best` kernel function.
    """
    return (target > target.max() * 0.05)


def normalize_template(template: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalize the template structure cf. A.M. Roseman (2000)."""
    norm_template = template * mask
    # normalize template;
    ind = mask != 0
    norm_template[ind] -= norm_template[ind].mean()
    norm_template[ind] /= norm_template[ind].std()
    # multiply again for core-weighted correlation score
    return norm_template * mask


def get_ft_shape(target: np.ndarray) -> tuple:
    """Returns shape of fourier transformed target."""
    return target.shape[:-1] + (target.shape[-1] // 2 + 1,)

def get_normalization_factor(mask: np.ndarray) -> np.float32:
    """Precompute the normalization factor for use in the LCC computing kernel"""
    norm_factor = np.not_equal(mask, 0).sum(dtype=np.float32)
    if norm_factor == 0:
        raise ValueError('Zero-filled mask is not allowed.')
    return norm_factor

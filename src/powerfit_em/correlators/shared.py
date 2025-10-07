"""Shared functionality between GPU and CPU correlators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Generic, TypeVar
import numpy as np
from tqdm import tqdm
from scipy.ndimage import laplace as laplace_filter

if TYPE_CHECKING:
    from pyopencl.array import Array as ClArray
    from pyopencl import Image

f32 = np.float32
i32 = np.int32

T = TypeVar("T", np.ndarray, "ClArray")
I = TypeVar("I", np.ndarray, "Image")


@dataclass
class Vars(Generic[T, I]):
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
class VarsFT(Generic[T]):
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


class Correlator(ABC):
    vars: Vars
    vars_ft: VarsFT
    rfftn: Callable
    irfftn: Callable
    conj_multiply: Callable
    square: Callable
    laplace: bool
    target: np.ndarray
    lcc: np.ndarray
    rot: np.ndarray

    @abstractmethod
    def __init__(self):
        """Initialize the correlator along with the above class properties."""
        pass

    @abstractmethod
    def _set_template_var(self, template: np.ndarray):
        """Set the Vars.template variable in-place."""
        pass

    @abstractmethod
    def _set_mask_var(self, mask: np.ndarray):
        """Set the Vars.mask variable in-place."""

    def set_template(self, template: np.ndarray, mask: np.ndarray):
        """Set the template structure that you want to fit in the target density.

        Can be used to try to fit a different template to the same target structure
        without recomputing the kernels.
        
        Args:
            template: the template structure that you want to fit in the target density,
                should have been regridded to the same grid as the target density.
        """
        if template.shape != self.target.shape:
            raise ValueError("Shape of template does not match the target.")

        if self.laplace:
            template = laplace_filter(template, mode='wrap')

        # Precompute the normalization factor for use in the LCC computing kernel
        self.norm_factor = get_normalization_factor(mask)

        template = normalize_template(template, mask)
        self._set_template_var(template)
        self._set_mask_var(mask)

        # Reset lcc and rot values after (re)setting the template
        self.lcc[:] = 0.0
        self.rot[:] = 0

    @abstractmethod
    def rotate_grids(self, rotmat: np.ndarray):
        """Rotate the template and mask using the rotational matrix."""
        pass

    def compute_gcc(self):
        """Compute the global cross-correlation.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 3."""
        self.rfftn(self.vars.rot_template, self.vars_ft.template)
        self.conj_multiply(
            self.vars_ft.template,
            self.vars_ft.target,
            self.vars_ft.gcc
        )
        self.irfftn(self.vars_ft.gcc, self.vars.gcc)

    def compute_sq_avg_density(self):
        """Compute the square of the average core-weighted density.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 4."""
        self.rfftn(self.vars.rot_mask, self.vars_ft.mask)
        self.conj_multiply(
            self.vars_ft.mask,
            self.vars_ft.target,
            self.vars_ft.ave
        )
        self.irfftn(self.vars_ft.ave, self.vars.ave)

    def compute_avg_sq_density(self):
        """Compute the average of the squared core-weighted density.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 5."""
        self.square(self.vars.rot_mask, self.vars.rot_mask2)
        self.rfftn(self.vars.rot_mask2, self.vars_ft.mask2)
        self.conj_multiply(self.vars_ft.mask2, self.vars_ft.target2, self.vars_ft.ave2)
        self.irfftn(self.vars_ft.ave2, self.vars.ave2)

    @abstractmethod
    def compute_lcc_score_and_take_best(self, n: int):
        """Compute the LCC score and store best result.
        
        Args:
            n: iteration number.
        """
        pass

    def compute_rotation(self, n: int, rotmat: np.ndarray):
        """Compute a single rotation.
        
        Args:
            n: rotation number.
            rotmat: rotation matrix for this rotation.
        """
        self.rotate_grids(rotmat)
        self.compute_gcc()
        self.compute_sq_avg_density()
        self.compute_avg_sq_density()
        self.compute_lcc_score_and_take_best(n)

    @abstractmethod
    def scan(self, progress: partial[tqdm] | None):
        pass

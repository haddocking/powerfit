"""Shared functionality between GPU and CPU correlators."""

import contextlib
import logging
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, overload

if sys.version_info >= (3, 12):
    from itertools import batched
else:
    from more_itertools import batched

import numpy as np
from numpy.typing import DTypeLike
from scipy.ndimage import laplace as laplace_filter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cupy import ndarray as CpArray
    from pyopencl import Image
    from pyopencl.array import Array as ClArray

f32 = np.float32
i32 = np.int32
# best batch size based on performance measurements in docs/performances.md and docs/timings.csv.
DEFAULT_BATCH_SIZE = 100

T = TypeVar("T", np.ndarray, "ClArray", "CpArray")


class NvidiaTexture(Protocol):
    """Minimal protocol for CUDA/NVIDIA texture-like handles."""

    @property
    def ptr(self) -> int: ...


# CPU uses ndarray, OpenCL uses Image, CUDA uses texture-like wrappers.
I = TypeVar("I", np.ndarray, "Image", NvidiaTexture)  # noqa: E741


class HasShape(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


R = TypeVar("R", bound=HasShape)


class ProgressBar(Protocol):
    """Progress bar object returned by progress factories."""

    n: int

    def update(self, n: int = 1) -> object: ...

    def __enter__(self) -> "ProgressBar": ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class ProgressFactory(Protocol):
    """Factory for progress wrappers used by scan routines."""

    @overload
    def __call__(self, iterable: range) -> Iterable[int]: ...

    @overload
    def __call__(self, *, total: int) -> ProgressBar: ...


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
    return target > target.max() * 0.05


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
        raise ValueError("Zero-filled mask is not allowed.")
    return norm_factor


def init_correlator_vars(
    target: np.ndarray,
    laplace: bool,
    *,
    array_from_host: Callable[[np.ndarray], T],
    zeros_array: Callable[[tuple[int, ...], DTypeLike], T],
    make_image: Callable[[np.ndarray], I],
    batch_size: int | None = None,
    empty_lcc_ft: bool = False,
    lcc_mask_dtype: DTypeLike = i32,
) -> tuple[Vars[T, I], VarsFT[T]]:
    """Create Vars and VarsFT containers for serial or batched correlators."""
    vol = target.shape
    ft = get_ft_shape(target)
    rot_vol = vol if batch_size is None else (batch_size,) + vol
    rot_ft = ft if batch_size is None else (batch_size,) + ft
    lcc_ft_shape = (0,) if empty_lcc_ft else ft

    lcc_mask = get_lcc_mask(target)
    filtered_target = laplace_filter(target, mode="wrap") if laplace else target
    zeros_vol = np.zeros(vol, dtype=f32)

    vars = Vars(
        target=array_from_host(filtered_target.astype(f32)),
        template=make_image(zeros_vol),
        mask=make_image(zeros_vol),
        lcc_mask=array_from_host(lcc_mask.astype(lcc_mask_dtype)),
        target2=zeros_array(vol, f32),
        rot_template=zeros_array(rot_vol, f32),
        rot_mask=zeros_array(rot_vol, f32),
        rot_mask2=zeros_array(rot_vol, f32),
        gcc=zeros_array(rot_vol, f32),
        ave=zeros_array(rot_vol, f32),
        ave2=zeros_array(rot_vol, f32),
        lcc=zeros_array(vol, f32),
        rot=array_from_host(np.zeros(vol, dtype=i32)),
    )
    vars_ft = VarsFT(
        target=zeros_array(ft, np.complex64),
        target2=zeros_array(ft, np.complex64),
        template=zeros_array(rot_ft, np.complex64),
        mask=zeros_array(rot_ft, np.complex64),
        mask2=zeros_array(rot_ft, np.complex64),
        ave=zeros_array(rot_ft, np.complex64),
        ave2=zeros_array(rot_ft, np.complex64),
        lcc=zeros_array(lcc_ft_shape, np.complex64),
        gcc=zeros_array(rot_ft, np.complex64),
    )
    return vars, vars_ft


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
            template = laplace_filter(template, mode="wrap")

        # Precompute the normalization factor for use in the LCC computing kernel
        self.norm_factor = get_normalization_factor(mask)

        template = normalize_template(template, mask)
        self._set_template_var(template)
        self._set_mask_var(mask)

        # Reset lcc and rot values after (re)setting the template
        self.lcc[:] = 0.0
        self.rot[:] = 0

    @abstractmethod
    def scan(self, progress: ProgressFactory | None = None):
        pass

    def compute_gcc(self):
        """Compute the global cross-correlation.

        Ref doi:10.3934/biophy.2015.2.73. Equation 3."""
        self.rfftn(self.vars.rot_template, self.vars_ft.template)
        self.conj_multiply(self.vars_ft.template, self.vars_ft.target, self.vars_ft.gcc)
        self.irfftn(self.vars_ft.gcc, self.vars.gcc)

    def compute_sq_avg_density(self):
        """Compute the square of the average core-weighted density.

        Ref doi:10.3934/biophy.2015.2.73. Equation 4."""
        self.rfftn(self.vars.rot_mask, self.vars_ft.mask)
        self.conj_multiply(self.vars_ft.mask, self.vars_ft.target, self.vars_ft.ave)
        self.irfftn(self.vars_ft.ave, self.vars.ave)

    def compute_avg_sq_density(self):
        """Compute the average of the squared core-weighted density.

        Ref doi:10.3934/biophy.2015.2.73. Equation 5."""
        self.square(self.vars.rot_mask, self.vars.rot_mask2)
        self.rfftn(self.vars.rot_mask2, self.vars_ft.mask2)
        self.conj_multiply(self.vars_ft.mask2, self.vars_ft.target2, self.vars_ft.ave2)
        self.irfftn(self.vars_ft.ave2, self.vars.ave2)


class SerialCorrelator(Correlator, ABC):
    """Base class for correlators that process rotations one at a time."""

    @abstractmethod
    def rotate_grids(self, rotmat: np.ndarray):
        """Rotate the template and mask using the rotational matrix."""
        pass

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


class BatchedCorrelator(Correlator, ABC, Generic[R]):
    """Base class for correlators that process rotations in batches.

    Provides a concrete `compute_batch` orchestration method analogous to
    `SerialCorrelator.compute_rotation`, with three abstract methods for the
    GPU-backend-specific operations.
    """

    batch_size: int
    max_batch_size: int
    rotations: R

    @abstractmethod
    def retrieve_results(self):
        """Retrieve the LCC and rotation results from the GPU to CPU."""
        pass

    def _scan_context(self):
        """Return a context manager wrapping the scan loop.

        Subclasses may override this to provide a GPU stream context (e.g.
        ``with self.cuda_stream:``). The default returns a no-op context.
        """
        return contextlib.nullcontext()

    def scan(self, progress: ProgressFactory | None = None):
        """Scan all provided rotations to find the best fit.

        Args:
            progress: optional factory for progress bars. Note that this is ignored
                for batched correlators, as this slows down computations.
        """
        n_rot = self.rotations.shape[0]
        logger.info(f"Batching {n_rot} rotations with batch size {self.batch_size} (max {self.max_batch_size}).")
        with self._scan_context():
            self.vars.lcc.fill(0)
            self.vars.rot.fill(0)
            for chunk in batched(range(n_rot), self.batch_size):
                self.compute_batch(chunk[0], len(chunk))
        self.retrieve_results()

    @abstractmethod
    def rotate_grids_batch(self, batch_start: int, chunk_size: int):
        """Rotate the template and mask for a whole batch of rotations.

        Args:
            batch_start: index of the first rotation in this batch.
            chunk_size: number of rotations in this batch.
        """
        pass

    @abstractmethod
    def compute_batch_lcc_score_and_take_best(self, batch_start: int, chunk_size: int):
        """Reduce per-batch LCC scores to the per-voxel best result.

        Args:
            batch_start: index of the first rotation in this batch.
            chunk_size: number of rotations in this batch.
        """
        pass

    def compute_batch(self, batch_start: int, chunk_size: int):
        """Compute correlation for a batch of rotations and reduce to global best.

        Batched equivalent of `compute_rotation`: rotate → GCC → sq-avg density
        → avg-sq density → LCC reduction.

        Args:
            batch_start: index of the first rotation in this batch.
            chunk_size: number of rotations in this batch.
        """
        self.rotate_grids_batch(batch_start, chunk_size)
        self.compute_gcc()
        self.compute_sq_avg_density()
        self.compute_avg_sq_density()
        self.compute_batch_lcc_score_and_take_best(batch_start, chunk_size)

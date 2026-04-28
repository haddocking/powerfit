import logging
import sys

if sys.version_info >= (3, 12):
    from itertools import batched
else:
    from more_itertools import batched

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl import Image
from pyopencl.array import Array as ClArray
from pyvkfft.opencl import VkFFTApp
from scipy.ndimage import laplace as laplace_filter

from powerfit_em.correlators.clkernels import CLKernels
from powerfit_em.correlators.shared import (
    Correlator,
    Vars,
    VarsFT,
    f32,
    get_ft_shape,
    get_lcc_mask,
    i32,
)

# Conservative batch-memory target for auto sizing on OpenCL devices.
_BATCH_MEM_TARGET = 0.70


logger = logging.getLogger(__name__)


def init_gpu_vars(
    queue: cl.CommandQueue,
    target: np.ndarray,
    laplace: bool,
) -> tuple[Vars[ClArray, Image], VarsFT[ClArray]]:
    """Initialize all GPU variables on the specified queue."""
    lcc_mask = get_lcc_mask(target)
    _t = laplace_filter(target, mode="wrap") if laplace else target
    zeros = np.zeros(target.shape, f32)
    gpu_vars = Vars(
        target=cl_array.to_device(queue, _t.astype(f32)),
        template=cl.image_from_array(queue.context, zeros),  # template is set through separate method
        mask=cl.image_from_array(queue.context, zeros),  # mask is set through separate method
        lcc_mask=cl_array.to_device(queue, lcc_mask.astype(i32)),
        target2=cl_array.to_device(queue, zeros),
        rot_template=cl_array.to_device(queue, zeros),
        rot_mask=cl_array.to_device(queue, zeros),
        rot_mask2=cl_array.to_device(queue, zeros),
        gcc=cl_array.to_device(queue, zeros),
        ave=cl_array.to_device(queue, zeros),
        ave2=cl_array.to_device(queue, zeros),
        lcc=cl_array.to_device(queue, zeros),
        rot=cl_array.to_device(queue, np.zeros(target.shape, i32)),
    )
    zeros_ft = np.zeros(get_ft_shape(target), dtype=np.complex64)
    gpu_vars_ft = VarsFT(
        target=cl_array.to_device(queue, zeros_ft),
        target2=cl_array.to_device(queue, zeros_ft),
        template=cl_array.to_device(queue, zeros_ft),
        mask=cl_array.to_device(queue, zeros_ft),
        mask2=cl_array.to_device(queue, zeros_ft),
        ave=cl_array.to_device(queue, zeros_ft),
        ave2=cl_array.to_device(queue, zeros_ft),
        lcc=cl_array.to_device(queue, zeros_ft),
        gcc=cl_array.to_device(queue, zeros_ft),
    )
    return gpu_vars, gpu_vars_ft


def generate_kernels(queue: cl.CommandQueue, target: np.ndarray):
    """Generate the custom OpenCL kernels based on the target's shape"""
    kernel_values = {
        "shape_x": target.shape[2],
        "shape_y": target.shape[1],
        "shape_z": target.shape[0],
        "llength": i32(min(target.shape) // 2),
    }
    return CLKernels(queue.context, kernel_values)


def build_opencl_ffts(shape: tuple[int, ...], queue: cl.CommandQueue):
    """Build planned OpenCL FFT and inverse FFT wrappers for reuse."""
    plan = VkFFTApp(
        shape,
        np.float32,
        queue,
        ndim=len(shape),
        inplace=False,
        r2c=True,
        norm=1,
    )

    def rfftn(src, dst):
        plan.fft(src, dst, queue=queue)

    def irfftn(src, dst):
        plan.ifft(src, dst, queue=queue)

    return rfftn, irfftn


def build_opencl_ffts_batched(vol_shape: tuple[int, ...], batch_size: int, queue: cl.CommandQueue):
    """Build batched 3D FFT wrappers over a leading batch axis."""
    plan = VkFFTApp(
        (batch_size,) + tuple(vol_shape),
        np.float32,
        queue,
        ndim=3,
        inplace=False,
        r2c=True,
        norm=1,
    )

    def rfftn_batch(src, dst):
        plan.fft(src, dst, queue=queue)

    def irfftn_batch(src, dst):
        plan.ifft(src, dst, queue=queue)

    return rfftn_batch, irfftn_batch


def _probe_batch_size(queue: cl.CommandQueue, vol_shape: tuple[int, int, int]) -> int:
    """Estimate an OpenCL batch size from device memory limits."""
    z, y, x = vol_shape
    ft_x = x // 2 + 1
    real_bytes = z * y * x * np.dtype(np.float32).itemsize
    complex_bytes = z * y * ft_x * np.dtype(np.complex64).itemsize
    bytes_per_rot = 6 * real_bytes + 6 * complex_bytes

    global_mem = int(queue.device.global_mem_size)
    max_alloc = int(queue.device.max_mem_alloc_size)
    budget = int(global_mem * _BATCH_MEM_TARGET)
    by_total = budget // bytes_per_rot
    by_alloc_real = max_alloc // real_bytes
    by_alloc_complex = max_alloc // complex_bytes
    batch_size = min(by_total, by_alloc_real, by_alloc_complex)
    if batch_size < 1:
        raise RuntimeError("Unable to auto-tune a valid OpenCL batch size for this device.")
    return int(batch_size)


def precompute_squared_targets(
    gpu_vars: Vars[ClArray, Image],
    gpu_vars_ft: VarsFT[ClArray],
    kernels: CLKernels,
    rfftn,
):
    """Compute the squared target and fourier transformed target on GPU for reuse."""
    rfftn(gpu_vars.target, gpu_vars_ft.target)
    kernels.multiply(gpu_vars.target, gpu_vars.target, gpu_vars.target2)
    rfftn(gpu_vars.target2, gpu_vars_ft.target2)


def transform_rotations(rotations: np.ndarray) -> np.ndarray:
    """Transform rotation array for input into OpenCL kernels.

    The OpenCL kernel requires a Float16 input (struct containing 16 single-
    precision floats). The rotation matrices need to occupy the first 9 entries.
    """
    rot_trans = np.zeros((rotations.shape[0], 16), dtype=np.float32)
    rot_trans[:, :9] = rotations.reshape(-1, 9)
    return rot_trans


class OpenCLCorrelator(Correlator):
    """Compute the LCC score for a target and template combination."""

    def __init__(
        self,
        target: np.ndarray,
        template: np.ndarray,
        rotations: np.ndarray,
        mask: np.ndarray,
        queue: cl.CommandQueue,
        laplace: bool = False,
        batch_size: int | None = None,
    ):
        """Initialize the GPU correlator.

        Args:
            target: the target density on which you want to fit a template structure.
            template: the template structure that you want to fit in the target density,
                should have been regridded to the same grid as the target density.
            rotations: array of 3D-rotation matrices, of shape (n_rotations, 3, 3).
                for each rotation the local cross correlation is computed for every
                possible translation (in parallel using Fourier transforms).
            mask: core-weighted mask. See doi:10.3934/biophy.2015.2.73, Figure 1.
            queue: the OpenCL command queue on which to execute the computations.
            laplace: if true, a Laplace pre-filter is applied to the target density and
                template to enhance the sensitivity of the scoring function.
            batch_size: If >0, use this fixed batch size for processing rotations.
                If 0, disable batching and process rotations one-by-one.
                If None, auto-tune a batch size from OpenCL device memory limits.
        """
        self.target: np.ndarray = target / target.max()
        self.laplace = laplace
        self.mask = mask
        self.queue = queue
        self.norm_factor = 0.0  # to be set by set_template

        self._rotations = transform_rotations(rotations)
        self._volume_size = int(np.prod(self.target.shape))
        self._ft_vol_size = int(np.prod(get_ft_shape(self.target)))

        self.vars, self.vars_ft = init_gpu_vars(queue, self.target, self.laplace)

        self.lcc = np.zeros(self.target.shape, dtype=np.float32)
        self.rot = np.zeros(self.target.shape, dtype=np.int32)

        self.cl_kernels = generate_kernels(queue, self.target)
        self.conj_multiply = self.cl_kernels.conj_multiply
        self.square = lambda a, b: self.cl_kernels.multiply(a, a, b)
        self.rfftn, self.irfftn = build_opencl_ffts(self.target.shape, queue)

        if batch_size == 0:
            self._use_batch = False
            self.batch_size = 1
        else:
            self._use_batch = True
            if batch_size is None:
                self.batch_size = _probe_batch_size(queue, self.target.shape)
            else:
                if batch_size < 0:
                    raise ValueError("batch_size must be >= 0.")
                self.batch_size = batch_size
            self._allocate_batch_buffers(self.batch_size)
            self._rfftn_batch, self._irfftn_batch = build_opencl_ffts_batched(self.target.shape, self.batch_size, queue)

        self.set_template(template, mask)
        precompute_squared_targets(self.vars, self.vars_ft, self.cl_kernels, self.rfftn)

    def _allocate_batch_buffers(self, batch_size: int):
        """Allocate all OpenCL batch buffers; raises on allocation failure."""
        vol = self.target.shape
        ft = get_ft_shape(self.target)
        bvol = (batch_size,) + vol
        bft = (batch_size,) + ft
        try:
            self._batch_rot_template = cl_array.zeros(self.queue, bvol, dtype=f32)
            self._batch_rot_mask = cl_array.zeros(self.queue, bvol, dtype=f32)
            self._batch_rot_mask2 = cl_array.zeros(self.queue, bvol, dtype=f32)
            self._batch_gcc = cl_array.zeros(self.queue, bvol, dtype=f32)
            self._batch_ave = cl_array.zeros(self.queue, bvol, dtype=f32)
            self._batch_ave2 = cl_array.zeros(self.queue, bvol, dtype=f32)
            self._batch_template_ft = cl_array.zeros(self.queue, bft, dtype=np.complex64)
            self._batch_mask_ft = cl_array.zeros(self.queue, bft, dtype=np.complex64)
            self._batch_mask2_ft = cl_array.zeros(self.queue, bft, dtype=np.complex64)
            self._batch_gcc_ft = cl_array.zeros(self.queue, bft, dtype=np.complex64)
            self._batch_ave_ft = cl_array.zeros(self.queue, bft, dtype=np.complex64)
            self._batch_ave2_ft = cl_array.zeros(self.queue, bft, dtype=np.complex64)
        except cl.MemoryError as exc:
            raise RuntimeError(f"Failed to allocate OpenCL batch buffers for batch_size={batch_size}.") from exc

    def _set_template_var(self, template: np.ndarray):
        self.vars.template = cl.image_from_array(self.vars.template.context, template.astype(f32))

    def _set_mask_var(self, mask: np.ndarray):
        self.vars.mask = cl.image_from_array(self.vars.mask.context, mask.astype(f32))

    def rotate_grids(self, rotmat: np.ndarray):
        """Rotate the template and mask using the rotational matrix."""
        self.cl_kernels.rotate_image3d(self.queue, self.vars.template, rotmat, self.vars.rot_template)
        self.cl_kernels.rotate_image3d(self.queue, self.vars.mask, rotmat, self.vars.rot_mask, nearest=True)

    def compute_lcc_score_and_take_best(self, n: int):
        """Compute the LCC score and store best result.

        Args:
            n: iteration number.
        """
        self.cl_kernels.calc_lcc_and_take_best(
            self.vars.gcc,
            self.vars.ave,
            self.vars.ave2,
            self.vars.lcc_mask,
            self.norm_factor,
            np.int32(n),
            self.vars.lcc,
            self.vars.rot,
        )

    def _compute_batch(self, batch_start: int, batch_size: int):
        """Compute correlation for a batch of rotations and reduce to global best."""
        if batch_size > self.batch_size:
            raise ValueError("batch_size exceeds allocated OpenCL batch buffers.")

        rotmats = cl_array.to_device(self.queue, self._rotations[batch_start : batch_start + batch_size])
        self._batch_rot_template.fill(0)
        self._batch_rot_mask.fill(0)

        self.cl_kernels.rotate_image3d_batch(
            self.queue,
            self.vars.template,
            rotmats,
            0,
            batch_size,
            self._batch_rot_template,
        )
        self.cl_kernels.rotate_image3d_batch(
            self.queue,
            self.vars.mask,
            rotmats,
            0,
            batch_size,
            self._batch_rot_mask,
            nearest=True,
        )

        self._rfftn_batch(self._batch_rot_template, self._batch_template_ft)
        self.cl_kernels.batch_conj_multiply(
            self.queue, self._batch_template_ft, self.vars_ft.target, self._batch_gcc_ft, batch_size, self._ft_vol_size
        )
        self._irfftn_batch(self._batch_gcc_ft, self._batch_gcc)

        self._rfftn_batch(self._batch_rot_mask, self._batch_mask_ft)
        self.cl_kernels.batch_conj_multiply(
            self.queue, self._batch_mask_ft, self.vars_ft.target, self._batch_ave_ft, batch_size, self._ft_vol_size
        )
        self._irfftn_batch(self._batch_ave_ft, self._batch_ave)

        self.square(self._batch_rot_mask, self._batch_rot_mask2)
        self._rfftn_batch(self._batch_rot_mask2, self._batch_mask2_ft)
        self.cl_kernels.batch_conj_multiply(
            self.queue, self._batch_mask2_ft, self.vars_ft.target2, self._batch_ave2_ft, batch_size, self._ft_vol_size
        )
        self._irfftn_batch(self._batch_ave2_ft, self._batch_ave2)

        self.cl_kernels.batch_lcc_and_take_best(
            self.queue,
            self._batch_gcc,
            self._batch_ave,
            self._batch_ave2,
            self.vars.lcc_mask,
            self.vars.lcc,
            self.vars.rot,
            np.float32(self.norm_factor),
            batch_start,
            batch_size,
            self._volume_size,
        )

    def retrieve_results(self):
        """Retrieve the results from the GPU."""
        self.vars.lcc.get(ary=self.lcc)
        self.vars.rot.get(ary=self.rot)
        self.queue.finish()

    def scan(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Scan all provided rotations to find the best fit."""
        self.vars.lcc.fill(0)
        self.vars.rot.fill(0)

        n_rot = self._rotations.shape[0]
        if self._use_batch:
            batch = self.batch_size
            logger.info(f"Batching {n_rot} rotations with batch size {batch}.")
            for chunk in batched(range(n_rot), batch):
                self._compute_batch(chunk[0], len(chunk))
        else:
            logger.info(f"Processing {n_rot} rotations without batching.")
            for n in range(0, n_rot):
                self.compute_rotation(n, self._rotations[n])
        self.retrieve_results()

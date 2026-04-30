import logging

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl import Image
from pyopencl.array import Array as ClArray
from pyvkfft.opencl import VkFFTApp

from powerfit_em.correlators.clkernels import CLKernels
from powerfit_em.correlators.shared import (
    DEFAULT_BATCH_SIZE,
    BatchedCorrelator,
    Correlator,
    Vars,
    VarsFT,
    f32,
    get_ft_shape,
    i32,
    init_correlator_vars,
)

logger = logging.getLogger(__name__)


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


def max_batch_size(queue: cl.CommandQueue, vol_shape: tuple[int, int, int]) -> int:
    """Return the hard upper bound on batch size imposed by OpenCL device memory."""
    BATCH_FLOOR = 1
    # Conservative batch-memory target for auto sizing on OpenCL devices.
    VRAM_TARGET = 0.70

    z, y, x = vol_shape
    ft_x = x // 2 + 1
    real_bytes = z * y * x * np.dtype(np.float32).itemsize
    complex_bytes = z * y * ft_x * np.dtype(np.complex64).itemsize
    bytes_per_rot = 6 * real_bytes + 6 * complex_bytes

    global_mem = int(queue.device.global_mem_size)
    max_alloc = int(queue.device.max_mem_alloc_size)
    budget = int(global_mem * VRAM_TARGET)
    by_total = budget // bytes_per_rot
    by_alloc_real = max_alloc // real_bytes
    by_alloc_complex = max_alloc // complex_bytes
    batch_size = min(by_total, by_alloc_real, by_alloc_complex)
    if batch_size < BATCH_FLOOR:
        raise RuntimeError("Unable to compute a valid OpenCL memory upper bound for this device.")
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


class OpenCLSerialCorrelator(Correlator):
    """Compute LCC scores for each rotation one-by-one using OpenCL.

    No batch buffers are allocated; each rotation is processed individually.
    Use this class when memory is constrained or batch overhead is undesirable.
    """

    def __init__(
        self,
        target: np.ndarray,
        template: np.ndarray,
        rotations: np.ndarray,
        mask: np.ndarray,
        queue: cl.CommandQueue,
        laplace: bool = False,
    ):
        """Initialize the serial OpenCL correlator.

        Args:
            target: the target density on which you want to fit a template structure.
            template: the template structure that you want to fit in the target density,
                should have been regridded to the same grid as the target density.
            rotations: array of 3D-rotation matrices, of shape (n_rotations, 3, 3).
            mask: core-weighted mask. See doi:10.3934/biophy.2015.2.73, Figure 1.
            queue: the OpenCL command queue on which to execute the computations.
            laplace: if true, a Laplace pre-filter is applied to the target density and
                template to enhance the sensitivity of the scoring function.
        """
        self.target: np.ndarray = target / target.max()
        self.laplace = laplace
        self.queue = queue
        self.norm_factor = 0.0  # to be set by set_template

        self.rotations = transform_rotations(rotations)

        self.init_vars()

        self.lcc = np.zeros(self.target.shape, dtype=np.float32)
        self.rot = np.zeros(self.target.shape, dtype=np.int32)

        self.cl_kernels = generate_kernels(queue, self.target)
        self.conj_multiply = self.cl_kernels.conj_multiply
        self.square = lambda a, b: self.cl_kernels.multiply(a, a, b)
        self.rfftn, self.irfftn = build_opencl_ffts(self.target.shape, queue)

        self.set_template(template, mask)
        precompute_squared_targets(self.vars, self.vars_ft, self.cl_kernels, self.rfftn)

    def init_vars(self):
        self.vars, self.vars_ft = init_correlator_vars(
            self.target,
            self.laplace,
            array_from_host=lambda arr: cl_array.to_device(self.queue, arr),
            zeros_array=lambda shape, dtype: cl_array.zeros(self.queue, shape, dtype=dtype),
            make_image=lambda arr: cl.image_from_array(self.queue.context, arr),
        )

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

    def retrieve_results(self):
        """Retrieve the results from the GPU."""
        self.vars.lcc.get(ary=self.lcc)
        self.vars.rot.get(ary=self.rot)
        self.queue.finish()

    def scan(self, progress=None):
        """Scan all provided rotations to find the best fit."""
        self.vars.lcc.fill(0)
        self.vars.rot.fill(0)

        n_rot = self.rotations.shape[0]
        logger.info(f"Processing {n_rot} rotations without batching.")
        for n in range(n_rot):
            self.compute_rotation(n, self.rotations[n])
        self.retrieve_results()


class OpenCLBatchedCorrelator(BatchedCorrelator[cl_array.Array]):
    """Compute LCC scores in batches of rotations using OpenCL.

    Batch buffers are allocated upfront and rotations are processed in groups
    for higher GPU throughput.
    """

    def __init__(
        self,
        target: np.ndarray,
        template: np.ndarray,
        rotations: np.ndarray,
        mask: np.ndarray,
        queue: cl.CommandQueue,
        laplace: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """Initialize the batched OpenCL correlator.

        Args:
            target: the target density on which you want to fit a template structure.
            template: the template structure that you want to fit in the target density,
                should have been regridded to the same grid as the target density.
            rotations: array of 3D-rotation matrices, of shape (n_rotations, 3, 3).
            mask: core-weighted mask. See doi:10.3934/biophy.2015.2.73, Figure 1.
            queue: the OpenCL command queue on which to execute the computations.
            laplace: if true, a Laplace pre-filter is applied to the target density and
                template to enhance the sensitivity of the scoring function.
            batch_size: number of rotations per batch. Must be > 0;
                use OpenCLSerialCorrelator for serial processing.
        """
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be > 0 for OpenCLBatchedCorrelator. Use OpenCLSerialCorrelator for serial processing."
            )

        self.target: np.ndarray = target / target.max()
        self.laplace = laplace
        self.queue = queue
        self.norm_factor = 0.0  # to be set by set_template

        transformed_rotations = transform_rotations(rotations)
        self.rotations= cl_array.to_device(self.queue, transformed_rotations)
        self.volume_size = int(np.prod(self.target.shape))
        self.ft_vol_size = int(np.prod(get_ft_shape(self.target)))

        self.lcc = np.zeros(self.target.shape, dtype=np.float32)
        self.rot = np.zeros(self.target.shape, dtype=np.int32)

        self.cl_kernels = generate_kernels(queue, self.target)
        self.square = lambda a, b: self.cl_kernels.multiply(a, a, b)

        self.max_batch_size = max_batch_size(queue, self.target.shape)
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size={batch_size} exceeds the device memory upper bound {self.max_batch_size}. Reduce batch_size."
            )
        self.batch_size = batch_size

        self.init_vars()
        self.rfftn, self.irfftn = build_opencl_ffts_batched(self.target.shape, self.batch_size, queue)

        self.set_template(template, mask)

        serial_rfftn, _ = build_opencl_ffts(self.target.shape, queue)
        precompute_squared_targets(self.vars, self.vars_ft, self.cl_kernels, serial_rfftn)

    def init_vars(self):
        """Allocate all GPU arrays needed by the batched path; raises on allocation failure."""
        try:
            self.vars, self.vars_ft = init_correlator_vars(
                self.target,
                self.laplace,
                array_from_host=lambda arr: cl_array.to_device(self.queue, arr),
                zeros_array=lambda shape, dtype: cl_array.zeros(self.queue, shape, dtype=dtype),
                make_image=lambda arr: cl.image_from_array(self.queue.context, arr),
                batch_size=self.batch_size,
                empty_lcc_ft=True,
            )
        except cl.MemoryError as exc:
            raise RuntimeError(f"Failed to allocate OpenCL batch buffers for batch_size={self.batch_size}.") from exc

    def _set_template_var(self, template: np.ndarray):
        self.vars.template = cl.image_from_array(self.vars.template.context, template.astype(f32))

    def _set_mask_var(self, mask: np.ndarray):
        self.vars.mask = cl.image_from_array(self.vars.mask.context, mask.astype(f32))

    def rotate_grids_batch(self, batch_start: int, chunk_size: int):
        self.cl_kernels.rotate_image3d_batch(
            self.queue,
            self.vars.template,
            self.rotations,
            batch_start,
            chunk_size,
            self.vars.rot_template,
        )
        self.cl_kernels.rotate_image3d_batch(
            self.queue,
            self.vars.mask,
            self.rotations,
            batch_start,
            chunk_size,
            self.vars.rot_mask,
            nearest=True,
        )

    def batch_conj_multiply(self, a, b, out, chunk_size: int):
        self.cl_kernels.batch_conj_multiply(self.queue, a, b, out, chunk_size, self.ft_vol_size)

    def compute_batch_lcc_score_and_take_best(self, batch_start: int, chunk_size: int):
        self.cl_kernels.batch_lcc_and_take_best(
            self.queue,
            self.vars.gcc,
            self.vars.ave,
            self.vars.ave2,
            self.vars.lcc_mask,
            self.vars.lcc,
            self.vars.rot,
            np.float32(self.norm_factor),
            batch_start,
            chunk_size,
            self.volume_size,
        )

    def retrieve_results(self):
        """Retrieve the results from the GPU."""
        self.vars.lcc.get(ary=self.lcc)
        self.vars.rot.get(ary=self.rot)
        self.queue.finish()

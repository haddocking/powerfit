import logging
from dataclasses import dataclass

import cupy as cp
import numpy as np
from cupy.cuda.texture import (
    ChannelFormatDescriptor,
    CUDAarray,
    ResourceDescriptor,
    TextureDescriptor,
    TextureObject,
)
from pyvkfft.cuda import VkFFTApp

from powerfit_em.correlators.cudakernels import CUDAKernels
from powerfit_em.correlators.shared import (
    DEFAULT_BATCH_SIZE,
    BatchedCorrelator,
    SerialCorrelator,
    f32,
    i32,
    init_correlator_vars,
)

logger = logging.getLogger(__name__)


@dataclass
class CUDATexture:
    """Texture handle paired with its backing CUDA array."""

    texture: TextureObject
    cuda_array: CUDAarray

    @property
    def ptr(self) -> int:
        return self.texture.ptr


def _make_cuda_texture(arr: np.ndarray, filter_mode: int) -> CUDATexture:
    """Upload *arr* to a CUDA array and wrap it in a TextureObject."""
    data = np.ascontiguousarray(arr, dtype=np.float32)
    # ChannelFormatDescriptor: 32-bit float, single channel (x=32, y=z=w=0, kind=float=2)
    channel = ChannelFormatDescriptor(32, 0, 0, 0, cp.cuda.runtime.cudaChannelFormatKindFloat)
    # CUDAarray shape is (depth, height, width) matching (Z, Y, X)
    cuda_arr = CUDAarray(channel, data.shape[2], data.shape[1], data.shape[0])
    # Copy host data into the CUDA array
    cuda_arr.copy_from(data)
    res_desc = ResourceDescriptor(cp.cuda.runtime.cudaResourceTypeArray, cuArr=cuda_arr)
    tex_desc = TextureDescriptor(
        addressModes=[cp.cuda.runtime.cudaAddressModeWrap] * 3,
        filterMode=filter_mode,
        normalizedCoords=1,
    )
    return CUDATexture(texture=TextureObject(res_desc, tex_desc), cuda_array=cuda_arr)


def make_cuda_texture_linear(arr: np.ndarray) -> CUDATexture:
    """Create a CUDA texture with trilinear filtering."""
    return _make_cuda_texture(arr, cp.cuda.runtime.cudaFilterModeLinear)


def make_cuda_texture_nearerst(arr: np.ndarray) -> CUDATexture:
    """Create a CUDA texture with nearest-neighbour filtering."""
    return _make_cuda_texture(arr, cp.cuda.runtime.cudaFilterModePoint)


def _square(a: cp.ndarray, out: cp.ndarray):
    cp.square(a, out=out)


def build_cuda_conj_multiply_kernel():
    return cp.ElementwiseKernel(
        in_params="complex64 a, complex64 b",
        out_params="complex64 out",
        operation="out = conj(a) * b;",
        name="powerfit_conj_multiply",
    )


def build_cuda_ffts(shape: tuple, cuda_stream=None):
    plan = VkFFTApp(
        shape,
        np.float32,
        ndim=len(shape),
        inplace=False,
        r2c=True,
        stream=cuda_stream,
        norm=1,
    )

    def rfftn(src, dst):
        plan.fft(src, dst)

    def irfftn(src, dst):
        plan.ifft(src, dst)

    return rfftn, irfftn


def build_cuda_ffts_batched(vol_shape: tuple, batch_size: int, cuda_stream=None):
    """Build a VkFFTApp plan that performs 3-D FFTs over the last three axes.

    By setting *ndim=3* on a 4-D array of shape *(batch_size, Z, Y, X)*, VkFFT
    treats the leading axis as a batch dimension and executes one independent
    3-D r2c/c2r transform per batch slot in a single launch.
    """
    batched_shape = (batch_size,) + tuple(vol_shape)
    plan = VkFFTApp(
        batched_shape,
        np.float32,
        ndim=3,
        inplace=False,
        r2c=True,
        stream=cuda_stream,
        norm=1,
    )

    def rfftn_batch(src, dst):
        plan.fft(src, dst)

    def irfftn_batch(src, dst):
        plan.ifft(src, dst)

    return rfftn_batch, irfftn_batch


def max_batch_size(vol_shape: tuple) -> int:
    """Return the hard upper bound on batch size imposed by CUDA device memory.

    Targets *_VRAM_TARGET* fraction of total device VRAM, capped by what is
    currently free (leaving a 10% headroom on free memory for driver overhead
    and VkFFT scratch buffers).

    Per rotation in a batch we allocate:
    * 6 real (float32) arrays of shape vol_shape
    * 6 complex (complex64) arrays of shape ft_shape
    """
    # Hard floor/ceiling used when converting VRAM budget to a legal batch size.
    BATCH_FLOOR = 1
    BATCH_CEIL = 8192
    # Fraction of total VRAM to target for batch buffers.
    VRAM_TARGET = 0.80
    VRAM_MAX = 0.90

    z, y, x = vol_shape
    ft_x = x // 2 + 1
    bytes_per_rot = 6 * z * y * x * 4 + 6 * z * y * ft_x * 8
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    # Primary target: a fraction of total VRAM.
    budget = int(total_mem * VRAM_TARGET)
    # Never exceed 90% of what's currently free to avoid OOM from driver overhead.
    budget = min(budget, int(free_mem * VRAM_MAX))
    batch = max(BATCH_FLOOR, min(BATCH_CEIL, budget // bytes_per_rot))
    # CUDA hard limit: gridDim.z <= 65535. The batch kernel packs (batch * Z)
    # into the Z grid dimension using block size 4 (CUDAKernels._block[2]).
    # See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
    # ("Maximum y- or z-dimension of a grid of thread blocks: 65535")
    BLOCK_Z = 4
    MAX_GRID_Z = 65535
    batch = min(batch, (MAX_GRID_Z * BLOCK_Z) // z)
    if batch < BATCH_FLOOR:
        raise RuntimeError(
            "Unable to fit even a single rotation in CUDA memory. "
            f"Required bytes per rotation: {bytes_per_rot}, free VRAM: {free_mem}."
        )
    return max(BATCH_FLOOR, int(batch))


def build_cuda_lcc_kernel():
    return cp.ElementwiseKernel(
        in_params=(
            "float32 gcc, float32 ave, float32 ave2, int32 lcc_mask, "
            "raw float32 prev_lcc, raw int32 prev_rot, float32 norm_factor, int32 nrot"
        ),
        out_params="float32 next_lcc, int32 next_rot",
        operation="""
        const float var = ave2 * norm_factor - ave * ave;
        float best_lcc = prev_lcc[i];
        int best_rot = prev_rot[i];
        if (lcc_mask != 0 && var > 0.0f) {
            const float score = gcc / sqrtf(var);
            if (score > best_lcc) {
                best_lcc = score;
                best_rot = nrot;
            }
        }
        next_lcc = best_lcc;
        next_rot = best_rot;
        """,
        name="powerfit_calc_lcc_and_take_best",
    )


class CUDACorrelator:
    """Shared CUDA state and helpers for serial and batched correlators."""

    def _init_cuda_common(
        self,
        target: np.ndarray,
        rotations: np.ndarray,
        cuda_stream: cp.cuda.Stream,
        laplace: bool = False,
    ):
        self.target: np.ndarray = target / target.max()
        self.laplace = laplace
        self.rotations = cp.asarray(rotations.reshape(rotations.shape[0], -1), dtype=f32)
        self.cuda_stream = cuda_stream
        self.norm_factor = 0.0  # to be set by set_template

        self.lcc = np.zeros(self.target.shape, dtype=f32)
        self.rot = np.zeros(self.target.shape, dtype=i32)
        self.volume_size = int(np.prod(self.target.shape))
        self.cuda_kernels = CUDAKernels(self.target.shape)
        self.conj_multiply_kernel = build_cuda_conj_multiply_kernel()
        self.square = _square

    def _init_vars_common(self, batch_size: int | None = None, empty_lcc_ft: bool = False):
        self.vars, self.vars_ft = init_correlator_vars(
            self.target,
            self.laplace,
            array_from_host=cp.asarray,
            zeros_array=lambda shape, dtype: cp.zeros(shape, dtype=dtype),
            make_image=make_cuda_texture_linear,
            batch_size=batch_size,
            empty_lcc_ft=empty_lcc_ft,
        )

    def _synchronize(self):
        self.cuda_stream.synchronize()

    def _set_template_var(self, template: np.ndarray):
        self.vars.template = make_cuda_texture_linear(template)

    def _set_mask_var(self, mask: np.ndarray):
        self.vars.mask = make_cuda_texture_nearerst(mask)


class CUDASerialCorrelator(CUDACorrelator, SerialCorrelator):
    """GPU-accelerated correlator that processes rotations one-by-one.

    No batch buffers are allocated; each rotation is processed individually.
    Use this class when memory is constrained or batch overhead is undesirable.
    """

    def __init__(
        self,
        target: np.ndarray,
        template: np.ndarray,
        rotations: np.ndarray,
        mask: np.ndarray,
        cuda_stream: cp.cuda.Stream,
        laplace: bool = False,
    ):
        self._init_cuda_common(target, rotations, cuda_stream, laplace)

        self.init_vars()
        self.lcc_kernel = build_cuda_lcc_kernel()
        self.rfftn, self.irfftn = build_cuda_ffts(self.target.shape, self.cuda_stream)

        with self.cuda_stream:
            self.set_template(template, mask)
            self.rfftn(self.vars.target, self.vars_ft.target)
            self.square(self.vars.target, self.vars.target2)
            self.rfftn(self.vars.target2, self.vars_ft.target2)
        self._synchronize()

    def init_vars(self):
        self._init_vars_common()

    def conj_multiply(self, a: cp.ndarray, b: cp.ndarray, out: cp.ndarray):
        self.conj_multiply_kernel(a, b, out)

    def rotate_grids(self, rotmat):
        with self.cuda_stream:
            self.cuda_kernels.rotate_image3d(self.vars.template, rotmat, self.vars.rot_template)
            self.cuda_kernels.rotate_image3d(self.vars.mask, rotmat, self.vars.rot_mask, nearest=True)

    def compute_lcc_score_and_take_best(self, n: int):
        self.lcc_kernel(
            self.vars.gcc,
            self.vars.ave,
            self.vars.ave2,
            self.vars.lcc_mask,
            self.vars.lcc,
            self.vars.rot,
            np.float32(self.norm_factor),
            np.int32(n),
            self.vars.lcc,
            self.vars.rot,
        )

    def retrieve_results(self):
        self._synchronize()
        self.lcc = cp.asnumpy(self.vars.lcc)
        self.rot = cp.asnumpy(self.vars.rot)

    def scan(self, progress=None):
        n_rot = self.rotations.shape[0]
        with self.cuda_stream:
            self.vars.lcc.fill(0)
            self.vars.rot.fill(0)
            logger.info(f"Processing {n_rot} rotations without batching.")
            for n in range(n_rot):
                self.compute_rotation(n, self.rotations[n])
        self.retrieve_results()


class CUDABatchedCorrelator(CUDACorrelator, BatchedCorrelator[cp.ndarray]):
    """GPU-accelerated correlator that processes rotations in batches.

    Batch buffers are allocated upfront and rotations are processed in groups
    for higher GPU throughput.
    """

    def __init__(
        self,
        target: np.ndarray,
        template: np.ndarray,
        rotations: np.ndarray,
        mask: np.ndarray,
        cuda_stream: cp.cuda.Stream,
        laplace: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """GPU-accelerated batched correlator using CuPy and custom CUDA kernels.

        Args:
            target: 3-D array representing the target volume.
            template: 3-D array representing the template volume.
            rotations: Array of shape (N, 3, 3) containing N rotation matrices
                to apply to the template and mask.
            mask: 3-D array representing the mask volume.
            cuda_stream: CuPy CUDA stream for asynchronous execution.
            laplace: Whether to apply a Laplacian filter to the target volume.
            batch_size: Number of rotations per batch. Must be > 0;
                use CUDASerialCorrelator for serial processing.
        """
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be > 0 for CUDABatchedCorrelator. Use CUDASerialCorrelator for serial processing."
            )

        self._init_cuda_common(target, rotations, cuda_stream, laplace)
        self.batch_lcc_kernel = self.cuda_kernels.batch_lcc_kernel

        self.max_batch_size = max_batch_size(self.target.shape)
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size={batch_size} exceeds the device memory upper bound"
                f" {self.max_batch_size}. Reduce batch_size."
            )
        self.batch_size = batch_size

        self.init_vars()
        self.rfftn, self.irfftn = build_cuda_ffts_batched(self.target.shape, self.batch_size, self.cuda_stream)

        with self.cuda_stream:
            self.set_template(template, mask)
            rfftn_serial, _ = build_cuda_ffts(self.target.shape, self.cuda_stream)
            rfftn_serial(self.vars.target, self.vars_ft.target)
            self.square(self.vars.target, self.vars.target2)
            rfftn_serial(self.vars.target2, self.vars_ft.target2)
        self._synchronize()

    def init_vars(self):
        """Allocate GPU arrays needed by the batched path; raises on OOM."""
        try:
            self._init_vars_common(batch_size=self.batch_size, empty_lcc_ft=True)
        except cp.cuda.memory.OutOfMemoryError as exc:
            raise RuntimeError(
                f"Failed to allocate CUDA batch buffers for batch_size={self.batch_size}. "
                "Reduce --batch-size or disable batching with --batch-size 0."
            ) from exc

    def rotate_grids_batch(self, batch_start: int, chunk_size: int):
        rotmats = self.rotations[batch_start : batch_start + chunk_size]
        self.cuda_kernels.rotate_image3d_batch(self.vars.template, rotmats, self.vars.rot_template, chunk_size)
        self.cuda_kernels.rotate_image3d_batch(self.vars.mask, rotmats, self.vars.rot_mask, chunk_size, nearest=True)

    def conj_multiply(self, a, b, out):
        # self.vars_ft.target has shape (Z, Y, X//2+1); the ElementwiseKernel
        # broadcasts it over the leading batch axis automatically.
        self.conj_multiply_kernel(a, b, out)

    def compute_batch_lcc_score_and_take_best(self, batch_start: int, chunk_size: int):
        # Picked as 256 as it multiple of 32 cuda warp,
        # tested 64, 128, 256 and 512 blocksizes with no significant difference in runtime.
        block = 512
        grid = (self.volume_size + block - 1) // block
        self.batch_lcc_kernel(
            (grid,),
            (block,),
            (
                self.vars.gcc,
                self.vars.ave,
                self.vars.ave2,
                self.vars.lcc_mask,
                self.vars.lcc,
                self.vars.rot,
                np.float32(self.norm_factor),
                np.int32(batch_start),
                np.int32(chunk_size),
                np.int32(self.volume_size),
            ),
        )

    def retrieve_results(self):
        self._synchronize()
        self.lcc = cp.asnumpy(self.vars.lcc)
        self.rot = cp.asnumpy(self.vars.rot)

    def _scan_context(self):
        return self.cuda_stream

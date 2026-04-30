import logging
import sys

if sys.version_info >= (3, 12):
    from itertools import batched
else:
    from more_itertools import batched

import cupy as cp
import numpy as np
from pyvkfft.cuda import VkFFTApp
from scipy.ndimage import laplace as laplace_filter

from powerfit_em.correlators.cudakernels import CUDAKernels
from powerfit_em.correlators.shared import (
    Correlator,
    Vars,
    VarsFT,
    f32,
    get_ft_shape,
    get_lcc_mask,
    i32,
)

# Minimum and maximum batch sizes considered during batch size guessing.
_BATCH_MIN = 1
_BATCH_MAX = 8192
# Fraction of total VRAM to target for batch buffers.
_VRAM_TARGET = 0.80
# Empirical performance-scaling constant for batch size guessing.
# Calibrated on an RTX 3050 (82 SMs, 1695 MHz) to target the largest batch
# within 10% of peak throughput. May need re-tuning for very different GPUs.
_K_CUDA_PERF = 81
# Practical bounds from docs/times.csv CUDA benchmarks (m3, m4).
# They prevent pathological tiny batches (batch=1) and oversized degraded ones.
_TUNED_BATCH_FLOOR = 20
_TUNED_BATCH_CEIL = 256
# Conservative device-attribute fallbacks used when CUDA reports zeros/missing.
_DEFAULT_SMS = 32
_DEFAULT_CLOCK_MHZ = 1500


logger = logging.getLogger(__name__)


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
    z, y, x = vol_shape
    ft_x = x // 2 + 1
    bytes_per_rot = 6 * z * y * x * 4 + 6 * z * y * ft_x * 8
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    # Primary target: a fraction of total VRAM.
    budget = int(total_mem * _VRAM_TARGET)
    # Never exceed 90% of what's currently free to avoid OOM from driver overhead.
    budget = min(budget, int(free_mem * 0.90))
    batch = max(_BATCH_MIN, min(_BATCH_MAX, budget // bytes_per_rot))
    # CUDA hard limit: gridDim.z <= 65535. The batch kernel packs (batch * Z)
    # into the Z grid dimension using block size 4 (CUDAKernels._block[2]).
    # See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications
    # ("Maximum y- or z-dimension of a grid of thread blocks: 65535")
    _BLOCK_Z = 4
    _MAX_GRID_Z = 65535
    batch = min(batch, (_MAX_GRID_Z * _BLOCK_Z) // z)
    return max(_BATCH_MIN, int(batch))


def guess_batch_size(vol_shape: tuple, device: None | cp.cuda.Device = None) -> int:
    """Estimate a practical CUDA batch size from device throughput proxies.

    Starts from a compute-capability proxy (SM count × clock frequency) scaled
    by *_K_CUDA_PERF*, then clamps into an empirically robust interval.
    This avoids pathological tiny auto-batches on some GPUs and oversized
    batches that can hurt throughput.

    Args:
        vol_shape: Spatial shape of the target volume.
        device: Optional CUDA device object. If None, uses current default
            device from CuPy.

    Returns:
        An estimated batch size that should yield good performance on the given device.
    """
    z, y, x = vol_shape
    ft_x = x // 2 + 1
    bytes_per_rot = 6 * z * y * x * 4 + 6 * z * y * ft_x * 8
    dev = cp.cuda.Device() if device is None else device
    n_sms = int(dev.attributes.get("MultiProcessorCount", 0))
    if n_sms <= 0:
        logger.warning(
            "CUDA device reported invalid MultiProcessorCount=%s; using fallback %s.",
            n_sms,
            _DEFAULT_SMS,
        )
        n_sms = _DEFAULT_SMS

    clock_khz = int(dev.attributes.get("ClockRate", 0))
    clock_mhz = clock_khz // 1000  # kHz -> MHz
    if clock_mhz <= 0:
        logger.warning(
            "CUDA device reported invalid ClockRate=%s kHz; using fallback %s MHz.",
            clock_khz,
            _DEFAULT_CLOCK_MHZ,
        )
        clock_mhz = _DEFAULT_CLOCK_MHZ

    raw = int(_K_CUDA_PERF * n_sms * clock_mhz / bytes_per_rot)
    return max(_TUNED_BATCH_FLOOR, min(_TUNED_BATCH_CEIL, raw))


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


def init_cuda_vars(
    target: np.ndarray,
    laplace: bool,
) -> tuple[Vars, VarsFT]:
    lcc_mask = get_lcc_mask(target)
    filtered_target = laplace_filter(target, mode="wrap") if laplace else target
    zeros = cp.zeros(target.shape, dtype=f32)
    vars = Vars(
        target=cp.asarray(filtered_target.astype(f32)),
        template=zeros.copy(),
        mask=zeros.copy(),
        lcc_mask=cp.asarray(lcc_mask.astype(i32)),
        target2=zeros.copy(),
        rot_template=zeros.copy(),
        rot_mask=zeros.copy(),
        rot_mask2=zeros.copy(),
        gcc=zeros.copy(),
        ave=zeros.copy(),
        ave2=zeros.copy(),
        lcc=zeros.copy(),
        rot=cp.zeros(target.shape, dtype=i32),
    )
    zeros_ft = cp.zeros(get_ft_shape(target), dtype=cp.complex64)
    vars_ft = VarsFT(
        target=zeros_ft.copy(),
        target2=zeros_ft.copy(),
        template=zeros_ft.copy(),
        mask=zeros_ft.copy(),
        mask2=zeros_ft.copy(),
        ave=zeros_ft.copy(),
        ave2=zeros_ft.copy(),
        lcc=zeros_ft.copy(),
        gcc=zeros_ft.copy(),
    )
    return vars, vars_ft


class CUDASerialCorrelator(Correlator):
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
        self.target: np.ndarray = target / target.max()
        self.laplace = laplace
        self.rotations = cp.asarray(rotations.reshape(rotations.shape[0], -1), dtype=f32)
        self.cuda_stream = cuda_stream

        self.vars, self.vars_ft = init_cuda_vars(self.target, self.laplace)

        self.lcc = np.zeros(self.target.shape, dtype=f32)
        self.rot = np.zeros(self.target.shape, dtype=i32)
        self._volume_size = int(np.prod(self.target.shape))
        self.cuda_kernels = CUDAKernels(self.target.shape)
        self.lcc_kernel = build_cuda_lcc_kernel()
        self.conj_multiply_kernel = build_cuda_conj_multiply_kernel()

        self.square = _square
        self.rfftn, self.irfftn = build_cuda_ffts(self.target.shape, self.cuda_stream)

        with self.cuda_stream:
            self.set_template(template, mask)
            self.rfftn(self.vars.target, self.vars_ft.target)
            self.square(self.vars.target, self.vars.target2)
            self.rfftn(self.vars.target2, self.vars_ft.target2)
        self._synchronize()

    def _synchronize(self):
        self.cuda_stream.synchronize()

    def _set_template_var(self, template: np.ndarray):
        self.vars.template = cp.asarray(template, dtype=f32)

    def _set_mask_var(self, mask: np.ndarray):
        self.vars.mask = cp.asarray(mask, dtype=f32)

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


class CUDABatchedCorrelator(Correlator):
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
        batch_size: int | None = None,
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
            batch_size: Number of rotations per batch. If None, auto-tune from
                available GPU memory. Must be > 0; use CUDASerialCorrelator for
                serial processing.
        """
        if batch_size is not None and batch_size <= 0:
            raise ValueError(
                "batch_size must be > 0 for CUDABatchedCorrelator. Use CUDASerialCorrelator for serial processing."
            )

        self.target: np.ndarray = target / target.max()
        self.laplace = laplace
        self.rotations = cp.asarray(rotations.reshape(rotations.shape[0], -1), dtype=f32)
        self.cuda_stream = cuda_stream

        self.lcc = np.zeros(self.target.shape, dtype=f32)
        self.rot = np.zeros(self.target.shape, dtype=i32)
        self._volume_size = int(np.prod(self.target.shape))
        self.cuda_kernels = CUDAKernels(self.target.shape)
        self._batch_lcc_kernel = self.cuda_kernels.batch_lcc_kernel
        self.conj_multiply_kernel = build_cuda_conj_multiply_kernel()

        self.square = _square
        self.rfftn, self.irfftn = build_cuda_ffts(self.target.shape, self.cuda_stream)

        self._max_batch = max_batch_size(self.target.shape)
        if batch_size is None:
            auto_batch = guess_batch_size(self.target.shape)
            if auto_batch > self._max_batch:
                raise ValueError(
                    f"Auto-tuned batch size {auto_batch} exceeds the device memory upper bound {self._max_batch}."
                )
            self.batch_size = auto_batch
        else:
            if batch_size > self._max_batch:
                raise ValueError(
                    f"batch_size={batch_size} exceeds the device memory upper bound "
                    f"{self._max_batch}. Reduce batch_size."
                )
            self.batch_size = batch_size

        self._allocate_batch_buffers()
        self._rfftn_batch, self._irfftn_batch = build_cuda_ffts_batched(
            self.target.shape, self.batch_size, self.cuda_stream
        )

        with self.cuda_stream:
            self.set_template(template, mask)
            self.rfftn(self.vars.target, self.vars_ft.target)
            self.square(self.vars.target, self.vars.target2)
            self.rfftn(self.vars.target2, self.vars_ft.target2)
        self._synchronize()

    def _allocate_batch_buffers(self):
        """Allocate GPU arrays needed by the batched path; raises on OOM."""
        try:
            vol = self.target.shape
            ft = get_ft_shape(self.target)
            bvol = (self.batch_size,) + vol
            bft = (self.batch_size,) + ft

            lcc_mask = get_lcc_mask(self.target)
            filtered_target = laplace_filter(self.target, mode="wrap") if self.laplace else self.target
            zeros = cp.zeros(vol, dtype=f32)
            zeros_ft = cp.zeros(ft, dtype=cp.complex64)
            self.vars = Vars(
                target=cp.asarray(filtered_target.astype(f32)),
                template=zeros.copy(),
                mask=zeros.copy(),
                lcc_mask=cp.asarray(lcc_mask.astype(i32)),
                target2=zeros.copy(),
                rot_template=cp.zeros(bvol, dtype=f32),
                rot_mask=cp.zeros(bvol, dtype=f32),
                rot_mask2=cp.zeros(bvol, dtype=f32),
                gcc=cp.zeros(bvol, dtype=f32),
                ave=cp.zeros(bvol, dtype=f32),
                ave2=cp.zeros(bvol, dtype=f32),
                lcc=cp.zeros(vol, dtype=f32),
                rot=cp.zeros(vol, dtype=i32),
            )
            self.vars_ft = VarsFT(
                target=zeros_ft.copy(),
                target2=zeros_ft.copy(),
                template=cp.zeros(bft, dtype=cp.complex64),
                mask=cp.zeros(bft, dtype=cp.complex64),
                mask2=cp.zeros(bft, dtype=cp.complex64),
                ave=cp.zeros(bft, dtype=cp.complex64),
                ave2=cp.zeros(bft, dtype=cp.complex64),
                lcc=cp.zeros(0, dtype=cp.complex64),
                gcc=cp.zeros(bft, dtype=cp.complex64),
            )
        except cp.cuda.memory.OutOfMemoryError as exc:
            raise RuntimeError(
                f"Failed to allocate CUDA batch buffers for batch_size={self.batch_size}. "
                "Reduce --batch-size or disable batching with --batch-size 0."
            ) from exc

    def _synchronize(self):
        self.cuda_stream.synchronize()

    def _set_template_var(self, template: np.ndarray):
        self.vars.template = cp.asarray(template, dtype=f32)

    def _set_mask_var(self, mask: np.ndarray):
        self.vars.mask = cp.asarray(mask, dtype=f32)

    def rotate_grids(self, rotmat: np.ndarray):
        raise NotImplementedError("rotate_grids is not used in the batched correlator.")

    def compute_lcc_score_and_take_best(self, n: int):
        raise NotImplementedError("compute_lcc_score_and_take_best is not used in the batched correlator.")

    def _compute_batch(self, batch_start: int, batch_size: int, rotmats: cp.ndarray):
        """Compute correlation for *batch_size* rotations and reduce to global best."""
        # Rotate template (linear interp) and mask (nearest) for the whole batch.
        self.cuda_kernels.rotate_image3d_batch(self.vars.template, rotmats, self.vars.rot_template, batch_size)
        self.cuda_kernels.rotate_image3d_batch(self.vars.mask, rotmats, self.vars.rot_mask, batch_size, nearest=True)

        # Batched equivalent of Correlator.compute_gcc().
        # GCC: rfftn(rot_template) then conj-multiply with target_ft, then irfftn.
        # self.vars_ft.target has shape (Z, Y, X//2+1); the ElementwiseKernel
        # broadcasts it over the leading batch axis automatically.
        self._rfftn_batch(self.vars.rot_template, self.vars_ft.template)
        self.conj_multiply_kernel(self.vars_ft.template, self.vars_ft.target, self.vars_ft.gcc)
        self._irfftn_batch(self.vars_ft.gcc, self.vars.gcc)

        # Batched equivalent of Correlator.compute_sq_avg_density().
        # ave: rfftn(rot_mask), conj-multiply with target_ft, irfftn.
        self._rfftn_batch(self.vars.rot_mask, self.vars_ft.mask)
        self.conj_multiply_kernel(self.vars_ft.mask, self.vars_ft.target, self.vars_ft.ave)
        self._irfftn_batch(self.vars_ft.ave, self.vars.ave)

        # Batched equivalent of Correlator.compute_avg_sq_density().
        # ave2: square(rot_mask), rfftn, conj-multiply with target2_ft, irfftn.
        cp.square(self.vars.rot_mask, out=self.vars.rot_mask2)
        self._rfftn_batch(self.vars.rot_mask2, self.vars_ft.mask2)
        self.conj_multiply_kernel(self.vars_ft.mask2, self.vars_ft.target2, self.vars_ft.ave2)
        self._irfftn_batch(self.vars_ft.ave2, self.vars.ave2)

        # Batched equivalent of Correlator.compute_lcc_score_and_take_best().
        # Per-voxel batch reduction: updates vars.lcc and vars.rot in-place.
        block = 256
        grid = (self._volume_size + block - 1) // block
        self._batch_lcc_kernel(
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
                np.int32(batch_size),
                np.int32(self._volume_size),
            ),
        )

    def retrieve_results(self):
        self._synchronize()
        self.lcc = cp.asnumpy(self.vars.lcc)
        self.rot = cp.asnumpy(self.vars.rot)

    def scan(self, progress=None):
        n_rot = self.rotations.shape[0]
        B = self.batch_size

        with self.cuda_stream:
            self.vars.lcc.fill(0)
            self.vars.rot.fill(0)

            logger.info(f"Batching {n_rot} rotations with batch size {B} (max {self._max_batch}).")
            for chunk in batched(range(n_rot), B):
                start = chunk[0]
                self._compute_batch(start, len(chunk), self.rotations[start : start + len(chunk)])

        self.retrieve_results()

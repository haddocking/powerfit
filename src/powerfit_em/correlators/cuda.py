import logging

import cupy as cp
import numpy as np
from pyvkfft.cuda import VkFFTApp
from scipy.ndimage import laplace as laplace_filter

from powerfit_em.correlators.cudakernels import CUDAKernels
from powerfit_em.correlators.shared import (
    Correlator,
    ProgressFactory,
    Vars,
    VarsFT,
    f32,
    get_ft_shape,
    get_lcc_mask,
    i32,
)

# Minimum and maximum batch sizes considered during auto-tuning.
_BATCH_MIN = 1
_BATCH_MAX = 8192
# Fraction of total VRAM to target for batch buffers.
_VRAM_TARGET = 0.80


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


def _probe_batch_size(vol_shape: tuple) -> int:
    """Estimate a safe batch size using the CUDA memory info.

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
    return int(batch)


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


class CUDACorrelator(Correlator):
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
        self._batch_lcc_kernel = self.cuda_kernels.batch_lcc_kernel
        self.conj_multiply_kernel = build_cuda_conj_multiply_kernel()

        self.square = _square
        self.rfftn, self.irfftn = build_cuda_ffts(self.target.shape, self.cuda_stream)

        # --- Batched scan setup ---
        self.batch_size = _probe_batch_size(self.target.shape)
        while self.batch_size >= _BATCH_MIN:
            if self._allocate_batch_buffers(self.batch_size):
                break
            self.batch_size //= 2

        self._rfftn_batch, self._irfftn_batch = build_cuda_ffts_batched(
            self.target.shape, self.batch_size, self.cuda_stream
        )

        with self.cuda_stream:
            self.set_template(template, mask)
            self.rfftn(self.vars.target, self.vars_ft.target)
            self.square(self.vars.target, self.vars.target2)
            self.rfftn(self.vars.target2, self.vars_ft.target2)
        self._synchronize()

    def _allocate_batch_buffers(self, batch_size: int) -> bool:
        """Try to allocate GPU buffers for *batch_size* parallel rotations.

        Returns True on success, False if allocation raises OOM.
        """
        try:
            vol = self.target.shape
            ft = get_ft_shape(self.target)
            bvol = (batch_size,) + vol
            bft = (batch_size,) + ft
            self._batch_rot_template = cp.zeros(bvol, dtype=f32)
            self._batch_rot_mask = cp.zeros(bvol, dtype=f32)
            self._batch_rot_mask2 = cp.zeros(bvol, dtype=f32)
            self._batch_gcc = cp.zeros(bvol, dtype=f32)
            self._batch_ave = cp.zeros(bvol, dtype=f32)
            self._batch_ave2 = cp.zeros(bvol, dtype=f32)
            self._batch_template_ft = cp.zeros(bft, dtype=cp.complex64)
            self._batch_mask_ft = cp.zeros(bft, dtype=cp.complex64)
            self._batch_mask2_ft = cp.zeros(bft, dtype=cp.complex64)
            self._batch_gcc_ft = cp.zeros(bft, dtype=cp.complex64)
            self._batch_ave_ft = cp.zeros(bft, dtype=cp.complex64)
            self._batch_ave2_ft = cp.zeros(bft, dtype=cp.complex64)
            return True
        except cp.cuda.memory.OutOfMemoryError:
            return False

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

    def _compute_batch(self, batch_start: int, batch_size: int, rotmats: cp.ndarray):
        """Compute correlation for *batch_size* rotations and reduce to global best."""
        # Rotate template (linear interp) and mask (nearest) for the whole batch.
        self.cuda_kernels.rotate_image3d_batch(self.vars.template, rotmats, self._batch_rot_template, batch_size)
        self.cuda_kernels.rotate_image3d_batch(self.vars.mask, rotmats, self._batch_rot_mask, batch_size, nearest=True)

        # Batched equivalent of Correlator.compute_gcc().
        # GCC: rfftn(rot_template) then conj-multiply with target_ft, then irfftn.
        # self.vars_ft.target has shape (Z, Y, X//2+1); the ElementwiseKernel
        # broadcasts it over the leading batch axis automatically.
        self._rfftn_batch(self._batch_rot_template, self._batch_template_ft)
        self.conj_multiply_kernel(self._batch_template_ft, self.vars_ft.target, self._batch_gcc_ft)
        self._irfftn_batch(self._batch_gcc_ft, self._batch_gcc)

        # Batched equivalent of Correlator.compute_sq_avg_density().
        # ave: rfftn(rot_mask), conj-multiply with target_ft, irfftn.
        self._rfftn_batch(self._batch_rot_mask, self._batch_mask_ft)
        self.conj_multiply_kernel(self._batch_mask_ft, self.vars_ft.target, self._batch_ave_ft)
        self._irfftn_batch(self._batch_ave_ft, self._batch_ave)

        # Batched equivalent of Correlator.compute_avg_sq_density().
        # ave2: square(rot_mask), rfftn, conj-multiply with target2_ft, irfftn.
        cp.square(self._batch_rot_mask, out=self._batch_rot_mask2)
        self._rfftn_batch(self._batch_rot_mask2, self._batch_mask2_ft)
        self.conj_multiply_kernel(self._batch_mask2_ft, self.vars_ft.target2, self._batch_ave2_ft)
        self._irfftn_batch(self._batch_ave2_ft, self._batch_ave2)

        # Batched equivalent of Correlator.compute_lcc_score_and_take_best().
        # Per-voxel batch reduction: updates vars.lcc and vars.rot in-place.
        block = 256
        grid = (self._volume_size + block - 1) // block
        self._batch_lcc_kernel(
            (grid,),
            (block,),
            (
                self._batch_gcc,
                self._batch_ave,
                self._batch_ave2,
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

    def scan(self, progress: ProgressFactory | None):
        n_rot = self.rotations.shape[0]
        B = self.batch_size
        logger.info(f"Batching {n_rot} rotations into {n_rot // B} batches. Batch size: {B}.")
        n_full = n_rot // B
        tail_start = n_full * B

        with self.cuda_stream:
            self.vars.lcc.fill(0)
            self.vars.rot.fill(0)

            for chunk in range(n_full):
                base = chunk * B
                self._compute_batch(base, B, self.rotations[base : base + B])

            # Tail: fewer than B rotations remain — use the single-rotation path.
            for n in range(tail_start, n_rot):
                self.compute_rotation(n, self.rotations[n])

        self.retrieve_results()

import warnings
from contextlib import nullcontext

import cupy as cp
import numpy as np
from pyvkfft import fft as vkfft_fft
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


def build_cuda_ffts(cuda_stream=None):
    if getattr(vkfft_fft, "has_cupy", False):
        return (
            lambda src, dst: vkfft_fft.rfftn(src, dst, cuda_stream=cuda_stream),
            lambda src, dst: vkfft_fft.irfftn(src, dst, cuda_stream=cuda_stream),
        )

    warnings.warn(
        "pyvkfft CUDA backend is unavailable; falling back to cupy.fft for CUDA transforms.",
        stacklevel=2,
    )

    def rfftn(src, dst):
        dst[...] = cp.fft.rfftn(src)

    def irfftn(src, dst):
        dst[...] = cp.fft.irfftn(src, s=dst.shape)

    return rfftn, irfftn


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
        cuda_stream=None,
        laplace: bool = False,
    ):
        self.target: np.ndarray = target / target.max()
        self.laplace = laplace
        self.rotations = rotations
        self.cuda_stream = cuda_stream

        self.vars, self.vars_ft = init_cuda_vars(self.target, self.laplace)

        self.lcc = np.zeros(self.target.shape, dtype=f32)
        self.rot = np.zeros(self.target.shape, dtype=i32)
        self.cuda_kernels = CUDAKernels(self.target.shape)

        self.conj_multiply = lambda a, b, c: cp.multiply(cp.conj(a), b, out=c)
        self.square = lambda a, b: cp.square(a, out=b)
        self.rfftn, self.irfftn = build_cuda_ffts(self.cuda_stream)

        with self._stream_context():
            self.set_template(template, mask)
            self.rfftn(self.vars.target, self.vars_ft.target)
            self.square(self.vars.target, self.vars.target2)
            self.rfftn(self.vars.target2, self.vars_ft.target2)
        self._synchronize()

    def _stream_context(self):
        if self.cuda_stream is None:
            return nullcontext()
        return self.cuda_stream

    def _synchronize(self):
        if self.cuda_stream is not None:
            self.cuda_stream.synchronize()

    def _set_template_var(self, template: np.ndarray):
        self.vars.template = cp.asarray(template.astype(f32))

    def _set_mask_var(self, mask: np.ndarray):
        self.vars.mask = cp.asarray(mask.astype(f32))

    def rotate_grids(self, rotmat: np.ndarray):
        with self._stream_context():
            self.vars.rot_template.fill(0)
            self.vars.rot_mask.fill(0)
            self.cuda_kernels.rotate_image3d(self.vars.template, rotmat.astype(f32), self.vars.rot_template)
            self.cuda_kernels.rotate_image3d(self.vars.mask, rotmat.astype(f32), self.vars.rot_mask, nearest=True)

    def compute_lcc_score_and_take_best(self, n: int):
        self.vars.ave2 *= self.norm_factor
        var = self.vars.ave2 - self.vars.ave**2
        positive_var = var > 0
        denominator = cp.sqrt(cp.where(positive_var, var, 1.0))
        safe_score = self.vars.gcc / denominator
        lcc_scan = cp.where(self.vars.lcc_mask & positive_var, safe_score, 0.0)
        ind = lcc_scan > self.vars.lcc
        self.vars.lcc = cp.where(ind, lcc_scan, self.vars.lcc)
        self.vars.rot = cp.where(ind, np.int32(n), self.vars.rot)

    def retrieve_results(self):
        self._synchronize()
        self.lcc = cp.asnumpy(self.vars.lcc)
        self.rot = cp.asnumpy(self.vars.rot)

    def scan(self, progress: ProgressFactory | None):
        with self._stream_context():
            self.vars.lcc.fill(0)
            self.vars.rot.fill(0)

            scan_range = range(self.rotations.shape[0])
            if progress is not None:
                scan_range = progress(scan_range)

            for n in scan_range:
                self.compute_rotation(n, self.rotations[n])
                if progress is not None:
                    self._synchronize()
        self.retrieve_results()

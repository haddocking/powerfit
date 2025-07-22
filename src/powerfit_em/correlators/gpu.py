from dataclasses import dataclass
from functools import partial
from pathlib import Path
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.array import Array as ClArray
from pyopencl.elementwise import ElementwiseKernel
from pyvkfft.fft import rfftn, irfftn
from tqdm import tqdm
from scipy.ndimage import laplace

from string import Template

import os


f32 = np.float32
i32 = np.int32

@dataclass
class GPUVars:
    target: ClArray
    template: ClArray
    mask: ClArray
    lcc_mask: ClArray
    target2: ClArray
    rot_template: ClArray
    rot_mask: ClArray
    rot_mask2: ClArray
    gcc: ClArray
    ave: ClArray
    ave2: ClArray
    lcc: ClArray
    rot: ClArray


@dataclass
class GPUVarsFT:
    """Fourier transformed (complex) arrays."""
    target: ClArray
    target2: ClArray
    template: ClArray
    mask: ClArray
    mask2: ClArray
    ave: ClArray
    ave2: ClArray
    lcc: ClArray
    gcc: ClArray


def laplace_filter(array):
    """Laplace transform"""
    return laplace(array, mode='wrap')


def get_lcc_mask(target: np.ndarray) -> np.ndarray:
    return (target > target.max() * 0.05).astype(np.uint8)


def get_ft_shape(target: np.ndarray) -> tuple:
    """Returns shape of fourier transformed target."""
    return target.shape[:-1] + (target.shape[-1] // 2 + 1,)


def init_gpu_vars(
    queue: cl.CommandQueue, target: np.ndarray, template: np.ndarray, mask: np.ndarray
):
    lcc_mask = get_lcc_mask(target)
    zeros = np.zeros(target.shape, f32)
    gpu_vars = GPUVars(
        target = cl_array.to_device(queue, target.astype(f32)),
        template = cl.image_from_array(queue.context, template.astype(f32)),
        mask = cl.image_from_array(queue.context, mask.astype(f32)),
        lcc_mask = cl_array.to_device(queue, lcc_mask.astype(np.int32)),
        target2 = cl_array.to_device(queue, zeros),
        rot_template = cl_array.to_device(queue, zeros),
        rot_mask = cl_array.to_device(queue, zeros),
        rot_mask2 = cl_array.to_device(queue, zeros),
        gcc = cl_array.to_device(queue, zeros),
        ave = cl_array.to_device(queue, zeros),
        ave2 = cl_array.to_device(queue, zeros),
        lcc = cl_array.to_device(queue, zeros),
        rot = cl_array.to_device(queue, np.zeros(target.shape, i32)),
    )
    zeros_ft = np.zeros(get_ft_shape(target), dtype=np.complex64)
    gpu_vars_ft = GPUVarsFT(
        target = cl_array.to_device(queue, zeros_ft),
        target2 = cl_array.to_device(queue, zeros_ft),
        template = cl_array.to_device(queue, zeros_ft),
        mask = cl_array.to_device(queue, zeros_ft),
        mask2 = cl_array.to_device(queue, zeros_ft),
        ave = cl_array.to_device(queue, zeros_ft),
        ave2 = cl_array.to_device(queue, zeros_ft),
        lcc = cl_array.to_device(queue, zeros_ft),
        gcc = cl_array.to_device(queue, zeros_ft),
    )
    return gpu_vars, gpu_vars_ft


def generate_kernels(queue: cl.CommandQueue, shape: tuple, rmax: int):
    kernel_values = {
        'shape_x': shape[2],
        'shape_y': shape[1],
        'shape_z': shape[0],
        'llength': rmax,
    }
    return CLKernels(queue.context, kernel_values)


class GPUCorrelator:
    """
    
    Glossary
    ========
    Template; the template structure that you want to fit in the target density.
        Should have been regridded to the same grid as the target density
    
    Mask; core-weighted mask. See ref. paper Figure 1.

    Target: ...

    LCC: local cross-correlation.

    """
    def __init__(self,
        target: np.ndarray,
        rotations: np.ndarray,
        template: np.ndarray,
        mask: np.ndarray,
        queue: cl.CommandQueue,
        laplace: bool = False
    ):
        if template.shape != target.shape:
            raise ValueError("Shape of template does not match the target.")
        mask = mask.copy()
        template = template.copy()
        target = target / target.max()

        self._queue = queue
        self._rmax = min(target.shape) // 2
        # Set rotations;
        rotations = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
        self._rotations = np.zeros((rotations.shape[0], 16), dtype=np.float32)
        self._rotations[:, :9] = rotations.reshape(-1, 9)

        ## Transform mask and template;
        ind = mask != 0

        self._norm_factor = np.float32(ind.sum())
        if self._norm_factor == 0:
            raise ValueError('Zero-filled mask is not allowed.')

        if laplace:
            template = laplace_filter(template)

        template *= mask
        # normalize template;
        template[ind] -= template[ind].mean()
        template[ind] /= template[ind].std()
        # multiply again for core-weighted correlation score
        template *= mask

        self.cpu_vars: dict[str, np.ndarray] = {}

        if laplace:
            self._target = laplace_filter(target)
        else:
            self._target = target

        self.gpu_vars, self.gpu_vars_ft = init_gpu_vars(queue, target, template, mask)

        self._lcc = np.zeros(target.shape, dtype=np.float32)
        self._rot = np.zeros(target.shape, dtype=np.int32)

        self._k = generate_kernels(queue, target.shape, self._rmax)

        # Do some one-time precalculations
        self.gpu_vars_ft.target = rfftn(self.gpu_vars.target)
        self._k.multiply(self.gpu_vars.target, self.gpu_vars.target, self.gpu_vars.target2)
        rfftn(self.gpu_vars.target2, self.gpu_vars_ft.target2)

    @property
    def queue(self) -> cl.CommandQueue:
        return self._queue

    def rotate_grids(self, rotmat: np.ndarray):
        self._k.rotate_image3d(self.queue, self.gpu_vars.template, rotmat,
                self.gpu_vars.rot_template)
        self._k.rotate_image3d(self.queue, self.gpu_vars.mask, rotmat,
                self.gpu_vars.rot_mask, nearest=True)

    def compute_gcc(self):
        """Compute the global cross-correlation.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 3."""
        rfftn(self.gpu_vars.rot_template, self.gpu_vars_ft.template)
        self._k.conj_multiply(
            self.gpu_vars_ft.template,
            self.gpu_vars_ft.target,
            self.gpu_vars_ft.gcc
        )
        irfftn(self.gpu_vars_ft.gcc, self.gpu_vars.gcc)

    def compute_sq_avg_density(self):
        """Compute the square of the average core-weighted density.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 4."""
        rfftn(self.gpu_vars.rot_mask, self.gpu_vars_ft.mask)
        self._k.conj_multiply(
            self.gpu_vars_ft.mask,
            self.gpu_vars_ft.target,
            self.gpu_vars_ft.ave
        )
        irfftn(self.gpu_vars_ft.ave, self.gpu_vars.ave)

    def compute_avg_sq_density(self):
        """Compute the average of the squared core-weighted density.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 5."""
        self._k.multiply(self.gpu_vars.rot_mask, self.gpu_vars.rot_mask, self.gpu_vars.rot_mask2)
        rfftn(self.gpu_vars.rot_mask2, self.gpu_vars_ft.mask2)
        self._k.conj_multiply(self.gpu_vars_ft.mask2, self.gpu_vars_ft.target2, self.gpu_vars_ft.ave2)
        irfftn(self.gpu_vars_ft.ave2, self.gpu_vars.ave2)

    def compute_lcc_score_and_take_best(self, n: int):
        """Compute the LCC score and store best result.
        
        Args:
            n: iteration number.
        """
        self._k.calc_lcc_and_take_best(
            self.gpu_vars.gcc,
            self.gpu_vars.ave,
            self.gpu_vars.ave2,
            self.gpu_vars.lcc_mask,
            self._norm_factor,
            np.int32(n),
            self.gpu_vars.lcc,
            self.gpu_vars.rot,
        )

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

    def retrieve_results(self):
        self.gpu_vars.lcc.get(ary=self._lcc)
        self.gpu_vars.rot.get(ary=self._rot)

    def scan(self, progress: partial[tqdm] = lambda x: x):
        self.gpu_vars.lcc.fill(0)
        self.gpu_vars.rot.fill(0)

        for n in progress(range(0, self._rotations.shape[0])):
            self.compute_rotation(n, self._rotations[n])
            self.queue.finish() # only necessary if we want to track it/s accuratly

        self.retrieve_results()
        self.queue.finish()


class CLKernels(object):
    def __init__(self, ctx, values):
        self.sampler_nearest = cl.Sampler(ctx, True,
                cl.addressing_mode.REPEAT, cl.filter_mode.NEAREST)
        self.sampler_linear = cl.Sampler(ctx, True,
                cl.addressing_mode.REPEAT, cl.filter_mode.LINEAR)
        self.multiply = ElementwiseKernel(ctx,
                "float *x, float *y, float *z",
                "z[i] = x[i] * y[i];"
                )
        self.conj_multiply = ElementwiseKernel(ctx,
                "cfloat_t *x, cfloat_t *y, cfloat_t *z",
                "z[i] = cfloat_mul(cfloat_conj(x[i]), y[i]);"
                )
        self.calc_lcc_and_take_best = ElementwiseKernel(ctx,
                """float *gcc, float *ave, float *ave2, int *mask,
                    float norm_factor, int nrot, float *lcc, int *grot""",
                """float _lcc;
                    if (mask[i] > 0) {
                        _lcc = gcc[i] / sqrt(ave2[i] * norm_factor - ave[i] * ave[i]);
                        if (_lcc > lcc[i]) {
                            lcc[i] = _lcc;
                            grot[i] = nrot;
                        };
                    };
                """
                )

        kernel_file = Path(__file__).parent.parent / "kernels.cl"
        with kernel_file.open() as f:
            t = Template(f.read()).substitute(**values)

        self._program = cl.Program(ctx, t).build()
        self._rotate_image3d = self._program.rotate_image3d
        self._gws_rotate_grid3d = (96, 64, 1)

    def rotate_grid3d(
        self,
        queue: cl.CommandQueue,
        grid:ClArray,
        rotmat: np.ndarray,
        out: ClArray,
        nearest: bool = False
    ):
        args = (grid.data, rotmat, out.data, np.int32(nearest))
        self._program.rotate_grid3d(queue, self._gws_rotate_grid3d, None, *args)

    def rotate_image3d(self, queue: cl.CommandQueue, image, rotmat, out, nearest=False):
        if nearest:
            args = (image, self.sampler_nearest, rotmat, out.data)
        else:
            args = (image, self.sampler_linear, rotmat, out.data)
        self._rotate_image3d(queue, self._gws_rotate_grid3d, None, *args)

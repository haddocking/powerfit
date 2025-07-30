from dataclasses import dataclass
from functools import partial
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.array import Array as ClArray
from pyopencl import Image
from pyvkfft.fft import rfftn, irfftn
from tqdm import tqdm
from scipy.ndimage import laplace as laplace_filter


from powerfit_em.correlators.clkernels import CLKernels
from powerfit_em.correlators.shared import Vars, VarsFT, get_ft_shape, get_lcc_mask, get_normalization_factor, normalize_template, f32, i32


def init_gpu_vars(
    queue: cl.CommandQueue, target: np.ndarray, mask: np.ndarray, laplace: bool,
) -> tuple[Vars[ClArray, Image], VarsFT[ClArray]]:
    """Initialize all GPU variables on the specified queue."""
    lcc_mask = get_lcc_mask(target)
    _t = laplace_filter(target, mode='wrap') if laplace else target
    zeros = np.zeros(target.shape, f32)
    gpu_vars = Vars(
        target = cl_array.to_device(queue, _t.astype(f32)),
        template = cl.image_from_array(queue.context, zeros),  # template is set through separate method
        mask = cl.image_from_array(queue.context, mask.astype(f32)),
        lcc_mask = cl_array.to_device(queue, lcc_mask.astype(i32)),
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
    gpu_vars_ft = VarsFT(
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


def generate_kernels(queue: cl.CommandQueue, target: np.ndarray):
    """Generate the custom OpenCL kernels based on the target's shape"""
    kernel_values = {
        'shape_x': target.shape[2],
        'shape_y': target.shape[1],
        'shape_z': target.shape[0],
        'llength': i32(min(target.shape) // 2),
    }
    return CLKernels(queue.context, kernel_values)


def precompute_squared_targets(gpu_vars: Vars[ClArray, Image], gpu_vars_ft: VarsFT[ClArray], kernels: CLKernels):
    """Compute the squared target and fourier transformed target on GPU for reuse."""
    gpu_vars_ft.target = rfftn(gpu_vars.target)
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


class GPUCorrelator:
    """Compute the LCC score for a target and template combination."""
    def __init__(
        self,
        target: np.ndarray,
        template: np.ndarray,
        rotations: np.ndarray,
        mask: np.ndarray,
        queue: cl.CommandQueue,
        laplace: bool = False,
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
        """
        self.target: np.ndarray = target / target.max()
        self.laplace = laplace
        self.mask = mask
        self.queue = queue

        self._rotations = transform_rotations(rotations)

        # Precompute the normalization factor for use in the LCC computing kernel
        self.norm_factor = get_normalization_factor(mask)

        self.vars, self.vars_ft = init_gpu_vars(queue, self.target, mask, self.laplace)

        self.lcc = np.zeros(self.target.shape, dtype=np.float32)
        self.rot = np.zeros(self.target.shape, dtype=np.int32)

        self.set_template(template)

        self.cl_kernels = generate_kernels(queue, self.target)
        precompute_squared_targets(self.vars, self.vars_ft, self.cl_kernels)

    def set_template(self, template: np.ndarray):
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
        
        template = normalize_template(template, self.mask)
        self.vars.template = cl.image_from_array(self.vars.template.context, template.astype(f32))

        # Reset lcc and rot values after (re)setting the template
        self.lcc[:] = 0.0
        self.rot[:] = 0

    def rotate_grids(self, rotmat: np.ndarray):
        self.cl_kernels.rotate_image3d(self.queue, self.vars.template, rotmat,
                self.vars.rot_template)
        self.cl_kernels.rotate_image3d(self.queue, self.vars.mask, rotmat,
                self.vars.rot_mask, nearest=True)

    def compute_gcc(self):
        """Compute the global cross-correlation.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 3."""
        rfftn(self.vars.rot_template, self.vars_ft.template)
        self.cl_kernels.conj_multiply(
            self.vars_ft.template,
            self.vars_ft.target,
            self.vars_ft.gcc
        )
        irfftn(self.vars_ft.gcc, self.vars.gcc)

    def compute_sq_avg_density(self):
        """Compute the square of the average core-weighted density.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 4."""
        rfftn(self.vars.rot_mask, self.vars_ft.mask)
        self.cl_kernels.conj_multiply(
            self.vars_ft.mask,
            self.vars_ft.target,
            self.vars_ft.ave
        )
        irfftn(self.vars_ft.ave, self.vars.ave)

    def compute_avg_sq_density(self):
        """Compute the average of the squared core-weighted density.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 5."""
        self.cl_kernels.multiply(self.vars.rot_mask, self.vars.rot_mask, self.vars.rot_mask2)
        rfftn(self.vars.rot_mask2, self.vars_ft.mask2)
        self.cl_kernels.conj_multiply(self.vars_ft.mask2, self.vars_ft.target2, self.vars_ft.ave2)
        irfftn(self.vars_ft.ave2, self.vars.ave2)

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
        """Retrieve the results from the GPU."""
        self.vars.lcc.get(ary=self.lcc)
        self.vars.rot.get(ary=self.rot)

    def scan(self, progress: partial[tqdm] = lambda x: x):
        """Scan all provided rotations to find the best fit."""
        self.vars.lcc.fill(0)
        self.vars.rot.fill(0)

        for n in progress(range(0, self._rotations.shape[0])):
            self.compute_rotation(n, self._rotations[n])
            self.queue.finish() # only necessary if we want to track it/s accuratly

        self.retrieve_results()
        self.queue.finish()

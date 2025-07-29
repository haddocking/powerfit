from dataclasses import dataclass
from functools import partial
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.array import Array as ClArray
from pyvkfft.fft import rfftn, irfftn
from tqdm import tqdm
from scipy.ndimage import laplace as laplace_filter


from powerfit_em.correlators.clkernels import CLKernels
from powerfit_em.correlators.shared import get_lcc_mask


f32 = np.float32
i32 = np.int32

@dataclass
class GPUVars:
    """Non-complex GPU arrays."""
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



def get_ft_shape(target: np.ndarray) -> tuple:
    """Returns shape of fourier transformed target."""
    return target.shape[:-1] + (target.shape[-1] // 2 + 1,)


def init_gpu_vars(
    queue: cl.CommandQueue, target: np.ndarray, mask: np.ndarray, laplace: bool,
):
    """Initialize all GPU variables on the specified queue."""
    lcc_mask = get_lcc_mask(target)
    _t = laplace_filter(target, mode='wrap') if laplace else target
    zeros = np.zeros(target.shape, f32)
    gpu_vars = GPUVars(
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


def generate_kernels(queue: cl.CommandQueue, target: np.ndarray):
    """Generate the custom OpenCL kernels based on the target's shape"""
    kernel_values = {
        'shape_x': target.shape[2],
        'shape_y': target.shape[1],
        'shape_z': target.shape[0],
        'llength': i32(min(target.shape) // 2),
    }
    return CLKernels(queue.context, kernel_values)


def normalize_template(template: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalize the template structure cf. A.M. Roseman (2000)."""
    norm_template = template * mask
    # normalize template;
    ind = mask != 0
    norm_template[ind] -= norm_template[ind].mean()
    norm_template[ind] /= norm_template[ind].std()
    # multiply again for core-weighted correlation score
    return norm_template * mask


def precompute_squared_targets(gpu_vars: GPUVars, gpu_vars_ft: GPUVarsFT, kernels: CLKernels):
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
        self._queue = queue

        self._rotations = transform_rotations(rotations)

        # Precompute the normalization factor for use in the LCC computing kernel
        self._norm_factor = np.float32((mask != 0).sum())
        if self._norm_factor == 0:
            raise ValueError('Zero-filled mask is not allowed.')

        self.gpu_vars, self.gpu_vars_ft = init_gpu_vars(queue, self.target, mask, self.laplace)

        self.lcc = np.zeros(self.target.shape, dtype=np.float32)
        self.rot = np.zeros(self.target.shape, dtype=np.int32)

        self.set_template(template)

        self.cl_kernels = generate_kernels(queue, self.target)
        precompute_squared_targets(self.gpu_vars, self.gpu_vars_ft, self.cl_kernels)

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
        self.gpu_vars.template = cl.image_from_array(self.gpu_vars.template.context, template.astype(f32))

        # Reset lcc and rot values after (re)setting the template
        self.lcc[:] = 0.0
        self.rot[:] = 0

    @property
    def queue(self) -> cl.CommandQueue:
        return self._queue

    def rotate_grids(self, rotmat: np.ndarray):
        self.cl_kernels.rotate_image3d(self.queue, self.gpu_vars.template, rotmat,
                self.gpu_vars.rot_template)
        self.cl_kernels.rotate_image3d(self.queue, self.gpu_vars.mask, rotmat,
                self.gpu_vars.rot_mask, nearest=True)

    def compute_gcc(self):
        """Compute the global cross-correlation.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 3."""
        rfftn(self.gpu_vars.rot_template, self.gpu_vars_ft.template)
        self.cl_kernels.conj_multiply(
            self.gpu_vars_ft.template,
            self.gpu_vars_ft.target,
            self.gpu_vars_ft.gcc
        )
        irfftn(self.gpu_vars_ft.gcc, self.gpu_vars.gcc)

    def compute_sq_avg_density(self):
        """Compute the square of the average core-weighted density.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 4."""
        rfftn(self.gpu_vars.rot_mask, self.gpu_vars_ft.mask)
        self.cl_kernels.conj_multiply(
            self.gpu_vars_ft.mask,
            self.gpu_vars_ft.target,
            self.gpu_vars_ft.ave
        )
        irfftn(self.gpu_vars_ft.ave, self.gpu_vars.ave)

    def compute_avg_sq_density(self):
        """Compute the average of the squared core-weighted density.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 5."""
        self.cl_kernels.multiply(self.gpu_vars.rot_mask, self.gpu_vars.rot_mask, self.gpu_vars.rot_mask2)
        rfftn(self.gpu_vars.rot_mask2, self.gpu_vars_ft.mask2)
        self.cl_kernels.conj_multiply(self.gpu_vars_ft.mask2, self.gpu_vars_ft.target2, self.gpu_vars_ft.ave2)
        irfftn(self.gpu_vars_ft.ave2, self.gpu_vars.ave2)

    def compute_lcc_score_and_take_best(self, n: int):
        """Compute the LCC score and store best result.
        
        Args:
            n: iteration number.
        """
        self.cl_kernels.calc_lcc_and_take_best(
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
        """Retrieve the results from the GPU."""
        self.gpu_vars.lcc.get(ary=self.lcc)
        self.gpu_vars.rot.get(ary=self.rot)

    def scan(self, progress: partial[tqdm] = lambda x: x):
        """Scan all provided rotations to find the best fit."""
        self.gpu_vars.lcc.fill(0)
        self.gpu_vars.rot.fill(0)

        for n in progress(range(0, self._rotations.shape[0])):
            self.compute_rotation(n, self._rotations[n])
            self.queue.finish() # only necessary if we want to track it/s accuratly

        self.retrieve_results()
        self.queue.finish()

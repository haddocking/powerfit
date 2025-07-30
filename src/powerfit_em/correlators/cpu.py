from functools import partial
import numpy as np
from scipy.ndimage import laplace as laplace_filter
import warnings

from tqdm import tqdm

from powerfit_em.correlators.shared import Vars, VarsFT, get_ft_shape, get_lcc_mask, get_normalization_factor, normalize_template, f32, i32

try:
    from pyfftw.builders import rfftn as rfftn_builder, irfftn as irfftn_builder
    from pyfftw import zeros_aligned, simd_alignment
    PYFFTW = True
except ImportError:
    PYFFTW = False

from powerfit_em._extensions import rotate_grid3d


def build_ffts(target: np.ndarray, gcc: np.ndarray, ft_gcc: np.ndarray, fftw: bool):
    """Build the FFTs (in case of pyfftw), or patch the numpy fft interface to resemble pyfftw."""
    if fftw:
        rfftn = rfftn_builder(gcc)
        irfftn = irfftn_builder(ft_gcc, s=target.shape)
    else:
        warnings.warn("Using numpy for calculating score. Install pyFFTW for faster calculation.")
        def rfftn(src: np.ndarray, dst: np.ndarray):
            np.fft.rfftn(src, out=dst)
        def irfftn(src: np.ndarray, dst: np.ndarray):
            np.fft.irfft(src, out=dst)
    return rfftn, irfftn


def rmax(target: np.ndarray) -> int:
    return (min(target.shape) // 2)


def zeros_array(shape: tuple[int], dtype: np.dtype, fftw: bool) -> np.ndarray:
    """Returns optimally SIMD aligned array if PyFFTW is used, for faster computation."""
    if fftw:
        return zeros_aligned(shape, dtype, n=simd_alignment)
    else:
        return np.zeros(shape, dtype)


def init_cpu_vars(
    target: np.ndarray, mask: np.ndarray, laplace: bool, fftw: bool,
)-> tuple[Vars[np.ndarray, np.ndarray], VarsFT[np.ndarray]]:
    """Initialize all CPU variables on the specified queue."""

    lcc_mask = get_lcc_mask(target)
    _t = laplace_filter(target, mode='wrap') if laplace else target
    
    vars = Vars(
        target = _t.astype(f32),
        template = zeros_array(target.shape, f32, fftw),
        mask = mask.astype(f32),
        lcc_mask = lcc_mask.astype(np.uint8),
        target2 = zeros_array(target.shape, f32, fftw),
        rot_template = zeros_array(target.shape, f32, fftw),
        rot_mask = zeros_array(target.shape, f32, fftw),
        rot_mask2 = zeros_array(target.shape, f32, fftw),
        gcc = zeros_array(target.shape, f32, fftw),
        ave = zeros_array(target.shape, f32, fftw),
        ave2 = zeros_array(target.shape, f32, fftw),
        lcc = zeros_array(target.shape, f32, fftw),
        rot = zeros_array(target.shape, i32, fftw),
    )

    vars_ft = VarsFT(
        target = zeros_array(get_ft_shape(target), np.complex64, fftw),
        target2 = zeros_array(get_ft_shape(target), np.complex64, fftw),
        template = zeros_array(get_ft_shape(target), np.complex64, fftw),
        mask = zeros_array(get_ft_shape(target), np.complex64, fftw),
        mask2 = zeros_array(get_ft_shape(target), np.complex64, fftw),
        ave = zeros_array(get_ft_shape(target), np.complex64, fftw),
        ave2 = zeros_array(get_ft_shape(target), np.complex64, fftw),
        lcc = zeros_array(get_ft_shape(target), np.complex64, fftw),
        gcc = zeros_array(get_ft_shape(target), np.complex64, fftw),
    )
    return vars, vars_ft


class CPUCorrelator:
    """Compute the LCC score for a target and template combination."""
    def __init__(
        self,
        target: np.ndarray,
        template: np.ndarray,
        rotations: np.ndarray,
        mask: np.ndarray,
        laplace: bool = False,
        fftw: bool = True,
    ):
        """Initialize the CPU correlator.

        Args:
            target: the target density on which you want to fit a template structure.
            template: the template structure that you want to fit in the target density,
                should have been regridded to the same grid as the target density.
            rotations: array of 3D-rotation matrices, of shape (n_rotations, 3, 3).
                for each rotation the local cross correlation is computed for every
                possible translation (in parallel using Fourier transforms).
            mask: core-weighted mask. See doi:10.3934/biophy.2015.2.73, Figure 1.
            laplace: if true, a Laplace pre-filter is applied to the target density and
                template to enhance the sensitivity of the scoring function.
            fftw: if true, the PyFFTW library will be used to compute the fourier transforms.
        """
        self.target: np.ndarray = target / target.max()
        self.laplace = laplace
        self.mask = mask
        self.rotations = rotations
        self.norm_factor = get_normalization_factor(mask)

        self.vars, self.vars_ft = init_cpu_vars(self.target, mask, self.laplace, fftw)
    
        self.lcc_scan = np.zeros(self.target.shape, dtype=f32)
        self.lcc = np.zeros(self.target.shape, dtype=f32)
        self.rot = np.zeros(self.target.shape, dtype=i32)

        self.set_template(template)

        self.rfftn, self.irfftn = build_ffts(self.target, self.vars.gcc, self.vars_ft.gcc, fftw)
        # pre-calculate the FFTs of the target
        self.rfftn(self.vars.target, self.vars_ft.target)
        self.rfftn(self.vars.target**2, self.vars_ft.target2)

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
        self.vars.template = template.astype(f32)

        # Reset lcc and rot values after (re)setting the template
        self.lcc[:] = 0.0
        self.rot[:] = 0

    def rotate_grids(self, rotmat: np.ndarray):
        """Rotate template and mask."""
        rotate_grid3d(
            self.vars.template, rotmat.astype(f32), rmax(self.target),
            self.vars.rot_template, False
        )
        rotate_grid3d(
            self.vars.mask, rotmat.astype(f32), rmax(self.target),
            self.vars.rot_mask, True
        )

    def compute_gcc(self):
        """Compute the global cross-correlation.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 3."""
        self.rfftn(self.vars.rot_template, self.vars_ft.template)
        self.vars_ft.gcc[:] = np.multiply(np.conjugate(self.vars_ft.template), self.vars_ft.target)
        self.irfftn(self.vars_ft.gcc, self.vars.gcc)

    def compute_sq_avg_density(self):
        """Compute the square of the average core-weighted density.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 4."""
        self.rfftn(self.vars.rot_mask, self.vars_ft.mask)
        self.vars_ft.ave[:] = np.multiply(np.conjugate(self.vars_ft.mask), self.vars_ft.target)
        self.irfftn(self.vars_ft.ave, self.vars.ave)

    def compute_avg_sq_density(self):
        """Compute the average of the squared core-weighted density.
        
        Ref doi:10.3934/biophy.2015.2.73. Equation 5."""
        self.vars.rot_mask2[:] = self.vars.rot_mask ** 2
        self.rfftn(self.vars.rot_mask2, self.vars_ft.mask2)
        self.vars_ft.ave2[:] = np.multiply(np.conjugate(self.vars_ft.mask2), self.vars_ft.target2)
        self.irfftn(self.vars_ft.ave2, self.vars.ave2)

    def compute_lcc_score_and_take_best(self, n: int):
        """Compute the LCC score and store best result.
        
        Args:
            n: iteration number.
        """
        self.vars.ave2 *= self.norm_factor

        var = self.vars.ave2 - self.vars.ave**2
        self.lcc_scan = np.where(self.vars.lcc_mask, self.vars.gcc / np.sqrt(var), 0.0)
        ind = np.greater(self.lcc_scan, self.lcc)
        # store lcc and rotation index
        self.lcc[ind] = self.lcc_scan[ind]
        self.rot[ind] = n

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

    def scan(self, progress: partial[tqdm] = lambda x: x):
        """Scan all provided rotations to find the best fit."""
        self.vars.lcc.fill(0)
        self.vars.rot.fill(0)

        for n in progress(range(0, self.rotations.shape[0])):
            self.compute_rotation(n, self.rotations[n])

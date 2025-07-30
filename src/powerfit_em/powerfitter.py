

from functools import partial
from os import remove
from os.path import join, abspath, isdir
import os.path
from multiprocessing import RawValue, Lock, Process, cpu_count
from string import Template
import warnings

import numpy as np
from numpy.fft import irfftn as np_irfftn, rfftn as np_rfftn
from scipy.ndimage import binary_erosion, laplace
from tqdm.auto import tqdm

from powerfit_em.correlators.shared import get_lcc_mask
from powerfit_em.volume import Volume
try:
    from pyfftw import zeros_aligned, simd_alignment
    from pyfftw.builders import rfftn as rfftn_builder, irfftn as irfftn_builder
    PYFFTW = True
except ImportError:
    PYFFTW = False
try:
    import pyopencl as _
    OPENCL = True
except:
    OPENCL = False


from ._powerfit import conj_multiply, calc_lcc, dilate_points
from ._extensions import rotate_grid3d


class _Counter(object):
    """Thread-safe counter object to follow PowerFit progress"""

    def __init__(self):
        self.val = RawValue('i', 0)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


class PowerFitter(object):
    """Wrapper around the Correlator classes for multiprocessing and GPU
    accelerated searches providing an easy interface.
    """

    def __init__(
        self, target: Volume, rotations: np.ndarray, template: Volume, mask: Volume, queues, laplace: bool = False
    ):
        self._target = target
        self._rotations = rotations
        self._template = template
        self._mask = mask
        self._queues = queues
        self._nproc = 1
        self._directory = abspath('./')
        self._laplace = laplace

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, directory):
        if isdir(directory):
            self._directory = abspath(directory)
        else:
            raise ValueError("Directory does not exist.")

    def scan(self, progress: partial[tqdm]):
        if self._queues is None:
            if self._nproc == 1:
                self._single_cpu_scan(progress)
            else:
                self._multi_cpu_scan(progress)
        else:
            self._gpu_scan(progress)

    def _gpu_scan(self, progress: partial[tqdm]):
        if OPENCL:
            from powerfit_em.correlators.gpu import GPUCorrelator
        self._corr = GPUCorrelator(
            self._target.array,
            self._template.array,
            self._rotations,
            self._mask.array,
            self._queues[0],
            self._laplace,
        )
        self._corr.scan(progress)
        self._lcc = self._corr.lcc
        self._rot = self._corr.rot

    def _multi_cpu_scan(self, progress: partial[tqdm]):
        nrot = self._rotations.shape[0]
        self._nrot_per_job = nrot // self._nproc
        processes = []
        self._counter = _Counter()
        self._njobs = self._nproc
        if self._queues is not None:
            self._njobs = len(self._queues)

        for n in range(self._njobs):
            init_rot = n * self._nrot_per_job
            end_rot = init_rot + self._nrot_per_job
            if n == self._njobs - 1:
                end_rot = None
            sub_rotations = self._rotations[init_rot: end_rot]
            processes.append(Process(
                  target=self._run_correlator_instance,
                  args=(self._target, self._template, self._mask,
                        sub_rotations, self._laplace, self._counter, n,
                        self._queues, self._directory)
                  ))

        for n in range(self._njobs):
            processes[n].start()

        with progress(total=nrot) as pbar:
            while self._counter.value() < nrot:
                current_count = self._counter.value()
                pbar.update(current_count - pbar.n)
        
        for n in range(self._njobs):
            processes[n].join()
        self._combine()

    def _single_cpu_scan(self, progress):
        target = self._target
        laplace = self._laplace
        correlator = CPUCorrelator(target.array, laplace=laplace)
        correlator.template = self._template.array
        correlator.mask = self._mask.array
        correlator.rotations = self._rotations
        correlator.scan(progress)
        self._lcc = correlator.lcc
        self._rot = correlator.rot


    @staticmethod
    def _run_correlator_instance(target, template, mask, rotations, laplace,
            counter, jobid, queues, directory):
        correlator = CPUCorrelator(target.array, laplace=laplace)
        correlator.template = template.array
        correlator.mask = mask.array
        correlator.rotations = rotations
        correlator._counter = counter
        correlator.scan(lambda x: x)
        np.save(join(directory, '_lcc_part_{:d}.npy').format(jobid), correlator._lcc)
        np.save(join(directory, '_rot_part_{:d}.npy').format(jobid), correlator._rot)

    def _combine(self):
        # Combine all the intermediate results
        lcc = np.zeros(self._target.shape)
        rot = np.zeros(self._target.shape)
        ind = np.zeros(lcc.shape, dtype=np.bool)
        for n in range(self._njobs):
            lcc_file = join(self._directory, '_lcc_part_{:d}.npy').format(n)
            rot_file = join(self._directory, '_rot_part_{:d}.npy').format(n)
            part_lcc = np.load(lcc_file)
            part_rot = np.load(rot_file)
            np.greater(part_lcc, lcc, ind)
            lcc[ind] = part_lcc[ind]
            # take care of the rotation index offset for each independent job
            rot[ind] = part_rot[ind] + self._nrot_per_job * n
            remove(lcc_file)
            remove(rot_file)
        self._lcc = lcc
        self._rot = rot


class BaseCorrelator(object):
    """Base class that calculates the local cross-correlation"""

    def __init__(self, target, laplace=False):
        self._target = target / target.max()
        self._rotations = None
        self._template = None
        self._mask = None
        self._laplace = laplace
        self._lcc_mask = get_lcc_mask(self._target)
        self._rmax = min(target.shape) // 2

    @property
    def target(self):
        return self._target

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        if self._template is None:
            raise ValueError("First set the template.")
        if self._target.shape != mask.shape:
            raise ValueError("Shape of the mask is different from target.")

        ind = mask != 0
        # remember the normalization factor for the cross-correlation
        self._norm_factor = ind.sum()
        # If mask is only zeros, raise error
        if self._norm_factor == 0:
            raise ValueError('Zero-filled mask is not allowed.')

        self._mask = mask.copy()
        if self._laplace:
            self._template = self._laplace_filter(self._template)
        self._template *= self._mask
        self._normalize_template(ind)
        # multiply again for core-weighted correlation score
        self._template *= self._mask

    @staticmethod
    def _laplace_filter(array):
        """Laplace transform"""
        return laplace(array, mode='wrap')

    def _normalize_template(self, ind):
        # normalize the template over the mask
        self._template[ind] -= self._template[ind].mean()
        self._template[ind] /= self._template[ind].std()

    @property
    def rotations(self):
        return self._rotations

    @rotations.setter
    def rotations(self, rotations):
        """Set the rotations that will be sampled."""
        rotations = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
        self._rotations = rotations

    @property
    def template(self):
        return self._template

    @template.setter
    def template(self, template):
        if template.shape != self._target.shape:
            raise ValueError("Shape of template does not match the target.")

        # reset the mask
        self._mask = None
        self._template = template.copy()

    @property
    def lcc(self):
        return self._lcc

    @property
    def rot(self):
        return self._rot

    def scan(self):
        if any([req is None for req in (self._template, self._mask, self._rotations)]):
            raise ValueError("First set the template, mask, and rotations.")


class CPUCorrelator(BaseCorrelator):

    """CPU implementation for calculating the local cross-correlation."""

    def __init__(self, target, laplace=False, fftw=True):
        super(CPUCorrelator, self).__init__(target, laplace=laplace)
        self._fftw = PYFFTW and fftw
        self._allocate_arrays(self._target.shape)
        self._build_ffts()

        target = self._target
        if self._laplace:
            target = self._laplace_filter(self._target)
        # pre-calculate the FFTs of the target
        if self._fftw:
            self._rfftn(target, self._ft_target)
            self._rfftn(target**2, self._ft_target2)
        else:
            self._ft_target = self._rfftn(target)
            self._ft_target2 = self._rfftn(target**2)


    def _allocate_arrays(self, shape):
        # allocate all the required arrays
        # real arrays
        arrays = '_rot_template _rot_mask _rot_mask2 _gcc _ave _ave2 _lcc_scan _lcc _rot'.split()
        for arr in arrays:
            setattr(self, arr, self._allocate_array(shape, np.float64, self._fftw))
        self._ind = np.zeros(shape, dtype=np.bool)

        # complex arrays
        self._ft_shape = self._get_ft_shape(shape)
        arrays = '_target _target2 _template _mask _mask2 _gcc _ave _ave2'.split()
        for arr in arrays:
            setattr(self, '_ft' + arr,
                    self._allocate_array(self._ft_shape, np.complex128, self._fftw))

    @staticmethod
    def _allocate_array(shape, dtype, fftw):
        if fftw:
            return zeros_aligned(shape, dtype=dtype, n=simd_alignment)
        else:
            return np.zeros(shape, dtype)

    @staticmethod
    def _get_ft_shape(shape):
        return list(shape[:-1]) + [shape[-1] // 2 + 1]

    def _build_ffts(self):
        # build the ffts
        if self._fftw:
            self._rfftn = rfftn_builder(self._gcc)
            self._irfftn = irfftn_builder(self._ft_gcc, s=self._target.shape)
        else:
            warnings.warn("Using numpy for calculating score. Install pyFFTW for faster calculation.")
            # monkey patch the numpy fft interface
            self._rfftn = np_rfftn
            self._irfftn = np_irfftn

    def scan(self, progress):
        super(CPUCorrelator, self).scan()

        self._lcc.fill(0)
        self._rot.fill(0)

        for n in progress(range(self._rotations.shape[0])):
            # rotate template and mask
            self._translational_scan(self._rotations[n])
            # get the indices where the scanned lcc is greater
            np.greater(self._lcc_scan, self._lcc, self._ind)
            # remember lcc and rotation index
            self._lcc[self._ind] = self._lcc_scan[self._ind]
            self._rot[self._ind] = n

            if hasattr(self, '_counter'):
                self._counter.increment()

    def _translational_scan(self, rotmat):
        self._rotate_grids(rotmat)
        self._get_lcc()

    def _rotate_grids(self, rotmat):
        rotate_grid3d(
              self._template, rotmat, self._rmax,
              self._rot_template, False
              )
        rotate_grid3d(
              self._mask, rotmat, self._rmax,
              self._rot_mask, True
              )

    def _get_lcc(self):
        np.multiply(self._rot_mask, self._rot_mask, self._rot_mask2)

        self._forward_ffts()

        conj_multiply(
              self._ft_template.ravel(), self._ft_target.ravel(),
              self._ft_gcc.ravel()
              )
        conj_multiply(
              self._ft_mask.ravel(), self._ft_target.ravel(),
              self._ft_ave.ravel()
              )
        conj_multiply(
              self._ft_mask2.ravel(), self._ft_target2.ravel(),
              self._ft_ave2.ravel()
              )

        self._backward_ffts()

        self._ave2 *= self._norm_factor
        calc_lcc(
              self._gcc.ravel(), self._ave.ravel(), self._ave2.ravel(),
              self._lcc_mask.ravel(), self._lcc_scan.ravel()
              )

    def _forward_ffts(self):
        if self._fftw:
            self._rfftn(self._rot_template, self._ft_template)
            self._rfftn(self._rot_mask, self._ft_mask)
            self._rfftn(self._rot_mask2, self._ft_mask2)
        else:
            self._ft_template = self._rfftn(self._rot_template)
            self._ft_mask = self._rfftn(self._rot_mask)
            self._ft_mask2 = self._rfftn(self._rot_mask2)

    def _backward_ffts(self):
        if self._fftw:
            self._irfftn(self._ft_gcc, self._gcc)
            self._irfftn(self._ft_ave, self._ave)
            self._irfftn(self._ft_ave2, self._ave2)
        else:
            self._gcc = self._irfftn(self._ft_gcc, s=self.target.shape)
            self._ave = self._irfftn(self._ft_ave, s=self.target.shape)
            self._ave2 = self._irfftn(self._ft_ave2, s=self.target.shape)

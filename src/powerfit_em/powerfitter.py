

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
try:
    from pyfftw import zeros_aligned, simd_alignment
    from pyfftw.builders import rfftn as rfftn_builder, irfftn as irfftn_builder
    PYFFTW = True
except ImportError:
    PYFFTW = False
try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    from pyopencl.elementwise import ElementwiseKernel
    from gpyfft import GpyFFT
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

    def __init__(self, target, laplace=False):
        self._target = target
        self._rotations = None
        self._template = None
        self._mask = None
        self._rotations = None
        self._queues = None
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
        self._corr = GPUCorrelator(self._target.array, self._queues[0],
                laplace=self._laplace)

        self._corr.template = self._template.array
        self._corr.mask = self._mask.array
        self._corr.rotations = self._rotations
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
        correlator.scan()
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
        self._lcc_mask = self._get_lcc_mask(self._target)
        self._rmax = min(target.shape) // 2

    @staticmethod
    def _get_lcc_mask(target):
        return (target > target.max() * 0.05).astype(np.uint8)

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


if OPENCL:
    class GPUCorrelator(BaseCorrelator):

        def __init__(self, target, queue, laplace=False):
            super(GPUCorrelator, self).__init__(target, laplace=laplace)
            self._queue = queue
            self._ctx = self._queue.context
            self._gpu = self._queue.device


            self._allocate_arrays()
            self._build_ffts()
            self._generate_kernels()

            target = self._target
            if self._laplace:
                target = self._laplace_filter(self._target)
            # move some arrays to the GPU
            self._gtarget = cl_array.to_device(self._queue, target.astype(np.float32))
            self._lcc_mask = cl_array.to_device(self._queue,
                    self._lcc_mask.astype(np.int32))
            # Do some one-time precalculations
            self._rfftn(self._gtarget, self._ft_target)
            self._k.multiply(self._gtarget, self._gtarget, self._target2)
            self._rfftn(self._target2, self._ft_target2)

            self._gshape = np.asarray(
                    list(self._target.shape) + [np.prod(self._target.shape)],
                    dtype=np.int32)

        def _allocate_arrays(self):

            # Determine the required shape and size of an array
            self._ft_shape = tuple(
                    [self._target.shape[0] // 2 + 1] + list(self._target.shape[1:])
                    )
            self._shape = self._target.shape

            # Allocate arrays on CPU
            self._lcc = np.zeros(self._target.shape, dtype=np.float32)
            self._rot = np.zeros(self._target.shape, dtype=np.int32)

            # Allocate arrays on GPU
            arrays = '_target2 _rot_template _rot_mask _rot_mask2 _gcc _ave _ave2 _glcc'.split()
            for array in arrays:
                setattr(self, array, 
                        cl_array.zeros( self._queue, self._shape, dtype=np.float32)
                        )
            self._grot = cl_array.zeros(self._queue, self._shape, dtype=np.int32)

            # Allocate all complex arrays
            ft_arrays = 'target target2 template mask mask2 gcc ave ave2 lcc'.split()
            for ft_array in ft_arrays:
                setattr(self, '_ft_' + ft_array, 
                        cl_array.to_device(self._queue,
                            np.zeros(self._ft_shape, dtype=np.complex64))
                        )

        def _build_ffts(self, batch_size=1):
            self._rfftn = grfftn_builder(self._ctx, self._target.shape,
                    batch_size=batch_size)
            self._irfftn = grfftn_builder(self._ctx, self._target.shape,
                    forward=False, batch_size=batch_size)
            self._rfftn.bake(self._queue)
            self._irfftn.bake(self._queue)

        @property
        def mask(self):
            return BaseCorrelator.mask

        @mask.setter
        def mask(self, mask):
            BaseCorrelator.mask.fset(self, mask)
            self._norm_factor = np.float32(self._norm_factor)
            self._rmax = np.int32(self._rmax)
            self._gtemplate = cl.image_from_array(self._queue.context,
                    self._template.astype(np.float32))
            self._gmask = cl.image_from_array(self._queue.context,
                    self._mask.astype(np.float32))

        @property
        def rotations(self):
            return BaseCorrelator.rotations

        @rotations.setter
        def rotations(self, rotations):
            BaseCorrelator.rotations.fset(self, rotations)
            self._cl_rotations = np.zeros((self._rotations.shape[0], 16),
                    dtype=np.float32)
            self._cl_rotations[:, :9] = self._rotations.reshape(-1, 9)

        def _cl_rotate_grids(self, rotmat):
            self._k.rotate_image3d(self._queue, self._gtemplate, rotmat,
                    self._rot_template)
            self._k.rotate_image3d(self._queue, self._gmask, rotmat,
                    self._rot_mask, nearest=True)
            self._queue.finish()

        def _cl_get_gcc(self):
            self._rfftn(self._rot_template, self._ft_template)
            self._k.conj_multiply(self._ft_template, self._ft_target, self._ft_gcc)
            self._irfftn(self._ft_gcc, self._gcc)
            self._queue.finish()

        def _cl_get_ave(self):
            self._rfftn(self._rot_mask, self._ft_mask)
            self._k.conj_multiply(self._ft_mask, self._ft_target, self._ft_ave)
            self._irfftn(self._ft_ave, self._ave)
            self._queue.finish()

        def _cl_get_ave2(self):
            self._k.multiply(self._rot_mask, self._rot_mask, self._rot_mask2)
            self._rfftn(self._rot_mask2, self._ft_mask2)
            self._k.conj_multiply(self._ft_mask2, self._ft_target2, self._ft_ave2)
            self._irfftn(self._ft_ave2, self._ave2)
            self._queue.finish()

        def scan(self, progress: partial[tqdm] = lambda x: x):
            super(GPUCorrelator, self).scan()

            self._glcc.fill(0)
            self._grot.fill(0)
            for n in progress(range(0, self._rotations.shape[0])):

                rotmat = self._cl_rotations[n]

                self._cl_rotate_grids(rotmat)

                self._cl_get_gcc()
                self._cl_get_ave()
                self._cl_get_ave2()

                self._k.calc_lcc_and_take_best(self._gcc, self._ave,
                        self._ave2, self._lcc_mask, self._norm_factor,
                        np.int32(n), self._glcc, self._grot)

                self._queue.finish()

            self._glcc.get(ary=self._lcc)
            self._grot.get(ary=self._rot)
            self._queue.finish()

        def _generate_kernels(self):
            kernel_values = {'shape_x': self._shape[2],
                             'shape_y': self._shape[1],
                             'shape_z': self._shape[0],
                             'llength': self._rmax,
                             }
            self._k = CLKernels(self._ctx, kernel_values)


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

            kernel_file = os.path.join(os.path.dirname(__file__), 'kernels.cl')
            with open(kernel_file) as f:
                t = Template(f.read()).substitute(**values)

            self._program = cl.Program(ctx, t).build()
            self._gws_rotate_grid3d = (96, 64, 1)

        def rotate_grid3d(self, queue, grid, rotmat, out, nearest=False):
            args = (grid.data, rotmat, out.data, np.int32(nearest))
            self._program.rotate_grid3d(queue, self._gws_rotate_grid3d, None, *args)

        def rotate_image3d(self, queue, image, rotmat, out, nearest=False):
            if nearest:
                args = (image, self.sampler_nearest, rotmat, out.data)
            else:
                args = (image, self.sampler_linear, rotmat, out.data)
            self._program.rotate_image3d(queue, self._gws_rotate_grid3d, None, *args)


    class grfftn_builder(object):
        _G = GpyFFT()
        CLFFT_HERMITIAN_INTERLEAVED = 3
        CLFFT_REAL = 5

        def __init__(self, ctx, shape, forward=True, batch_size=1):
            self.ctx = ctx
            self.shape = shape
            self.plan = self._G.create_plan(self.ctx, shape)
            if forward:
                layouts = (self.CLFFT_REAL, self.CLFFT_HERMITIAN_INTERLEAVED)
            else:
                layouts = (self.CLFFT_HERMITIAN_INTERLEAVED, self.CLFFT_REAL)
            self.plan.layouts = layouts
            self.plan.inplace = False
            size = np.prod(shape)
            ft_size = np.prod([shape[0] // 2 + 1] + list(shape)[1:])
            if forward:
                self.distances = (size, ft_size)
            else:
                self.distances = (ft_size, size)
            self.plan.batch_size = batch_size
            strides = (shape[-2] * shape[-1], shape[-1], 1)
            self.plan.strides_in = strides
            self.plan.strides_out = strides
            self.forward = forward

        def bake(self, queue):
            self.queue = queue
            self.plan.bake(queue)

        def __call__(self, inarray, outarray):
            self.plan.enqueue_transform(self.queue, inarray.data,
                    outarray.data, direction_forward=self.forward)


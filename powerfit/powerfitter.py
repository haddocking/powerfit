from __future__ import absolute_import, division

from sys import stdout
from os import remove
from os.path import join, abspath, isdir
import os.path
from time import time, sleep
from multiprocessing import RawValue, Lock, Process, cpu_count

import numpy as np
from numpy.fft import irfftn as np_irfftn, rfftn as np_rfftn
from scipy.ndimage import binary_erosion, laplace
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

from ._powerfit import rotate_image3d, conj_multiply, calc_lcc, dilate_points


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

    def scan(self):
        if self._queues is None:
            self._cpu_scan()
        else:
            self._gpu_scan()

    def _gpu_scan(self):
        self._corr = GPUCorrelator(self._target.array, self._queues[0],
                laplace=self._laplace)

        self._corr.template = self._template.array
        self._corr.mask = self._mask.array
        self._corr.rotations = self._rotations
        self._corr.scan()
        self._lcc = self._corr.lcc
        self._rot = self._corr.rot

    def _cpu_scan(self):
        nrot = self._rotations.shape[0]
        self._nrot_per_job = nrot // self._nproc
        processes = []
        self._counter = _Counter()
        self._njobs = self._nproc
        if self._queues is not None:
            self._njobs = len(self._queues)

        for n in xrange(self._njobs):
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

        time0 = time()
        for n in xrange(self._njobs):
            processes[n].start()

        while self._counter.value() < nrot:
            n = self._counter.value()
            p_done = (n + 1) / float(nrot) * 100
            now = time()
            eta = ((now - time0) / p_done) * (100 - p_done)
            total = (now - time0) / p_done * (100)
            stdout.write('{:7.2%} {:.0f}s {:.0f}s       \r'.format(n / float(nrot), eta, total))
            stdout.flush()
            sleep(0.5)
        stdout.write('\n')
        for n in xrange(self._njobs):
            processes[n].join()
        self._combine()

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
        # get center of grid
        self._center = self._get_center(self._target.shape)
        self._lcc_mask = self._get_lcc_mask(self._target)

    @staticmethod
    def _get_center(shape):
        """Get the center of the grid to rotate around"""
        #self._center = (np.asarray(template.shape, dtype=np.float64)[::-1] - 1)/ 2
        return (np.asarray(shape, dtype=np.float64) / 2)[::-1]

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
        # calculate the maximum radius from the center (in zyx)
        self._rmax = self._get_rmax(ind, self._center[::-1])

    @staticmethod
    def _laplace_filter(array):
        """Laplace transform"""
        return laplace(array, mode='constant')

    def _normalize_template(self, ind):
        # normalize the template over the mask
        self._template[ind] -= self._template[ind].mean()
        self._template[ind] /= self._template[ind].std()

    @staticmethod
    def _get_rmax(mask, center):
        # Calculate the maximum radius of the template. This helps in the
        # direct rotation during the search.
        mask = mask != 0
        surface = mask - binary_erosion(mask)
        coor = np.asarray((surface > 0).nonzero()).astype(np.float64)
        coor -= center.reshape(-1, 1)
        return int(np.ceil(np.linalg.norm(coor, axis=0).max()))

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
        self._rfftn(target, self._ft_target)
        self._rfftn(target**2, self._ft_target2)

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
            # monkey patch the numpy fft interface
            self._rfftn = lambda in_array, out_array: np_rfftn(in_array)
            self._irfftn = (lambda in_array, out_array:
                                np_irfftn(in_array, s=self._target.shape)
                            )

    def scan(self):
        super(CPUCorrelator, self).scan()

        self._lcc.fill(0)
        self._rot.fill(0)

        for n in xrange(self._rotations.shape[0]):
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
        rotate_image3d(
              self._template, rotmat, self._center, self._rmax,
              self._rot_template
              )
        rotate_image3d(
              self._mask, rotmat, self._center, self._rmax,
              self._rot_mask, nearest=True
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
        self._rfftn(self._rot_template, self._ft_template)
        self._rfftn(self._rot_mask, self._ft_mask)
        self._rfftn(self._rot_mask2, self._ft_mask2)

    def _backward_ffts(self):
        self._irfftn(self._ft_gcc, self._gcc)
        self._irfftn(self._ft_ave, self._ave)
        self._irfftn(self._ft_ave2, self._ave2)


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
            self._lcc_mask = cl_array.to_device(self._queue, self._lcc_mask.astype(np.int32))
            # Do some one-time precalculations
            self._rfftn(self._gtarget, self._ft_target)
            self._k.multiply(self._gtarget, self._gtarget, self._target2)
            self._rfftn(self._target2, self._ft_target2)

            self._gcenter = np.asarray(list(self._center) + [0], dtype=np.float32)
            self._gshape = np.asarray(
                    list(self._target.shape) + [np.product(self._target.shape)],
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
            self._gtemplate = cl.image_from_array(
                    self._ctx, self._template.astype(np.float32)
                    )
            self._gmask = cl.image_from_array(
                    self._ctx, self._mask.astype(np.float32)
                    )
            max_items = self._queue.device.max_compute_units * 32 * 16
            gws = [0] * 3
            gws[0] = min(2 * self._rmax, max_items)
            gws[1] = min(max_items // gws[0], 2 * self._rmax)
            gws[2] = min(max(max_items // (gws[0] * gws[0]), 1), 2 * self._rmax)
            self._gws = tuple(gws)

        @property
        def rotations(self):
            return BaseCorrelator.rotations

        @rotations.setter
        def rotations(self, rotations):
            BaseCorrelator.rotations.fset(self, rotations)
            self._grotations = cl_array.to_device(self._queue,
                    rotations.ravel().astype(np.float32))

        def scan(self):
            super(GPUCorrelator, self).scan()

            self._glcc.fill(0)
            self._grot.fill(0)
            for n in xrange(0, self._rotations.shape[0]):

                args = (self._gtemplate, self._gmask, self._grotations.data,
                        self._k._sampler_linear, self._k._sampler_nearest,
                        self._gcenter, self._gshape, self._rmax,
                        self._rot_template.data, self._rot_mask.data,
                        self._rot_mask2.data, np.int32(n))
                self._k.rotate_grids_and_multiply(self._queue, self._gws, None, *args)
                self._rfftn(self._rot_template, self._ft_template)
                self._rfftn(self._rot_mask, self._ft_mask)
                self._rfftn(self._rot_mask2, self._ft_mask2)

                self._k.conj_multiply(self._ft_template, self._ft_target, self._ft_gcc)
                self._k.conj_multiply(self._ft_mask, self._ft_target, self._ft_ave)
                self._k.conj_multiply(self._ft_mask2, self._ft_target2, self._ft_ave2)

                self._irfftn(self._ft_gcc, self._gcc)
                self._irfftn(self._ft_ave, self._ave)
                self._irfftn(self._ft_ave2, self._ave2)
                self._k.calc_lcc_and_take_best(self._gcc, self._ave, self._ave2, self._lcc_mask,
                        self._norm_factor, np.int32(n), self._glcc, self._grot)
                self._queue.finish()

                print n
        #        print n, self._rotations.shape[0]
        #        if hasattr(self, '_counter'):
        #            self._counter.increment()
            self._glcc.get(ary=self._lcc)
            self._queue.finish()

        def _generate_kernels(self):
            self._k = CLKernels(self._ctx)


    class CLKernels(object):
        def __init__(self, ctx):
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
            self._program = cl.Program(ctx, open(kernel_file).read()).build()
            self.rotate_grids_and_multiply = self._program.rotate_grids_and_multiply

            # Two samplers
            self._sampler_linear = cl.Sampler(
                    ctx, False, cl.addressing_mode.CLAMP, cl.filter_mode.LINEAR
                    )
            self._sampler_nearest = cl.Sampler(
                    ctx, False, cl.addressing_mode.CLAMP, cl.filter_mode.NEAREST
                    )


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


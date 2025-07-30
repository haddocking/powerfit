

from functools import partial
from os import remove
from os.path import join, abspath, isdir
from multiprocessing import RawValue, Lock, Process

import numpy as np
from tqdm.auto import tqdm

from powerfit_em.correlators.cpu import CPUCorrelator
from powerfit_em.volume import Volume
try:
    import pyfftw as _
    PYFFTW = True
except ImportError:
    PYFFTW = False
try:
    import pyopencl as _
    OPENCL = True
except:
    OPENCL = False


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
        correlator = CPUCorrelator(
            self._target.array,
            self._template.array,
            self._rotations,
            self._mask.array,
            self._laplace,
        )
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

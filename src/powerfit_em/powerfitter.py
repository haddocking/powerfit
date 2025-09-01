

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


def run_correlator_instance(
    target: Volume,
    template: Volume,
    mask: Volume,
    rotations: np.ndarray,
    laplace: bool,
    jobid: int,
    directory: str,
    counter: _Counter | None
):
    correlator = CPUCorrelator(
        target.array,
        template.array,
        rotations,
        mask.array,
        laplace,
    )
    for n in range(len(rotations)):
        correlator.compute_rotation(n, rotmat=correlator.rotations[n])
        if counter is not None:
            counter.increment()

    np.save(join(directory, '_lcc_part_{:d}.npy').format(jobid), correlator.lcc)
    np.save(join(directory, '_rot_part_{:d}.npy').format(jobid), correlator.rot)


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

    def scan(self, progress: partial[tqdm] | None):
        if self._queues is None:
            if self._nproc == 1:
                self._single_cpu_scan(progress)
            else:
                self._multi_cpu_scan(progress)
        else:
            self._gpu_scan(progress)

    def _gpu_scan(self, progress: partial[tqdm] | None):
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

    def _multi_cpu_scan(self, progress: partial[tqdm] | None):
        nrot = self._rotations.shape[0]
        self._nrot_per_job = nrot // self._nproc
        processes: list[Process] = []
        self._counter = None if processes is None else _Counter()
        self._njobs = self._nproc
        if self._queues is not None:
            self._njobs = len(self._queues)

        for id in range(self._njobs):
            start = id * self._nrot_per_job
            stop = start + self._nrot_per_job
            if id == self._njobs - 1:
                stop = len(self._rotations)
            partial_rotations = self._rotations[start:stop]
            processes.append(
                Process(
                  target=run_correlator_instance,
                  args=(self._target, self._template, self._mask,
                        partial_rotations, self._laplace, id,
                        self._directory, self._counter)
                )
            )

        for id in range(self._njobs):
            processes[id].start()

        if progress is not None:
            with progress(total=nrot) as pbar:
                while self._counter.value() < nrot:
                    current_count = self._counter.value()
                    pbar.update(current_count - pbar.n)
        
        for id in range(self._njobs):
            processes[id].join()
        self._combine()

    def _single_cpu_scan(self, progress: partial[tqdm] | None):
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

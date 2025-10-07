

from functools import partial
import multiprocessing
from multiprocessing.managers import DictProxy
from multiprocessing import RawValue, Lock, Process
from typing import Any

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
    import pyopencl as cl
    OPENCL = True
except:
    OPENCL = False

if OPENCL:
    from powerfit_em.correlators.gpu import GPUCorrelator


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
    counter: _Counter | None,
    results: dict[int, Any]
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

    results[jobid] = (correlator.lcc, correlator.rot)


class PowerFitter(object):
    """Wrapper around the Correlator classes for multiprocessing and GPU
    accelerated searches providing an easy interface.
    """

    def __init__(
        self,
        target: Volume,
        rotations: np.ndarray,
        template: Volume,
        mask: Volume,
        queue: "cl.CommandQueue | None",
        nproc: int = 1,
        laplace: bool = False
    ):
        self._target = target
        self._rotations = rotations
        self._template = template
        self._mask = mask
        self._queue = queue
        self._nproc = nproc
        self._laplace = laplace
        self._corr = None
        self._lcc = np.zeros(0, dtype=np.float32)
        self._rot = np.zeros(0, dtype=np.float32)

    @property
    def lcc(self):
        return self._lcc.copy()

    @property
    def rot(self):
        return self._rot.copy()

    def scan(self, progress: partial[tqdm] | None):
        if self._queue is None:
            if self._nproc == 1:
                self._single_cpu_scan(progress)
            else:
                self._multi_cpu_scan(progress)
        else:
            self._gpu_scan(progress)

    def set_template(self, template: Volume, mask: Volume):
        if not self._corr:
            msg = f"No correlator available yet. First run scan."
            raise ValueError(msg)
        self._corr.set_template(template.array, mask.array)
        
    def _gpu_scan(self, progress: partial[tqdm] | None):
        if OPENCL:
            if self._corr is None:
                self._corr = GPUCorrelator(
                    self._target.array,
                    self._template.array,
                    self._rotations,
                    self._mask.array,
                    self._queue,
                    self._laplace,
                )
            self._corr.scan(progress)
            self._lcc = self._corr.lcc
            self._rot = self._corr.rot
        else:
            raise ValueError("No OpenCL")

    def _multi_cpu_scan(self, progress: partial[tqdm] | None):
        nrot = self._rotations.shape[0]
        self._nrot_per_job = nrot // self._nproc
        processes: list[Process] = []
        self._counter = None if processes is None else _Counter()
        self._njobs = self._nproc
        if self._queue is not None:
            self._njobs = len(self._queue)

        ids = tuple(range(self._njobs))
        manager = multiprocessing.Manager()
        results = manager.dict()

        for id in ids:
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
                        self._counter, results)
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
        self._combine(ids, results)

    def _single_cpu_scan(self, progress: partial[tqdm] | None):
        self._corr = CPUCorrelator(
            self._target.array,
            self._template.array,
            self._rotations,
            self._mask.array,
            self._laplace,
        )
        self._corr.scan(progress)
        self._lcc = self._corr.lcc
        self._rot = self._corr.rot

    def _combine(self, ids: tuple[int, ...], results: DictProxy):
        # Combine all the intermediate results
        lcc = np.zeros(self._target.shape)
        rot = np.zeros(self._target.shape)
        ind = np.zeros(lcc.shape, dtype=np.bool)
        for n in ids:
            # Get LCC and rotations from results
            part_lcc = results[n][0]
            part_rot = results[n][1]

            np.greater(part_lcc, lcc, ind)
            lcc[ind] = part_lcc[ind]
            # take care of the rotation index offset for each independent job
            rot[ind] = part_rot[ind] + self._nrot_per_job * n
        self._lcc = lcc
        self._rot = rot

from unittest.mock import patch

import numpy as np
import pytest

from powerfit_em.gpu import get_opencl_queue, opencl_available

OPENCL_AVAILABLE = opencl_available()

pytestmark = pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL resources are not available.")

from powerfit_em.correlators.cpu import CPUCorrelator  # noqa: E402
from powerfit_em.correlators.opencl import (  # noqa: E402
    OpenCLBatchedCorrelator,
    OpenCLSerialCorrelator,
    _max_batch_size,
    _tuned_batch_size,
)


@pytest.fixture(scope="module")
def opencl_queue():
    try:
        return get_opencl_queue("0:0")
    except (RuntimeError, ValueError) as exc:
        pytest.skip(str(exc))


def _make_inputs():
    target = np.zeros((8, 8, 8), dtype=np.float32)
    target[2:6, 2:6, 2:6] = 1.0
    template = target.copy()
    mask = np.ones_like(target)
    rotations = np.asarray(
        [
            np.eye(3, dtype=np.float32),
            np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
        ]
    )
    return target, template, mask, rotations


def test_scan_serial_matches_cpu(opencl_queue):
    target, template, mask, rotations = _make_inputs()

    cpu_corr = CPUCorrelator(target, template, rotations, mask, laplace=False)
    ocl_corr = OpenCLSerialCorrelator(target, template, rotations, mask, opencl_queue, laplace=False)

    cpu_corr.scan()
    ocl_corr.scan()

    assert np.allclose(cpu_corr.lcc, ocl_corr.lcc, atol=1e-4, rtol=1e-4)
    assert np.array_equal(cpu_corr.rot, ocl_corr.rot)


def test_scan_forced_batch_size_matches_cpu(opencl_queue):
    target, template, mask, rotations = _make_inputs()

    cpu_corr = CPUCorrelator(target, template, rotations, mask, laplace=False)
    ocl_corr = OpenCLBatchedCorrelator(target, template, rotations, mask, opencl_queue, laplace=False, batch_size=1)

    cpu_corr.scan(progress=None)
    ocl_corr.scan()

    assert np.allclose(cpu_corr.lcc, ocl_corr.lcc, atol=1e-4, rtol=1e-4)
    assert np.array_equal(cpu_corr.rot, ocl_corr.rot)


def test_scan_batch_size_two_matches_cpu(opencl_queue):
    target, template, mask, rotations = _make_inputs()

    cpu_corr = CPUCorrelator(target, template, rotations, mask, laplace=False)
    ocl_corr = OpenCLBatchedCorrelator(target, template, rotations, mask, opencl_queue, laplace=False, batch_size=2)

    cpu_corr.scan()
    ocl_corr.scan()

    assert np.allclose(cpu_corr.lcc, ocl_corr.lcc, atol=1e-4, rtol=1e-4)
    assert np.array_equal(cpu_corr.rot, ocl_corr.rot)


def test_scan_partial_batch_remainder_matches_cpu(opencl_queue):
    """Batch size does not divide n_rotations evenly; tail must use _compute_batch."""
    target = np.zeros((8, 8, 8), dtype=np.float32)
    target[2:6, 2:6, 2:6] = 1.0
    template = target.copy()
    mask = np.ones_like(target)
    rotations = np.asarray(
        [
            np.eye(3, dtype=np.float32),
            np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
            np.asarray([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32),
        ]
    )

    cpu_corr = CPUCorrelator(target, template, rotations, mask, laplace=False)
    ocl_corr = OpenCLBatchedCorrelator(target, template, rotations, mask, opencl_queue, laplace=False, batch_size=2)

    cpu_corr.scan()
    ocl_corr.scan()

    assert np.allclose(cpu_corr.lcc, ocl_corr.lcc, atol=1e-4, rtol=1e-4)
    assert np.array_equal(cpu_corr.rot, ocl_corr.rot)


def test_serial_and_batched_own_independent_gpu_state(opencl_queue):
    """Serial and batched correlators must own distinct vars, vars_ft, and cl_kernels."""
    target, template, mask, rotations = _make_inputs()

    serial = OpenCLSerialCorrelator(target, template, rotations, mask, opencl_queue, laplace=False)
    batched = OpenCLBatchedCorrelator(target, template, rotations, mask, opencl_queue, laplace=False, batch_size=1)

    assert serial.vars is not batched.vars
    assert serial.vars_ft is not batched.vars_ft
    assert serial.cl_kernels is not batched.cl_kernels


def test_batched_invalid_batch_size_raises(opencl_queue):
    target, template, mask, rotations = _make_inputs()
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        OpenCLBatchedCorrelator(target, template, rotations, mask, opencl_queue, batch_size=0)


def test_max_batch_size_returns_positive(opencl_queue):
    result = _max_batch_size(opencl_queue, (32, 32, 32))
    assert result >= 1


def test_tuned_batch_size_returns_positive(opencl_queue):
    result = _tuned_batch_size(opencl_queue, (32, 32, 32))
    assert result >= 1


def test_tuned_batch_size_at_most_max(opencl_queue):
    shape = (32, 32, 32)
    assert _tuned_batch_size(opencl_queue, shape) <= _max_batch_size(opencl_queue, shape)


def test_batched_explicit_batch_size_exceeds_max_raises(opencl_queue):
    target, template, mask, rotations = _make_inputs()
    with patch("powerfit_em.correlators.opencl._max_batch_size", return_value=1):  # noqa: SIM117
        with pytest.raises(ValueError, match="exceeds the device memory upper bound"):
            OpenCLBatchedCorrelator(target, template, rotations, mask, opencl_queue, batch_size=2)


def test_batched_auto_tuned_exceeds_max_raises(opencl_queue):
    target, template, mask, rotations = _make_inputs()
    with (  # noqa: SIM117
        patch("powerfit_em.correlators.opencl._max_batch_size", return_value=1),
        patch("powerfit_em.correlators.opencl._tuned_batch_size", return_value=999),
    ):
        with pytest.raises(ValueError, match="Auto-tuned batch size"):
            OpenCLBatchedCorrelator(target, template, rotations, mask, opencl_queue)

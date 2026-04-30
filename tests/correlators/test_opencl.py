from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from powerfit_em.gpu import get_opencl_queue, opencl_available

OPENCL_AVAILABLE = opencl_available()

pytestmark = pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL resources are not available.")

from powerfit_em.correlators.cpu import CPUCorrelator  # noqa: E402
from powerfit_em.correlators.opencl import (  # noqa: E402
    _K_OPENCL_PERF,
    _TUNED_BATCH_CEIL,
    _TUNED_BATCH_FLOOR,
    OpenCLBatchedCorrelator,
    OpenCLSerialCorrelator,
    guess_batch_size,
    max_batch_size,
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
    result = max_batch_size(opencl_queue, (32, 32, 32))
    assert result >= 1


def test_guess_batch_size_returns_positive(opencl_queue):
    result = guess_batch_size(opencl_queue, (32, 32, 32))
    assert result >= 1


def test_guess_batch_size_at_most_max(opencl_queue):
    shape = (32, 32, 32)
    assert guess_batch_size(opencl_queue, shape) <= max_batch_size(opencl_queue, shape)


class TestGuessedBatchSize:
    def test_uses_floor_when_raw_estimate_is_tiny(self):
        class _FakeDevice:
            max_compute_units = 1
            max_clock_frequency = 1

        class _FakeQueue:
            device = _FakeDevice()

        # Huge volume ensures raw estimate is effectively 0 before clamping.
        shape = (512, 512, 512)
        assert guess_batch_size(cast(Any, _FakeQueue()), shape) == _TUNED_BATCH_FLOOR

    def test_uses_ceiling_for_extreme_raw_estimate(self):
        class _FakeDevice:
            max_compute_units = 1_000_000
            max_clock_frequency = 3_000_000

        class _FakeQueue:
            device = _FakeDevice()

        shape = (8, 8, 8)
        assert guess_batch_size(cast(Any, _FakeQueue()), shape) == _TUNED_BATCH_CEIL

    def test_returns_expected_clamped_raw_for_m2_profile(self):
        class _FakeDevice:
            max_compute_units = 10
            max_clock_frequency = 1398

        class _FakeQueue:
            device = _FakeDevice()

        shape = (32, 32, 32)
        z, y, x = shape
        ft_x = x // 2 + 1
        real_bytes = z * y * x * np.dtype(np.float32).itemsize
        complex_bytes = z * y * ft_x * np.dtype(np.complex64).itemsize
        bytes_per_rot = 6 * real_bytes + 6 * complex_bytes
        expected_raw = int(
            _K_OPENCL_PERF * _FakeDevice.max_compute_units * _FakeDevice.max_clock_frequency / bytes_per_rot
        )
        expected = max(_TUNED_BATCH_FLOOR, min(_TUNED_BATCH_CEIL, expected_raw))

        assert guess_batch_size(cast(Any, _FakeQueue()), shape) == expected


def test_batched_explicit_batch_size_exceeds_max_raises(opencl_queue):
    target, template, mask, rotations = _make_inputs()
    with patch("powerfit_em.correlators.opencl.max_batch_size", return_value=1):  # noqa: SIM117
        with pytest.raises(ValueError, match="exceeds the device memory upper bound"):
            OpenCLBatchedCorrelator(target, template, rotations, mask, opencl_queue, batch_size=2)


def test_batched_auto_tuned_exceeds_max_raises(opencl_queue):
    target, template, mask, rotations = _make_inputs()
    with (  # noqa: SIM117
        patch("powerfit_em.correlators.opencl.max_batch_size", return_value=1),
        patch("powerfit_em.correlators.opencl.guess_batch_size", return_value=999),
    ):
        with pytest.raises(ValueError, match="Auto-tuned batch size"):
            OpenCLBatchedCorrelator(target, template, rotations, mask, opencl_queue)

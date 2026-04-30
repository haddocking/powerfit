from unittest.mock import patch

import numpy as np
import pytest

from powerfit_em.gpu import cuda_available, get_cuda_stream

CUDA_AVAILABLE = cuda_available()

pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA resources are not available.")

cp = pytest.importorskip("cupy", reason="CUDA resources are not available.")

from powerfit_em.correlators.cpu import CPUCorrelator  # noqa: E402
from powerfit_em.correlators.cuda import (  # noqa: E402
    _DEFAULT_CLOCK_MHZ,
    _DEFAULT_SMS,
    _K_CUDA_PERF,
    _TUNED_BATCH_CEIL,
    _TUNED_BATCH_FLOOR,
    CUDABatchedCorrelator,
    CUDASerialCorrelator,
    build_cuda_ffts,
    guess_batch_size,
    max_batch_size,
)


@pytest.fixture(scope="module")
def cuda_stream():
    try:
        return get_cuda_stream(0)
    except (RuntimeError, ValueError) as exc:
        pytest.skip(str(exc))


def test_build_cuda_ffts_matches_cupy_fft(cuda_stream):
    src = cp.zeros((8, 8, 8), dtype=cp.float32)
    src[2:6, 2:6, 2:6] = 1.0
    fft_out = cp.empty((8, 8, 5), dtype=cp.complex64)
    inv_out = cp.empty_like(src)
    rfftn, irfftn = build_cuda_ffts((8, 8, 8), cuda_stream)

    with cuda_stream:
        rfftn(src, fft_out)
        irfftn(fft_out, inv_out)
    cuda_stream.synchronize()

    assert fft_out.shape == (8, 8, 5)
    assert inv_out.shape == src.shape
    assert bool(cp.isfinite(fft_out).all())
    assert bool(cp.isfinite(inv_out).all())


def test_scan_batched_matches_cpu_correlator(cuda_stream):
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

    cpu_corr = CPUCorrelator(target, template, rotations, mask, laplace=False)
    cuda_corr = CUDABatchedCorrelator(target, template, rotations, mask, cuda_stream, laplace=False)

    cpu_corr.scan()
    cuda_corr.scan()

    assert np.allclose(cpu_corr.lcc, cuda_corr.lcc, atol=1e-4, rtol=1e-4)
    assert np.array_equal(cpu_corr.rot, cuda_corr.rot)


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


def test_scan_serial_matches_cpu(cuda_stream):
    target, template, mask, rotations = _make_inputs()

    cpu_corr = CPUCorrelator(target, template, rotations, mask, laplace=False)
    cuda_corr = CUDASerialCorrelator(target, template, rotations, mask, cuda_stream, laplace=False)

    cpu_corr.scan()
    cuda_corr.scan()

    assert np.allclose(cpu_corr.lcc, cuda_corr.lcc, atol=1e-4, rtol=1e-4)
    assert np.array_equal(cpu_corr.rot, cuda_corr.rot)


def test_scan_forced_batch_size_matches_cpu(cuda_stream):
    target, template, mask, rotations = _make_inputs()

    cpu_corr = CPUCorrelator(target, template, rotations, mask, laplace=False)
    cuda_corr = CUDABatchedCorrelator(target, template, rotations, mask, cuda_stream, laplace=False, batch_size=1)

    cpu_corr.scan()
    cuda_corr.scan()

    assert np.allclose(cpu_corr.lcc, cuda_corr.lcc, atol=1e-4, rtol=1e-4)
    assert np.array_equal(cpu_corr.rot, cuda_corr.rot)


def test_scan_partial_batch_remainder_matches_cpu(cuda_stream):
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
    cuda_corr = CUDABatchedCorrelator(target, template, rotations, mask, cuda_stream, laplace=False, batch_size=2)

    cpu_corr.scan()
    cuda_corr.scan()

    assert np.allclose(cpu_corr.lcc, cuda_corr.lcc, atol=1e-4, rtol=1e-4)
    assert np.array_equal(cpu_corr.rot, cuda_corr.rot)


def test_serial_and_batched_own_independent_gpu_state(cuda_stream):
    """Serial and batched correlators must own distinct vars, vars_ft, and cuda_kernels."""
    target, template, mask, rotations = _make_inputs()

    serial = CUDASerialCorrelator(target, template, rotations, mask, cuda_stream, laplace=False)
    batched = CUDABatchedCorrelator(target, template, rotations, mask, cuda_stream, laplace=False, batch_size=1)

    assert serial.vars is not batched.vars
    assert serial.vars_ft is not batched.vars_ft
    assert serial.cuda_kernels is not batched.cuda_kernels


def test_batched_invalid_batch_size_raises(cuda_stream):
    target, template, mask, rotations = _make_inputs()
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        CUDABatchedCorrelator(target, template, rotations, mask, cuda_stream, batch_size=0)


def test_max_batch_size_returns_positive():
    result = max_batch_size((32, 32, 32))
    assert result >= 1


class TestGuessedBatchSize:
    def test_returns_positive(self):
        result = guess_batch_size((32, 32, 32))
        assert result >= 1

    def test_at_most_max(self):
        shape = (32, 32, 32)
        assert guess_batch_size(shape) <= max_batch_size(shape)

    def test_uses_floor_when_raw_estimate_is_tiny(self):
        class _FakeDevice:
            attributes = {
                "MultiProcessorCount": 1,
                "ClockRate": 1_000_000,
            }

        # Huge volume ensures raw estimate is effectively 0 before clamping.
        shape = (512, 512, 512)
        assert guess_batch_size(shape, _FakeDevice()) == _TUNED_BATCH_FLOOR

    def test_uses_fallbacks_for_invalid_device_attributes(self):
        class _FakeDevice:
            attributes = {
                "MultiProcessorCount": 0,
                "ClockRate": 0,
            }

        shape = (64, 64, 64)
        z, y, x = shape
        ft_x = x // 2 + 1
        bytes_per_rot = 6 * z * y * x * 4 + 6 * z * y * ft_x * 8
        expected_raw = int(_K_CUDA_PERF * _DEFAULT_SMS * _DEFAULT_CLOCK_MHZ / bytes_per_rot)
        expected = max(_TUNED_BATCH_FLOOR, min(_TUNED_BATCH_CEIL, expected_raw))

        assert guess_batch_size(shape, _FakeDevice()) == expected

    def test_uses_ceiling_for_extreme_raw_estimate(self):
        class _FakeDevice:
            attributes = {
                "MultiProcessorCount": 1_000_000,
                "ClockRate": 3_000_000,
            }

        shape = (8, 8, 8)
        assert guess_batch_size(shape, _FakeDevice()) == _TUNED_BATCH_CEIL


def test_batched_explicit_batch_size_exceeds_max_raises(cuda_stream):
    target, template, mask, rotations = _make_inputs()
    with patch("powerfit_em.correlators.cuda.max_batch_size", return_value=1):  # noqa: SIM117
        with pytest.raises(ValueError, match="exceeds the device memory upper bound"):
            CUDABatchedCorrelator(target, template, rotations, mask, cuda_stream, batch_size=2)


def test_batched_auto_tuned_exceeds_max_raises(cuda_stream):
    target, template, mask, rotations = _make_inputs()
    with (  # noqa: SIM117
        patch("powerfit_em.correlators.cuda.max_batch_size", return_value=1),
        patch("powerfit_em.correlators.cuda.guess_batch_size", return_value=999),
    ):
        with pytest.raises(ValueError, match="Auto-tuned batch size"):
            CUDABatchedCorrelator(target, template, rotations, mask, cuda_stream)

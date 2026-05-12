import numpy as np
import pytest

from powerfit_em.gpu import cuda_available, get_cuda_stream

CUDA_AVAILABLE = cuda_available()

pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA resources are not available.")

cp = pytest.importorskip("cupy", reason="CUDA resources are not available.")

from powerfit_em.correlators.cpu import CPUCorrelator  # noqa: E402
from powerfit_em.correlators.cuda import (  # noqa: E402
    CUDABatchedCorrelator,
    CUDASerialCorrelator,
    build_cuda_ffts,
    make_cuda_texture_linear,
    make_cuda_texture_nearerst,
    max_batch_size,
)
from powerfit_em.correlators.cudakernels import CUDAKernels  # noqa: E402


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


# LLENGTH = min(shape)//2 = 8; voxel at (z=3,y=0,x=0) has dist2=9 < 64,
# so it is inside the valid sphere and must survive an identity rotation.
_TEXTURE_SHAPE = (16, 16, 16)
_TEXTURE_PARAMS = [
    pytest.param(make_cuda_texture_linear, False, id="linear"),
    pytest.param(make_cuda_texture_nearerst, True, id="nearest"),
]


def _make_texture_fixtures(make_tex):
    vol = np.zeros(_TEXTURE_SHAPE, dtype=np.float32)
    vol[3, 0, 0] = 1.0
    return make_tex(vol), CUDAKernels(_TEXTURE_SHAPE)


@pytest.mark.parametrize("make_tex, nearest", _TEXTURE_PARAMS)
def test_rotate_image3d_identity(make_tex, nearest):
    tex, kernels = _make_texture_fixtures(make_tex)
    out = cp.zeros(_TEXTURE_SHAPE, dtype=cp.float32)

    kernels.rotate_image3d(tex, np.eye(3, dtype=np.float32), out, nearest=nearest)
    cp.cuda.Stream.null.synchronize()

    assert bool(cp.isfinite(out).all()), "output contains non-finite values"
    assert float(out[3, 0, 0]) == pytest.approx(1.0, abs=1e-4), "identity rotation must reproduce the source voxel"


@pytest.mark.parametrize("make_tex, nearest", _TEXTURE_PARAMS)
def test_rotate_image3d_batch_identity(make_tex, nearest):
    tex, kernels = _make_texture_fixtures(make_tex)
    rotmats = np.stack(
        [
            np.eye(3, dtype=np.float32),
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
        ]
    )
    out = cp.zeros((2, *_TEXTURE_SHAPE), dtype=cp.float32)

    kernels.rotate_image3d_batch(tex, rotmats, out, batch_size=2, nearest=nearest)
    cp.cuda.Stream.null.synchronize()

    assert bool(cp.isfinite(out).all()), "output contains non-finite values"
    assert float(out[0, 3, 0, 0]) == pytest.approx(1.0, abs=1e-4), "identity rotation must reproduce the source voxel"

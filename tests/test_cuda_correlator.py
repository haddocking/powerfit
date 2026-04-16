import numpy as np
import pytest

from powerfit_em.helpers import cuda_available
from powerfit_em.powerfit import get_cuda_stream

pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA resources are not available.")

if cuda_available():
    import cupy as cp

    from powerfit_em.correlators.cpu import CPUCorrelator
    from powerfit_em.correlators.cuda import CUDACorrelator, build_cuda_ffts, vkfft_fft


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
    rfftn, irfftn = build_cuda_ffts(cuda_stream)

    with cuda_stream:
        rfftn(src, fft_out)
        irfftn(fft_out, inv_out)
    cuda_stream.synchronize()

    assert fft_out.shape == (8, 8, 5)
    assert inv_out.shape == src.shape
    assert bool(cp.isfinite(fft_out).all())
    assert bool(cp.isfinite(inv_out).all())

    if not getattr(vkfft_fft, "has_cupy", False):
        expected_fft = cp.fft.rfftn(src)
        expected_inv = cp.fft.irfftn(fft_out, s=src.shape)

        assert cp.allclose(fft_out, expected_fft, atol=1e-4, rtol=1e-4)
        assert cp.allclose(inv_out, expected_inv, atol=1e-4, rtol=1e-4)


def test_scan_matches_cpu_correlator(cuda_stream):
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
    cuda_corr = CUDACorrelator(target, template, rotations, mask, cuda_stream, laplace=False)

    cpu_corr.scan(progress=None)
    cuda_corr.scan(progress=None)

    assert np.allclose(cpu_corr.lcc, cuda_corr.lcc, atol=1e-4, rtol=1e-4)
    assert np.array_equal(cpu_corr.rot, cuda_corr.rot)

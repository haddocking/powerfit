import numpy as np
import pytest

from powerfit_em.gpu import cuda_available
from powerfit_em.powerfit import get_cuda_stream

CUDA_AVAILABLE = cuda_available()

pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA resources are not available.")

cp = pytest.importorskip("cupy", reason="CUDA resources are not available.")

from powerfit_em.correlators.cpu import CPUCorrelator  # noqa: E402
from powerfit_em.correlators.cuda import CUDACorrelator, build_cuda_ffts  # noqa: E402


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

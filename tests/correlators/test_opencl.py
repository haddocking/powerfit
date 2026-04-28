import numpy as np
import pytest

from powerfit_em.gpu import get_opencl_queue, opencl_available

OPENCL_AVAILABLE = opencl_available()

pytestmark = pytest.mark.skipif(not OPENCL_AVAILABLE, reason="OpenCL resources are not available.")

from powerfit_em.correlators.cpu import CPUCorrelator  # noqa: E402
from powerfit_em.correlators.opencl import OpenCLCorrelator  # noqa: E402


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


def test_scan_batch_size_zero_matches_cpu(opencl_queue):
    target, template, mask, rotations = _make_inputs()

    cpu_corr = CPUCorrelator(target, template, rotations, mask, laplace=False)
    ocl_corr = OpenCLCorrelator(target, template, rotations, mask, opencl_queue, laplace=False, batch_size=0)

    cpu_corr.scan()
    ocl_corr.scan()

    assert np.allclose(cpu_corr.lcc, ocl_corr.lcc, atol=1e-4, rtol=1e-4)
    assert np.array_equal(cpu_corr.rot, ocl_corr.rot)


def test_scan_forced_batch_size_matches_cpu(opencl_queue):
    target, template, mask, rotations = _make_inputs()

    cpu_corr = CPUCorrelator(target, template, rotations, mask, laplace=False)
    ocl_corr = OpenCLCorrelator(target, template, rotations, mask, opencl_queue, laplace=False, batch_size=1)

    cpu_corr.scan(progress=None)
    ocl_corr.scan()

    assert np.allclose(cpu_corr.lcc, ocl_corr.lcc, atol=1e-4, rtol=1e-4)
    assert np.array_equal(cpu_corr.rot, ocl_corr.rot)


def test_scan_batch_size_two_matches_cpu(opencl_queue):
    target, template, mask, rotations = _make_inputs()

    cpu_corr = CPUCorrelator(target, template, rotations, mask, laplace=False)
    ocl_corr = OpenCLCorrelator(target, template, rotations, mask, opencl_queue, laplace=False, batch_size=2)

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
    ocl_corr = OpenCLCorrelator(target, template, rotations, mask, opencl_queue, laplace=False, batch_size=2)

    cpu_corr.scan()
    ocl_corr.scan()

    assert np.allclose(cpu_corr.lcc, ocl_corr.lcc, atol=1e-4, rtol=1e-4)
    assert np.array_equal(cpu_corr.rot, ocl_corr.rot)

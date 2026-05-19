import numpy as np
import pytest

from powerfit_em.correlators.cpu import CPUCorrelator
from powerfit_em.gpu import cuda_available, get_cuda_stream, get_opencl_queue, opencl_available
from powerfit_em._extensions import rotate_grid3d

def _rot_z_90() -> np.ndarray:
    return np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _rot_x_90() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def _rot_y_90() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


def _make_template(shape: tuple[int, int, int] = (24, 24, 24)) -> np.ndarray:
    vol = np.zeros(shape, dtype=np.float32)

    z0, y0, x0 = 1, 2, 1
    dz, dy, dx = 8, 8, 6

    # Right-hand-rule inspired asymmetric body plus axis-specific protrusions.
    vol[z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx] = 1.0
    vol[z0 : z0 + dz, y0 : y0 + 2, x0 + dx : x0 + dx + 3] = 0.9
    vol[z0 : z0 + 2, y0 + dy : y0 + dy + 5, x0 : x0 + dx] = 0.55
    vol[z0 + dz : z0 + dz + 3, y0 : y0 + dy, x0 : x0 + 2] = 0.35
    return vol


def _build_synthetic_case():
    template = _make_template()
    mask = np.ones_like(template, dtype=np.float32)

    true_rot = _rot_z_90()
    rotations = np.asarray(
        [
            np.eye(3, dtype=np.float32),
            _rot_x_90(),
            _rot_y_90(),
            true_rot,
        ],
        dtype=np.float32,
    )
    true_rot_idx = 3

    rotated = np.zeros_like(template, dtype=np.float32)
    rotate_grid3d(template, true_rot, min(template.shape) // 2, rotated, False)

    shift_zyx = (3, 2, 1)
    target = np.roll(rotated, shift=shift_zyx, axis=(0, 1, 2)).astype(np.float32, copy=False)
    expected_peak_zyx = (4, 3, 23)

    return {
        "target": target,
        "template": template,
        "mask": mask,
        "rotations": rotations,
        "expected_rot_idx": true_rot_idx,
        "expected_peak_zyx": expected_peak_zyx,
    }


def _maxima_positions(lcc: np.ndarray, eps: float = 1e-6) -> set[tuple[int, int, int]]:
    top = float(lcc.max())
    maxima = np.argwhere(lcc >= top - eps)
    return {(int(row[0]), int(row[1]), int(row[2])) for row in maxima}


def _assert_expected_outputs(correlator, expected_rot_idx: int, expected_peak_zyx: tuple[int, int, int]):
    lcc = correlator.lcc
    rot = correlator.rot

    assert float(lcc.max()) > 0.40

    maxima = _maxima_positions(lcc)
    assert expected_peak_zyx in maxima

    expected_rot_at_peak = int(rot[expected_peak_zyx])
    assert expected_rot_at_peak == expected_rot_idx

    argmax_peak = tuple(int(v) for v in np.unravel_index(np.argmax(lcc), lcc.shape))
    assert argmax_peak == expected_peak_zyx
    assert int(rot[argmax_peak]) == expected_rot_idx


def test_cpu_synthetic_expected_output():
    case = _build_synthetic_case()

    corr = CPUCorrelator(
        case["target"],
        case["template"],
        case["rotations"],
        case["mask"],
        laplace=False,
    )
    corr.scan()

    _assert_expected_outputs(corr, case["expected_rot_idx"], case["expected_peak_zyx"])


@pytest.mark.skipif(not cuda_available(), reason="CUDA resources are not available.")
def test_cuda_serial_synthetic_expected_output():
    cp = pytest.importorskip("cupy", reason="CUDA resources are not available.")
    from powerfit_em.correlators.cuda import CUDASerialCorrelator

    case = _build_synthetic_case()

    try:
        stream = get_cuda_stream(0)
    except (RuntimeError, ValueError) as exc:
        pytest.skip(str(exc))

    corr = CUDASerialCorrelator(
        case["target"],
        case["template"],
        case["rotations"],
        case["mask"],
        stream,
        laplace=False,
    )
    corr.scan()
    cp.cuda.Stream.null.synchronize()

    _assert_expected_outputs(corr, case["expected_rot_idx"], case["expected_peak_zyx"])


@pytest.mark.skipif(not cuda_available(), reason="CUDA resources are not available.")
def test_cuda_batched_synthetic_expected_output():
    cp = pytest.importorskip("cupy", reason="CUDA resources are not available.")
    from powerfit_em.correlators.cuda import CUDABatchedCorrelator

    case = _build_synthetic_case()

    try:
        stream = get_cuda_stream(0)
    except (RuntimeError, ValueError) as exc:
        pytest.skip(str(exc))

    corr = CUDABatchedCorrelator(
        case["target"],
        case["template"],
        case["rotations"],
        case["mask"],
        stream,
        laplace=False,
        batch_size=2,
    )
    corr.scan()
    cp.cuda.Stream.null.synchronize()

    _assert_expected_outputs(corr, case["expected_rot_idx"], case["expected_peak_zyx"])


@pytest.mark.skipif(not opencl_available(), reason="OpenCL resources are not available.")
def test_opencl_serial_synthetic_expected_output():
    pytest.importorskip("pyopencl", reason="OpenCL resources are not available.")
    from powerfit_em.correlators.opencl import OpenCLSerialCorrelator

    case = _build_synthetic_case()

    try:
        queue = get_opencl_queue("0:0")
    except (RuntimeError, ValueError) as exc:
        pytest.skip(str(exc))

    corr = OpenCLSerialCorrelator(
        case["target"],
        case["template"],
        case["rotations"],
        case["mask"],
        queue,
        laplace=False,
    )
    corr.scan()

    _assert_expected_outputs(corr, case["expected_rot_idx"], case["expected_peak_zyx"])


@pytest.mark.skipif(not opencl_available(), reason="OpenCL resources are not available.")
def test_opencl_batched_synthetic_expected_output():
    pytest.importorskip("pyopencl", reason="OpenCL resources are not available.")
    from powerfit_em.correlators.opencl import OpenCLBatchedCorrelator

    case = _build_synthetic_case()

    try:
        queue = get_opencl_queue("0:0")
    except (RuntimeError, ValueError) as exc:
        pytest.skip(str(exc))

    corr = OpenCLBatchedCorrelator(
        case["target"],
        case["template"],
        case["rotations"],
        case["mask"],
        queue,
        laplace=False,
        batch_size=2,
    )
    corr.scan()

    _assert_expected_outputs(corr, case["expected_rot_idx"], case["expected_peak_zyx"])

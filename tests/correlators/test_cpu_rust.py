import numpy as np
import pytest

from powerfit_em.correlators.cpu import CPUCorrelator

powerfitrs = pytest.importorskip("powerfit_em.powerfitrs", reason="Rust extension is not available")
CpuRustCorrelator = powerfitrs.CpuRustCorrelator


def _make_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def test_scan_matches_cpu_correlator_no_laplace():
    target, template, mask, rotations = _make_inputs()

    cpu_corr = CPUCorrelator(target, template, rotations, mask, laplace=False)
    rust_corr = CpuRustCorrelator(target, template, rotations, mask, False, 1)

    cpu_corr.scan()
    rust_corr.scan()

    assert np.allclose(cpu_corr.lcc, rust_corr.lcc, atol=1e-4, rtol=1e-4)
    assert np.array_equal(cpu_corr.rot, rust_corr.rot)


def test_scan_matches_cpu_correlator_with_laplace():
    target, template, mask, rotations = _make_inputs()

    cpu_corr = CPUCorrelator(target, template, rotations, mask, laplace=True)
    rust_corr = CpuRustCorrelator(target, template, rotations, mask, True, 1)

    cpu_corr.scan()
    rust_corr.scan()

    assert np.allclose(cpu_corr.lcc, rust_corr.lcc, atol=1e-4, rtol=1e-4)
    # Tie-breaking can differ at near-zero LCC values; compare rotation ids where
    # either backend has a meaningful score.
    meaningful = (np.abs(cpu_corr.lcc) > 1e-5) | (np.abs(rust_corr.lcc) > 1e-5)
    assert np.array_equal(cpu_corr.rot[meaningful], rust_corr.rot[meaningful])


def test_scan_nproc2_matches_nproc1():
    target, template, mask, rotations = _make_inputs()

    rust_nproc1 = CpuRustCorrelator(target, template, rotations, mask, False, 1)
    rust_nproc2 = CpuRustCorrelator(target, template, rotations, mask, False, 2)

    rust_nproc1.scan()
    rust_nproc2.scan()

    assert np.allclose(rust_nproc1.lcc, rust_nproc2.lcc, atol=1e-4, rtol=1e-4)
    assert np.array_equal(rust_nproc1.rot, rust_nproc2.rot)

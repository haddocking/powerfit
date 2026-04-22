import shlex
from argparse import ArgumentParser
from pathlib import Path

import pytest

from powerfit_em.helpers import cuda_available, opencl_available
from powerfit_em.powerfit import add_computational_resources2parser

CUDA_AVAILABLE = cuda_available()
OPENCL_AVAILABLE = opencl_available()

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# The fixtures below where copied from https://github.com/haddocking/powerfit-tutorial

@pytest.fixture(scope="session")
def ribosome_map() -> Path:
    return FIXTURES_DIR / "ribosome-KsgA.map"


@pytest.fixture(scope="session")
def ksga_pdb() -> Path:
    return FIXTURES_DIR / "KsgA.pdb"


@pytest.fixture(scope="session")
def baseline_solutions() -> Path:
    return FIXTURES_DIR / "solutions.out"


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: test requires CUDA hardware and cupy package")
    config.addinivalue_line("markers", "requires_opencl: test requires OpenCL platform/device and pyopencl package")
    config.addinivalue_line("markers", "gpu_integration: test is a GPU integration test requiring hardware")


def pytest_addoption(parser):
    parser.addoption(
        "--powerfit",
        action="store",
        default="",
        help="Extra PowerFit arguments to pass through to regression tests (e.g., '--gpu' or '--nproc 6')",
    )


@pytest.fixture
def powerfit_args(request) -> list[str]:
    """Parse and validate --powerfit option using PowerFit's argument parser.

    Only allows --gpu and --nproc arguments to control execution profile.
    Rejects all other arguments to preserve test determinism.
    """
    powerfit_str = request.config.getoption("--powerfit", default="").strip()

    if not powerfit_str:
        return []

    parser = ArgumentParser()
    add_computational_resources2parser(parser)
    raw_args = shlex.split(powerfit_str)
    try:
        parser.parse_args(raw_args)
    except SystemExit as e:
        pytest.fail(f"Failed to parse --powerfit: {powerfit_str}\nParser error: {e}")

    return raw_args

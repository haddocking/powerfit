from pathlib import Path

import pytest

from powerfit_em.helpers import cuda_available, opencl_available

CUDA_AVAILABLE = cuda_available()
OPENCL_AVAILABLE = opencl_available()

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: test requires CUDA hardware and cupy package")
    config.addinivalue_line("markers", "requires_opencl: test requires OpenCL platform/device and pyopencl package")
    config.addinivalue_line("markers", "gpu_integration: test is a GPU integration test requiring hardware")


@pytest.fixture(scope="session")
def ribosome_map() -> Path:
    return FIXTURES_DIR / "ribosome-KsgA.map"


@pytest.fixture(scope="session")
def ksga_pdb() -> Path:
    return FIXTURES_DIR / "KsgA.pdb"

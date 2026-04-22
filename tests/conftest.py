import shlex
from argparse import ArgumentParser

import pytest

from powerfit_em.gpu import cuda_available, opencl_available
from powerfit_em.powerfit import add_computational_resources2parser

CUDA_AVAILABLE = cuda_available()
OPENCL_AVAILABLE = opencl_available()


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

    Only allows computational resource arguments such as --gpu, --nproc,
    and --progressbar/--no-progressbar to control execution profile.
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

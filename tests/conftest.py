"""Shared pytest fixtures and CLI options for the test suite."""

import shlex
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest

from powerfit_em.powerfit import add_computational_resources2parser
from powerfit_em.volume import Volume


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

    Only allows computational resource arguments,
    use `pytest ... --powerfit="--help"` to see available options.
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


@pytest.fixture
def example_mrc_file(tmp_path: Path, example_volume: Volume) -> Path:
    """Write `example_volume` to a file and return its Path."""
    fn = tmp_path / "example_volume.mrc"
    example_volume.tofile(fn)
    return fn


@pytest.fixture
def example_volume() -> Volume:
    """Synthetic `Volume` with three non-zero voxels."""
    array = np.zeros((3, 4, 5))
    array[0, 0, 0] = 1.1
    array[1, 2, 3] = 2.2
    array[2, 3, 4] = 3.3
    return Volume(array)

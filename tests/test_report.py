"""Test the generation of the report."""

import shutil
from pathlib import Path

import pytest

from powerfit_em import report as report_module

from . import FIXTURE_DATA_DIR


@pytest.fixture
def solutions_output_path() -> Path:
    """Path to the golden `solutions.out`."""
    return FIXTURE_DATA_DIR / "solutions.out"


def test_generate_report(tmp_path: Path, solutions_output_path: Path, example_mrc_file: Path) -> None:
    """Test that the report is generated."""
    run_dir = tmp_path

    src = solutions_output_path
    dst = run_dir / "solutions.out"

    shutil.copy(src, dst)

    report_module.generate_report(
        directory=str(run_dir),
        target=str(example_mrc_file),
        num=1,
        delimiter=",",
        options={},
    )

    state_path = run_dir / "state.mvsj"
    # check if the state contains non-ASCII chars as expected
    assert "\u03b1" in state_path.read_text(encoding="utf-8")

    report_path = run_dir / "report.html"
    assert report_path.exists()

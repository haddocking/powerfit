"""Test the generation of the report."""

from pathlib import Path

import pytest

from powerfit_em import report as report_module


def test_generate_report_writes_files_as_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the report is written with `utf-8` encoding."""
    run_dir = tmp_path
    target_path = run_dir / "target.mrc"
    solution = {
        "rank": "1",
        "fitted_model_file": run_dir / "fit_1.pdb",
        "cc": "0.9",
        "Fish-z": "1.0",
        "rel-z": "1.0",
        "sigma_dif": 0.1,
        "x": "0",
        "y": "0",
        "z": "0",
        "a11": "1",
        "a12": "0",
        "a13": "0",
        "a21": "0",
        "a22": "1",
        "a23": "0",
        "a31": "0",
        "a32": "0",
        "a33": "1",
    }

    monkeypatch.setattr(report_module, "copy_target_to_report_dir", lambda _target, _run_dir: target_path)
    monkeypatch.setattr(
        report_module,
        "_calc_rel_isovalue",
        lambda _volume_path: report_module.Iso(value=1.0, min=0.0, max=2.0, step=0.1),
    )
    monkeypatch.setattr(report_module, "_read_solutions", lambda _path, _delimiter=None: [solution])
    monkeypatch.setattr(report_module, "generate_html", lambda *_args, **_kwargs: "<html></html>")

    report_module.generate_report(directory=str(run_dir), target=str(target_path), num=1, delimiter=",", options={})

    state_path = run_dir / "state.mvsj"
    # check if the state contains non-ASCII chars as expected
    assert "\u03b1" in state_path.read_text(encoding="utf-8")

    report_path = run_dir / "report.html"
    assert report_path.exists()

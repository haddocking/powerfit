from pathlib import Path

from powerfit_em import report as report_module


def _make_solution(run_dir: Path) -> dict:
    return {
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


def test_generate_report_writes_files_as_utf8(tmp_path, monkeypatch):
    calls = []
    original_write_text = Path.write_text

    def _write_text(self, data, *args, **kwargs):
        calls.append((self, kwargs.get("encoding")))
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _write_text)

    run_dir = tmp_path
    target_path = run_dir / "target.mrc"
    solution = _make_solution(run_dir)

    monkeypatch.setattr(report_module, "copy_target_to_report_dir", lambda target, run_dir: target_path)
    monkeypatch.setattr(
        report_module,
        "_calc_rel_isovalue",
        lambda volume_path: report_module.Iso(value=1.0, min=0.0, max=2.0, step=0.1),
    )
    monkeypatch.setattr(report_module, "_read_solutions", lambda path, delimiter=None: [solution])
    monkeypatch.setattr(report_module, "generate_html", lambda *args, **kwargs: "<html></html>")

    report_module.generate_report(str(run_dir), str(target_path), num=1, delimiter=",", options={})

    assert calls, "expected generate_report to write at least one file"
    for path, encoding in calls:
        assert encoding == "utf-8", f"{path} was written without explicit utf-8 encoding"

    state_path = run_dir / "state.mvsj"
    assert "α" in state_path.read_text(encoding="utf-8")

    report_path = run_dir / "report.html"
    assert report_path.exists()

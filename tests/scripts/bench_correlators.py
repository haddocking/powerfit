"""
Usage:
    uv run python tests/scripts/bench_correlators.py
"""

import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

RESOLUTION = 13
NPROCS = (1, 4, 16)
ANGLES = (20, 10)
ENGINES = [("cython", "9a11eed", None), ("python", "ab1800d", None), ("rust", "ab1800d", "--rust")]


def fetch_data() -> tuple[Path, Path]:
    data_dir = Path(tempfile.mkdtemp(prefix="pf-bench-data"))
    map_url = "https://github.com/haddocking/powerfit-tutorial/raw/refs/heads/master/ribosome-KsgA.map"
    pdb_url = "https://github.com/haddocking/powerfit-tutorial/raw/refs/heads/master/KsgA.pdb"

    map_path = data_dir / "ribosome-KsgA.map"
    pdb_path = data_dir / "KsgA.pdb"
    urllib.request.urlretrieve(map_url, map_path)  # noqa: S310
    urllib.request.urlretrieve(pdb_url, pdb_path)  # noqa: S310

    return map_path, pdb_path


def setup_clone(commit: str) -> Path:
    clone_dir = Path(tempfile.mkdtemp(prefix="pf-bench-"))
    subprocess.run(["git", "clone", "https://github.com/haddocking/powerfit.git", str(clone_dir)], check=True)
    subprocess.run(["git", "-C", str(clone_dir), "checkout", commit], check=True)
    subprocess.run(["uv", "venv", ".venv", "--python=3.14"], cwd=clone_dir, check=True)
    subprocess.run(
        ["uv", "pip", "install", "--python", ".venv/bin/python", "-e", "."],
        cwd=clone_dir,
        check=True,
    )
    return clone_dir


def parse_search_seconds(log_text: str) -> float:
    match = re.compile(r"Time for search: (?:(\d+)m )?([\d.]+)\s*s").search(log_text)
    if match is None:
        msg = f"could not find 'Time for search' line in log:\n{log_text}"
        raise ValueError(msg)
    minutes, seconds = match.groups()
    return (int(minutes) * 60 if minutes else 0) + float(seconds)


def run_powerfit(
    clone_dir: Path,
    flag: str | None,
    map_path: Path,
    pdb_path: Path,
    angle: float,
    nproc: int,
    out_dir: Path,
) -> float:
    cmd = []
    cmd.append(clone_dir / ".venv" / "bin" / "powerfit")
    if flag:
        cmd.append(flag)
    cmd += [
        str(map_path),
        str(RESOLUTION),
        str(pdb_path),
        "-a",
        str(angle),
        "--delimiter",
        ",",
        "-n",
        "0",
        "--nproc",
        str(nproc),
        "-d",
        str(out_dir),
        "--log-level",
        "INFO",
    ]
    result = subprocess.run(cmd, cwd=clone_dir, capture_output=True, text=True, check=True)
    log_text = result.stdout + result.stderr
    (out_dir / "powerfit.log").write_text(log_text)
    return parse_search_seconds(log_text)


def main() -> None:
    map_path, pdb_path = fetch_data()

    # one clone per unique commit -- python and rust share ab1800d
    commits = {commit for _, commit, _ in ENGINES}
    clone_by_commit: dict[str, Path] = {}
    for commit in commits:
        clone_by_commit[commit] = setup_clone(commit)

    try:
        rows = []
        for angle in ANGLES:
            for nproc in NPROCS:
                for identifier, commit, flag in ENGINES:
                    clone_dir = clone_by_commit[commit]
                    # NOTE: dump results, we only need the times
                    with tempfile.TemporaryDirectory() as tmpdir:
                        seconds = run_powerfit(clone_dir, flag, map_path, pdb_path, angle, nproc, Path(tmpdir))

                    rows.append((identifier, nproc, angle, seconds))
    finally:
        for clone_dir in clone_by_commit.values():
            shutil.rmtree(clone_dir, ignore_errors=True)

    summary_path = Path.cwd() / "summary.csv"
    with summary_path.open("w") as f:
        f.write("engine,nproc,angle,search_seconds\n")
        for engine, nproc, angle, seconds in rows:
            f.write(f"{engine},{nproc},{angle},{seconds:.3f}\n")

    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

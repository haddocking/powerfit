"""
Usage:
    uv run python tests/scripts/bench_correlators.py
"""

import csv
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Engine:
    identifier: str
    commit: str
    flag: str | None = None
    optimize: bool = False

    def setup_clone(self) -> Path:
        clone_dir = Path(tempfile.mkdtemp(prefix="pf-bench-"))
        print(f"-> cloning {self.commit}")
        self.run(["git", "clone", "https://github.com/haddocking/powerfit.git", str(clone_dir)])
        self.run(["git", "-C", str(clone_dir), "checkout", self.commit])
        print("-> building venv")
        self.run(["uv", "venv", ".venv", "--python=3.14"], cwd=clone_dir)

        print("-> installing" + (" (target-cpu=native)" if self.optimize else ""))
        env = {"RUSTFLAGS": "-C target-cpu=native"} if self.optimize else None
        self.run(["uv", "pip", "install", "--python", ".venv/bin/python", "-e", "."], cwd=clone_dir, env=env)

        return clone_dir

    @contextmanager
    def cloned(self) -> Iterator[Path]:
        clone_dir = self.setup_clone()
        try:
            yield clone_dir
        finally:
            shutil.rmtree(clone_dir, ignore_errors=True)

    def run_powerfit(
        self,
        clone_dir: Path,
        map_path: Path,
        pdb_path: Path,
        angle: float,
        resolution: float,
        nproc: int,
        out_dir: Path,
    ) -> float:
        cmd = [str(clone_dir / ".venv" / "bin" / "powerfit")]
        if self.flag:
            cmd.append(self.flag)
        cmd += [
            str(map_path),
            str(resolution),
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
        log_text = self.run(cmd, cwd=clone_dir)
        (out_dir / "powerfit.log").write_text(log_text)
        return self.parse_search_seconds(log_text)

    @staticmethod
    def run(cmd: list, cwd: Path | None = None, env: dict | None = None) -> str:
        full_env = {**os.environ, **env} if env else None
        result = subprocess.run(cmd, cwd=cwd, env=full_env, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            result.check_returncode()
        return result.stdout + result.stderr

    @staticmethod
    def parse_search_seconds(log_text: str) -> float:
        match = re.compile(r"Time for search: (?:(\d+)m )?([\d.]+)\s*s").search(log_text)
        if match is None:
            msg = f"could not find 'Time for search' line in log:\n{log_text}"
            raise ValueError(msg)
        minutes, seconds = match.groups()
        return (int(minutes) * 60 if minutes else 0) + float(seconds)


@contextmanager
def fetch_data(map_url: str, pdb_url: str) -> Iterator[tuple[Path, Path]]:
    print("-> fetching data")
    data_dir = Path(tempfile.mkdtemp(prefix="pf-bench-data-"))
    try:
        map_path = data_dir / Path(urlparse(map_url).path).name
        pdb_path = data_dir / Path(urlparse(pdb_url).path).name

        urllib.request.urlretrieve(map_url, map_path)
        urllib.request.urlretrieve(pdb_url, pdb_path)

        yield map_path, pdb_path
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def print_chart(summary_path: Path, bar_width: int = 40) -> None:
    """Make an ASCII chart."""
    with summary_path.open(newline="") as f:
        rows = [
            (r["engine"], int(r["nproc"]), float(r["angle"]), float(r["search_seconds"])) for r in csv.DictReader(f)
        ]
    if not rows:
        return

    max_seconds = max(seconds for _, _, _, seconds in rows)

    print("\n-> chart (search seconds)")
    for angle in sorted({a for _, _, a, _ in rows}):
        for nproc in sorted({n for _, n, a, _ in rows if a == angle}):
            print(f"\n  nproc={nproc}")
            for identifier, n, a, seconds in rows:
                if n != nproc or a != angle:
                    continue
                bar_len = round(seconds / max_seconds * bar_width) if max_seconds else 0
                bar = "#" * bar_len
                print(f"    {identifier:<16} {bar} {seconds:.2f}s")


def main() -> None:

    nprocs = (1, 16)

    # paper data
    map_url = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-1046/map/emd_1046.map.gz"
    pdb_url = "https://files.rcsb.org/download/9A2G.cif.gz"
    resolution = 20
    angle = 4.71

    # "tutorial" data
    # map_url = "https://github.com/haddocking/powerfit-tutorial/raw/refs/heads/master/ribosome-KsgA.map"
    # pdb_url = "https://github.com/haddocking/powerfit-tutorial/raw/refs/heads/master/KsgA.pdb"
    # resolution = 13
    # angle = 20

    # naming: <orchestration>+<rotation kernel>[-fma][-native]
    # python+cython = pyFFTW + C/Cython rotation (v5.0.2 baseline)
    # python+rust   = pyFFTW + Rust rotation
    # rust          = ndrustfft + Rust rotation (pure rust, compute-wise)
    #
    # NOTE: `fma` = fused multiply-add -> <https://doc.rust-lang.org/stable/core/primitive.f32.html#algebraic-operators>
    ENGINES = [
        Engine("python+cython", "cf1e3f8"),  # v5.0.2
        Engine("python+rust", "ab1800d"),
        Engine("python+rust-fma", "a78c7ab"),
        Engine("rust", "ab1800d", flag="--rust"),
        Engine("rust-fma", "a78c7ab", flag="--rust"),
        Engine("rust-native", "ab1800d", flag="--rust", optimize=True),
        Engine("rust-fma-native", "a78c7ab", flag="--rust", optimize=True),
    ]

    print("-> running benchmarks")
    rows = []
    # NOTE: context here to make cleaning easier
    with fetch_data(map_url=map_url, pdb_url=pdb_url) as (map_path, pdb_path):
        for engine in ENGINES:
            # NOTE: context here to make cleaning easier
            with engine.cloned() as clone_dir:
                for nproc in nprocs:
                    print(f"   {engine.identifier:<16} nproc={nproc:<2} ", end="", flush=True)
                    # NOTE: dump results, we only need the times
                    with tempfile.TemporaryDirectory() as tmpdir:
                        seconds = engine.run_powerfit(
                            clone_dir=clone_dir,
                            map_path=map_path,
                            pdb_path=pdb_path,
                            angle=angle,
                            resolution=resolution,
                            nproc=nproc,
                            out_dir=Path(tmpdir),
                        )

                    rows.append((engine.identifier, nproc, angle, seconds))
                    print(f"{seconds:.3f}s")

    summary_path = Path.cwd() / "summary.csv"
    with summary_path.open("w") as f:
        f.write("engine,nproc,angle,search_seconds\n")
        for engine, nproc, angle, seconds in rows:
            f.write(f"{engine},{nproc},{angle},{seconds:.3f}\n")

    print(f"-> summary: {summary_path}")
    print_chart(summary_path)


if __name__ == "__main__":
    main()

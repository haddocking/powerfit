#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path

BATCH_SIZE_RE = re.compile(r"rotations with batch size\s+(\d+)")
RUN_NUMBER_RE = re.compile(r"-r(\d+)$")


def parse_run_number(run_dir_name: str):
    match = RUN_NUMBER_RE.search(run_dir_name)
    if match is None:
        return None
    return int(match.group(1))


def parse_duration_from_line(line: str, marker: str):
    if marker not in line:
        return None

    tail = line.split(marker, 1)[1].strip()
    parts = tail.split()

    # Format 1: "0m 16s"
    if len(parts) >= 2 and parts[0].endswith("m") and parts[1].endswith("s"):
        minutes = int(parts[0][:-1])
        seconds = float(parts[1][:-1])
        return minutes * 60 + seconds

    # Format 2: "6.998 s"
    if len(parts) >= 2 and parts[1] == "s":
        return float(parts[0])

    return None


def parse_log(log_path: Path):
    run = log_path.parent.name
    run_number = parse_run_number(run)
    pending_search_seconds = None
    pending_batch_size = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            batch_match = BATCH_SIZE_RE.search(line)
            if batch_match is not None:
                pending_batch_size = int(batch_match.group(1))

            search_seconds = parse_duration_from_line(line, "Time for search:")
            if search_seconds is not None:
                pending_search_seconds = search_seconds
                continue

            total_seconds = parse_duration_from_line(line, "Total time:")
            if total_seconds is not None:
                # Total time is expected in m/s format; cast to int for clean CSV.
                yield run, run_number, int(total_seconds), pending_search_seconds, pending_batch_size
                pending_search_seconds = None
                pending_batch_size = None


def main():
    parser = argparse.ArgumentParser(
        description="Parse powerfit logs and output run,run_number,total_seconds,search_seconds as CSV"
    )
    parser.add_argument("log_glob", help='Glob for log files, e.g. "runs/*/*.log"')
    args = parser.parse_args()

    logs = sorted(Path().glob(args.log_glob))

    writer = csv.writer(sys.stdout)
    writer.writerow(["run", "run_number", "total_seconds", "search_seconds", "batch_size"])

    for log_path in logs:
        if log_path.is_file():
            for run, run_number, total_s, search_s, batch_size in parse_log(log_path):
                # Keep search precision when it is sub-second-like float.
                if isinstance(search_s, float) and search_s.is_integer():  # noqa: SIM108
                    search_out = int(search_s)
                else:
                    search_out = search_s
                writer.writerow([run, run_number, total_s, search_out, batch_size])


if __name__ == "__main__":
    main()

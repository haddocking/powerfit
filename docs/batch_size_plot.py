#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

EXPECTED_COLUMNS = {"run", "run_number", "batch_size", "search_seconds"}


def normalize_run(raw_run: str) -> str:
    parts = raw_run.split("-")
    if len(parts) < 2:
        raise ValueError(f"Cannot normalize run name: {raw_run}")
    return "-".join(parts[:2])


def is_numeric(value: str) -> bool:
    if value is None:
        return False
    text = value.strip()
    if not text:
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def transform_rows(reader: csv.DictReader):
    missing = EXPECTED_COLUMNS.difference(reader.fieldnames or [])
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing expected columns: {missing_text}")

    for row in reader:
        raw_run = row["run"]
        run_number = row["run_number"].strip()
        batch_size = row["batch_size"]
        search_seconds = row["search_seconds"]
        if "-autobs" in raw_run or not run_number:
            continue
        if not is_numeric(batch_size) or not is_numeric(search_seconds):
            continue

        yield {
            "run": normalize_run(raw_run),
            "run_number": run_number,
            "batch_size": batch_size.strip(),
            "search_seconds": search_seconds.strip(),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Create batch-size vs search-time CSV from docs/times.csv"
    )
    parser.add_argument("input_csv", type=Path, help="Input times.csv path")
    parser.add_argument("output_csv", type=Path, help="Output batchsize_vs_search.csv path")
    args = parser.parse_args()

    with args.input_csv.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        rows = sorted(
            transform_rows(reader),
            key=lambda row: (
                row["run"],
                float(row["batch_size"]),
                float(row["run_number"]) if row["run_number"] else -1,
                float(row["search_seconds"]),
            ),
        )

    with args.output_csv.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=["run", "run_number", "batch_size", "search_seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
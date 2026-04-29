# Performance

How PowerFit performs on different computational resources, such as CPU and GPU, and with different batch sizes. This can help users understand the trade-offs between different configurations and choose the best one for their needs.

## Measurements

Fetch the map and the structure for the test case:

```shell
wget https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-1046/map/emd_1046.map.gz
wget https://files.rcsb.org/download/9A2G.cif.gz
```

Runs on a machines with the following specifications:

* m1: AMD Ryzen 5 5600G and NVIDIA GeForce RTX 3050
* m2: AMD Ryzen 7 7800X3D and AMD Radeon RX 7900 XTX

<details>
  <summary>Run commands to test different computational resources
    </summary>

```shell
# On machine 1
mkdir -p runs
for run in 1 2 3 4 5; do
	# powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --nproc 1 -d runs/m1-cpu1-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --nproc 6 -d runs/m1-cpu6-r${run}
	# powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --nproc 1 --progressbar -d runs/m1-cpu1-pb-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --nproc 6 --progressbar -d runs/m1-cpu6-pb-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 -d runs/m1-cuda-autobs-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 1000 -d runs/m1-cuda-bs1000-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 0 -d runs/m1-cuda-nobs-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 -d runs/m1-opencl-autobs-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 1000 -d runs/m1-opencl-bs1000-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 0 -d runs/m1-opencl-nobs-r${run}
done
# On machine 2
mkdir -p runs
for run in 1 2 3 4 5; do
	# powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --nproc 1 -d runs/m2-cpu1-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --nproc 6 -d runs/m2-cpu6-r${run}
	# powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --nproc 1 --progressbar -d runs/m2-cpu1-pb-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --nproc 6 --progressbar -d runs/m2-cpu6-pb-r${run}
	# CUDA not supported on AMD GPU
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 -d runs/m2-opencl-autobs-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 1000 -d runs/m2-opencl-bs1000-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 3500 -d runs/m2-opencl-bs3500-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 0 -d runs/m2-opencl-nobs-r${run}
done
```

</details>

<details>
  <summary>Convert log to table</summary>

```python
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
                if isinstance(search_s, float) and search_s.is_integer():
                    search_out = int(search_s)
                else:
                    search_out = search_s
                writer.writerow([run, run_number, total_s, search_out, batch_size])

if __name__ == "__main__":
    main()
```

```shell
python3 docs/parse_times.py "runs/*/*.log" > runs/times.csv
```

Group by and markdown table with duckdb

```sql
.mode markdown
WITH runs AS (
  SELECT
    regexp_replace(run, '-r[0-9]+$', '') AS run_group,
    total_seconds::DOUBLE AS total,
    search_seconds::DOUBLE AS search,
    batch_size
  FROM read_csv_auto('runs/times.csv')
)
SELECT
  run_group,
  round(avg(search), 2) AS avg_search,
  round(stddev_samp(search), 2) AS stddev_search,
  min(search) AS min_search,
  max(search) AS max_search,
  median(search) AS median_search,

  round(avg(total), 2) AS avg_total,
  round(stddev_samp(total), 2) AS stddev_total,
  min(total) AS min_total,
  max(total) AS max_total,
  median(total) AS median_total,

  min(batch_size) AS batch_size
FROM runs
GROUP BY run_group
ORDER BY run_group;
```

</details>

|    run_group     | avg_search | stddev_search | min_search | max_search | median_search | avg_total | stddev_total | min_total | max_total | median_total | batch_size |
|------------------|-----------:|--------------:|-----------:|-----------:|--------------:|----------:|-------------:|----------:|----------:|-------------:|------------|
| m1-cpu6          | 36.0       | 2.92          | 34.0       | 41.0       | 35.0          | 36.6      | 3.21         | 34.0      | 42.0      | 35.0         | NULL       |
| m1-cpu6-pb       | 40.2       | 2.95          | 37.0       | 45.0       | 40.0          | 40.8      | 2.59         | 38.0      | 45.0      | 40.0         | NULL       |
| m1-cuda-autobs   | 6.8        | 0.6           | 6.282      | 7.832      | 6.615         | 7.6       | 0.89         | 7.0       | 9.0       | 7.0          | 1703       |
| m1-cuda-bs1000   | 6.11       | 0.2           | 5.837      | 6.405      | 6.09          | 7.0       | 0.0          | 7.0       | 7.0       | 7.0          | 1000       |
| m1-cuda-nobs     | 8.67       | 0.07          | 8.607      | 8.792      | 8.636         | 9.2       | 0.45         | 9.0       | 10.0      | 9.0          | NULL       |
| m1-opencl-autobs | 7.54       | 0.9           | 6.969      | 9.132      | 7.238         | 8.4       | 0.89         | 8.0       | 10.0      | 8.0          | 1587       |
| m1-opencl-bs1000 | 7.79       | 1.03          | 6.972      | 9.411      | 7.348         | 8.6       | 0.89         | 8.0       | 10.0      | 8.0          | 1000       |
| m1-opencl-nobs   | 16.0       | 0.0           | 16.0       | 16.0       | 16.0          | 17.0      | 0.0          | 17.0      | 17.0      | 17.0         | NULL       |
| m2-cpu6          | 22.0       | 0.0           | 22.0       | 22.0       | 22.0          | 22.0      | 0.0          | 22.0      | 22.0      | 22.0         | NULL       |
| m2-cpu6-pb       | 22.0       | 0.0           | 22.0       | 22.0       | 22.0          | 22.0      | 0.0          | 22.0      | 22.0      | 22.0         | NULL       |
| m2-opencl-autobs | 2.16       | 0.02          | 2.147      | 2.185      | 2.15          | 3.0       | 0.0          | 3.0       | 3.0       | 3.0          | 6717       |
| m2-opencl-bs1000 | 2.01       | 0.01          | 2.006      | 2.032      | 2.01          | 3.0       | 0.0          | 3.0       | 3.0       | 3.0          | 1000       |
| m2-opencl-bs3500 | 2.16       | 0.01          | 2.153      | 2.164      | 2.162         | 3.0       | 0.0          | 3.0       | 3.0       | 3.0          | 3500       |
| m2-opencl-nobs   | 8.95       | 0.08          | 8.845      | 9.064      | 8.949         | 9.8       | 0.45         | 9.0       | 10.0      | 10.0         | NULL       |

Legend:

* `cpuN`: CPU with N processes
* `pb`: Run with progress bar enabled
* `opencl`: GPU with OpenCL backend
* `cuda`: GPU with CUDA backend
* `autobs`: GPU with automatic batch size and the respective backend
* `nobs`: GPU with serial rotations and the respective backend
* `bsNNNN`: GPU with batch size of NNNN and the respective backend
* `rN`: Run number N
* `total`: Total time taken for the run, includes reading input, writing output, and all computations
* `search`: Time taken for the all computations
* `batch_size`: Batch size used for the run

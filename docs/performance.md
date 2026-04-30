# Performance

How PowerFit performs on different computational resources, such as CPU and GPU, and with different batch sizes. This can help users understand the trade-offs between different configurations and choose the best one for their needs.

## Measurements

Fetch the map and the structure for the test case:

```shell
wget https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-1046/map/emd_1046.map.gz
wget https://files.rcsb.org/download/9A2G.cif.gz
```

Runs on machines with the following specifications:

* m1: AMD Ryzen 5 5600G and NVIDIA GeForce RTX 3050
* m2: AMD Ryzen 7 7800X3D and AMD Radeon RX 7900 XTX
* m3: AMD EPYC 9554 and NVIDIA RTX 6000 Ada
* m4: Intel i7-13700H and NVIDIA RTX 4050 Laptop via WSL

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
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 250 -d runs/m2-opencl-bs250-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 200 -d runs/m2-opencl-bs200-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 150 -d runs/m2-opencl-bs150-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 100 -d runs/m2-opencl-bs100-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 75 -d runs/m2-opencl-bs75-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 50 -d runs/m2-opencl-bs50-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 10 -d runs/m2-opencl-bs10-r${run}    
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 0 -d runs/m2-opencl-nobs-r${run}
done
# On machine 3
mkdir -p runs
for run in 1 2 3 4 5; do
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 -d runs/m3-cuda-autobs-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 4000 -d runs/m3-cuda-bs4000-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 3000 -d runs/m3-cuda-bs3000-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 2000 -d runs/m3-cuda-bs2000-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 1000 -d runs/m3-cuda-bs1000-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 500 -d runs/m3-cuda-bs500-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 250 -d runs/m3-cuda-bs250-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 200 -d runs/m3-cuda-bs200-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 150 -d runs/m3-cuda-bs150-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 100 -d runs/m3-cuda-bs100-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 75 -d runs/m3-cuda-bs75-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 50 -d runs/m3-cuda-bs50-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 10 -d runs/m3-cuda-bs10-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 0 -d runs/m3-cuda-nobs-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 -d runs/m3-opencl-autobs-r${run}
    powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 4000 -d runs/m3-opencl-bs4000-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 3000 -d runs/m3-opencl-bs3000-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 2000 -d runs/m3-opencl-bs2000-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 1000 -d runs/m3-opencl-bs1000-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 500 -d runs/m3-opencl-bs500-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 250 -d runs/m3-opencl-bs250-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 200 -d runs/m3-opencl-bs200-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 150 -d runs/m3-opencl-bs150-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 100 -d runs/m3-opencl-bs100-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 75 -d runs/m3-opencl-bs75-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 50 -d runs/m3-opencl-bs50-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 10 -d runs/m3-opencl-bs10-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size 0 -d runs/m3-opencl-nobs-r${run}
done
# On machine 4
mkdir runs
for run in 1 2 3 4 5; do
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 -d runs/m4-cuda-autobs-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 1000 -d runs/m4-cuda-bs1000-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 500 -d runs/m4-cuda-bs500-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 250 -d runs/m4-cuda-bs250-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 200 -d runs/m4-cuda-bs200-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 150 -d runs/m4-cuda-bs150-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 100 -d runs/m4-cuda-bs100-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 75 -d runs/m4-cuda-bs75-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 50 -d runs/m4-cuda-bs50-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 10 -d runs/m4-cuda-bs10-r${run}
	powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size 0 -d runs/m4-cuda-nobs-r${run}
done
```

</details>

<details>
  <summary>Convert logs to table</summary>

```shell
python3 docs/parse_times.py "runs/*/*.log" > docs/times.csv
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
  FROM read_csv_auto('docs/times.csv')
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

  min(batch_size) AS batch_size,
  count(*) AS nr_runs
FROM runs
GROUP BY run_group
ORDER BY run_group;
```

</details>

The [times.csv](times.csv) contains the parsed measurements taken
around 30 April 2026 on commit 0e60abd4f69d3d438ddaee0651519a79d99fa0f3 of code.

## Batch size impact

```vegalite
{
    "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
    "title": "m3 CUDA: Batch Size vs Search Seconds",
    "data": {
        "url": "../times.csv",
        "format": {
            "type": "csv"
        }
    },
    "transform": [
        {
            "filter": {
                "and": ["slice(datum.run, 0, 10) == 'm3-cuda-bs'", {
                    "field": "batch_size",
                    "lt": 1000
                }]
            }
        }
    ],
    "params": [{
        "name": "grid",
        "select":"interval",
        "bind":"scales"
    }],
    "mark": {"type": "point", "tooltip": true},
    "encoding": {
        "x": {
            "field": "batch_size",
            "type": "quantitative",
            "title": "Batch Size"
        },
        "y": {
            "field": "search_seconds",
            "type": "quantitative",
            "title": "Search (s)"
        }
    }
}
```

```vegalite
{
    "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
    "title": "m3 OpenCL: Batch Size vs Search Seconds",
    "data": {
        "url": "../times.csv",
        "format": {
            "type": "csv"
        }
    },
    "transform": [
        {
            "filter": {
                "and": ["slice(datum.run, 0, 12) == 'm3-opencl-bs'", {
                    "field": "batch_size",
                    "lt": 1000
                }]
            }
        }
    ],
    "params": [{
        "name": "grid",
        "select":"interval",
        "bind":"scales"
    }],
    "mark": {"type": "point", "tooltip": true},
    "encoding": {
        "x": {
            "field": "batch_size",
            "type": "quantitative",
            "title": "Batch Size"
        },
        "y": {
            "field": "search_seconds",
            "type": "quantitative",
            "title": "Search (s)"
        }
    }
}
```

```vegalite
{
    "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
    "title": "m4 CUDA: Batch Size vs Search Seconds",
    "data": {
        "url": "../times.csv",
        "format": {
            "type": "csv"
        }
    },
    "transform": [
        {
            "filter": {
                "and": ["slice(datum.run, 0, 10) == 'm4-cuda-bs'", {
                    "field": "batch_size",
                    "lt": 1000
                }]
            }
        }
    ],
    "params": [{
        "name": "grid",
        "select":"interval",
        "bind":"scales"
    }],
    "mark": {"type": "point", "tooltip": true},
    "encoding": {
        "x": {
            "field": "batch_size",
            "type": "quantitative",
            "title": "Batch Size"
        },
        "y": {
            "field": "search_seconds",
            "type": "quantitative",
            "title": "Search (s)"
        }
    }
}
```

## Legend

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

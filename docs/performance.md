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
    for batch_size in 1000 500 250 200 150 100 75 50 10 0; do
        powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size $batch_size -d runs/m1-cuda-bs${batch_size}-r${run}
    done
    for batch_size in 1000 500 250 200 150 100 75 50 10 0; do
        powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size $batch_size -d runs/m1-opencl-bs${batch_size}-r${run}
    done
done
# On machine 2
mkdir -p runs
for run in 1 2 3 4 5; do
    powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --nproc 6 -d runs/m2-cpu6-r${run}
    powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --nproc 6 --progressbar -d runs/m2-cpu6-pb-r${run}
    for batch_size in 3500 1000 250 200 150 100 75 50 10 0; do
        powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size $batch_size -d runs/m2-opencl-bs${batch_size}-r${run}
    done
done

# On machine 3
mkdir -p runs
for run in 1 2 3 4 5; do
    for batch_size in 1000 500 250 200 150 100 75 50 10 0; do
        powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size $batch_size -d runs/m3-cuda-bs${batch_size}-r${run}
    done
    for batch_size in 4000 3000 2000 1000 500 250 200 150 100 75 50 10 0; do
        powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu 0:0 --batch-size $batch_size -d runs/m3-opencl-bs${batch_size}-r${run}
    done
done

# On machine 4
mkdir runs
for run in 1 2 3 4 5; do
    for batch_size in 1000 500 250 200 150 100 75 50 10 0; do
        powerfit emd_1046.map.gz 20 9A2G.cif.gz -a 4.71 --delimiter , -n 0 --gpu cuda:0 --batch-size $batch_size -d runs/m4-cuda-bs${batch_size}-r${run}
    done
done
```

</details>

<details>
  <summary>Convert logs to table</summary>

```shell
python3 docs/parse_times.py "runs/*/*.log" > docs/times.csv
python3 docs/batch_size_plot.py docs/times.csv docs/batchsize_vs_search.csv
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
The [batchsize_vs_search.csv](batchsize_vs_search.csv) file is a chart-ready view
with normalized run labels and preserved `run_number` values for replicate spread.

## Batch size impact

```vegalite
{
    "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
    "title": "Batch Size vs Search Seconds",
    "data": {
        "url": "../batchsize_vs_search.csv",
        "format": {
            "type": "csv"
        }
    },
    "transform": [
        {
            "filter": {
                "field": "batch_size",
                "lt": 1000
            }
        }
    ],
    "params": [
        {
            "name": "series",
            "select": {
                "type": "point",
                "fields": ["run"]
            },
            "bind": "legend"
        },
        {
            "name": "grid",
            "select":"interval",
            "bind":"scales"
        }
    ],
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
        },
        "color": {
            "field": "run",
            "type": "nominal",
            "title": "Run",
            "scale": {
                "scheme": "category10"
            }
        },
        "opacity": {
            "condition": {
                "param": "series",
                "value": 1
            },
            "value": 0.01
        },
        "tooltip": [
            {
                "field": "run",
                "type": "nominal",
                "title": "Run"
            },
            {
                "field": "run_number",
                "type": "ordinal",
                "title": "Run Number"
            },
            {
                "field": "batch_size",
                "type": "quantitative",
                "title": "Batch Size"
            },
            {
                "field": "search_seconds",
                "type": "quantitative",
                "title": "Search (s)"
            }
        ]
    }
}
```
(If you do not see plot, please reload the page.)

Based on plots, the default batch size is set to 100.

When batch size is set 0 then the rotations are processed one by one instead of in batches.

When you have very few rotations (`--angle 20`) setting batch size to 0 can be faster.

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

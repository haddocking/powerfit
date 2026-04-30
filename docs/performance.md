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


|         run_group         | avg_search | stddev_search | min_search | max_search | median_search | avg_total | stddev_total | min_total | max_total | median_total | batch_size | nr_runs |
|---------------------------|-----------:|--------------:|-----------:|-----------:|--------------:|----------:|-------------:|----------:|----------:|-------------:|------------|--------:|
| m1-cpu6                   | 36.0       | 2.92          | 34.0       | 41.0       | 35.0          | 36.6      | 3.21         | 34.0      | 42.0      | 35.0         | NULL       | 5       |
| m1-cpu6-pb                | 40.2       | 2.95          | 37.0       | 45.0       | 40.0          | 40.8      | 2.59         | 38.0      | 45.0      | 40.0         | NULL       | 5       |
| m1-cuda-autobs            | 6.8        | 0.6           | 6.282      | 7.832      | 6.615         | 7.6       | 0.89         | 7.0       | 9.0       | 7.0          | 1703       | 5       |
| m1-cuda-autobs2           | 8.34       | NULL          | 8.342      | 8.342      | 8.342         | 9.0       | NULL         | 9.0       | 9.0       | 9.0          | 1          | 1       |
| m1-cuda-bs100             | 6.44       | NULL          | 6.442      | 6.442      | 6.442         | 7.0       | NULL         | 7.0       | 7.0       | 7.0          | 100        | 1       |
| m1-cuda-bs1000            | 6.11       | 0.2           | 5.837      | 6.405      | 6.09          | 7.0       | 0.0          | 7.0       | 7.0       | 7.0          | 1000       | 5       |
| m1-cuda-bs250             | 5.77       | NULL          | 5.765      | 5.765      | 5.765         | 7.0       | NULL         | 7.0       | 7.0       | 7.0          | 250        | 1       |
| m1-cuda-bs300             | 5.79       | NULL          | 5.789      | 5.789      | 5.789         | 7.0       | NULL         | 7.0       | 7.0       | 7.0          | 300        | 1       |
| m1-cuda-bs400             | 5.76       | NULL          | 5.758      | 5.758      | 5.758         | 7.0       | NULL         | 7.0       | 7.0       | 7.0          | 400        | 1       |
| m1-cuda-bs500             | 5.76       | NULL          | 5.763      | 5.763      | 5.763         | 7.0       | NULL         | 7.0       | 7.0       | 7.0          | 500        | 1       |
| m1-cuda-bs600             | 5.76       | NULL          | 5.756      | 5.756      | 5.756         | 7.0       | NULL         | 7.0       | 7.0       | 7.0          | 600        | 1       |
| m1-cuda-bs800             | 5.81       | NULL          | 5.81       | 5.81       | 5.81          | 7.0       | NULL         | 7.0       | 7.0       | 7.0          | 800        | 1       |
| m1-cuda-nobs              | 8.67       | 0.07          | 8.607      | 8.792      | 8.636         | 9.2       | 0.45         | 9.0       | 10.0      | 9.0          | NULL       | 5       |
| m1-opencl-autobs          | 7.54       | 0.9           | 6.969      | 9.132      | 7.238         | 8.4       | 0.89         | 8.0       | 10.0      | 8.0          | 1587       | 5       |
| m1-opencl-autobs-r-master | 13.0       | NULL          | 13.0       | 13.0       | 13.0          | 14.0      | NULL         | 14.0      | 14.0      | 14.0         | NULL       | 1       |
| m1-opencl-bs1000          | 7.79       | 1.03          | 6.972      | 9.411      | 7.348         | 8.6       | 0.89         | 8.0       | 10.0      | 8.0          | 1000       | 5       |
| m1-opencl-nobs            | 16.0       | 0.0           | 16.0       | 16.0       | 16.0          | 17.0      | 0.0          | 17.0      | 17.0      | 17.0         | NULL       | 5       |
| m2-cpu6                   | 22.0       | 0.0           | 22.0       | 22.0       | 22.0          | 22.0      | 0.0          | 22.0      | 22.0      | 22.0         | NULL       | 5       |
| m2-cpu6-pb                | 22.0       | 0.0           | 22.0       | 22.0       | 22.0          | 22.0      | 0.0          | 22.0      | 22.0      | 22.0         | NULL       | 5       |
| m2-opencl-autobs          | 2.16       | 0.02          | 2.147      | 2.185      | 2.15          | 3.0       | 0.0          | 3.0       | 3.0       | 3.0          | 6717       | 5       |
| m2-opencl-bs100           | 1.44       | NULL          | 1.439      | 1.439      | 1.439         | 2.0       | NULL         | 2.0       | 2.0       | 2.0          | 100        | 1       |
| m2-opencl-bs1000          | 2.01       | 0.01          | 2.006      | 2.032      | 2.01          | 3.0       | 0.0          | 3.0       | 3.0       | 3.0          | 1000       | 5       |
| m2-opencl-bs150           | 1.44       | NULL          | 1.436      | 1.436      | 1.436         | 2.0       | NULL         | 2.0       | 2.0       | 2.0          | 150        | 1       |
| m2-opencl-bs200           | 1.43       | NULL          | 1.434      | 1.434      | 1.434         | 2.0       | NULL         | 2.0       | 2.0       | 2.0          | 200        | 1       |
| m2-opencl-bs250           | 1.48       | NULL          | 1.477      | 1.477      | 1.477         | 2.0       | NULL         | 2.0       | 2.0       | 2.0          | 250        | 1       |
| m2-opencl-bs300           | 1.56       | NULL          | 1.556      | 1.556      | 1.556         | 2.0       | NULL         | 2.0       | 2.0       | 2.0          | 300        | 1       |
| m2-opencl-bs3500          | 2.16       | 0.01          | 2.153      | 2.164      | 2.162         | 3.0       | 0.0          | 3.0       | 3.0       | 3.0          | 3500       | 5       |
| m2-opencl-bs50            | 1.44       | NULL          | 1.442      | 1.442      | 1.442         | 2.0       | NULL         | 2.0       | 2.0       | 2.0          | 50         | 1       |
| m2-opencl-bs500           | 1.99       | NULL          | 1.99       | 1.99       | 1.99          | 3.0       | NULL         | 3.0       | 3.0       | 3.0          | 500        | 1       |
| m2-opencl-nobs            | 8.95       | 0.08          | 8.845      | 9.064      | 8.949         | 9.8       | 0.45         | 9.0       | 10.0      | 10.0         | NULL       | 5       |
| m3-cuda-autobs            | 1.24       | 0.0           | 1.237      | 1.244      | 1.238         | 3.0       | 0.0          | 3.0       | 3.0       | 3.0          | 5825       | 5       |
| m3-cuda-bs10              | 0.94       | 0.0           | 0.937      | 0.948      | 0.942         | 2.0       | 0.0          | 2.0       | 2.0       | 2.0          | 10         | 5       |
| m3-cuda-bs100             | 0.63       | 0.01          | 0.62       | 0.644      | 0.625         | 2.0       | 0.0          | 2.0       | 2.0       | 2.0          | 100        | 5       |
| m3-cuda-bs1000            | 1.17       | 0.0           | 1.171      | 1.176      | 1.175         | 2.2       | 0.45         | 2.0       | 3.0       | 2.0          | 1000       | 5       |
| m3-cuda-bs150             | 0.64       | 0.0           | 0.634      | 0.638      | 0.636         | 2.0       | 0.0          | 2.0       | 2.0       | 2.0          | 150        | 5       |
| m3-cuda-bs200             | 0.66       | 0.0           | 0.653      | 0.661      | 0.658         | 2.0       | 0.0          | 2.0       | 2.0       | 2.0          | 200        | 5       |
| m3-cuda-bs2000            | 1.18       | 0.0           | 1.184      | 1.186      | 1.185         | 2.2       | 0.45         | 2.0       | 3.0       | 2.0          | 2000       | 5       |
| m3-cuda-bs250             | 0.7        | 0.0           | 0.696      | 0.7        | 0.7           | 2.0       | 0.0          | 2.0       | 2.0       | 2.0          | 250        | 5       |
| m3-cuda-bs3000            | 1.19       | 0.0           | 1.186      | 1.191      | 1.186         | 2.0       | 0.0          | 2.0       | 2.0       | 2.0          | 3000       | 5       |
| m3-cuda-bs4000            | 1.19       | 0.0           | 1.187      | 1.19       | 1.188         | 2.0       | 0.0          | 2.0       | 2.0       | 2.0          | 4000       | 5       |
| m3-cuda-bs50              | 0.61       | 0.0           | 0.612      | 0.614      | 0.613         | 2.0       | 0.0          | 2.0       | 2.0       | 2.0          | 50         | 5       |
| m3-cuda-bs500             | 1.17       | 0.0           | 1.172      | 1.176      | 1.174         | 2.0       | 0.0          | 2.0       | 2.0       | 2.0          | 500        | 5       |
| m3-cuda-bs75              | 0.61       | 0.0           | 0.61       | 0.615      | 0.613         | 2.0       | 0.0          | 2.0       | 2.0       | 2.0          | 75         | 5       |
| m3-cuda-nobs              | 7.11       | 0.05          | 7.054      | 7.154      | 7.146         | 8.0       | 0.0          | 8.0       | 8.0       | 8.0          | NULL       | 5       |
| m3-opencl-autobs          | 1.26       | 0.0           | 1.255      | 1.258      | 1.257         | 2.0       | 0.0          | 2.0       | 2.0       | 2.0          | 13277      | 5       |
| m3-opencl-bs100           | 0.48       | 0.0           | 0.476      | 0.48       | 0.477         | 2.0       | 0.0          | 2.0       | 2.0       | 2.0          | 100        | 5       |
| m3-opencl-bs1000          | 1.08       | 0.0           | 1.074      | 1.08       | 1.076         | 2.0       | 0.0          | 2.0       | 2.0       | 2.0          | 1000       | 5       |
| m3-opencl-nobs            | 14.0       | 0.0           | 14.0       | 14.0       | 14.0          | 15.2      | 0.45         | 15.0      | 16.0      | 15.0         | NULL       | 5       |
| m4-cuda-autobs            | 15.2       | 1.79          | 14.0       | 18.0       | 14.0          | 16.0      | 2.0          | 14.0      | 19.0      | 15.0         | 1          | 5       |
| m4-cuda-bs10              | 2.68       | 0.15          | 2.527      | 2.901      | 2.623         | 3.2       | 0.45         | 3.0       | 4.0       | 3.0          | 10         | 5       |
| m4-cuda-bs100             | 4.25       | 1.32          | 3.646      | 6.602      | 3.662         | 5.4       | 1.95         | 4.0       | 8.0       | 4.0          | 100        | 5       |
| m4-cuda-bs1000            | 4.45       | 0.03          | 4.419      | 4.503      | 4.433         | 5.0       | 0.0          | 5.0       | 5.0       | 5.0          | 1000       | 5       |
| m4-cuda-bs150             | 4.39       | 0.01          | 4.373      | 4.4        | 4.396         | 5.2       | 0.45         | 5.0       | 6.0       | 5.0          | 150        | 5       |
| m4-cuda-bs200             | 4.39       | 0.01          | 4.368      | 4.403      | 4.39          | 5.0       | 0.0          | 5.0       | 5.0       | 5.0          | 200        | 5       |
| m4-cuda-bs250             | 5.56       | 1.61          | 4.38       | 7.34       | 4.386         | 6.2       | 1.64         | 5.0       | 8.0       | 5.0          | 250        | 5       |
| m4-cuda-bs50              | 2.83       | 1.29          | 2.198      | 5.123      | 2.212         | 3.6       | 1.34         | 3.0       | 6.0       | 3.0          | 50         | 5       |
| m4-cuda-bs500             | 4.4        | 0.01          | 4.384      | 4.414      | 4.395         | 5.6       | 1.34         | 5.0       | 8.0       | 5.0          | 500        | 5       |
| m4-cuda-bs75              | 2.68       | 0.01          | 2.673      | 2.695      | 2.685         | 3.2       | 0.45         | 3.0       | 4.0       | 3.0          | 75         | 5       |
| m4-cuda-nobs              | 18.4       | 2.97          | 15.0       | 22.0       | 17.0          | 19.4      | 2.97         | 16.0      | 23.0      | 18.0         | NULL       | 5       |

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

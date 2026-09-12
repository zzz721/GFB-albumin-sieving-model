# GBM 100-network demonstration

The manuscript-review demonstration contains 50 WT and 50 AS synthetic GBM
networks. 

The data archive is distributed as the GitHub Release asset
[gbm-demo-100.zip](https://github.com/zzz721/GFB-albumin-sieving-model/releases/download/gfb-model-v1.0.2/gbm-demo-100.zip); it is not committed to the Git repository because its size is approximately 107 MiB.

After extracting the archive, follow its top-level `README.md`. The full demo
runs 100 networks under both conditions, giving 200 calculations. The expected
flow-weighted integrated sieving coefficients are:

| Group | Exclusion radius (nm) | Hydrodynamic radius (nm) | Expected integrated sieving |
|---|---:|---:|---:|
| WT | 3.55 | 3.55 | 0.0162575191 |
| AS | 3.55 | 3.55 | 0.1269321692 |
| WT | 4.25 | 3.55 | 0.0047920408 |
| AS | 4.25 | 3.55 | 0.0735055270 |

`manifest.csv` lists the network identifiers in the demonstration, and
`population_comparison.csv` records the demo and complete-ensemble aggregates.

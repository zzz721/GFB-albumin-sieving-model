# Generate two GBM demonstration networks

This demonstration samples one AS and one WT network from the published
Phase 1 thickness distribution and Phase 2 structural parameters, then runs
classification and albumin transport on both networks.

From the `gbm-albumin-sieving` directory, install the package and run:

```text
python -m pip install -e .
python workflows/simulate_fitted_networks.py --n-runs 1 --sample-types AS WT --seed 20260912 --run-result-dir outputs/generation_demo --no-retry-disconnected
```

`--n-runs 1` means one network for each requested sample type, giving two
networks in total. No separate Phase 1 or Phase 2 path is required because the
fitted parameters are included under `data/fitted_parameters/`.

On the tested Windows 11 desktop, generation and transport took approximately
60 seconds. The principal generated files are:

```text
outputs/generation_demo/phase3_synthetic/networks/synthetic_AS_run0_plane_pores.xlsx
outputs/generation_demo/phase3_synthetic/networks/synthetic_AS_run0_plane_throats.xlsx
outputs/generation_demo/phase3_synthetic/networks/synthetic_WT_run0_plane_pores.xlsx
outputs/generation_demo/phase3_synthetic/networks/synthetic_WT_run0_plane_throats.xlsx
outputs/generation_demo/phase4_sieving/sieving_per_run.xlsx
outputs/generation_demo/phase4_sieving/sieving_statistics.json
```

With the tested dependency versions in `requirements-tested.txt`, the fixed
seed produces the values in `expected_summary.csv`. Both rows should report
`phase3_status=ok`, `phase4_status=ok`, and `Stable=Yes`.

# Generated Network Data Asset

The main fitted-network analysis used 10,000 generated GBM pore-throat networks:
5,000 Healthy/WT networks and 5,000 Alport syndrome/AS networks. To keep the
GitHub source package small, these generated Phase3 network tables are distributed
as a separate data asset rather than stored directly in the source folder.

After extracting the generated-network data asset, the expected layout is:

```text
gbm-albumin-sieving/
  data/
    generated_networks/
      phase3_synthetic/
        manifest_10000_generated_networks.csv
        networks/
          synthetic_WT_run0_plane_pores.xlsx
          synthetic_WT_run0_plane_throats.xlsx
          ...
```

The Phase3 directory passed to the workflow should be:

```text
data/generated_networks/phase3_synthetic
```

Example:

```powershell
python workflows\simulate_fitted_networks.py `
  --n-runs 5000 `
  --sample-types AS WT `
  --phase3-dir data\generated_networks\phase3_synthetic `
  --run-result-dir outputs\main_5000_from_restored_networks `
  --no-clear `
  --skip-existing
```

Use `--no-clear` to avoid deleting the restored Phase3 network tables. The
`--skip-existing` option makes the run restart-safe and allows downstream Phase4
calculations to be resumed.

The matching run-level reproducibility metadata is stored in
`configs/reproducibility/`.

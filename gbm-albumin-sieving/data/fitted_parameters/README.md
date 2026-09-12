# Fitted parameters for synthetic GBM networks

These JSON files are the fitted model inputs required by
`workflows/simulate_fitted_networks.py` to generate synthetic WT and AS
pore-throat networks.

- `phase1_thickness/thickness_distribution_parameters.json` describes the
  fitted WT and AS thickness distributions.
- `phase2_parameters/` contains the fitted thickness-dependent structural
  relations used to sample pore and throat properties.

The workflow uses these directories automatically when project-local fitted
results are unavailable. The files are a compact publication set of fitted
parameters; the fitting code is under `src/gbm_sieving/analysis/`.

To generate one WT and one AS demonstration network, follow
`examples/generation_2/README.md`.

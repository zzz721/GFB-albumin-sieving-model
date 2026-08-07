# Structure-Resolved GBM Albumin Sieving

This project contains the analysis and simulation code for converting reconstructed
glomerular basement membrane (GBM) pore-throat networks into structure-resolved
predictions of solvent flow and albumin sieving.

The repository is being assembled as a clean copy of the working research code.
The original `analyze_and_calculate_2` and `gbm_full_model` directories remain
unchanged and are not runtime dependencies of this project.

## Code organization

- `src/gbm_sieving/analysis`: sample cutting, network extraction, thickness analysis,
  structural statistics, and parameter fitting.
- `src/gbm_sieving/simulation`: synthetic-network generation, transport, electrostatic
  exclusion effective-radius analysis, and parameter-replacement experiments.
- `src/gbm_sieving/data_io`: result schemas, unit-safe readers/writers, naming, and aggregation.
- `src/gbm_sieving/visualization`: manuscript, supplementary, and diagnostic figures.
- `workflows`: user-facing pipelines that connect the package modules.

## Naming policy

Generated artifacts use:

`{sample_id}__{artifact}[__{variant}].{extension}`

Examples:

- `WT118_sub01__pore_classification.xlsx`
- `synthetic_WT_run0001__sieving_summary__radius-4p25nm.xlsx`

Result readers use the canonical names produced by this project. Intermediate Excel
reader/writer pairs will be renamed together so the pipeline always remains consistent.

## Unit policy

Internal calculations use SI units. Units are included in exported column names.
Conversions are centralized in `gbm_sieving.data_io.units`; in particular,
`1 nL/s = 1e-12 m^3/s`.

## Albumin transport model

The cylindrical-throat solver uses the Dechadilok-Deen (2006) approximations as
its hydrodynamic hindrance model. For each albumin-accessible throat, the code
stores the DD diffusive hindrance factor as `K_D` and calculates
`D_eff = D0 * K_D`; this `K_D` is the reciprocal form of a drag-enhancement
factor. The albumin advective velocity is `v_alb = K_C * v_water`, and steric
exclusion is represented by the albumin-center accessible area
`A_eff = pi * (r_throat - r_albumin)^2`.

Every standard concentration solve enables final low-connectivity cluster pruning
by default. Internal abnormal clusters are identified using `C < 1e-6` or
`C > 1e4`, removed only when an inlet-to-outlet albumin pathway remains, and the
field is solved repeatedly until no further cluster can be safely removed (up to
100 passes). The thresholds can be changed with
`--auto-prune-oob-low-threshold` and `--auto-prune-oob-high-threshold`.

See `docs/FILE_GUIDE.md` for a description of every file.

## Local setup and checks

```text
python -m pip install -e ".[dev]"
```

The two user-facing entry points are:

```text
python workflows/analyze_real_networks.py --help
python workflows/simulate_fitted_networks.py --help
```

Input defaults may reuse data in the parent research workspace. All new default
outputs are written below `outputs/`, which is excluded from Git.

Random seeds and small realized inputs for the principal 5,000-run, electrostatic
exclusion-radius, and parameter-replacement experiments are versioned under
`configs/reproducibility/`.

The current main-text figure whitelist, frozen inputs, commands, and approved PNG
hashes are documented in `src/gbm_sieving/visualization/manuscript/README.md`.

The cleaned first-stage workflow intentionally excludes the old iterative validation
solver and the separate whole-sample (`overall_*`) branch.

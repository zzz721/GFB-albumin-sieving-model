# Structure-Resolved GBM Albumin Sieving

This project contains the analysis and simulation code for converting reconstructed
glomerular basement membrane (GBM) pore-throat networks into structure-resolved
predictions of solvent flow and albumin sieving.

The package contains the reviewed source code and small reproducibility records.
The complete 5,000-network-per-group manuscript pool is not committed; the
standalone 100-network demonstration is distributed separately.

## Code organization

- `src/gbm_sieving/analysis`: sample cutting, network extraction, thickness analysis,
  structural statistics, and parameter fitting.
- `src/gbm_sieving/simulation`: synthetic-network generation, transport, electrostatic
  exclusion effective-radius analysis, and parameter-replacement experiments.
- `src/gbm_sieving/data_io`: result schemas, unit-safe readers/writers, naming, and aggregation.
- `workflows`: user-facing pipelines that connect the package modules.
- `data/fitted_parameters`: published Phase 1 thickness and Phase 2 structural
  parameters for synthetic WT and AS network generation.

## Naming policy

Generated artifacts use:

`{sample_id}__{artifact}[__{variant}].{extension}`

Examples:

- `WT118_sub01__pore_classification.xlsx`
- `synthetic_WT_run0001__sieving_summary__radius-4p25nm.xlsx`

Result readers use the canonical names produced by this project. Intermediate Excel
reader/writer pairs use the same canonical names throughout the pipeline.

## Unit policy

Internal calculations use SI units. Units are included in exported column names.
Conversions are centralized in `gbm_sieving.data_io.units`; in particular,
`1 nL/s = 1e-12 m^3/s`.

## Albumin transport model used for the manuscript

The manuscript solver is
`gbm_sieving.simulation.transport.dd2006_conservative_solver`. It separates the
radius used for geometric/electrostatic exclusion from the physical hydrated
radius used for hydrodynamic hindrance. The exclusion radius controls throat
accessibility and `Phi`; the hydrodynamic radius controls the Dechadilok-Deen
(2006) `K_D` and `K_C` factors. Diffusion therefore uses
`D0 * Phi * K_D * A/L`, and advection uses `Phi * K_C * Q_water`.

| Condition | Exclusion radius | Hydrodynamic radius for `K_D`, `K_C` |
|---|---:|---:|
| Steric only | 3.55 nm | 3.55 nm |
| Electrostatic effective-radius approximation | 4.25 nm | 3.55 nm |

The electrostatic-radius workflow now calls this solver and passes both values
explicitly. The older `sieving_solver.py` remains for historical workflow
comparison; it is not used to reproduce the final electrostatic manuscript
results.

Every standard concentration solve enables final low-connectivity cluster pruning
by default. Internal abnormal clusters are identified using `C < 1e-6` or
`C > 1e4`, removed only when an inlet-to-outlet albumin pathway remains, and the
field is solved repeatedly until no further cluster can be safely removed (up to
100 passes). The thresholds can be changed with
`--auto-prune-oob-low-threshold` and `--auto-prune-oob-high-threshold`.

See `docs/FILE_GUIDE.md` for a description of every file.

## Local setup and checks

For normal use:

```text
python -m pip install -e .
```

To install the additional development and test tools:

```text
python -m pip install -e ".[dev]"
```

The exact package versions tested on Windows 11 with Python 3.12.6 are listed
in `requirements-tested.txt`. No GPU or non-standard hardware is required.
A fresh Windows environment containing the exact dependencies and all three
repository packages was created in approximately 2 minutes 18 seconds on the
tested desktop. Download time and hardware affect this value.

The principal user-facing entry points are:

```text
python workflows/analyze_real_networks.py --help
python workflows/simulate_fitted_networks.py --help
python workflows/run_existing_network_dd2006.py --help
```

To generate one WT and one AS demonstration network from the included fitted
parameters, run:

```text
python workflows/simulate_fitted_networks.py --n-runs 1 --sample-types AS WT --seed 20260912 --run-result-dir outputs/generation_demo --no-retry-disconnected
```

This took approximately 60 seconds on the tested desktop. It creates the pore
and throat workbooks below `outputs/generation_demo/phase3_synthetic/networks/`
and transport summaries below `outputs/generation_demo/phase4_sieving/`. Exact
expected values and output paths are documented in
`examples/generation_2/README.md`.

To use a new network, place `NAME_pores.xlsx` and `NAME_throats.xlsx` in
`DATA/NAME`, generate the boundary and solvent-connectivity classifications,
and then run the transport solver:

```text
python src/gbm_sieving/simulation/transport/analyze_network.py --sample-name NAME --sample-dir DATA/NAME --output-dir DATA --solute-radius-nm 3.55 --no-html
python workflows/run_existing_network_dd2006.py --sample-name NAME --sample-dir DATA/NAME --analysis-root DATA --output-dir outputs/example --solute-radius-nm 4.25 --hydrodynamic-radius-nm 3.55
```

For one previously classified network, run only the second command and pass
both radii explicitly. Use `--solute-radius-nm 3.55` for the steric condition;
use `4.25` for the electrostatic effective-radius condition while retaining
`--hydrodynamic-radius-nm 3.55`.

This command expects the five input workbooks documented in the README inside
the GBM demo archive. Geometry changes require consistent regeneration of the
boundary and solvent-connectivity classifications.

Refitting the distributions and analyzing reconstructed experimental networks
requires the corresponding input data. The included fitted parameters are
sufficient for synthetic-network generation. All default generated outputs are
written below `outputs/`, which is excluded from Git.

Random seeds and small realized inputs for the principal 5,000-run, electrostatic
exclusion-radius, and parameter-replacement experiments are versioned under
`configs/reproducibility/`.

## Demonstration data

The 100-network review demonstration is described in `examples/demo_100/`.
Because the compressed input archive is about 107 MiB, it is supplied as the
GitHub Release asset `gbm-demo-100.zip`; the ordinary Git repository contains
only its small manifest, aggregate comparison, and instructions.

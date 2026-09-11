# GBM package file guide

This guide describes the files included in the current release candidate.
Generated data, caches, large network inputs, and manuscript plotting code that
has not yet been migrated are not part of this package.

## Repository files

- `README.md` gives installation, model, demo, and single-network use
  instructions.
- `pyproject.toml` defines the Python package and runtime dependencies.
- `requirements-tested.txt` records exact package versions used in validation.
- `.gitignore` excludes generated outputs, environments, and local research data.
- `configs/reproducibility/` records realized seeds, thickness arrays, and
  settings for the main synthetic-network and sensitivity calculations.
- `examples/demo_100/` contains the small manifest, aggregate checkpoints,
  validation summary, and instructions for downloading the separate Release
  asset.

## Analysis package

- `analysis/cutting/` contains the interactive sample and large-subsample
  selection tools.
- `analysis/network_extraction/extract_network.py` identifies inlet/outlet
  surfaces, computes thickness, classifies solvent and albumin-accessible
  networks, and writes the classification tables required by transport.
- `analysis/structure/` calculates structural statistics and compares real
  subsamples.
- `analysis/thickness/` performs thickness quality control and distribution
  fitting.
- `analysis/parameter_fitting/` fits thickness-dependent pore and throat
  distributions used by synthetic-network generation.

## Result handling

- `data_io/units.py` contains the shared physical unit conversions.
- `data_io/filenames.py` defines canonical generated filenames.
- `data_io/aggregation.py` reads solver results, validates ranges, computes
  flow-weighted aggregates, and prepares comparison outputs.
- `paths.py` defines the package, project, workspace, and default output roots.

## Simulation package

- `simulation/network_generation/generator.py` generates fitted synthetic WT
  and AS pore-throat networks.
- `simulation/transport/analyze_network.py` creates boundary and solvent/
  albumin connectivity classifications for a synthetic network.
- `simulation/transport/hindrance.py` is the shared Dechadilok-Deen (2006)
  implementation for diffusive and convective hindrance factors.
- `simulation/transport/dd2006_conservative_solver.py` is the manuscript
  production transport solver. It re-solves pressure, separates exclusion and
  hydrodynamic radii, solves concentration with outlet closure, performs the
  standardized pruning step, and writes the final flow and sieving results.
- `simulation/transport/sieving_solver.py` is retained for traceability to
  earlier workflow outputs. It is not the final electrostatic manuscript
  solver.
- `simulation/experiments/electrostatic_exclusion_radius/run_analysis.py`
  reruns saved networks at the two exclusion radii while keeping the
  hydrodynamic radius at 3.55 nm, then calculates paired summaries.
- `simulation/experiments/parameter_replacement/run_analysis.py` runs the
  configured parameter-replacement experiments.

## User-facing workflows

- `workflows/analyze_real_networks.py` runs the real-subsample analysis path.
- `workflows/simulate_fitted_networks.py` runs fitted network generation,
  classification, transport, restart/retry handling, and aggregation.
- `workflows/run_existing_network_dd2006.py` runs the final transport solver on
  one previously classified network and is the smallest public GBM entry point.

The current package does not claim one-command reproduction of every manuscript
figure. The repository-level `MANUSCRIPT_RESULTS_MAP.md` lists which figure
inputs and plotting workflows remain to be migrated before that optional scope
can be claimed.


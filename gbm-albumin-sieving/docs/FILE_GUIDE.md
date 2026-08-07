# File guide

This guide describes every version-controlled file in the first cleaned copy.
Generated data, caches, and temporary smoke-test outputs are not part of the project.

## Repository root

- `.gitignore` — prevents local environments, caches, research data, Excel outputs,
  HTML figures, logs, and `outputs/` from being committed; permits deliberately small
  fixtures below `tests/fixtures/` and `examples/data/`.
- `pyproject.toml` — Python package metadata, supported Python version, runtime
  dependencies, optional test/lint dependencies, package discovery, and test/lint
  configuration.
- `README.md` — project purpose, directory layout, naming and unit policies, setup,
  entry points, and the current migration boundary.
- `configs/README.md` — policy for version-controlled run configurations.
- `configs/reproducibility/README.md` — explains the seed records required for the
  principal experiments.
- `configs/reproducibility/main_5000_simulation.json` — main 5,000-AS + 5,000-WT
  experiment metadata, seed rules, source hashes, and status counts.
- `configs/reproducibility/main_5000_run_seeds.csv` — exact final Phase 3 seed, retry
  indices, thickness, and no-exit RNG seed for every one of the 10,000 networks.
- `configs/reproducibility/main_5000_thickness_AS.json` — realized AS thickness array;
  required because the original thickness draw had no fixed seed.
- `configs/reproducibility/main_5000_thickness_WT.json` — realized WT thickness array;
  required because the original thickness draw had no fixed seed.
- `configs/reproducibility/electrostatic_exclusion_radius.json` — reused network pool,
  radii, bootstrap seed, and plot-jitter seed.
- `configs/reproducibility/parameter_replacement.json` — actual sensitivity seed,
  variant list, paired design, and deterministic WT/AS per-run seed formulas.
- `examples/README.md` — policy for future small redistributable examples.
- `docs/FILE_GUIDE.md` — this per-file reference.

## Package root

- `src/gbm_sieving/__init__.py` — marks `gbm_sieving` as the public Python package and
  exposes package-level metadata.
- `src/gbm_sieving/paths.py` — defines package, source, project, output, and surrounding
  research-workspace roots. Existing data may be read from the workspace; new default
  output stays inside this project.

## Analysis: cutting

- `analysis/__init__.py` — package marker for all real-network analysis code.
- `analysis/cutting/__init__.py` — package marker for cutting and selection tools.
- `analysis/cutting/sample_splitter.py` — interactive selection of two cutting lines,
  projection of pore coordinates, subdivision of a sample, throat filtering, and
  writing each cut subsample plus its normal-vector metadata.
- `analysis/cutting/large_subsample_selector.py` — interactive three-point rectangle
  selection for a larger local subsample, polygon inclusion tests, associated throat
  filtering, and metadata/output writing.
- `analysis/cutting/subsample_overview.py` — loads all subsamples for one specimen and
  produces combined 3D geometry/radius views for visual checking of the cut regions.

## Analysis: network extraction

- `analysis/network_extraction/__init__.py` — package marker for pore-network extraction.
- `analysis/network_extraction/extract_network.py` — the real-subsample analyzer: loads
  pore/throat Excel tables, estimates an alpha shape, identifies inlet/outlet surface
  pores by ray casting, calculates normal-aware thickness, classifies solvent and
  albumin-accessible networks, finds penetrating components, and writes intermediate
  tables/optional 3D views.

## Analysis: structure

- `analysis/structure/__init__.py` — package marker for structural statistics.
- `analysis/structure/statistics.py` — calculates convex-hull/expanded volumes, pore and
  throat volume fractions, degree distributions, radius statistics, global and local
  orientation tensors, and radial distribution functions.
- `analysis/structure/compare_subsamples.py` — applies the structural metrics to all
  valid subsamples of one specimen, optionally compares them with the whole specimen,
  and produces summary tables, violin plots, and distribution tests.
- `analysis/structure/thickness_relationships.py` — combines thickness and structural
  metrics across samples, fits thickness–structure relationships, and creates WT, AS,
  and combined correlation figures.

## Analysis: thickness and parameter fitting

- `analysis/thickness/__init__.py` — package marker for thickness analysis.
- `analysis/thickness/qc_large_subsamples.py` — quality control for local large-sample
  thickness: reads surface classifications and summaries, recomputes local-window
  thickness statistics, compares methods, and plots QC summaries.
- `analysis/thickness/fit_distribution.py` — collects corrected thickness values,
  checks permeation-direction angles, visualizes distributions, detects multimodality,
  and fits/saves WT and AS thickness-distribution parameters.
- `analysis/parameter_fitting/__init__.py` — package marker for fitted model parameters.
- `analysis/parameter_fitting/fit_thickness_trends.py` — aggregates real-network
  summaries and fits thickness-dependent trends for density, radii, throat length,
  degree, connectivity, and related Phase 1.5 relationships.
- `analysis/parameter_fitting/fit_pore_throat_parameters.py` — the main Phase 2 fitting
  module: pools pore/throat observations by sample type and thickness, fits empirical,
  parametric, KDE, and GMM distributions/correlations, handles sparse AS bins, and
  exports the parameter JSON/figures used by network generation.

## Result storage and aggregation

The code package is called `data_io` to distinguish result-handling code from the
generated `outputs/` directory.

- `data_io/__init__.py` — package marker for result readers, writers, units, and
  aggregation.
- `data_io/units.py` — authoritative physical conversion constants and functions,
  including `1 nL/s = 1e-12 m^3/s`, nm-to-m, area, and mmHg-to-Pa conversions.
- `data_io/filenames.py` — sanitizes identifiers and constructs/finds canonical
  `{sample}__{artifact}[__{variant}].ext` filenames, including radius variants.
- `data_io/aggregation.py` — canonical sieving-result reader, physical-range
  validation, single-network Phase 4 orchestration, multi-run aggregation, and WT/AS
  comparison plots.

## Simulation: network generation

- `simulation/__init__.py` — package marker for simulation code.
- `simulation/network_generation/__init__.py` — package marker for synthetic generation.
- `simulation/network_generation/generator.py` — the fitted synthetic GBM generator:
  samples Phase 1/2 distributions, places pores, constructs overlap/non-overlap throats,
  enforces density/degree/radius relationships and geometry constraints, and writes
  synthetic pore/throat networks plus optional diagnostics.

## Simulation: transport

- `simulation/transport/__init__.py` — package marker for transport calculation.
- `simulation/transport/analyze_network.py` — Phase 4 preprocessing for synthetic
  networks: fixes permeation to Z, extracts inlet/outlet surfaces, writes the axis into
  the classification table, and classifies solvent/albumin penetrating components.
- `simulation/transport/hindrance.py` — the single authoritative implementation of
  the Dechadilok-Deen (2006) cylindrical-pore diffusive and convective hindrance
  factors, plus the standard equivalent-concentration pruning predicate.
- `simulation/transport/sieving_solver.py` — current production solvent/albumin solver:
  builds the hydraulic pressure system, computes throat flow, applies the DD2006
  hindrance factors, solves the concentration system with global outlet closure,
  repeatedly prunes removable clusters outside `1e-6 <= C <= 1e4`, calculates
  `C_out/C_in`, and writes canonical pressure, concentration, coefficient, Peclet, and
  hindrance results. It reads the explicit permeation axis produced by the analyzer and
  safely exports small-system debug matrices.

## Tests

- `tests/test_hindrance.py` — regression tests for DD2006 limiting and representative
  values, its near-occlusion branch, and the default two-sided concentration thresholds.

## Simulation: experiments

- `simulation/experiments/__init__.py` — package marker for simulation experiments.
- `simulation/experiments/electrostatic_exclusion_radius/__init__.py` — package marker
  for the renamed charge-exclusion-radius analysis.
- `simulation/experiments/electrostatic_exclusion_radius/run_analysis.py` — reuses the
  same generated networks at multiple effective albumin radii, invokes the packaged
  Phase 4 analyzer/solver, sanitizes invalid or near-zero coefficients, aggregates
  paired radius results, bootstraps summaries, and creates comparison outputs.
- `simulation/experiments/parameter_replacement/__init__.py` — package marker for
  parameter-replacement/ablation experiments.
- `simulation/experiments/parameter_replacement/run_analysis.py` — creates controlled
  variants that replace selected WT/AS thickness, radius, density, degree, length, or
  correlation parameters; generates paired small networks, runs transport, and writes
  compact sensitivity summaries.

## Visualization

- `visualization/__init__.py` — package marker for reusable plotting code.
- `visualization/style.py` — applies the shared Arial/DejaVu Sans and vector-font
  Matplotlib settings.
- `visualization/diagnostics/__init__.py` — placeholder package for future promoted
  diagnostic plots; no diagnostic script has yet passed the inclusion audit.
- `visualization/manuscript/__init__.py` — package marker for main-text figures.
- `visualization/manuscript/fiber_slice.py` — shared rasterizer used only by the selected
  WT311 and five-sample concentration/pressure section figures.
- `visualization/manuscript/fiber_slices.py` — low-level single-section renderer used by
  the shared rasterizer.
- `visualization/manuscript/voxelization.py` — pore/throat-to-voxel conversion required
  by the section renderer.
- `visualization/manuscript/figure4_wt311_sections/plot_fig4E.py` — creates the WT311
  pressure section; `plot_fig4F.py` creates the concentration section. Both use the
  internal `_shared.py` renderer.
- `visualization/manuscript/figure5_concentration_comparison/plot_fig5I.py` — creates only the
  selected rotated five-sample concentration row.
- `visualization/manuscript/figure5_binned_distributions/plot_fig5E_H.py` — creates only the
  selected 40 nm/tail-400 bimodal-component distribution panel.
- `visualization/manuscript/figure6_sieving/plot_fig6C_E.py` — creates only the selected Phase 4
  panel with disconnected networks.
- `visualization/manuscript/figure6_electrostatic_exclusion/plot_fig6A_B.py` — creates only the
  selected three-panel electrostatic-exclusion row with the integrated bar third.
- `visualization/supplementary/__init__.py` — package marker for supplementary figures.
- `visualization/supplementary/README.md` — approved Fig. S1–S8 and S10–S12 entry
  points, frozen-data policy, and PNG output names; Fig. S9 is intentionally absent.
- `visualization/supplementary/parameter_distribution_panels.py` — shared frozen-CSV
  renderer used by Figs. S4 and S5.
- Each `visualization/supplementary/figureS*/plot_figS*.py` script produces only the
  selected PNG panel(s) for that figure. Its adjacent `frozen_data/` directory stores
  the smallest practical reproducibility input copied or derived from read-only
  original results.
- SD-specific sieving and Fig. 3G rendering are maintained in the separate
  `sd-albumin-sieving` repository package.

## Workflows

- `workflows/analyze_real_networks.py` — user-facing real-subsample workflow: discovers
  cut or large subsamples, reads normal metadata, calls packaged extraction and
  sieving modules, aggregates summaries, and plots thickness versus sieving. The old
  iterative and whole-sample branches are explicitly gated in this first copy.
- `workflows/simulate_fitted_networks.py` — user-facing fitted-model workflow: samples
  thickness, generates repeated WT/AS networks, runs Phase 4, supports restart/retry
  behavior, aggregates physical/sieving outputs, and produces the main run summaries.

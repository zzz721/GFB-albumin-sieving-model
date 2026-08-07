# Reproducibility records

These files describe the random state used for the principal reported experiments.
They are small enough to commit to Git and do not contain simulation networks or
Phase 4 result tables.

- `main_5000_simulation.json` records the 5,000-AS + 5,000-WT experiment and its seed
  derivation rules.
- `main_5000_run_seeds.csv` contains the actual final Phase 3 seed and retry state for
  every one of the 10,000 networks.
- `main_5000_thickness_AS.json` and `main_5000_thickness_WT.json` preserve the actual
  sampled thickness arrays. These are necessary because the original thickness draw
  was run without a global seed.
- `electrostatic_exclusion_radius.json` records the reused network pool, radii,
  bootstrap seed, and plot-jitter seed.
- `parameter_replacement.json` records the actual sensitivity-experiment seed and the
  deterministic per-sample seed formulas.

An empty or `null` seed means that no fixed seed was supplied for that random draw. It
must not be interpreted as seed zero. Where this occurred in the main experiment, the
realized values are stored explicitly.

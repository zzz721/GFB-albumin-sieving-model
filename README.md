# GFB albumin sieving model

This repository contains the code used to calculate structure-derived albumin
sieving through the glomerular filtration barrier (GFB). The model is organized
into three connected packages:

| Directory | Role |
|---|---|
| `gbm-albumin-sieving/` | Pore-network water flow and albumin transport through the GBM, including separate exclusion and hydrodynamic radii. |
| `sd-albumin-sieving/` | Slit-diaphragm geometry, orientation-dependent accessible area, and lower/upper sieving calculation. |
| `gfb-integration/` | Self-consistent bulk–GBM–bulk–SD calculation with common water and albumin flow rates. |

Each package has its own installation and use instructions. Install and run it
from its directory so outputs remain separated.

## Quick verification

```text
cd sd-albumin-sieving
python -m pip install -e .
python workflows/calculate_sd_sieving.py

cd ../gfb-integration
python workflows/run_gfb_model.py
```

The GBM software demonstration is provided separately as the GitHub Release
asset `gbm-demo-100.zip` because the archive is about 107 MiB. It contains 50 WT
and 50 AS networks and verifies both exclusion-radius conditions. See
`gbm-albumin-sieving/examples/demo_100/README.md`; replace its Release URL
placeholder after the release is published.

## Manuscript calculations

The manuscript-to-code map is in `MANUSCRIPT_RESULTS_MAP.md`. It records the
input, command, and expected result for the main SD, GBM, electrostatic, and
whole-GFB calculations. `VALIDATION_REPORT.md` records the completed local
checks, `CHANGE_SUMMARY_ZH.md` explains this update in Chinese, and
`NATURE_CODE_CHECKLIST.md` records the remaining submission tasks.

The electrostatic effective-radius approximation uses 4.25 nm only for
accessibility and the partition factor. The physical hydrated radius used for
DD2006 `K_D` and `K_C` remains 3.55 nm.

The serial GFB model uses a common `Q_water` and a common `Q_albumin` through
all layers. Layer transfer ratios are reported after solving this conserved-flow
system.

## License

A software license has not yet been selected. The authors must add a `LICENSE`
file and update this section before the first public release.

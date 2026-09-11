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
asset [gbm-demo-100.zip](https://github.com/zzz721/GFB-albumin-sieving-model/releases/download/gbm-example-v1.0.0/gbm-demo-100.zip). It contains 50 WT and 50 AS networks and verifies both exclusion-radius conditions. See
`gbm-albumin-sieving/examples/demo_100/README.md` for instructions and expected
outputs.

## Reproducibility and validation

Each package README gives the commands and expected outputs for its supplied
example data. `VALIDATION_REPORT.md` summarizes the fresh-environment,
numerical, flow-conservation, and public Release checks. Reproducibility inputs
for the GBM ensemble calculations are versioned under
`gbm-albumin-sieving/configs/reproducibility/`.

## License

This software is released under the [MIT License](LICENSE).

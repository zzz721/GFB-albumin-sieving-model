# Manuscript result and software map

This map follows the September 6 manuscript version. Expected numerical values
are checkpoints, not replacements for the complete output files.

| Manuscript item | Input | Calculation / command | Expected checkpoint | Status |
|---|---|---|---|---|
| Fig. 3G, SD orientation bounds and sieving | `sd-albumin-sieving/src/sd_albumin_sieving/data/` | `python workflows/recalculate_orientation_bounds.py` followed by `python workflows/calculate_sd_sieving.py` | lower 0.189359; upper 0.764459; midpoint 0.476909 | Code integrated; manuscript run uses 101 width points × 5,000 random orientations per point |
| Fig. 6E, healthy GFB with electrostatic exclusion | Final WT GBM coefficient, SD midpoint, bulk parameters | `python gfb-integration/workflows/run_gfb_model.py` | plasma-to-GBM ≈1.0016; GBM ≈0.005; GBM-to-SD ≈1.0008; GFB ≈0.0024 | Conserved-flow module included |
| Fig. 6F, whole-GFB ensembles | Integrated GBM result table and fixed rounded figure factors | GFB workflow output column `figure6_fixed_factor_product` | WT 0.0078/0.0024; AS 0.060/0.035 without/with electrostatic exclusion | Calculation represented; plotting workflow still to be migrated |
| Extended Data Fig. 10, fitted network descriptors | Fitted parameter tables and generated network summaries | GBM fitting and generation workflows | Distributions in the final figure source data | Fitting code present; final fitted inputs/source-data bundle still to be added |
| Extended Data Fig. 12, integrated GBM sieving | 5,000 WT and 5,000 AS generated networks | GBM conservative solver and flow-weighted aggregation | WT 0.01625797; AS 0.12543062 | Final saved values traced; full network pool is not committed |
| Extended Data Fig. 14, electrostatic exclusion | Same generated networks at 4.25-nm exclusion and 3.55-nm hydrodynamic radius | `gbm_sieving.simulation.experiments.electrostatic_exclusion_radius.run_analysis` | WT 0.00500058; AS 0.07212139 | Correct dual-radius path integrated; full rerun remains to be archived |

For the ensemble coefficient, sum albumin and water flows first:

```text
S_integrated = sum(Q_albumin) / (C_GBM,in * sum(Q_water))
```

The GBM calculations use `C_GBM,in = 1`. Networks without a connected albumin
path retain their water flow and contribute zero albumin flow.

The historical supplementary plotting filename `plot_figS12.py` is retained for
traceability. The authors must confirm its final Extended Data numbering when
the manuscript figures are frozen.

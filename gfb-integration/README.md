# Conserved-flow four-layer GFB integration

This module couples four sequential transport elements:

```text
plasma bulk region -> GBM -> GBM-to-SD bulk region -> slit diaphragm
```

It uses one total water flow `Q_water` and one total albumin flow `Q_albumin`
through every layer. The reported layer transfer ratios are derived after the
self-consistent concentration solution; they are not treated as independent
flow rates.

With plasma concentration normalized to one and
`C* = Q_albumin / Q_water`, the model solves:

```text
C_GBM,in  = C* + (1 - C*) exp(Pe_bulk1)
C_GBM,out = S_GBM C_GBM,in
C_SD,in   = C* + (C_GBM,out - C*) exp(Pe_bulk2)
C*        = S_SD C_SD,in
Pe_bulk   = Q_water L_bulk / (A_bulk D_albumin)
```

The same `Q_water` appears in both bulk Péclet numbers and is reported for all
four layers. The same `Q_albumin = Q_water C*` is also reported for all layers,
together with equation residuals.

## Installation and demo

The implementation uses only the Python standard library and requires Python
3.10 or newer. From this directory:

```text
python workflows/run_gfb_model.py
```

Results are written to `outputs/manuscript_conditions/`. The included four
conditions use the final integrated GBM coefficients, the SD midpoint
`0.4769089716`, the manuscript bulk dimensions, and the representative water
flow used for the serial calculation.

Expected coupled GFB sieving coefficients are:

| Condition | Group | Expected sieving coefficient |
|---|---|---:|
| Steric only | WT | 0.0077722877 |
| Steric only | AS | 0.0599583902 |
| Electrostatic exclusion | WT | 0.0023905966 |
| Electrostatic exclusion | AS | 0.0344768872 |

The command writes `gfb_serial_results.csv` and `validation.json`. A successful
run reports `all_passed: true`, identical common-flow columns for every layer,
and a maximum absolute equation residual below `1e-12`.

The output also reports the values obtained from the rounded Fig. 6F factors
`1.0016 × 1.0008 × 0.48`. That column is retained as a figure-value comparison.
The conservation model itself solves the layer concentrations and obtains the
bulk transfer ratios from `Q_water`, area, length, and diffusion.

## Inputs and scope

Edit or replace `data/manuscript_conditions.csv` to evaluate other effective
GBM or SD coefficients. Each row must provide the common water flow, both bulk
geometries, albumin diffusion coefficient, `S_GBM`, and `S_SD`.

The supplied rows use the effective GBM coefficients from the full
5,000-network-per-group manuscript ensemble. The 100-network Release asset and
the two-network generation example are software demonstrations with their own
reported GBM results.

This integration module consumes effective GBM and SD sieving coefficients. It
does not rerun the pore-network solver or SD orientation geometry internally.
Their source calculations are provided in the neighboring packages.

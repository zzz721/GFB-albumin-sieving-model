# Validation report

Validation date: 2026-09-1.

## Test environment and installation

- Windows 11, build 26200, 64 bit
- Python 3.12.6, 64 bit
- Intel Core i9-13980HX processor and 15.63 GiB RAM
- No GPU or non-standard hardware required

Exact tested dependency versions are listed in the package-level
`requirements-tested.txt` files. A new virtual environment was created, the
exact dependencies and all three repository packages were installed, and
`pip check` reported no broken requirements. Environment creation and
installation took approximately 2 minutes 18 seconds on the test system.

## GBM demonstration

The public demonstration contains 50 WT and 50 AS networks and runs every
network under both exclusion-radius conditions. All 200 calculations completed
successfully in approximately 59.8 seconds with four workers.

| Group | Exclusion radius (nm) | Hydrodynamic radius (nm) | Integrated sieving coefficient |
|---|---:|---:|---:|
| WT | 3.55 | 3.55 | 0.0162575191 |
| AS | 3.55 | 3.55 | 0.1269321692 |
| WT | 4.25 | 3.55 | 0.0047920408 |
| AS | 4.25 | 3.55 | 0.0735055270 |

Reference-result, water-flow, albumin-flow, outlet-mixing, and fixed
hydrodynamic-radius checks all passed. Changing the exclusion radius from
3.55 nm to 4.25 nm did not alter the solvent-flow calculation, and the
hydrodynamic radius used for the DD2006 hindrance factors remained 3.55 nm.

The new-data path was also tested by regenerating boundary and
solvent-connectivity classifications from raw pore/throat workbooks and running
the DD2006 transport solver on the regenerated inputs. The solver returned a
valid result.

The public Release asset is:

```text
gbm-demo-100.zip
112332505 bytes
SHA256 17a3ed7d05e0dcd029d29a1ac37acc791439d8e4b8328cf8e6631e949e0a871f
```

The archive contains 512 ZIP entries, including 500 Excel input files. The ZIP
integrity check passed, and all 500 hashes recorded in `checksums.json` matched.

## Slit-diaphragm calculation

The full orientation geometry calculation used 101 width points, 5,000 random
three-dimensional orientations per point, a 2-degree deterministic scan, and
seed `20260804`. It completed in approximately 63.8 seconds and matched the
frozen geometry tables exactly in the post-release check.

| Quantity | Recalculated value |
|---|---:|
| Lower sieving coefficient | 0.1893594363581635 |
| Upper sieving coefficient | 0.7644585069335398 |
| Arithmetic midpoint | 0.47690897164585166 |

The workflow integrates the lower and upper accessible-area curves separately,
converts each integrated area to a sieving coefficient, and then calculates
their arithmetic mean. Both supplied plotting workflows completed successfully.

## Conserved-flow GFB integration

The four supplied conditions were recalculated with a common total water flow
and common total albumin flow through all layers.

| Condition | Group | Coupled GFB sieving |
|---|---|---:|
| Steric only | WT | 0.0077722877 |
| Steric only | AS | 0.0599583902 |
| Electrostatic exclusion | WT | 0.0023905966 |
| Electrostatic exclusion | AS | 0.0344768872 |

All four results matched the expected outputs. Flow-conservation checks passed,
and the maximum absolute equation residual was
`1.3877787807814457e-17`.

## Package integrity

All 59 Python files passed syntax parsing. The repository contains no tracked
Python bytecode, cache directories, or generated output directories. The
software is distributed under the MIT License.

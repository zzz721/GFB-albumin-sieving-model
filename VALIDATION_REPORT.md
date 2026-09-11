# Validation report

Validation date: 2026-09-11.

## Environment

- Windows 11, build 26200, 64 bit
- Python 3.12.6, 64 bit
- No GPU or non-standard hardware used
- Exact tested package versions are listed in each package's
  `requirements-tested.txt` where third-party packages are required.

## GBM transport

The staged `dd2006_conservative_solver` was run on
`synthetic_WT_run463_plane` using the final demonstration input files.

| Exclusion radius (nm) | Hydrodynamic radius (nm) | Status | Sieving coefficient | Water flow (m3/s) | Albumin flow at inlet concentration 1 (m3/s) |
|---:|---:|---|---:|---:|---:|
| 3.55 | 3.55 | valid | 0.002738304129289562 | 9.422825354006606e-20 | 2.580256157645067e-22 |
| 4.25 | 3.55 | valid disconnected zero | 0 | 9.422825354006606e-20 | 0 |

The identical water-flow values confirm that changing exclusion radius does
not alter the solvent calculation. The 4.25-nm run retains a 3.55-nm physical
hydrodynamic radius for the DD2006 hindrance factors. A direct command omitting
`--hydrodynamic-radius-nm` was also checked at 4.25-nm exclusion; the recorded
hydrodynamic radius remained 3.55 nm.

The bundled 100-network demonstration contains 50 WT and 50 AS networks and
has a recorded complete run of 200 calculations. All reference, water-flow,
albumin-flow, outlet-mixing, and fixed-hydrodynamic-radius checks passed with
zero failures. Its recorded runtime was 62.019 seconds with four workers.

| Group | Exclusion radius (nm) | Demo result | Complete-ensemble result | Relative difference |
|---|---:|---:|---:|---:|
| WT | 3.55 | 0.0162575191 | 0.01625797 | -0.003% |
| AS | 3.55 | 0.1269321692 | 0.12543062 | +1.197% |
| WT | 4.25 | 0.0047920408 | 0.00500058 | -4.170% |
| AS | 4.25 | 0.0735055270 | 0.07212139 | +1.919% |

All four demonstration aggregates are within 10% of the corresponding full
population values. The internal selection procedure is intentionally omitted
from the public package.

Two networks investigated during curation are absent from the final manifest:
`synthetic_AS_run2490_plane` (singular matrix) and
`synthetic_AS_run1466_plane` (historical-value mismatch). The latter had already
been replaced in the same thickness/water-flow cell by
`synthetic_AS_run1533_plane`, which passed both radius conditions. Because
neither anomalous network is in the final demonstration, no further network
replacement is required for this release candidate.

Current Release asset:

```text
gbm-demo-100.zip
112332505 bytes
SHA256 17a3ed7d05e0dcd029d29a1ac37acc791439d8e4b8328cf8e6631e949e0a871f
```

The archive contains 500 input Excel files, a standalone solver, expected
results, documentation, and recorded validation outputs. Its ZIP structure and
all input checksums were checked after rebuilding the archive.

## Slit diaphragm calculation

The frozen manuscript calculation was rerun from the staged package. The
orientation grid uses 101 width points, 5,000 random three-dimensional
orientations per point, a 2-degree deterministic scan, and seed `20260804`.
The recorded full geometry runtime was approximately 70.1 seconds.

| Quantity | Recalculated value |
|---|---:|
| Lower sieving coefficient | 0.1893594363581635 |
| Upper sieving coefficient | 0.7644585069335398 |
| Arithmetic midpoint | 0.47690897164585166 |

The workflow integrates the lower and upper accessible-area curves separately,
converts each integrated area to a sieving coefficient, and then averages the
two coefficients. The optional orientation summary is used only as provenance
and cannot overwrite the new integration. Recomputed geometry tables matched
the frozen tables within `1e-10` (largest observed numerical difference
approximately `1.7e-13`). A deliberately shifted width input correctly failed
with an instruction to regenerate the orientation tables.

## Conserved-flow GFB integration

The four manuscript conditions were rerun through the serial conserved-flow
model.

| Condition | Group | Coupled GFB sieving |
|---|---|---:|
| Steric only | WT | 0.0077722877 |
| Steric only | AS | 0.0599583902 |
| Electrostatic exclusion | WT | 0.0023905966 |
| Electrostatic exclusion | AS | 0.0344768872 |

All four layers reported identical total water flow and identical total
albumin flow. The maximum absolute equation residual was
`1.3877787807814457e-17`; all conservation checks passed.

## Package checks and limits

- Parsed all 59 Python files successfully.
- Removed generated output directories, bytecode, caches, and accidental nested
  copies from the upload candidate.
- The clean candidate contains 113 files and is approximately 3.22 MiB before
  compression.
- The current checks validate the supplied inputs and stated calculations. An
  installation by an unfamiliar colleague and the remaining manuscript
  figure/source-data migration are still pending.

## Public GitHub and Release check

The public `main` branch and Release `gbm-example-v1.0.0` were independently
downloaded after publication. The tag points to the published `main` commit.
The Release is public and contains one asset named `gbm-demo-100.zip`.

- Public asset size: 112332505 bytes
- Public asset SHA256:
  `17a3ed7d05e0dcd029d29a1ac37acc791439d8e4b8328cf8e6631e949e0a871f`
- ZIP entries: 512, including 500 input Excel files
- Manifest: 50 WT and 50 AS networks
- Input checksum mismatches: 0
- Full public-asset demo: 200 calculations, 0 failures, 59.8 seconds
- Full SD geometry: 101 width points, exact match to frozen tables, 63.8 seconds
- SD lower/upper/midpoint outputs: exact match to expected values
- GFB four-condition outputs: exact match to expected values; all conservation
  checks passed
- SD manuscript and supplementary plotting commands: both completed
- GBM new-data path: raw pore/throat workbooks were reclassified and the
  regenerated classifications produced a valid DD2006 transport result

A new Windows virtual environment was created with Python 3.12.6. Environment
creation took 7.2 seconds, exact dependency installation took 113.1 seconds,
and installation of the three repository packages took 17.8 seconds, for a
combined time of approximately 2 minutes 18 seconds. `pip check` reported no
broken requirements. The test system used an Intel Core i9-13980HX processor
and 15.63 GiB RAM; no GPU or non-standard hardware was required.

The post-release public `main` audit also found 57 legacy tracked
cache/generated output files and one unresolved Release-link placeholder in the
GBM example README. These do not affect the numerical results. The prepared
cleanup commit removes the 57 files, fixes the link, records the measured
installation time, and documents the tested GBM new-data path.

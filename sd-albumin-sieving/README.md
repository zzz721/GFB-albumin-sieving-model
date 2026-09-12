# SD albumin sieving

This package calculates albumin sieving through the slit diaphragm (SD) from
the measured SD-width distribution and a deformable hexagonal pore model.

Two workflows are provided:

1. `recalculate_orientation_bounds.py` starts from the measured widths,
   reference hexagon, and ellipsoidal albumin dimensions. It deforms the pore,
   performs geometric erosion for three-dimensional orientations, and writes
   lower and upper accessible-area curves.
2. `calculate_sd_sieving.py` integrates a supplied orientation grid over the
   fitted normal width distribution, converts the lower and upper integrated
   areas separately to sieving coefficients, and then takes their arithmetic
   midpoint.

The default orientation calculation uses 5,000 uniform random SO(3)
orientations per width with seed `20260804`, plus a deterministic principal-axis
scan in 2-degree increments. The lower scenario is the random-orientation mean.
The upper scenario is the largest value evaluated across the random orientations
and deterministic scan; it is a finite-grid best-passage scenario rather than a
proof of the global continuous-orientation maximum.

## System requirements and installation

- Python 3.10 or newer.
- Tested on Windows 11 with Python 3.12.6.
- No GPU or non-standard hardware is required.
- Tested package versions are listed in `requirements-tested.txt`.
- A fresh Windows environment containing the exact dependencies and all three
  repository packages was created in approximately 2 minutes 18 seconds on the
  tested desktop. Download time and hardware affect this value.

```text
python -m pip install -e .
```

## Recalculate the orientation geometry

Full manuscript settings:

```text
python workflows/recalculate_orientation_bounds.py --random-orientations 5000 --grid-size 101 --theta-step-deg 2 --seed 20260804
```

Quick software check:

```text
python workflows/recalculate_orientation_bounds.py --random-orientations 50 --grid-size 11 --theta-step-deg 10 --output-dir outputs/sd_orientation_quick
```

The full workflow writes the observed-width table, width-grid table, scenario
summary, integrated-area summary, and overview plot below
`outputs/sd_orientation_geometry/`. The quick check does not reproduce the
manuscript values because it uses fewer orientations and grid points.

## Calculate sieving from an orientation grid

Run the frozen manuscript inputs:

```text
python workflows/calculate_sd_sieving.py
```

The outputs below `outputs/sd_sieving/` include `sd_sieving_curve.csv`,
`sd_observed_spacing_results.csv`, and `sd_sieving_summary.json`.

Expected integrated values:

```text
lower sieving = 0.1893594363
upper sieving = 0.7644585069
reported midpoint = (lower + upper) / 2 = 0.4769089716
```

The lower and upper accessible areas are integrated separately, each integrated
area is converted to a sieving coefficient, and only then are the two sieving
coefficients averaged. Averaging the areas before nonlinear conversion is not
the manuscript algorithm.

An optional `--orientation-alpha-summary-csv` is accepted as a provenance
record only. It never overrides values integrated from the supplied grid.
Observed-width results are matched by numerical width, independent of CSV row
order.

To calculate from a newly generated grid:

```text
python workflows/calculate_sd_sieving.py --orientation-grid-csv outputs/sd_orientation_geometry/sd_orientation_bounds_by_width_grid.csv --orientation-observed-csv outputs/sd_orientation_geometry/sd_orientation_bounds_by_observed_width.csv
```

If widths, hexagon geometry, albumin dimensions, sampling settings, or seed are
changed, regenerate both orientation tables before calculating the new result.
The manuscript grid contains 101 width points.

## Figures

The plotting workflows read frozen data so submitted figures can be rendered
without rerunning the orientation scan:

```text
python workflows/plot_fig3G.py
python workflows/plot_figS12.py
```

The historical `FigS12` filename is retained solely for compatibility with the
saved plotting workflow.

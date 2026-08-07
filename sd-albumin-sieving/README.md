# SD albumin sieving

Deterministic calculation of albumin sieving through the slit diaphragm (SD).

The calculation uses the observed SD-spacing distribution, a fixed-side flexible
hexagon model, orientation-bounded albumin-accessible centroid areas, and a
Peclet conversion to obtain the integrated sieving coefficient. The lower bound
uses uniformly random three-dimensional albumin orientations, the upper bound
uses the best-passage three-dimensional orientation at each SD width, and the
reported SD sieving coefficient is the arithmetic midpoint of the lower and
upper sieving coefficients.

Run from this project directory:

```powershell
python workflows\calculate_sd_sieving.py
```

The default outputs are written to `outputs/sd_sieving/`:

- `sd_sieving_curve.csv`
- `sd_observed_spacing_results.csv`
- `sd_sieving_summary.json`

The default integrated midpoint SD sieving coefficient is approximately 0.477.

Render the SD manuscript and supplementary figures from frozen data:

```powershell
python workflows\plot_fig3G.py
python workflows\plot_figS12.py
```

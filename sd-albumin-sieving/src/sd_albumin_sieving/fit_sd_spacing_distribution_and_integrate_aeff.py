from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .albumin_aeff_from_sd_spacing import (
    compute_rows,
    erode_convex_polygon_by_ellipse,
    polygon_area,
    scan_aeff_theta,
)
from .flexible_hexagon_from_slit_width import read_sides, read_slit_widths, symmetric_hexagon_from_major


@dataclass(frozen=True)
class FitResult:
    name: str
    params: tuple[float, ...]
    log_likelihood: float
    aic: float
    ks_statistic: float
    ks_pvalue: float


def fit_distributions(values: np.ndarray) -> list[FitResult]:
    fits: list[FitResult] = []

    candidates = [
        ("normal", stats.norm, stats.norm.fit(values)),
        ("lognormal_loc0", stats.lognorm, stats.lognorm.fit(values, floc=0)),
        ("gamma_loc0", stats.gamma, stats.gamma.fit(values, floc=0)),
    ]
    for name, distribution, params in candidates:
        logpdf = distribution.logpdf(values, *params)
        log_likelihood = float(np.sum(logpdf))
        k = len(params)
        aic = 2 * k - 2 * log_likelihood
        ks_statistic, ks_pvalue = stats.kstest(values, distribution.cdf, args=params)
        fits.append(
            FitResult(
                name=name,
                params=tuple(float(value) for value in params),
                log_likelihood=log_likelihood,
                aic=float(aic),
                ks_statistic=float(ks_statistic),
                ks_pvalue=float(ks_pvalue),
            )
        )
    return sorted(fits, key=lambda fit: fit.aic)


def distribution_object(fit: FitResult):
    if fit.name == "normal":
        return stats.norm, fit.params
    if fit.name == "lognormal_loc0":
        return stats.lognorm, fit.params
    if fit.name == "gamma_loc0":
        return stats.gamma, fit.params
    raise ValueError(f"Unknown fit: {fit.name}")


def aeff_for_spacing_grid(
    spacing_grid: np.ndarray,
    hexagon_csv: Path,
    reference_length_nm: float,
    ellipse_rx_nm: float,
    ellipse_ry_nm: float,
    theta_step_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slanted_side, vertical_side, current_hex_major, _ = read_sides(hexagon_csv)
    fixed_values = []
    mean_values = []
    best_values = []
    for sd_spacing in spacing_grid:
        major = current_hex_major * float(sd_spacing) / reference_length_nm
        try:
            vertices, _, _, _ = symmetric_hexagon_from_major(major, slanted_side, vertical_side)
        except ValueError:
            fixed_values.append(0.0)
            mean_values.append(0.0)
            best_values.append(0.0)
            continue
        fixed = polygon_area(erode_convex_polygon_by_ellipse(vertices, ellipse_rx_nm, ellipse_ry_nm, 0.0))
        _, areas, _ = scan_aeff_theta(vertices, ellipse_rx_nm, ellipse_ry_nm, theta_step_deg)
        fixed_values.append(fixed)
        mean_values.append(float(np.mean(areas)))
        best_values.append(float(np.max(areas)))
    return np.array(fixed_values), np.array(mean_values), np.array(best_values)


def weighted_integral(grid: np.ndarray, values: np.ndarray, weights: np.ndarray) -> float:
    numerator = float(np.trapz(values * weights, grid))
    denominator = float(np.trapz(weights, grid))
    return numerator / denominator if denominator > 0 else math.nan


def integrate_aeff(
    fits: list[FitResult],
    empirical_rows,
    hexagon_csv: Path,
    reference_length_nm: float,
    ellipse_rx_nm: float,
    ellipse_ry_nm: float,
    theta_step_deg: float,
) -> tuple[list[dict[str, float | str]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spacings = np.array([row.sd_spacing_nm for row in empirical_rows], dtype=float)
    grid_min = max(0.01, float(spacings.min() - 5.0))
    grid_max = float(spacings.max() + 5.0)
    grid = np.linspace(grid_min, grid_max, 2501)
    fixed_grid, mean_grid, best_grid = aeff_for_spacing_grid(
        grid,
        hexagon_csv,
        reference_length_nm,
        ellipse_rx_nm,
        ellipse_ry_nm,
        theta_step_deg,
    )

    rows: list[dict[str, float | str]] = []
    empirical_fixed = np.array([row.aeff_fixed_nm2 for row in empirical_rows], dtype=float)
    empirical_mean = np.array([row.aeff_mean_nm2 for row in empirical_rows], dtype=float)
    empirical_best = np.array([row.aeff_max_nm2 for row in empirical_rows], dtype=float)
    rows.append(
        {
            "distribution": "empirical_sample_mean",
            "integrated_aeff_fixed_nm2": float(np.mean(empirical_fixed)),
            "integrated_aeff_theta_mean_nm2": float(np.mean(empirical_mean)),
            "integrated_aeff_theta_best_nm2": float(np.mean(empirical_best)),
            "p_aeff_positive_fixed": float(np.mean(empirical_fixed > 1e-9)),
            "p_aeff_positive_best": float(np.mean(empirical_best > 1e-9)),
        }
    )

    for fit in fits:
        distribution, params = distribution_object(fit)
        pdf = distribution.pdf(grid, *params)
        rows.append(
            {
                "distribution": fit.name,
                "integrated_aeff_fixed_nm2": weighted_integral(grid, fixed_grid, pdf),
                "integrated_aeff_theta_mean_nm2": weighted_integral(grid, mean_grid, pdf),
                "integrated_aeff_theta_best_nm2": weighted_integral(grid, best_grid, pdf),
                "p_aeff_positive_fixed": weighted_integral(grid, (fixed_grid > 1e-9).astype(float), pdf),
                "p_aeff_positive_best": weighted_integral(grid, (best_grid > 1e-9).astype(float), pdf),
            }
        )
    return rows, grid, fixed_grid, mean_grid, best_grid


def write_fit_csv(path: Path, fits: list[FitResult]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["distribution", "params", "log_likelihood", "aic", "ks_statistic", "ks_pvalue"])
        for fit in fits:
            writer.writerow([fit.name, ";".join(f"{value:.10g}" for value in fit.params), fit.log_likelihood, fit.aic, fit.ks_statistic, fit.ks_pvalue])


def write_integral_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = [
        "distribution",
        "integrated_aeff_fixed_nm2",
        "integrated_aeff_theta_mean_nm2",
        "integrated_aeff_theta_best_nm2",
        "p_aeff_positive_fixed",
        "p_aeff_positive_best",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_fit_plot(path: Path, values: np.ndarray, fits: list[FitResult]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
    x_min = float(values.min() - 2.0)
    x_max = float(values.max() + 2.0)
    x = np.linspace(x_min, x_max, 600)

    axes[0].hist(values, bins=np.arange(math.floor(values.min()), math.ceil(values.max()) + 1, 1), density=True, alpha=0.45, color="#1b6a83", edgecolor="white", label="observed")
    colors = ["#d7263d", "#2ca25f", "#7a3cff"]
    for color, fit in zip(colors, fits):
        distribution, params = distribution_object(fit)
        axes[0].plot(x, distribution.pdf(x, *params), color=color, linewidth=2.0, label=f"{fit.name} (AIC {fit.aic:.1f})")
    axes[0].set_xlabel("SD spacing (nm)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("SD spacing distribution fit")
    axes[0].legend(fontsize=8)

    best_distribution, best_params = distribution_object(fits[0])
    sorted_values = np.sort(values)
    probs = (np.arange(1, len(sorted_values) + 1) - 0.5) / len(sorted_values)
    theoretical = best_distribution.ppf(probs, *best_params)
    axes[1].scatter(theoretical, sorted_values, color="#1b6a83", edgecolor="#143f4d", s=46)
    lo = min(float(theoretical.min()), float(sorted_values.min()))
    hi = max(float(theoretical.max()), float(sorted_values.max()))
    axes[1].plot([lo, hi], [lo, hi], color="#d7263d", linestyle="--", linewidth=1.6)
    axes[1].set_xlabel(f"Theoretical quantiles ({fits[0].name})")
    axes[1].set_ylabel("Observed SD spacing (nm)")
    axes[1].set_title("Best-fit Q-Q check")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_integrated_plot(
    path: Path,
    values: np.ndarray,
    fits: list[FitResult],
    grid: np.ndarray,
    fixed_grid: np.ndarray,
    mean_grid: np.ndarray,
    best_grid: np.ndarray,
) -> None:
    fig, ax1 = plt.subplots(figsize=(10.8, 5.2), constrained_layout=True)
    ax2 = ax1.twinx()

    ax1.plot(grid, fixed_grid, color="#f5a623", linewidth=2.0, label="Aeff fixed theta=0")
    ax1.plot(grid, mean_grid, color="#4f8fc0", linewidth=2.0, label="Aeff theta mean")
    ax1.plot(grid, best_grid, color="#2ca25f", linewidth=2.0, label="Aeff theta best")
    ax1.scatter(values, np.zeros_like(values), color="#111111", s=16, alpha=0.65, label="observed spacings")

    best_distribution, best_params = distribution_object(fits[0])
    pdf = best_distribution.pdf(grid, *best_params)
    ax2.fill_between(grid, pdf, color="#9aa3aa", alpha=0.25, label=f"{fits[0].name} density")

    ax1.set_xlabel("SD spacing (nm)")
    ax1.set_ylabel("Aeff (nm²)")
    ax2.set_ylabel("Fitted density")
    ax1.set_title("Integrated Aeff from fitted SD spacing distribution")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_index(output_dir: Path, fits: list[FitResult], integral_rows: list[dict[str, float | str]]) -> Path:
    best = fits[0]
    integral_best = next(row for row in integral_rows if row["distribution"] == best.name)
    out = output_dir / "index.html"
    out.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SD spacing distribution fit and integrated Aeff</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #f6f7f8; color: #1f272a; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px 22px 44px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p {{ line-height: 1.5; }}
    a {{ color: #096a6e; font-weight: 700; }}
    img {{ width: 100%; height: auto; border: 1px solid #d9dee2; background: white; margin: 12px 0 26px; }}
  </style>
</head>
<body>
  <main>
    <h1>SD spacing distribution fit and integrated Aeff</h1>
    <p>Best AIC fit: {html.escape(best.name)}. Integrated Aeff under this fit: fixed={integral_best["integrated_aeff_fixed_nm2"]:.3f} nm², theta mean={integral_best["integrated_aeff_theta_mean_nm2"]:.3f} nm², theta best={integral_best["integrated_aeff_theta_best_nm2"]:.3f} nm².</p>
    <p>
      <a href="sd_spacing_distribution_fits.csv">Fit CSV</a> ·
      <a href="integrated_aeff_by_distribution.csv">Integrated Aeff CSV</a>
    </p>
    <img src="sd_spacing_distribution_fit.png" alt="SD spacing distribution fit">
    <img src="integrated_aeff_distribution.png" alt="Integrated Aeff from distribution">
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit SD spacing distribution and integrate Aeff over it.")
    parser.add_argument(
        "--slit-width-xlsx",
        default="C:/Users/糖吉诃德/xwechat_files/wxid_5wtp6e154kya22_b9f0/msg/file/2026-07/Slit_width.xlsx",
    )
    parser.add_argument(
        "--hexagon-csv",
        default="protein_structure/selected_symmetric_hexagons_refined/SDsubunitCorrect_closed_angle_117_expanded_symmetric_hexagon.csv",
    )
    parser.add_argument("--output-dir", default="protein_structure/sd_spacing_distribution_aeff")
    parser.add_argument("--reference-length-nm", type=float, default=50.2439981515067)
    parser.add_argument("--ellipse-diameter-x-nm", type=float, default=5.45106753660192)
    parser.add_argument("--ellipse-diameter-y-nm", type=float, default=6.986832689834353)
    parser.add_argument("--theta-step-deg", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    slit_rows = read_slit_widths(Path(args.slit_width_xlsx))
    spacings = np.array([row[2] for row in slit_rows], dtype=float)
    fits = fit_distributions(spacings)

    empirical_rows = compute_rows(
        Path(args.slit_width_xlsx),
        Path(args.hexagon_csv),
        args.reference_length_nm,
        args.ellipse_diameter_x_nm / 2.0,
        args.ellipse_diameter_y_nm / 2.0,
        args.theta_step_deg,
    )
    integral_rows, grid, fixed_grid, mean_grid, best_grid = integrate_aeff(
        fits,
        empirical_rows,
        Path(args.hexagon_csv),
        args.reference_length_nm,
        args.ellipse_diameter_x_nm / 2.0,
        args.ellipse_diameter_y_nm / 2.0,
        args.theta_step_deg,
    )

    fit_csv = output_dir / "sd_spacing_distribution_fits.csv"
    integral_csv = output_dir / "integrated_aeff_by_distribution.csv"
    fit_png = output_dir / "sd_spacing_distribution_fit.png"
    integral_png = output_dir / "integrated_aeff_distribution.png"
    write_fit_csv(fit_csv, fits)
    write_integral_csv(integral_csv, integral_rows)
    save_fit_plot(fit_png, spacings, fits)
    save_integrated_plot(integral_png, spacings, fits, grid, fixed_grid, mean_grid, best_grid)
    index = write_index(output_dir, fits, integral_rows)

    print(f"Wrote {fit_csv}")
    print(f"Wrote {integral_csv}")
    print(f"Wrote {fit_png}")
    print(f"Wrote {integral_png}")
    print(f"Wrote {index}")
    print(f"Best fit: {fits[0].name}, AIC={fits[0].aic:.3f}, params={fits[0].params}")
    for row in integral_rows:
        print(
            f"{row['distribution']}: fixed={row['integrated_aeff_fixed_nm2']:.3f}, "
            f"theta_mean={row['integrated_aeff_theta_mean_nm2']:.3f}, "
            f"theta_best={row['integrated_aeff_theta_best_nm2']:.3f}, "
            f"P_best_positive={row['p_aeff_positive_best']:.3f}"
        )


if __name__ == "__main__":
    main()

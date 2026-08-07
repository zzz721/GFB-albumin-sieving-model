from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "frozen_data"
OUTPUT_ROOT = Path(__file__).resolve().parents[5] / "outputs"
OUTPUT_DIR = OUTPUT_ROOT / "manuscript" / "figure3_sd_sieving"

CURVE_CSV = DATA_DIR / "sieving_curve_mean.csv"
OBSERVED_CSV = DATA_DIR / "observed_spacing_points.csv"
SUMMARY_JSON = DATA_DIR / "sieving_summary.json"
ORIENTATION_GRID_CSV = DATA_DIR / "sd_orientation_bounds_by_width_grid.csv"
ORIENTATION_OBSERVED_CSV = DATA_DIR / "sd_orientation_bounds_by_observed_width.csv"
ORIENTATION_ALPHA_SUMMARY_CSV = DATA_DIR / "sd_orientation_alpha_summary.csv"

CURVE_COLOR = "#2F80B7"
CURVE_EDGE_COLOR = "#DDEFF8"
BAND_COLOR = "#9BD2EA"
BAND_EDGE_COLOR = "#6FB7D7"
DENSITY_COLOR = "#BFE8DF"
SAMPLE_WIDTH_COLOR = "#168AA3"
SAMPLE_WIDTH_BRIGHT_GREEN = "#35E987"
INTEGRATED_COLOR = "#006D5B"
INTEGRATED_TEXT_COLOR = "#00564A"


def read_csv_float_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            rows.append(parsed)
    return rows


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sd_sieving_from_aeff(aeff_per_large_pore_nm2: np.ndarray, area_per_large_pore_nm2: float, peclet: float) -> np.ndarray:
    alpha = np.asarray(aeff_per_large_pore_nm2, dtype=float) / float(area_per_large_pore_nm2)
    return alpha / (alpha + float(peclet))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render manuscript Fig. 3G from frozen data.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    curve_rows = read_csv_float_rows(CURVE_CSV)
    observed_rows = read_csv_float_rows(OBSERVED_CSV)
    orientation_rows = read_csv_float_rows(ORIENTATION_GRID_CSV)
    orientation_observed_rows = read_csv_float_rows(ORIENTATION_OBSERVED_CSV)
    orientation_alpha_summary = read_csv_rows(ORIENTATION_ALPHA_SUMMARY_CSV)
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

    x = np.array([row["sd_spacing_nm"] for row in curve_rows], dtype=float)
    density = np.array([row["normal_density"] for row in curve_rows], dtype=float)
    bound_x = np.array([row["spacing_nm"] for row in orientation_rows], dtype=float)
    lower_sieving = sd_sieving_from_aeff(
        np.array([row["aeff_3d_random_mean_nm2"] for row in orientation_rows], dtype=float),
        float(summary["water_accessible_area_per_complex_nm2"]) / float(summary["large_pore_count_per_complex"]),
        float(summary["peclet_used_for_figure"]),
    )
    upper_sieving = sd_sieving_from_aeff(
        np.array([row["aeff_3d_best_nm2"] for row in orientation_rows], dtype=float),
        float(summary["water_accessible_area_per_complex_nm2"]) / float(summary["large_pore_count_per_complex"]),
        float(summary["peclet_used_for_figure"]),
    )
    midpoint_sieving = 0.5 * (lower_sieving + upper_sieving)

    observed_x = np.array([row["spacing_nm"] for row in orientation_observed_rows], dtype=float)
    observed_lower = sd_sieving_from_aeff(
        np.array([row["aeff_3d_random_mean_nm2"] for row in orientation_observed_rows], dtype=float),
        float(summary["water_accessible_area_per_complex_nm2"]) / float(summary["large_pore_count_per_complex"]),
        float(summary["peclet_used_for_figure"]),
    )
    observed_upper = sd_sieving_from_aeff(
        np.array([row["aeff_3d_best_nm2"] for row in orientation_observed_rows], dtype=float),
        float(summary["water_accessible_area_per_complex_nm2"]) / float(summary["large_pore_count_per_complex"]),
        float(summary["peclet_used_for_figure"]),
    )
    observed_sieving = 0.5 * (observed_lower + observed_upper)
    observed_y = np.full_like(observed_x, -0.047, dtype=float)
    observed_order = np.argsort(observed_x)
    observed_x = observed_x[observed_order]
    observed_sieving = observed_sieving[observed_order]
    observed_y = observed_y[observed_order]

    area_per_large_pore = float(summary["water_accessible_area_per_complex_nm2"]) / float(summary["large_pore_count_per_complex"])
    integrated_by_scenario = {}
    for row in orientation_alpha_summary:
        scenario = str(row.get("scenario", ""))
        if scenario in {"lower_random_3d", "upper_best_3d"}:
            integrated_by_scenario[scenario] = float(row["integrated_aeff_nm2_normal"])
    integrated_lower_sieving = float(
        sd_sieving_from_aeff(
            np.array([integrated_by_scenario["lower_random_3d"]], dtype=float),
            area_per_large_pore,
            float(summary["peclet_used_for_figure"]),
        )[0]
    )
    integrated_upper_sieving = float(
        sd_sieving_from_aeff(
            np.array([integrated_by_scenario["upper_best_3d"]], dtype=float),
            area_per_large_pore,
            float(summary["peclet_used_for_figure"]),
        )[0]
    )
    integrated_sieving = 0.5 * (integrated_lower_sieving + integrated_upper_sieving)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=False)
    fig.subplots_adjust(left=0.16, right=0.85, bottom=0.39, top=0.95)
    ax2 = ax.twinx()
    ax.set_zorder(2)
    ax2.set_zorder(1)
    ax.patch.set_alpha(0.0)
    ax2.patch.set_alpha(0.0)

    ax2.fill_between(
        x,
        density,
        color=DENSITY_COLOR,
        alpha=0.62,
        linewidth=0,
        label="Fitted SD width distribution",
        zorder=0,
    )
    ax.fill_between(
        bound_x,
        lower_sieving,
        upper_sieving,
        color=BAND_COLOR,
        alpha=0.34,
        linewidth=0,
        label="S lower-upper range",
        zorder=2,
    )
    ax.plot(
        bound_x,
        lower_sieving,
        color=BAND_EDGE_COLOR,
        linewidth=0.75,
        alpha=0.65,
        zorder=2.2,
    )
    ax.plot(
        bound_x,
        upper_sieving,
        color=BAND_EDGE_COLOR,
        linewidth=0.75,
        alpha=0.65,
        zorder=2.2,
    )
    ax.plot(
        bound_x,
        midpoint_sieving,
        color=CURVE_COLOR,
        linewidth=1.9,
        label="Midpoint S curve",
        zorder=3,
    )
    ax.scatter(
        observed_x,
        observed_sieving,
        s=28,
        color=CURVE_COLOR,
        edgecolor=CURVE_EDGE_COLOR,
        linewidth=0.35,
        alpha=0.86,
        label="Midpoint S at width",
        zorder=4,
    )
    ax.scatter(
        observed_x,
        observed_y,
        marker="^",
        s=32,
        color=SAMPLE_WIDTH_BRIGHT_GREEN,
        linewidth=0,
        alpha=0.74,
        clip_on=False,
        label="Observed SD width",
        zorder=0.5,
    )
    ax.axhline(
        integrated_sieving,
        color=INTEGRATED_COLOR,
        linewidth=1.1,
        linestyle="--",
        label="Integrated S",
        zorder=10,
    )

    ax.set_xlabel("SD width (nm)", labelpad=2)
    ax.set_ylabel("Sieving coefficient")
    ax2.set_ylabel("Fitted SD width density")

    ax.set_xlim(45.0, 60.0)
    left_ylim = (-0.075, min(1.0, max(0.42, float(np.max(upper_sieving)) * 1.12)))
    right_ticks = np.arange(0.0, 0.251, 0.05)
    right_top = 0.25
    zero_fraction = (0.0 - left_ylim[0]) / (left_ylim[1] - left_ylim[0])
    right_bottom = -zero_fraction * right_top / (1.0 - zero_fraction)
    ax.set_ylim(*left_ylim)
    ax2.set_ylim(right_bottom, right_top)
    ax2.set_yticks(right_ticks)
    ax.set_xticks(np.arange(45.0, 60.1, 2.5))

    ax.spines["top"].set_visible(True)
    ax2.spines["top"].set_visible(True)
    ax.text(
        0.04,
        integrated_sieving + 0.012,
        f"{integrated_sieving:.3f}",
        transform=ax.get_yaxis_transform(),
        color=INTEGRATED_TEXT_COLOR,
        fontsize=5.8,
        ha="left",
        va="bottom",
        zorder=11,
    )
    handles, labels = ax.get_legend_handles_labels()
    density_handles, density_labels = ax2.get_legend_handles_labels()
    legend_by_label = dict(zip(labels + density_labels, handles + density_handles))
    sd_legend_order = [
        "Observed SD width",
        "Fitted SD width distribution",
    ]
    s_legend_order = [
        "S lower-upper range",
        "Midpoint S at width",
        "Midpoint S curve",
        "Integrated S",
    ]
    sd_legend = ax.legend(
        [legend_by_label[label] for label in sd_legend_order],
        sd_legend_order,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=4.85,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.75,
        borderaxespad=0.0,
        labelspacing=0.18,
    )
    ax.add_artist(sd_legend)
    s_legend = ax.legend(
        [legend_by_label[label] for label in s_legend_order],
        ["S lower-upper range", "Midpoint S at width", "Midpoint S curve", "Integrated midpoint S"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.315),
        ncol=2,
        fontsize=4.55,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.55,
        borderaxespad=0.0,
        labelspacing=0.18,
    )
    s_legend._legend_box.align = "center"

    out_base = args.out_dir / "fig3G_orientation_bounds"
    fig.savefig(out_base.with_suffix(".png"), dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    source_rows = []
    for row_x, low, high, mid in zip(bound_x, lower_sieving, upper_sieving, midpoint_sieving):
        source_rows.append(
            {
                "sd_spacing_nm": float(row_x),
                "sieving_lower_random_3d": float(low),
                "sieving_upper_best_3d": float(high),
                "sieving_midpoint": float(mid),
            }
        )
    with (args.out_dir / "fig3G_orientation_bounds_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0].keys()))
        writer.writeheader()
        writer.writerows(source_rows)
    (args.out_dir / "fig3G_orientation_bounds_summary.json").write_text(
        json.dumps(
            {
                "area_per_large_pore_nm2": area_per_large_pore,
                "peclet": float(summary["peclet_used_for_figure"]),
                "integrated_lower_sieving": integrated_lower_sieving,
                "integrated_upper_sieving": integrated_upper_sieving,
                "integrated_midpoint_sieving": integrated_sieving,
                "orientation_method": "lower=random 3D orientation, upper=best 3D orientation, final=midpoint of lower and upper S",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase, HandlerPatch
from matplotlib.patches import FancyArrowPatch


class HandlerArrow(HandlerPatch):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        arrow = FancyArrowPatch(
            (xdescent, ydescent + height / 2.0),
            (xdescent + width, ydescent + height / 2.0),
            arrowstyle="-|>",
            mutation_scale=fontsize * 1.3,
            linewidth=orig_handle.get_linewidth(),
            color=orig_handle.get_edgecolor(),
            transform=trans,
        )
        return [arrow]


class HandlerStackedLines(HandlerBase):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        colors = [handle.get_color() for handle in orig_handle]
        y_positions = [ydescent + height * 0.82, ydescent + height * 0.5, ydescent + height * 0.18]
        artists = []
        for color, y in zip(colors, y_positions):
            artists.append(
                Line2D(
                    [xdescent, xdescent + width],
                    [y, y],
                    color=color,
                    linewidth=3.0,
                    solid_capstyle="butt",
                    transform=trans,
                )
            )
        return artists


@dataclass(frozen=True)
class FlexibleHexagonRow:
    sample: int
    tomo: str
    slit_width_nm: float
    protein_length_nm: float
    length_scale: float
    hex_major_nm: float
    hex_minor_nm: float
    inradius_nm: float
    top_bottom_angle_deg: float
    side_angle_deg: float
    albumin_margin_nm: float
    passes_albumin: bool


def interior_angles(vertices: np.ndarray) -> list[float]:
    angles = []
    for idx, vertex in enumerate(vertices):
        previous_vertex = vertices[idx - 1]
        next_vertex = vertices[(idx + 1) % len(vertices)]
        v1 = previous_vertex - vertex
        v2 = next_vertex - vertex
        cosine = float(v1 @ v2) / (float(np.linalg.norm(v1)) * float(np.linalg.norm(v2)))
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return angles


def symmetric_hexagon_from_major(
    major_nm: float,
    slanted_side_nm: float,
    vertical_side_nm: float,
) -> tuple[np.ndarray, float, float, float]:
    half_vertical_side = vertical_side_nm / 2.0
    half_major = major_nm / 2.0
    delta = half_major - half_vertical_side
    if delta < -1e-9 or delta > slanted_side_nm + 1e-9:
        raise ValueError(
            f"Major axis {major_nm:.3f} nm is not compatible with fixed sides "
            f"{slanted_side_nm:.3f}/{vertical_side_nm:.3f} nm"
        )
    half_minor = math.sqrt(max(0.0, slanted_side_nm**2 - delta**2))
    vertices = np.array(
        [
            [0.0, -half_major],
            [half_minor, -half_vertical_side],
            [half_minor, half_vertical_side],
            [0.0, half_major],
            [-half_minor, half_vertical_side],
            [-half_minor, -half_vertical_side],
        ],
        dtype=float,
    )
    inradius = min(half_minor, half_major * half_minor / slanted_side_nm)
    angles = interior_angles(vertices)
    top_bottom_angle = (angles[0] + angles[3]) / 2.0
    side_angle = (angles[1] + angles[2] + angles[4] + angles[5]) / 4.0
    return vertices, inradius, top_bottom_angle, side_angle


def read_slit_widths(path: Path) -> list[tuple[int, str, float]]:
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
        if "sd_spacing_nm" not in raw.columns:
            raise ValueError(f"CSV input must contain sd_spacing_nm: {path}")
        rows = []
        for index, row in raw.iterrows():
            width = float(row["sd_spacing_nm"])
            if not math.isfinite(width):
                continue
            sample = int(row["sample"]) if "sample" in raw.columns else len(rows) + 1
            tomo = str(row["tomo"]) if "tomo" in raw.columns else str(index + 1)
            rows.append((sample, tomo, width))
        if not rows:
            raise RuntimeError(f"No numeric width values found in {path}")
        return rows
    raw = pd.read_excel(path, header=None)
    rows = []
    for _, row in raw.iterrows():
        try:
            width = float(row.iloc[5])
        except Exception:
            continue
        if not math.isfinite(width):
            continue
        tomo = row.iloc[4]
        rows.append((len(rows) + 1, str(tomo), width))
    if not rows:
        raise RuntimeError(f"No numeric width values found in {path}")
    return rows


def read_sides(path: Path) -> tuple[float, float, float, float]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    sides = [float(row["side_to_next_nm"]) for row in rows]
    vertices = np.array([[float(row["x_nm"]), float(row["y_nm"])] for row in rows], dtype=float)
    slanted_side = float(np.mean([sides[0], sides[2], sides[3], sides[5]]))
    vertical_side = float(np.mean([sides[1], sides[4]]))
    current_major = float(vertices[:, 1].max() - vertices[:, 1].min())
    current_minor = float(vertices[:, 0].max() - vertices[:, 0].min())
    return slanted_side, vertical_side, current_major, current_minor


def threshold_for_albumin(
    albumin_radius_nm: float,
    reference_length_nm: float,
    current_hex_major_nm: float,
    slanted_side_nm: float,
    vertical_side_nm: float,
) -> tuple[float, float, float]:
    half_vertical_side = vertical_side_nm / 2.0
    feasible = []
    for delta in np.linspace(0.0, slanted_side_nm, 1_000_001):
        half_minor = math.sqrt(max(0.0, slanted_side_nm**2 - delta**2))
        half_major = half_vertical_side + delta
        inradius = min(half_minor, half_major * half_minor / slanted_side_nm)
        if inradius >= albumin_radius_nm:
            feasible.append((delta, half_major, half_minor, inradius))
    if not feasible:
        raise RuntimeError("No fixed-side hexagon shape can pass the albumin radius")
    # Closest passable state to the current elongated SDsubunit shape is the largest passable delta.
    _, half_major, half_minor, inradius = feasible[-1]
    pass_major = 2.0 * half_major
    pass_minor = 2.0 * half_minor
    threshold_length = reference_length_nm * pass_major / current_hex_major_nm
    return threshold_length, pass_major, pass_minor


def write_rows(path: Path, rows: list[FlexibleHexagonRow]) -> None:
    fieldnames = [
        "sample",
        "tomo",
        "slit_width_nm",
        "estimated_protein_length_nm",
        "length_scale_vs_reference",
        "hex_major_axis_nm",
        "hex_minor_axis_nm",
        "hex_inradius_nm",
        "top_bottom_angle_deg",
        "side_angle_deg",
        "albumin_radius_margin_nm",
        "passes_albumin_3p55nm",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample": row.sample,
                    "tomo": row.tomo,
                    "slit_width_nm": row.slit_width_nm,
                    "estimated_protein_length_nm": row.protein_length_nm,
                    "length_scale_vs_reference": row.length_scale,
                    "hex_major_axis_nm": row.hex_major_nm,
                    "hex_minor_axis_nm": row.hex_minor_nm,
                    "hex_inradius_nm": row.inradius_nm,
                    "top_bottom_angle_deg": row.top_bottom_angle_deg,
                    "side_angle_deg": row.side_angle_deg,
                    "albumin_radius_margin_nm": row.albumin_margin_nm,
                    "passes_albumin_3p55nm": row.passes_albumin,
                }
            )


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "q1": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q3": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def write_summary(
    path: Path,
    rows: list[FlexibleHexagonRow],
    reference_length_nm: float,
    length_fraction: float,
    albumin_radius_nm: float,
    threshold_length_nm: float,
    pass_major_nm: float,
    pass_minor_nm: float,
) -> None:
    arrays = {
        "estimated_protein_length_nm": np.array([row.protein_length_nm for row in rows]),
        "hex_major_axis_nm": np.array([row.hex_major_nm for row in rows]),
        "hex_minor_axis_nm": np.array([row.hex_minor_nm for row in rows]),
        "hex_inradius_nm": np.array([row.inradius_nm for row in rows]),
        "top_bottom_angle_deg": np.array([row.top_bottom_angle_deg for row in rows]),
        "side_angle_deg": np.array([row.side_angle_deg for row in rows]),
    }
    pass_count = sum(row.passes_albumin for row in rows)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["reference_sdsubunit_length_nm", reference_length_nm])
        writer.writerow(["slit_width_to_protein_length_fraction", length_fraction])
        writer.writerow(["albumin_radius_nm", albumin_radius_nm])
        writer.writerow(["pass_threshold_protein_length_nm", threshold_length_nm])
        writer.writerow(["pass_threshold_hex_major_axis_nm", pass_major_nm])
        writer.writerow(["pass_threshold_hex_minor_axis_nm", pass_minor_nm])
        writer.writerow(["sample_count", len(rows)])
        writer.writerow(["pass_count", pass_count])
        writer.writerow(["pass_fraction", pass_count / len(rows)])
        for name, values in arrays.items():
            for key, value in percentile_summary(values).items():
                writer.writerow([f"{name}_{key}", value])


def plot_distribution(
    path: Path,
    rows: list[FlexibleHexagonRow],
    threshold_length_nm: float,
    albumin_radius_nm: float,
) -> None:
    lengths = np.array([row.protein_length_nm for row in rows])
    inradii = np.array([row.inradius_nm for row in rows])
    pass_mask = np.array([row.passes_albumin for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)

    bins = np.arange(math.floor(lengths.min()), math.ceil(lengths.max()) + 1, 1)
    axes[0].hist(lengths, bins=bins, color="#1b6a83", edgecolor="white", alpha=0.9)
    axes[0].axvline(threshold_length_nm, color="#d7263d", linewidth=2.0, linestyle="--", label="albumin pass threshold")
    axes[0].set_xlabel("Estimated SDsubunitCorrect length (nm)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Length distribution")
    axes[0].legend(fontsize=8)

    axes[1].scatter(lengths[~pass_mask], inradii[~pass_mask], color="#d7263d", s=44, label="not pass")
    axes[1].scatter(lengths[pass_mask], inradii[pass_mask], color="#138a36", s=44, label="pass")
    axes[1].axhline(albumin_radius_nm, color="#111111", linewidth=1.5, linestyle=":", label="albumin radius")
    axes[1].axvline(threshold_length_nm, color="#d7263d", linewidth=2.0, linestyle="--")
    axes[1].set_xlabel("Estimated SDsubunitCorrect length (nm)")
    axes[1].set_ylabel("Hexagon inradius (nm)")
    axes[1].set_title("Flexible hexagon passability")
    axes[1].legend(fontsize=8)

    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_hexagon_shapes(
    path: Path,
    rows: list[FlexibleHexagonRow],
    reference_length_nm: float,
    current_hex_major_nm: float,
    slanted_side_nm: float,
    vertical_side_nm: float,
    threshold_length_nm: float,
) -> None:
    lengths = np.array([row.protein_length_nm for row in rows])
    quantile_specs = [
        ("min", float(np.min(lengths))),
        ("Q1", float(np.percentile(lengths, 25))),
        ("median", float(np.median(lengths))),
        ("Q3", float(np.percentile(lengths, 75))),
        ("max", float(np.max(lengths))),
        ("threshold", threshold_length_nm),
    ]
    colors = {
        "min": "#0f9d58",
        "Q1": "#4caf50",
        "median": "#f5a623",
        "Q3": "#d97706",
        "max": "#d7263d",
        "threshold": "#111111",
    }
    styles = {"threshold": "--"}

    fig, ax = plt.subplots(figsize=(6.4, 6.2), constrained_layout=True)
    for label, protein_length in quantile_specs:
        major = current_hex_major_nm * protein_length / reference_length_nm
        vertices, inradius, _, _ = symmetric_hexagon_from_major(major, slanted_side_nm, vertical_side_nm)
        closed = np.vstack([vertices, vertices[0]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color=colors[label],
            linestyle=styles.get(label, "-"),
            linewidth=2.2,
            label=f"{label}: L={protein_length:.2f} nm, r={inradius:.2f} nm",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color="#d9dee2", linewidth=0.8)
    ax.axvline(0, color="#d9dee2", linewidth=0.8)
    ax.set_xlabel("Hexagon X (nm)")
    ax.set_ylabel("Hexagon Y (nm)")
    ax.set_title("Fixed-side flexible hexagon shape distribution")
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_appendix_hexagon_schematic(
    png_path: Path,
    svg_path: Path,
    rows: list[FlexibleHexagonRow],
    reference_length_nm: float,
    current_hex_major_nm: float,
    slanted_side_nm: float,
    vertical_side_nm: float,
    threshold_length_nm: float,
) -> None:
    lengths = np.array([row.protein_length_nm for row in rows], dtype=float)
    specs = [
        ("Maximum", float(np.max(lengths)), "#d92943"),
        ("Mean", float(np.mean(lengths)), "#f0a22e"),
        ("Minimum", float(np.min(lengths)), "#189a5b"),
    ]

    threshold_major = current_hex_major_nm * threshold_length_nm / reference_length_nm
    threshold_vertices, threshold_inradius, _, _ = symmetric_hexagon_from_major(
        threshold_major,
        slanted_side_nm,
        vertical_side_nm,
    )
    threshold_closed = np.vstack([threshold_vertices, threshold_vertices[0]])

    all_vertices = [threshold_vertices]
    panel_data = []
    for label, protein_length, color in specs:
        major = current_hex_major_nm * protein_length / reference_length_nm
        vertices, inradius, _, _ = symmetric_hexagon_from_major(
            major,
            slanted_side_nm,
            vertical_side_nm,
        )
        panel_data.append((label, protein_length, color, vertices, inradius))
        all_vertices.append(vertices)

    x_lim = 6.0
    y_lim = 10.0

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.55), constrained_layout=False)
    for ax, (label, protein_length, color, vertices, inradius) in zip(axes, panel_data):
        closed = np.vstack([vertices, vertices[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=3.0)
        ax.fill(closed[:, 0], closed[:, 1], color=color, alpha=0.08)
        ax.plot(
            threshold_closed[:, 0],
            threshold_closed[:, 1],
            color="#202020",
            linestyle=(0, (4, 3)),
            linewidth=2.0,
            alpha=0.9,
            zorder=5,
        )
        ax.axhline(0, color="#d9dee2", linewidth=0.8, zorder=0)
        ax.axvline(0, color="#d9dee2", linewidth=0.8, zorder=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_xticks(np.arange(-6, 7, 3))
        ax.set_title(f"{label} pore shape\nSD width={protein_length:.2f} nm, r={inradius:.2f} nm", fontsize=14, pad=14)
        ax.tick_params(labelsize=12)

    def add_max_to_min_arrows(ax, label: str | None = None) -> None:
        # Max -> min means the fixed-side hexagon shortens vertically and widens laterally.
        ax.annotate(
            "",
            xy=(0.0, 7.6),
            xytext=(0.0, 9.0),
            arrowprops=dict(arrowstyle="-|>", color="#5c6f82", linewidth=1.5),
        )
        ax.annotate(
            "",
            xy=(0.0, -7.6),
            xytext=(0.0, -9.0),
            arrowprops=dict(arrowstyle="-|>", color="#5c6f82", linewidth=1.5),
        )
        ax.annotate(
            "",
            xy=(-2.8, 0.0),
            xytext=(-1.2, 0.0),
            arrowprops=dict(arrowstyle="-|>", color="#5c6f82", linewidth=1.5),
        )
        ax.annotate(
            "",
            xy=(2.8, 0.0),
            xytext=(1.2, 0.0),
            arrowprops=dict(arrowstyle="-|>", color="#5c6f82", linewidth=1.5),
        )
        if label:
            ax.text(
                0.0,
                -y_lim + 0.25,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#5c6f82",
            )

    add_max_to_min_arrows(axes[0])
    add_max_to_min_arrows(axes[1])

    fig.suptitle("Large-pore shape varies with SD width", fontsize=20, weight="bold")
    fig.supxlabel("Short-axis length (nm)", fontsize=16, y=0.145)
    fig.supylabel("Long-axis length (nm)", fontsize=16, x=0.045)
    arrow_handle = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", mutation_scale=12, linewidth=1.8, color="#5c6f82")
    pore_shape_handle = (
        Line2D([0], [0], color="#d92943", linewidth=3.0),
        Line2D([0], [0], color="#f0a22e", linewidth=3.0),
        Line2D([0], [0], color="#189a5b", linewidth=3.0),
    )
    legend_handles = [
        Line2D([0], [0], color="#202020", linestyle=(0, (4, 3)), linewidth=2.2, label="albumin ellipsoid pass threshold"),
        pore_shape_handle,
        arrow_handle,
    ]
    arrow_handle.set_label("shape change: max -> min")
    fig.legend(
        handles=legend_handles,
        labels=["albumin ellipsoid pass threshold", "pore shape", "shape change: max -> min"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
        fontsize=12,
        handlelength=2.4,
        columnspacing=1.8,
        handler_map={FancyArrowPatch: HandlerArrow(), tuple: HandlerStackedLines()},
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.72, bottom=0.26, wspace=0.48)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_fixed_width_hexagon_schematic(
    png_path: Path,
    svg_path: Path,
    widths_nm: list[float],
    reference_length_nm: float,
    current_hex_major_nm: float,
    slanted_side_nm: float,
    vertical_side_nm: float,
    threshold_length_nm: float,
) -> None:
    threshold_major = current_hex_major_nm * threshold_length_nm / reference_length_nm
    threshold_vertices, _, _, _ = symmetric_hexagon_from_major(
        threshold_major,
        slanted_side_nm,
        vertical_side_nm,
    )
    threshold_closed = np.vstack([threshold_vertices, threshold_vertices[0]])

    colors = ["#189a5b", "#d92943", "#f0a22e", "#4c78a8"]
    panel_data = []
    all_vertices = [threshold_vertices]
    for width_nm, color in zip(widths_nm, colors):
        major = current_hex_major_nm * width_nm / reference_length_nm
        vertices, inradius, _, _ = symmetric_hexagon_from_major(
            major,
            slanted_side_nm,
            vertical_side_nm,
        )
        panel_data.append((width_nm, color, vertices, inradius))
        all_vertices.append(vertices)

    stacked = np.vstack(all_vertices)
    x_lim = max(4.3, float(np.max(np.abs(stacked[:, 0]))) + 1.0)
    y_lim = max(9.2, float(np.max(np.abs(stacked[:, 1]))) + 1.0)

    fig, axes = plt.subplots(1, len(panel_data), figsize=(8.9, 5.3), constrained_layout=False)
    if len(panel_data) == 1:
        axes = [axes]

    for ax, (width_nm, color, vertices, inradius) in zip(axes, panel_data):
        closed = np.vstack([vertices, vertices[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=3.0)
        ax.fill(closed[:, 0], closed[:, 1], color=color, alpha=0.08)
        ax.plot(
            threshold_closed[:, 0],
            threshold_closed[:, 1],
            color="#202020",
            linestyle=(0, (4, 3)),
            linewidth=2.0,
            alpha=0.9,
            zorder=5,
        )
        ax.axhline(0, color="#d9dee2", linewidth=0.8, zorder=0)
        ax.axvline(0, color="#d9dee2", linewidth=0.8, zorder=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_xticks(np.arange(-6, 7, 3))
        ax.set_title(f"SD width={width_nm:.1f} nm\nr={inradius:.2f} nm", fontsize=14, pad=12)
        ax.tick_params(labelsize=12)

    fig.suptitle("Flexible large-pore shape at selected SD widths", fontsize=18, weight="bold")
    fig.supxlabel("Short-axis length (nm)", fontsize=15, y=0.16)
    fig.supylabel("Long-axis length (nm)", fontsize=15, x=0.04)
    legend_handles = [
        Line2D([0], [0], color="#202020", linestyle=(0, (4, 3)), linewidth=2.2),
        Line2D([0], [0], color=colors[0], linewidth=3.0),
        Line2D([0], [0], color=colors[1], linewidth=3.0),
    ]
    fig.legend(
        handles=legend_handles,
        labels=[
            "albumin ellipsoid pass threshold",
            f"SD width={widths_nm[0]:.1f} nm",
            f"SD width={widths_nm[1]:.1f} nm",
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        frameon=False,
        fontsize=11,
        handlelength=2.4,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.72, bottom=0.28, wspace=0.38)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_index(
    output_dir: Path,
    rows: list[FlexibleHexagonRow],
    threshold_length_nm: float,
    pass_major_nm: float,
    pass_minor_nm: float,
) -> Path:
    pass_count = sum(row.passes_albumin for row in rows)
    out = output_dir / "index.html"
    out.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flexible SDsubunit Hexagon From Slit Width</title>
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
    <h1>Flexible SDsubunit Hexagon From Slit Width</h1>
    <p>Albumin pass threshold length: {threshold_length_nm:.3f} nm. Threshold hexagon major/minor axes: {pass_major_nm:.3f}/{pass_minor_nm:.3f} nm. Passing samples: {pass_count}/{len(rows)}.</p>
    <p>
      <a href="SDsubunit_flexible_hexagon_by_slit_width.csv">Sample CSV</a> ·
      <a href="SDsubunit_flexible_hexagon_summary.csv">Summary CSV</a>
    </p>
    <img src="SDsubunit_flexible_hexagon_length_distribution.png" alt="length and passability distribution">
    <img src="SDsubunit_flexible_hexagon_appendix_schematic.png" alt="appendix hexagon deformation schematic">
    <img src="SDsubunit_flexible_hexagon_width_47p5_57p5_schematic.png" alt="selected width hexagon deformation schematic">
    <img src="SDsubunit_flexible_hexagon_shape_distribution.png" alt="shape distribution">
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Map slit-width-derived SDsubunit lengths to fixed-side flexible hexagons.")
    parser.add_argument(
        "--slit-width-xlsx",
        default="C:/Users/糖吉诃德/xwechat_files/wxid_5wtp6e154kya22_b9f0/msg/file/2026-07/Slit_width.xlsx",
    )
    parser.add_argument(
        "--hexagon-csv",
        default="protein_structure/selected_symmetric_hexagons_refined/SDsubunitCorrect_closed_angle_117_expanded_symmetric_hexagon.csv",
    )
    parser.add_argument("--output-dir", default="protein_structure/flexible_hexagon_length_distribution")
    parser.add_argument("--reference-length-nm", type=float, default=50.2439981515067)
    parser.add_argument("--length-fraction", type=float, default=0.90)
    parser.add_argument("--albumin-radius-nm", type=float, default=3.55)
    parser.add_argument(
        "--appendix-ellipsoid-threshold-length-nm",
        type=float,
        default=52.379922487355266,
        help="SD spacing/length for the just-passable PCA albumin ellipsoid threshold used in the appendix schematic.",
    )
    args = parser.parse_args()

    slit_rows = read_slit_widths(Path(args.slit_width_xlsx))
    slanted_side, vertical_side, current_hex_major, current_hex_minor = read_sides(Path(args.hexagon_csv))
    threshold_length, pass_major, pass_minor = threshold_for_albumin(
        args.albumin_radius_nm,
        args.reference_length_nm,
        current_hex_major,
        slanted_side,
        vertical_side,
    )

    rows: list[FlexibleHexagonRow] = []
    for sample, tomo, slit_width in slit_rows:
        protein_length = slit_width * args.length_fraction
        length_scale = protein_length / args.reference_length_nm
        major = current_hex_major * length_scale
        vertices, inradius, top_bottom_angle, side_angle = symmetric_hexagon_from_major(
            major,
            slanted_side,
            vertical_side,
        )
        minor = float(vertices[:, 0].max() - vertices[:, 0].min())
        rows.append(
            FlexibleHexagonRow(
                sample=sample,
                tomo=tomo,
                slit_width_nm=slit_width,
                protein_length_nm=protein_length,
                length_scale=length_scale,
                hex_major_nm=major,
                hex_minor_nm=minor,
                inradius_nm=inradius,
                top_bottom_angle_deg=top_bottom_angle,
                side_angle_deg=side_angle,
                albumin_margin_nm=inradius - args.albumin_radius_nm,
                passes_albumin=inradius >= args.albumin_radius_nm,
            )
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = output_dir / "SDsubunit_flexible_hexagon_by_slit_width.csv"
    summary_csv = output_dir / "SDsubunit_flexible_hexagon_summary.csv"
    distribution_png = output_dir / "SDsubunit_flexible_hexagon_length_distribution.png"
    shape_png = output_dir / "SDsubunit_flexible_hexagon_shape_distribution.png"
    appendix_png = output_dir / "SDsubunit_flexible_hexagon_appendix_schematic.png"
    appendix_svg = output_dir / "SDsubunit_flexible_hexagon_appendix_schematic.svg"
    selected_width_png = output_dir / "SDsubunit_flexible_hexagon_width_47p5_57p5_schematic.png"
    selected_width_svg = output_dir / "SDsubunit_flexible_hexagon_width_47p5_57p5_schematic.svg"

    write_rows(sample_csv, rows)
    write_summary(
        summary_csv,
        rows,
        args.reference_length_nm,
        args.length_fraction,
        args.albumin_radius_nm,
        threshold_length,
        pass_major,
        pass_minor,
    )
    plot_distribution(distribution_png, rows, threshold_length, args.albumin_radius_nm)
    plot_hexagon_shapes(
        shape_png,
        rows,
        args.reference_length_nm,
        current_hex_major,
        slanted_side,
        vertical_side,
        threshold_length,
    )
    plot_appendix_hexagon_schematic(
        appendix_png,
        appendix_svg,
        rows,
        args.reference_length_nm,
        current_hex_major,
        slanted_side,
        vertical_side,
        args.appendix_ellipsoid_threshold_length_nm,
    )
    plot_fixed_width_hexagon_schematic(
        selected_width_png,
        selected_width_svg,
        [47.5, 57.5],
        args.reference_length_nm,
        current_hex_major,
        slanted_side,
        vertical_side,
        args.appendix_ellipsoid_threshold_length_nm,
    )
    index = write_index(output_dir, rows, threshold_length, pass_major, pass_minor)

    print(f"Wrote {sample_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {distribution_png}")
    print(f"Wrote {appendix_png}")
    print(f"Wrote {appendix_svg}")
    print(f"Wrote {selected_width_png}")
    print(f"Wrote {selected_width_svg}")
    print(f"Wrote {shape_png}")
    print(f"Wrote {index}")
    print(f"Reference SDsubunit length: {args.reference_length_nm:.3f} nm")
    print(f"Albumin pass threshold length: {threshold_length:.3f} nm")
    print(f"Appendix ellipsoid pass threshold length: {args.appendix_ellipsoid_threshold_length_nm:.3f} nm")
    print(f"Threshold hexagon major/minor: {pass_major:.3f}/{pass_minor:.3f} nm")
    print(f"Passing samples: {sum(row.passes_albumin for row in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()

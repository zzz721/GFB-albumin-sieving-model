from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .flexible_hexagon_from_slit_width import read_sides, read_slit_widths, symmetric_hexagon_from_major


@dataclass(frozen=True)
class AeffRow:
    sample: int
    tomo: str
    sd_spacing_nm: float
    hex_major_nm: float
    hex_minor_nm: float
    hex_inradius_nm: float
    aeff_fixed_nm2: float
    aeff_min_nm2: float
    aeff_mean_nm2: float
    aeff_max_nm2: float
    best_theta_deg: float
    positive_theta_fraction: float
    center_region_vertices_fixed: int
    passes_fixed: bool
    passes_any_theta: bool


def polygon_area(vertices: list[np.ndarray] | np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    pts = np.asarray(vertices, dtype=float)
    return float(abs(0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])))


def clip_polygon_halfplane(polygon: list[np.ndarray], normal: np.ndarray, offset: float) -> list[np.ndarray]:
    if not polygon:
        return []

    def inside(point: np.ndarray) -> bool:
        return float(normal @ point) <= offset + 1e-10

    clipped: list[np.ndarray] = []
    previous = np.asarray(polygon[-1], dtype=float)
    previous_inside = inside(previous)
    for current_raw in polygon:
        current = np.asarray(current_raw, dtype=float)
        current_inside = inside(current)
        if current_inside != previous_inside:
            direction = current - previous
            denominator = float(normal @ direction)
            if abs(denominator) > 1e-12:
                t = (offset - float(normal @ previous)) / denominator
                clipped.append(previous + t * direction)
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    return clipped


def erode_convex_polygon_by_ellipse(
    vertices_xy: np.ndarray,
    ellipse_rx_nm: float,
    ellipse_ry_nm: float,
    theta_deg: float = 0.0,
) -> list[np.ndarray]:
    vertices = np.asarray(vertices_xy, dtype=float)
    signed_area = 0.5 * np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) - np.roll(vertices[:, 0], -1) * vertices[:, 1])
    if signed_area < 0:
        vertices = vertices[::-1]

    extent = max(float(np.abs(vertices).max()) + ellipse_rx_nm + ellipse_ry_nm + 5.0, 20.0)
    center_region = [
        np.array([-extent, -extent], dtype=float),
        np.array([extent, -extent], dtype=float),
        np.array([extent, extent], dtype=float),
        np.array([-extent, extent], dtype=float),
    ]

    for idx, point in enumerate(vertices):
        next_point = vertices[(idx + 1) % len(vertices)]
        edge = next_point - point
        outward = np.array([edge[1], -edge[0]], dtype=float)
        outward /= float(np.linalg.norm(outward))
        boundary_offset = float(outward @ point)

        # Support function of a rotated ellipse centered at the albumin centroid.
        theta = math.radians(theta_deg)
        c = math.cos(theta)
        s = math.sin(theta)
        local_x = c * outward[0] + s * outward[1]
        local_y = -s * outward[0] + c * outward[1]
        albumin_support = math.sqrt((ellipse_rx_nm * local_x) ** 2 + (ellipse_ry_nm * local_y) ** 2)
        center_region = clip_polygon_halfplane(center_region, outward, boundary_offset - albumin_support)
        if len(center_region) < 3:
            return []
    return center_region


def scan_aeff_theta(
    vertices_xy: np.ndarray,
    ellipse_rx_nm: float,
    ellipse_ry_nm: float,
    theta_step_deg: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    # 0 and 180 degrees are equivalent for an ellipse; endpoint=False avoids duplicate sampling.
    thetas = np.arange(0.0, 180.0, theta_step_deg, dtype=float)
    areas = []
    for theta in thetas:
        center_region = erode_convex_polygon_by_ellipse(vertices_xy, ellipse_rx_nm, ellipse_ry_nm, theta)
        areas.append(polygon_area(center_region))
    areas_np = np.array(areas, dtype=float)
    best_theta = float(thetas[int(np.argmax(areas_np))]) if areas_np.size else math.nan
    return thetas, areas_np, best_theta


def compute_rows(
    slit_width_path: Path,
    hexagon_csv: Path,
    reference_length_nm: float,
    ellipse_rx_nm: float,
    ellipse_ry_nm: float,
    theta_step_deg: float,
) -> list[AeffRow]:
    slanted_side, vertical_side, current_hex_major, _ = read_sides(hexagon_csv)
    rows: list[AeffRow] = []
    for sample, tomo, sd_spacing in read_slit_widths(slit_width_path):
        major = current_hex_major * sd_spacing / reference_length_nm
        vertices, inradius, _, _ = symmetric_hexagon_from_major(major, slanted_side, vertical_side)
        center_region_fixed = erode_convex_polygon_by_ellipse(vertices, ellipse_rx_nm, ellipse_ry_nm, 0.0)
        fixed = polygon_area(center_region_fixed)
        _, theta_areas, best_theta = scan_aeff_theta(vertices, ellipse_rx_nm, ellipse_ry_nm, theta_step_deg)
        positive = theta_areas > 1e-9
        rows.append(
            AeffRow(
                sample=sample,
                tomo=tomo,
                sd_spacing_nm=sd_spacing,
                hex_major_nm=major,
                hex_minor_nm=float(np.ptp(vertices[:, 0])),
                hex_inradius_nm=inradius,
                aeff_fixed_nm2=fixed,
                aeff_min_nm2=float(np.min(theta_areas)) if theta_areas.size else 0.0,
                aeff_mean_nm2=float(np.mean(theta_areas)) if theta_areas.size else 0.0,
                aeff_max_nm2=float(np.max(theta_areas)) if theta_areas.size else 0.0,
                best_theta_deg=best_theta,
                positive_theta_fraction=float(np.mean(positive)) if theta_areas.size else 0.0,
                center_region_vertices_fixed=len(center_region_fixed),
                passes_fixed=fixed > 1e-9,
                passes_any_theta=bool(np.any(positive)),
            )
        )
    return rows


def find_threshold_spacing(
    hexagon_csv: Path,
    reference_length_nm: float,
    ellipse_rx_nm: float,
    ellipse_ry_nm: float,
    theta_step_deg: float,
    mode: str,
    lo_nm: float = 35.0,
    hi_nm: float = 60.0,
) -> float:
    slanted_side, vertical_side, current_hex_major, _ = read_sides(hexagon_csv)

    def aeff_at(sd_spacing: float) -> float:
        major = current_hex_major * sd_spacing / reference_length_nm
        try:
            vertices, _, _, _ = symmetric_hexagon_from_major(major, slanted_side, vertical_side)
        except ValueError:
            return 0.0
        if mode == "fixed":
            center_region = erode_convex_polygon_by_ellipse(vertices, ellipse_rx_nm, ellipse_ry_nm, 0.0)
            return polygon_area(center_region)
        _, areas, _ = scan_aeff_theta(vertices, ellipse_rx_nm, ellipse_ry_nm, theta_step_deg)
        if mode == "mean":
            return float(np.mean(areas))
        if mode == "max":
            return float(np.max(areas))
        if mode == "min":
            return float(np.min(areas))
        raise ValueError(f"Unknown threshold mode: {mode}")

    grid = np.linspace(lo_nm, hi_nm, 2001)
    positive = np.array([aeff_at(value) > 1e-8 for value in grid])
    if not positive.any():
        return math.nan
    max_positive = grid[positive].max()
    left = max_positive
    right_candidates = grid[grid > max_positive]
    if right_candidates.size == 0:
        return max_positive
    right = float(right_candidates[0])
    for _ in range(70):
        mid = (left + right) / 2.0
        if aeff_at(mid) > 1e-8:
            left = mid
        else:
            right = mid
    return left


def write_csv(path: Path, rows: list[AeffRow]) -> None:
    fieldnames = [
        "sample",
        "tomo",
        "sd_spacing_nm",
        "hex_major_axis_nm",
        "hex_minor_axis_nm",
        "hex_inradius_nm",
        "aeff_fixed_theta0_nm2",
        "aeff_min_nm2",
        "aeff_mean_nm2",
        "aeff_max_nm2",
        "best_theta_deg",
        "positive_theta_fraction",
        "center_region_vertices_fixed",
        "aeff_fixed_positive",
        "aeff_any_theta_positive",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample": row.sample,
                    "tomo": row.tomo,
                    "sd_spacing_nm": row.sd_spacing_nm,
                    "hex_major_axis_nm": row.hex_major_nm,
                    "hex_minor_axis_nm": row.hex_minor_nm,
                    "hex_inradius_nm": row.hex_inradius_nm,
                    "aeff_fixed_theta0_nm2": row.aeff_fixed_nm2,
                    "aeff_min_nm2": row.aeff_min_nm2,
                    "aeff_mean_nm2": row.aeff_mean_nm2,
                    "aeff_max_nm2": row.aeff_max_nm2,
                    "best_theta_deg": row.best_theta_deg,
                    "positive_theta_fraction": row.positive_theta_fraction,
                    "center_region_vertices_fixed": row.center_region_vertices_fixed,
                    "aeff_fixed_positive": row.passes_fixed,
                    "aeff_any_theta_positive": row.passes_any_theta,
                }
            )


def write_summary(
    path: Path,
    rows: list[AeffRow],
    ellipse_rx_nm: float,
    ellipse_ry_nm: float,
    threshold_fixed_nm: float,
    threshold_mean_nm: float,
    threshold_max_nm: float,
    theta_step_deg: float,
) -> None:
    fixed = np.array([row.aeff_fixed_nm2 for row in rows], dtype=float)
    mean = np.array([row.aeff_mean_nm2 for row in rows], dtype=float)
    max_values = np.array([row.aeff_max_nm2 for row in rows], dtype=float)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["model", "albumin PCA ellipsoid, long axis aligned with flow"])
        writer.writerow(["theta_scan_deg", f"0-180 step {theta_step_deg:g}"])
        writer.writerow(["ellipse_cross_section_diameter_x_nm", 2 * ellipse_rx_nm])
        writer.writerow(["ellipse_cross_section_diameter_y_nm", 2 * ellipse_ry_nm])
        writer.writerow(["critical_sd_spacing_fixed_theta0_nm", threshold_fixed_nm])
        writer.writerow(["critical_sd_spacing_mean_theta_nm", threshold_mean_nm])
        writer.writerow(["critical_sd_spacing_best_theta_nm", threshold_max_nm])
        writer.writerow(["sample_count", len(rows)])
        writer.writerow(["fixed_theta0_positive_count", int(np.sum(fixed > 1e-9))])
        writer.writerow(["any_theta_positive_count", int(np.sum(max_values > 1e-9))])
        writer.writerow(["fixed_theta0_positive_fraction", float(np.mean(fixed > 1e-9))])
        writer.writerow(["any_theta_positive_fraction", float(np.mean(max_values > 1e-9))])
        writer.writerow(["aeff_fixed_mean_nm2", float(np.mean(fixed))])
        writer.writerow(["aeff_theta_mean_mean_nm2", float(np.mean(mean))])
        writer.writerow(["aeff_best_mean_nm2", float(np.mean(max_values))])
        writer.writerow(["aeff_fixed_max_nm2", float(np.max(fixed))])
        writer.writerow(["aeff_theta_mean_max_nm2", float(np.max(mean))])
        writer.writerow(["aeff_best_max_nm2", float(np.max(max_values))])


def plot_bar(path: Path, rows: list[AeffRow], threshold_fixed_nm: float, threshold_max_nm: float) -> None:
    sorted_rows = sorted(rows, key=lambda row: row.sd_spacing_nm)
    x = np.arange(len(sorted_rows))
    fixed = np.array([row.aeff_fixed_nm2 for row in sorted_rows])
    mean = np.array([row.aeff_mean_nm2 for row in sorted_rows])
    best = np.array([row.aeff_max_nm2 for row in sorted_rows])
    labels = [f"{row.sd_spacing_nm:.1f}" for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(12.0, 5.4), constrained_layout=True)
    width = 0.26
    ax.bar(x - width, fixed, width=width, color="#f5a623", edgecolor="#333333", linewidth=0.35, label="fixed theta=0")
    ax.bar(x, mean, width=width, color="#4f8fc0", edgecolor="#333333", linewidth=0.35, label="theta mean")
    ax.bar(x + width, best, width=width, color="#2ca25f", edgecolor="#333333", linewidth=0.35, label="theta best")
    ax.axhline(0, color="#222222", linewidth=0.8)
    if not math.isnan(threshold_max_nm):
        insertion = sum(row.sd_spacing_nm <= threshold_max_nm for row in sorted_rows) - 0.5
        ax.axvline(insertion, color="#138a36", linestyle="--", linewidth=2.0, label=f"best threshold ~{threshold_max_nm:.2f} nm")
    if not math.isnan(threshold_fixed_nm):
        insertion = sum(row.sd_spacing_nm <= threshold_fixed_nm for row in sorted_rows) - 0.5
        ax.axvline(insertion, color="#d7263d", linestyle=":", linewidth=2.0, label=f"fixed threshold ~{threshold_fixed_nm:.2f} nm")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_xlabel("SD spacing / SDsubunit full length (nm)")
    ax.set_ylabel("Aeff for albumin centroid (nm²)")
    ax.set_title("Flow-aligned albumin ellipsoid: Aeff(theta) across in-plane rotation")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_scatter(path: Path, rows: list[AeffRow], threshold_max_nm: float) -> None:
    sd = np.array([row.sd_spacing_nm for row in rows])
    fixed = np.array([row.aeff_fixed_nm2 for row in rows])
    mean = np.array([row.aeff_mean_nm2 for row in rows])
    best = np.array([row.aeff_max_nm2 for row in rows])

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.scatter(sd, fixed, color="#f5a623", edgecolor="#7a4d00", s=50, label="fixed theta=0")
    ax.scatter(sd, mean, color="#4f8fc0", edgecolor="#254c66", s=50, label="theta mean")
    ax.scatter(sd, best, color="#2ca25f", edgecolor="#165a34", s=58, label="theta best")
    if not math.isnan(threshold_max_nm):
        ax.axvline(threshold_max_nm, color="#138a36", linestyle="--", linewidth=2.0, label=f"best threshold {threshold_max_nm:.2f} nm")
    ax.set_xlabel("SD spacing / SDsubunit full length (nm)")
    ax.set_ylabel("Aeff (nm²)")
    ax.set_title("Aeff decreases as SD spacing elongates the hexagon")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_index(output_dir: Path, rows: list[AeffRow], threshold_fixed_nm: float, threshold_mean_nm: float, threshold_max_nm: float) -> Path:
    positive_fixed = sum(row.passes_fixed for row in rows)
    positive_any = sum(row.passes_any_theta for row in rows)
    max_aeff = max(row.aeff_max_nm2 for row in rows)
    out = output_dir / "index.html"
    out.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Albumin Aeff vs SD Spacing</title>
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
    <h1>Albumin Aeff vs SD spacing</h1>
    <p>Model: orange PCA albumin ellipsoid, long axis aligned with flow. The in-plane footprint is an ellipse of 5.45 x 6.99 nm. Aeff(theta) is the centroid-accessible area after eroding each flexible SD hexagon by the rotated ellipse.</p>
    <p>Critical SD spacing for Aeff &gt; 0: fixed theta=0 is {threshold_fixed_nm:.3f} nm; theta mean is {threshold_mean_nm:.3f} nm; best theta is {threshold_max_nm:.3f} nm. Positive samples: fixed {positive_fixed}/{len(rows)}, any theta {positive_any}/{len(rows)}. Max best-theta Aeff: {max_aeff:.3f} nm².</p>
    <p>
      <a href="albumin_aeff_flow_aligned_by_sd_spacing.csv">Sample CSV</a> ·
      <a href="albumin_aeff_flow_aligned_summary.csv">Summary CSV</a>
    </p>
    <img src="albumin_aeff_flow_aligned_bar.png" alt="SD spacing Aeff bar chart">
    <img src="albumin_aeff_flow_aligned_scatter.png" alt="SD spacing Aeff scatter chart">
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute albumin centroid Aeff in flexible SD hexagons.")
    parser.add_argument(
        "--slit-width-xlsx",
        default="C:/Users/糖吉诃德/xwechat_files/wxid_5wtp6e154kya22_b9f0/msg/file/2026-07/Slit_width.xlsx",
    )
    parser.add_argument(
        "--hexagon-csv",
        default="protein_structure/selected_symmetric_hexagons_refined/SDsubunitCorrect_closed_angle_117_expanded_symmetric_hexagon.csv",
    )
    parser.add_argument("--output-dir", default="protein_structure/albumin_aeff_flow_aligned")
    parser.add_argument("--reference-length-nm", type=float, default=50.2439981515067)
    parser.add_argument("--ellipse-diameter-x-nm", type=float, default=5.45106753660192)
    parser.add_argument("--ellipse-diameter-y-nm", type=float, default=6.986832689834353)
    parser.add_argument("--theta-step-deg", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ellipse_rx = args.ellipse_diameter_x_nm / 2.0
    ellipse_ry = args.ellipse_diameter_y_nm / 2.0

    rows = compute_rows(
        Path(args.slit_width_xlsx),
        Path(args.hexagon_csv),
        args.reference_length_nm,
        ellipse_rx,
        ellipse_ry,
        args.theta_step_deg,
    )
    threshold_fixed = find_threshold_spacing(Path(args.hexagon_csv), args.reference_length_nm, ellipse_rx, ellipse_ry, args.theta_step_deg, "fixed")
    threshold_mean = find_threshold_spacing(Path(args.hexagon_csv), args.reference_length_nm, ellipse_rx, ellipse_ry, args.theta_step_deg, "mean")
    threshold_max = find_threshold_spacing(Path(args.hexagon_csv), args.reference_length_nm, ellipse_rx, ellipse_ry, args.theta_step_deg, "max")

    csv_path = output_dir / "albumin_aeff_flow_aligned_by_sd_spacing.csv"
    summary_path = output_dir / "albumin_aeff_flow_aligned_summary.csv"
    bar_path = output_dir / "albumin_aeff_flow_aligned_bar.png"
    scatter_path = output_dir / "albumin_aeff_flow_aligned_scatter.png"
    write_csv(csv_path, rows)
    write_summary(summary_path, rows, ellipse_rx, ellipse_ry, threshold_fixed, threshold_mean, threshold_max, args.theta_step_deg)
    plot_bar(bar_path, rows, threshold_fixed, threshold_max)
    plot_scatter(scatter_path, rows, threshold_max)
    index = write_index(output_dir, rows, threshold_fixed, threshold_mean, threshold_max)

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {bar_path}")
    print(f"Wrote {scatter_path}")
    print(f"Wrote {index}")
    print(f"Critical SD spacing for fixed Aeff>0: {threshold_fixed:.3f} nm")
    print(f"Critical SD spacing for mean Aeff>0: {threshold_mean:.3f} nm")
    print(f"Critical SD spacing for best Aeff>0: {threshold_max:.3f} nm")
    print(f"Aeff positive samples, fixed: {sum(row.passes_fixed for row in rows)}/{len(rows)}")
    print(f"Aeff positive samples, any theta: {sum(row.passes_any_theta for row in rows)}/{len(rows)}")
    print(f"Max best-theta Aeff: {max(row.aeff_max_nm2 for row in rows):.3f} nm^2")


if __name__ == "__main__":
    main()

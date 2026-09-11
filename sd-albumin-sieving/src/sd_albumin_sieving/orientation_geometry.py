from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from .albumin_aeff_from_sd_spacing import (
    clip_polygon_halfplane,
    erode_convex_polygon_by_ellipse,
    polygon_area,
    scan_aeff_theta,
)
from .fit_sd_spacing_distribution_and_integrate_aeff import distribution_object, fit_distributions, weighted_integral
from .flexible_hexagon_from_slit_width import read_sides, read_slit_widths, symmetric_hexagon_from_major


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data"
PROJECT_ROOT = HERE.parents[1]


def random_rotation_matrices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random SO(3) rotations using Shoemake quaternions."""
    u1 = rng.random(n)
    u2 = rng.random(n)
    u3 = rng.random(n)
    q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)

    x, y, z, w = q1, q2, q3, q4
    rotations = np.empty((n, 3, 3), dtype=float)
    rotations[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotations[:, 0, 1] = 2 * (x * y - z * w)
    rotations[:, 0, 2] = 2 * (x * z + y * w)
    rotations[:, 1, 0] = 2 * (x * y + z * w)
    rotations[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotations[:, 1, 2] = 2 * (y * z - x * w)
    rotations[:, 2, 0] = 2 * (x * z - y * w)
    rotations[:, 2, 1] = 2 * (y * z + x * w)
    rotations[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotations


def z_rotation(theta_rad: float) -> np.ndarray:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def principal_axis_rotation(axis_aligned_to_z: int) -> np.ndarray:
    """Return a local-to-world rotation with one local principal axis mapped to world z."""
    if axis_aligned_to_z == 0:
        # local major axis -> z; local middle/minor axes span the pore plane.
        return np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
    if axis_aligned_to_z == 1:
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
    if axis_aligned_to_z == 2:
        return np.eye(3, dtype=float)
    raise ValueError(axis_aligned_to_z)


def deterministic_orientation_grid(theta_step_deg: float) -> np.ndarray:
    rotations = []
    theta_values = np.deg2rad(np.arange(0.0, 180.0, theta_step_deg, dtype=float))
    for axis in range(3):
        base = principal_axis_rotation(axis)
        for theta in theta_values:
            rotations.append(z_rotation(float(theta)) @ base)
    return np.stack(rotations, axis=0)


def erode_convex_polygon_by_projected_ellipsoid(
    vertices_xy: np.ndarray,
    semi_axes_nm: np.ndarray,
    rotation: np.ndarray,
) -> list[np.ndarray]:
    vertices = np.asarray(vertices_xy, dtype=float)
    signed_area = 0.5 * np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1) - np.roll(vertices[:, 0], -1) * vertices[:, 1])
    if signed_area < 0:
        vertices = vertices[::-1]

    extent = max(float(np.abs(vertices).max()) + 2.0 * float(np.max(semi_axes_nm)) + 5.0, 20.0)
    center_region = [
        np.array([-extent, -extent], dtype=float),
        np.array([extent, -extent], dtype=float),
        np.array([extent, extent], dtype=float),
        np.array([-extent, extent], dtype=float),
    ]

    q_matrix = rotation @ np.diag(semi_axes_nm**2) @ rotation.T
    for idx, point in enumerate(vertices):
        next_point = vertices[(idx + 1) % len(vertices)]
        edge = next_point - point
        outward = np.array([edge[1], -edge[0]], dtype=float)
        outward /= float(np.linalg.norm(outward))
        normal_3d = np.array([outward[0], outward[1], 0.0], dtype=float)
        albumin_support = math.sqrt(max(0.0, float(normal_3d @ q_matrix @ normal_3d)))
        center_region = clip_polygon_halfplane(center_region, outward, float(outward @ point) - albumin_support)
        if len(center_region) < 3:
            return []
    return center_region


def aeff_for_rotations(vertices_xy: np.ndarray, semi_axes_nm: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    return np.array(
        [
            polygon_area(erode_convex_polygon_by_projected_ellipsoid(vertices_xy, semi_axes_nm, rotation))
            for rotation in rotations
        ],
        dtype=float,
    )


def water_area_for_spacing(
    spacing_nm: float,
    *,
    rectangle_width_nm: float,
    reference_height_nm: float,
    reference_protein_area_nm2: float,
) -> float:
    rectangle_area = rectangle_width_nm * spacing_nm
    scaled_protein_area = reference_protein_area_nm2 * spacing_nm / reference_height_nm
    return rectangle_area - scaled_protein_area


def compute_spacing_values(
    spacing_nm: float,
    *,
    slanted_side_nm: float,
    vertical_side_nm: float,
    current_hex_major_nm: float,
    reference_length_nm: float,
    ellipse_rx_nm: float,
    ellipse_ry_nm: float,
    semi_axes_nm: np.ndarray,
    deterministic_rotations: np.ndarray,
    random_rotations: np.ndarray,
    water_area_per_pore_nm2: float,
    theta_step_deg: float,
) -> dict[str, float]:
    try:
        major = current_hex_major_nm * spacing_nm / reference_length_nm
        vertices, _, _, _ = symmetric_hexagon_from_major(major, slanted_side_nm, vertical_side_nm)
    except ValueError:
        return {
            "spacing_nm": spacing_nm,
            "water_area_per_pore_nm2": water_area_per_pore_nm2,
            "aeff_end_on_mean_nm2": 0.0,
            "aeff_end_on_best_nm2": 0.0,
            "aeff_3d_best_nm2": 0.0,
            "aeff_3d_random_mean_nm2": 0.0,
            "p_random_positive": 0.0,
            "alpha_end_on_mean": 0.0,
            "alpha_end_on_best": 0.0,
            "alpha_3d_best": 0.0,
            "alpha_3d_random_mean": 0.0,
        }

    _, end_on_areas, _ = scan_aeff_theta(vertices, ellipse_rx_nm, ellipse_ry_nm, theta_step_deg)
    deterministic_areas = aeff_for_rotations(vertices, semi_axes_nm, deterministic_rotations)
    random_areas = aeff_for_rotations(vertices, semi_axes_nm, random_rotations)

    end_on_mean = float(np.mean(end_on_areas))
    end_on_best = float(np.max(end_on_areas))
    best_3d = float(max(np.max(deterministic_areas), np.max(random_areas)))
    random_mean = float(np.mean(random_areas))
    denom = max(water_area_per_pore_nm2, 1e-12)
    return {
        "spacing_nm": spacing_nm,
        "water_area_per_pore_nm2": water_area_per_pore_nm2,
        "aeff_end_on_mean_nm2": end_on_mean,
        "aeff_end_on_best_nm2": end_on_best,
        "aeff_3d_best_nm2": best_3d,
        "aeff_3d_random_mean_nm2": random_mean,
        "p_random_positive": float(np.mean(random_areas > 1e-9)),
        "alpha_end_on_mean": end_on_mean / denom,
        "alpha_end_on_best": end_on_best / denom,
        "alpha_3d_best": best_3d / denom,
        "alpha_3d_random_mean": random_mean / denom,
    }


def write_dict_rows(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_bounds(path: Path, grid_rows: list[dict[str, float]], spacings: np.ndarray, pdf: np.ndarray) -> None:
    grid = np.array([row["spacing_nm"] for row in grid_rows], dtype=float)
    alpha_random = np.array([row["alpha_3d_random_mean"] for row in grid_rows], dtype=float)
    alpha_mean = np.array([row["alpha_end_on_mean"] for row in grid_rows], dtype=float)
    alpha_best = np.array([row["alpha_3d_best"] for row in grid_rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax2 = ax1.twinx()
    ax1.fill_between(grid, alpha_random, alpha_best, color="#c7d9c4", alpha=0.45, label="3D orientation range")
    ax1.plot(grid, alpha_random, color="#6f9f72", linewidth=2.0, label="random 3D mean")
    ax1.plot(grid, alpha_mean, color="#4f8fc0", linewidth=2.0, label="current end-on in-plane mean")
    ax1.plot(grid, alpha_best, color="#1b6a41", linewidth=2.2, label="best 3D orientation")
    ax2.fill_between(grid, pdf, color="#9aa3aa", alpha=0.20, label="fitted SD width density")
    ax2.scatter(spacings, np.zeros_like(spacings), color="#333333", s=18, alpha=0.65, label="observed widths")
    ax1.set_xlabel("SD width (nm)")
    ax1.set_ylabel(r"$A_{\mathrm{eff},SD}/A_{\mathrm{SD}}$")
    ax2.set_ylabel("Fitted density")
    ax1.set_title("SD albumin-accessible area fraction under orientation assumptions")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8, loc="upper right")
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Estimate SD Aeff/A bounds from albumin orientation assumptions.")
    parser.add_argument(
        "--spacing-csv",
        "--slit-width-xlsx",
        dest="spacing_path",
        type=Path,
        default=DEFAULT_DATA_DIR / "sd_spacing_observations.csv",
        help="CSV with sample,tomo,sd_spacing_nm. The legacy Excel layout is also accepted.",
    )
    parser.add_argument("--hexagon-csv", type=Path, default=DEFAULT_DATA_DIR / "sd_reference_hexagon.csv")
    parser.add_argument("--water-area-summary", type=Path, default=DEFAULT_DATA_DIR / "sd_water_area_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sd_orientation_geometry")
    parser.add_argument("--reference-length-nm", type=float, default=50.2439981515067)
    parser.add_argument("--albumin-diameters-nm", type=float, nargs=3, default=[8.172944443691739, 6.986832689834353, 5.45106753660192])
    parser.add_argument("--theta-step-deg", type=float, default=2.0)
    parser.add_argument("--random-orientations", type=int, default=5000)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=101,
        help="Width-grid points. The manuscript calculation uses 101.",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    if args.random_orientations < 1 or args.grid_size < 2 or args.theta_step_deg <= 0:
        parser.error("random-orientations >= 1, grid-size >= 2, and theta-step-deg > 0 are required")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    water_summary = {}
    with Path(args.water_area_summary).open(newline="", encoding="utf-8-sig") as handle:
        for metric, value in csv.reader(handle):
            if metric == "metric":
                continue
            water_summary[metric] = float(value)

    rectangle_width_nm = water_summary["rectangle_width_nm"]
    reference_height_nm = water_summary["reference_projection_height_nm"]
    reference_protein_area_nm2 = water_summary["reference_protein_area_nm2"]

    slit_rows = read_slit_widths(Path(args.spacing_path))
    spacings = np.array([row[2] for row in slit_rows], dtype=float)
    fits = fit_distributions(spacings)
    best_fit = fits[0]
    distribution, params = distribution_object(best_fit)
    grid = np.linspace(max(0.01, float(spacings.min() - 5.0)), float(spacings.max() + 5.0), int(args.grid_size))
    pdf = distribution.pdf(grid, *params)

    slanted_side, vertical_side, current_hex_major, _ = read_sides(Path(args.hexagon_csv))
    diameters = np.array(args.albumin_diameters_nm, dtype=float)
    semi_axes = diameters / 2.0
    ellipse_rx_nm = diameters[2] / 2.0
    ellipse_ry_nm = diameters[1] / 2.0
    deterministic_rotations = deterministic_orientation_grid(float(args.theta_step_deg))
    rng = np.random.default_rng(int(args.seed))
    random_rotations = random_rotation_matrices(int(args.random_orientations), rng)

    common_kwargs = {
        "slanted_side_nm": slanted_side,
        "vertical_side_nm": vertical_side,
        "current_hex_major_nm": current_hex_major,
        "reference_length_nm": float(args.reference_length_nm),
        "ellipse_rx_nm": ellipse_rx_nm,
        "ellipse_ry_nm": ellipse_ry_nm,
        "semi_axes_nm": semi_axes,
        "deterministic_rotations": deterministic_rotations,
        "random_rotations": random_rotations,
        "theta_step_deg": float(args.theta_step_deg),
    }

    empirical_rows = []
    for spacing in spacings:
        water_area = water_area_for_spacing(
            float(spacing),
            rectangle_width_nm=rectangle_width_nm,
            reference_height_nm=reference_height_nm,
            reference_protein_area_nm2=reference_protein_area_nm2,
        ) / 2.0
        empirical_rows.append(compute_spacing_values(float(spacing), water_area_per_pore_nm2=water_area, **common_kwargs))

    grid_rows = []
    for spacing in grid:
        water_area = water_area_for_spacing(
            float(spacing),
            rectangle_width_nm=rectangle_width_nm,
            reference_height_nm=reference_height_nm,
            reference_protein_area_nm2=reference_protein_area_nm2,
        ) / 2.0
        grid_rows.append(compute_spacing_values(float(spacing), water_area_per_pore_nm2=water_area, **common_kwargs))

    value_keys = [
        "aeff_end_on_mean_nm2",
        "aeff_end_on_best_nm2",
        "aeff_3d_best_nm2",
        "aeff_3d_random_mean_nm2",
        "alpha_end_on_mean",
        "alpha_end_on_best",
        "alpha_3d_best",
        "alpha_3d_random_mean",
        "p_random_positive",
    ]
    summary_rows: list[dict[str, float | str]] = []
    for key in value_keys:
        empirical_values = np.array([row[key] for row in empirical_rows], dtype=float)
        grid_values = np.array([row[key] for row in grid_rows], dtype=float)
        summary_rows.append(
            {
                "metric": key,
                "empirical_width_mean": float(np.mean(empirical_values)),
                f"integrated_{best_fit.name}": weighted_integral(grid, grid_values, pdf),
            }
        )

    fixed_water_area_per_pore = water_summary["water_open_area_per_large_pore_nm2"]
    scenario_specs = [
        ("lower_random_3d", "aeff_3d_random_mean_nm2", "alpha_3d_random_mean"),
        ("current_end_on_in_plane_mean", "aeff_end_on_mean_nm2", "alpha_end_on_mean"),
        ("upper_best_3d", "aeff_3d_best_nm2", "alpha_3d_best"),
    ]
    alpha_summary_rows: list[dict[str, float | str]] = []
    for scenario, aeff_key, alpha_key in scenario_specs:
        empirical_aeff = np.array([row[aeff_key] for row in empirical_rows], dtype=float)
        grid_aeff = np.array([row[aeff_key] for row in grid_rows], dtype=float)
        empirical_scaled_alpha = np.array([row[alpha_key] for row in empirical_rows], dtype=float)
        grid_scaled_alpha = np.array([row[alpha_key] for row in grid_rows], dtype=float)
        alpha_summary_rows.append(
            {
                "scenario": scenario,
                "empirical_aeff_nm2": float(np.mean(empirical_aeff)),
                f"integrated_aeff_nm2_{best_fit.name}": weighted_integral(grid, grid_aeff, pdf),
                "empirical_alpha_fixed_A_SD": float(np.mean(empirical_aeff)) / fixed_water_area_per_pore,
                f"integrated_alpha_fixed_A_SD_{best_fit.name}": weighted_integral(grid, grid_aeff, pdf) / fixed_water_area_per_pore,
                "empirical_alpha_width_scaled_A_SD": float(np.mean(empirical_scaled_alpha)),
                f"integrated_alpha_width_scaled_A_SD_{best_fit.name}": weighted_integral(grid, grid_scaled_alpha, pdf),
            }
        )

    write_dict_rows(output_dir / "sd_orientation_bounds_by_observed_width.csv", empirical_rows)
    write_dict_rows(output_dir / "sd_orientation_bounds_by_width_grid.csv", grid_rows)
    write_summary(output_dir / "sd_orientation_bounds_summary.csv", summary_rows)
    write_summary(output_dir / "sd_orientation_alpha_summary.csv", alpha_summary_rows)
    plot_bounds(output_dir / "sd_orientation_bounds_alpha.png", grid_rows, spacings, pdf)
    metadata = {
        "spacing_observations": len(spacings),
        "width_grid_points": len(grid),
        "random_orientations_per_width": int(args.random_orientations),
        "deterministic_orientations": int(len(deterministic_rotations)),
        "theta_step_deg": float(args.theta_step_deg),
        "seed": int(args.seed),
        "best_width_distribution": best_fit.name,
        "elapsed_seconds": time.perf_counter() - started,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "pandas", "matplotlib"]
        },
        "upper_bound_scope": "maximum over the implemented deterministic and random sampled orientations, not a proof of the continuous global maximum",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Best SD width fit: {best_fit.name}")
    print(f"Observed widths: n={len(spacings)}, mean={np.mean(spacings):.4f} nm")
    print(f"Random orientations per width: {int(args.random_orientations)}")
    for row in summary_rows:
        print(f"{row['metric']}: empirical={row['empirical_width_mean']:.8g}, integrated={row[f'integrated_{best_fit.name}']:.8g}")
    print("Scenario alpha summary with fixed A_SD denominator:")
    for row in alpha_summary_rows:
        print(
            f"{row['scenario']}: "
            f"empirical_alpha={row['empirical_alpha_fixed_A_SD']:.8g}, "
            f"integrated_alpha={row[f'integrated_alpha_fixed_A_SD_{best_fit.name}']:.8g}"
        )
    print(f"Wrote {output_dir}")
    print(f"Elapsed seconds: {metadata['elapsed_seconds']:.3f}")


if __name__ == "__main__":
    main()

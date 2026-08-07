from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data"
PROJECT_ROOT = HERE.parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_ORIENTATION_GRID_CSV = DEFAULT_DATA_DIR / "sd_orientation_bounds_by_width_grid.csv"
DEFAULT_ORIENTATION_OBSERVED_CSV = DEFAULT_DATA_DIR / "sd_orientation_bounds_by_observed_width.csv"
DEFAULT_ORIENTATION_ALPHA_SUMMARY_CSV = DEFAULT_DATA_DIR / "sd_orientation_alpha_summary.csv"

REFERENCE_LENGTH_NM = 50.2439981515067
ALBUMIN_ELLIPSE_DIAMETER_X_NM = 5.45106753660192
ALBUMIN_ELLIPSE_DIAMETER_Y_NM = 6.986832689834353
THETA_STEP_DEG = 1.0
WATER_ACCESSIBLE_AREA_PER_COMPLEX_NM2 = 540.0
LARGE_PORE_COUNT_PER_COMPLEX = 2.0
QWATER_MEAN_M3_S = 7.8584e-20
WATER_ACCESSIBLE_AREA_FOR_VELOCITY_M2 = 5.40e-15
TRANSPORT_LENGTH_M = 1.0e-8
DIFFUSION_COEFFICIENT_M2_S = 9.2608e-11


def _read_spacing_rows(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "sample": int(raw["sample"]),
                    "tomo": str(raw["tomo"]),
                    "sd_spacing_nm": float(raw["sd_spacing_nm"]),
                }
            )
    if not rows:
        raise ValueError(f"No SD spacing observations found in {path}")
    return rows


def _read_float_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            parsed: dict[str, float] = {}
            for key, value in raw.items():
                if value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _read_reference_hexagon(path: Path) -> tuple[float, float, float]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6:
        raise ValueError(f"Expected six reference hexagon vertices in {path}")
    sides = np.array([float(row["side_to_next_nm"]) for row in rows], dtype=float)
    vertices = np.array([[float(row["x_nm"]), float(row["y_nm"])] for row in rows], dtype=float)
    slanted_side_nm = float(np.mean(sides[[0, 2, 3, 5]]))
    vertical_side_nm = float(np.mean(sides[[1, 4]]))
    current_major_nm = float(np.ptp(vertices[:, 1]))
    return slanted_side_nm, vertical_side_nm, current_major_nm


def symmetric_hexagon_from_major(
    major_nm: float,
    slanted_side_nm: float,
    vertical_side_nm: float,
) -> np.ndarray:
    half_vertical = vertical_side_nm / 2.0
    half_major = major_nm / 2.0
    delta = half_major - half_vertical
    if delta < -1e-9 or delta > slanted_side_nm + 1e-9:
        raise ValueError(f"Major axis {major_nm:.6g} nm is incompatible with the fixed side lengths")
    half_minor = math.sqrt(max(0.0, slanted_side_nm**2 - delta**2))
    return np.array(
        [
            [0.0, -half_major],
            [half_minor, -half_vertical],
            [half_minor, half_vertical],
            [0.0, half_major],
            [-half_minor, half_vertical],
            [-half_minor, -half_vertical],
        ],
        dtype=float,
    )


def _polygon_area(vertices: list[np.ndarray] | np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    points = np.asarray(vertices, dtype=float)
    return float(
        abs(
            0.5
            * np.sum(
                points[:, 0] * np.roll(points[:, 1], -1)
                - np.roll(points[:, 0], -1) * points[:, 1]
            )
        )
    )


def _clip_polygon_halfplane(
    polygon: list[np.ndarray], normal: np.ndarray, offset: float
) -> list[np.ndarray]:
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
                fraction = (offset - float(normal @ previous)) / denominator
                clipped.append(previous + fraction * direction)
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    return clipped


def accessible_centroid_area_nm2(
    vertices_xy: np.ndarray,
    ellipse_radius_x_nm: float,
    ellipse_radius_y_nm: float,
    theta_deg: float,
) -> float:
    vertices = np.asarray(vertices_xy, dtype=float)
    signed_area = 0.5 * np.sum(
        vertices[:, 0] * np.roll(vertices[:, 1], -1)
        - np.roll(vertices[:, 0], -1) * vertices[:, 1]
    )
    if signed_area < 0:
        vertices = vertices[::-1]
    extent = max(float(np.abs(vertices).max()) + ellipse_radius_x_nm + ellipse_radius_y_nm + 5.0, 20.0)
    center_region = [
        np.array([-extent, -extent]),
        np.array([extent, -extent]),
        np.array([extent, extent]),
        np.array([-extent, extent]),
    ]
    theta = math.radians(theta_deg)
    cosine, sine = math.cos(theta), math.sin(theta)
    for index, point in enumerate(vertices):
        edge = vertices[(index + 1) % len(vertices)] - point
        outward = np.array([edge[1], -edge[0]], dtype=float)
        outward /= float(np.linalg.norm(outward))
        local_x = cosine * outward[0] + sine * outward[1]
        local_y = -sine * outward[0] + cosine * outward[1]
        support = math.sqrt(
            (ellipse_radius_x_nm * local_x) ** 2
            + (ellipse_radius_y_nm * local_y) ** 2
        )
        center_region = _clip_polygon_halfplane(
            center_region, outward, float(outward @ point) - support
        )
        if len(center_region) < 3:
            return 0.0
    return _polygon_area(center_region)


def mean_accessible_area_for_spacing_nm2(
    spacing_nm: float,
    *,
    slanted_side_nm: float,
    vertical_side_nm: float,
    reference_major_nm: float,
    reference_length_nm: float,
    ellipse_radius_x_nm: float,
    ellipse_radius_y_nm: float,
    theta_step_deg: float,
) -> float:
    major_nm = reference_major_nm * float(spacing_nm) / reference_length_nm
    try:
        vertices = symmetric_hexagon_from_major(major_nm, slanted_side_nm, vertical_side_nm)
    except ValueError:
        return 0.0
    thetas = np.arange(0.0, 180.0, theta_step_deg, dtype=float)
    areas = [
        accessible_centroid_area_nm2(
            vertices, ellipse_radius_x_nm, ellipse_radius_y_nm, float(theta)
        )
        for theta in thetas
    ]
    return float(np.mean(areas)) if areas else 0.0


def _weighted_average(grid: np.ndarray, values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.trapezoid(values * weights, grid) / np.trapezoid(weights, grid))


def _sieving_from_aeff_per_large_pore(
    aeff_per_large_pore_nm2: np.ndarray | float,
    area_per_large_pore_nm2: float,
    peclet: float,
) -> np.ndarray:
    alpha = np.asarray(aeff_per_large_pore_nm2, dtype=float) / float(area_per_large_pore_nm2)
    return np.divide(alpha, alpha + float(peclet), out=np.zeros_like(alpha), where=(alpha + float(peclet)) > 0)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def calculate(
    spacing_csv: Path,
    hexagon_csv: Path,
    output_dir: Path,
    *,
    orientation_grid_csv: Path = DEFAULT_ORIENTATION_GRID_CSV,
    orientation_observed_csv: Path = DEFAULT_ORIENTATION_OBSERVED_CSV,
    orientation_alpha_summary_csv: Path = DEFAULT_ORIENTATION_ALPHA_SUMMARY_CSV,
    reference_length_nm: float = REFERENCE_LENGTH_NM,
    ellipse_diameter_x_nm: float = ALBUMIN_ELLIPSE_DIAMETER_X_NM,
    ellipse_diameter_y_nm: float = ALBUMIN_ELLIPSE_DIAMETER_Y_NM,
    theta_step_deg: float = THETA_STEP_DEG,
    water_area_per_complex_nm2: float = WATER_ACCESSIBLE_AREA_PER_COMPLEX_NM2,
    large_pore_count: float = LARGE_PORE_COUNT_PER_COMPLEX,
    qwater_mean_m3_s: float = QWATER_MEAN_M3_S,
    velocity_area_m2: float = WATER_ACCESSIBLE_AREA_FOR_VELOCITY_M2,
    transport_length_m: float = TRANSPORT_LENGTH_M,
    diffusion_coefficient_m2_s: float = DIFFUSION_COEFFICIENT_M2_S,
) -> dict[str, object]:
    observations = _read_spacing_rows(spacing_csv)
    spacings = np.array([float(row["sd_spacing_nm"]) for row in observations], dtype=float)
    fitted_mu_nm, fitted_sigma_nm = (float(value) for value in stats.norm.fit(spacings))
    # The manuscript calculation persisted fit parameters with 10 significant
    # digits before evaluating the curve. Preserve that published convention.
    normal_mu_nm = float(f"{fitted_mu_nm:.10g}")
    normal_sigma_nm = float(f"{fitted_sigma_nm:.10g}")
    _ = (hexagon_csv, reference_length_nm, ellipse_diameter_x_nm, ellipse_diameter_y_nm, theta_step_deg)

    orientation_rows = _read_float_rows(orientation_grid_csv)
    grid = np.array([row["spacing_nm"] for row in orientation_rows], dtype=float)
    lower_aeff = np.array([row["aeff_3d_random_mean_nm2"] for row in orientation_rows], dtype=float)
    upper_aeff = np.array([row["aeff_3d_best_nm2"] for row in orientation_rows], dtype=float)
    midpoint_aeff = 0.5 * (lower_aeff + upper_aeff)
    local_velocity_m_s = qwater_mean_m3_s / velocity_area_m2
    peclet = local_velocity_m_s * transport_length_m / diffusion_coefficient_m2_s
    area_per_large_pore_nm2 = water_area_per_complex_nm2 / large_pore_count
    lower_alpha = lower_aeff / area_per_large_pore_nm2
    upper_alpha = upper_aeff / area_per_large_pore_nm2
    midpoint_alpha = midpoint_aeff / area_per_large_pore_nm2
    lower_sieving = _sieving_from_aeff_per_large_pore(lower_aeff, area_per_large_pore_nm2, peclet)
    upper_sieving = _sieving_from_aeff_per_large_pore(upper_aeff, area_per_large_pore_nm2, peclet)
    midpoint_sieving = 0.5 * (lower_sieving + upper_sieving)
    density = stats.norm.pdf(grid, loc=normal_mu_nm, scale=normal_sigma_nm)

    curve_rows = [
        {
            "sd_spacing_nm": float(x),
            "aeff_lower_random_3d_per_large_pore_nm2": float(low_aeff),
            "aeff_upper_best_3d_per_large_pore_nm2": float(high_aeff),
            "aeff_midpoint_per_large_pore_nm2": float(mid_aeff),
            "alpha_lower_random_3d": float(low_alpha),
            "alpha_upper_best_3d": float(high_alpha),
            "alpha_midpoint": float(mid_alpha),
            "peclet": peclet,
            "sieving_lower_random_3d": float(low_s),
            "sieving_upper_best_3d": float(high_s),
            "sieving_midpoint": float(mid_s),
            "sieving_coefficient": float(mid_s),
            "normal_density": float(pdf),
        }
        for x, low_aeff, high_aeff, mid_aeff, low_alpha, high_alpha, mid_alpha, low_s, high_s, mid_s, pdf in zip(
            grid,
            lower_aeff,
            upper_aeff,
            midpoint_aeff,
            lower_alpha,
            upper_alpha,
            midpoint_alpha,
            lower_sieving,
            upper_sieving,
            midpoint_sieving,
            density,
        )
    ]

    orientation_observed_rows = _read_float_rows(orientation_observed_csv)
    observed_rows: list[dict[str, object]] = []
    for row, orientation_row in zip(observations, orientation_observed_rows):
        low_aeff = float(orientation_row["aeff_3d_random_mean_nm2"])
        high_aeff = float(orientation_row["aeff_3d_best_nm2"])
        mid_aeff = 0.5 * (low_aeff + high_aeff)
        low_s = float(_sieving_from_aeff_per_large_pore(low_aeff, area_per_large_pore_nm2, peclet))
        high_s = float(_sieving_from_aeff_per_large_pore(high_aeff, area_per_large_pore_nm2, peclet))
        mid_s = 0.5 * (low_s + high_s)
        observed_rows.append(
            {
                **row,
                "aeff_lower_random_3d_per_large_pore_nm2": low_aeff,
                "aeff_upper_best_3d_per_large_pore_nm2": high_aeff,
                "aeff_midpoint_per_large_pore_nm2": mid_aeff,
                "alpha_lower_random_3d": low_aeff / area_per_large_pore_nm2,
                "alpha_upper_best_3d": high_aeff / area_per_large_pore_nm2,
                "alpha_midpoint": mid_aeff / area_per_large_pore_nm2,
                "sieving_lower_random_3d": low_s,
                "sieving_upper_best_3d": high_s,
                "sieving_midpoint": mid_s,
                "sieving_coefficient": mid_s,
            }
        )

    integrated_lower_aeff = _weighted_average(grid, lower_aeff, density)
    integrated_upper_aeff = _weighted_average(grid, upper_aeff, density)
    for row in _read_csv_rows(orientation_alpha_summary_csv):
        scenario = row.get("scenario", "")
        if scenario == "lower_random_3d":
            integrated_lower_aeff = float(row["integrated_aeff_nm2_normal"])
        elif scenario == "upper_best_3d":
            integrated_upper_aeff = float(row["integrated_aeff_nm2_normal"])
    integrated_midpoint_aeff = 0.5 * (integrated_lower_aeff + integrated_upper_aeff)
    integrated_lower_alpha = integrated_lower_aeff / area_per_large_pore_nm2
    integrated_upper_alpha = integrated_upper_aeff / area_per_large_pore_nm2
    integrated_midpoint_alpha = integrated_midpoint_aeff / area_per_large_pore_nm2
    integrated_lower_sieving = float(
        _sieving_from_aeff_per_large_pore(integrated_lower_aeff, area_per_large_pore_nm2, peclet)
    )
    integrated_upper_sieving = float(
        _sieving_from_aeff_per_large_pore(integrated_upper_aeff, area_per_large_pore_nm2, peclet)
    )
    integrated_midpoint_sieving = 0.5 * (integrated_lower_sieving + integrated_upper_sieving)
    summary: dict[str, object] = {
        "model": "SD spacing-dependent flexible hexagon; orientation-bounded albumin accessibility; final sieving is the midpoint of lower and upper S",
        "orientation_method": "lower=random 3D albumin orientation, upper=best 3D albumin orientation at each SD width, final=arithmetic midpoint of lower and upper sieving coefficients",
        "normal_fit_mu_nm": normal_mu_nm,
        "normal_fit_sigma_nm": normal_sigma_nm,
        "normal_fit_parameter_significant_digits": 10,
        "source_orientation_grid_csv": orientation_grid_csv.name,
        "source_orientation_observed_csv": orientation_observed_csv.name,
        "source_orientation_alpha_summary_csv": orientation_alpha_summary_csv.name,
        "albumin_ellipse_diameter_x_nm": ellipse_diameter_x_nm,
        "albumin_ellipse_diameter_y_nm": ellipse_diameter_y_nm,
        "large_pore_count_per_complex": large_pore_count,
        "water_accessible_area_per_complex_nm2": water_area_per_complex_nm2,
        "water_accessible_area_per_large_pore_nm2": area_per_large_pore_nm2,
        "qwater_mean_m3_s": qwater_mean_m3_s,
        "water_accessible_area_for_velocity_m2": velocity_area_m2,
        "local_water_velocity_m_s": local_velocity_m_s,
        "transport_length_m": transport_length_m,
        "diffusion_coefficient_m2_s": diffusion_coefficient_m2_s,
        "peclet": peclet,
        "peclet_used_for_figure": peclet,
        "integrated_lower_aeff_per_large_pore_nm2": integrated_lower_aeff,
        "integrated_upper_aeff_per_large_pore_nm2": integrated_upper_aeff,
        "integrated_midpoint_aeff_per_large_pore_nm2": integrated_midpoint_aeff,
        "integrated_lower_alpha": integrated_lower_alpha,
        "integrated_upper_alpha": integrated_upper_alpha,
        "integrated_midpoint_alpha": integrated_midpoint_alpha,
        "integrated_lower_sieving": integrated_lower_sieving,
        "integrated_upper_sieving": integrated_upper_sieving,
        "integrated_midpoint_sieving": integrated_midpoint_sieving,
        "sieving_from_integrated_aeff_over_area": integrated_midpoint_sieving,
        "density_weighted_mean_sieving_coefficient": _weighted_average(grid, midpoint_sieving, density),
        "max_curve_sieving_coefficient": float(np.max(midpoint_sieving)),
        "max_curve_lower_sieving": float(np.max(lower_sieving)),
        "max_curve_upper_sieving": float(np.max(upper_sieving)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "sd_sieving_curve.csv", curve_rows)
    _write_csv(output_dir / "sd_observed_spacing_results.csv", observed_rows)
    (output_dir / "sd_sieving_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate slit-diaphragm albumin sieving from observed SD spacing."
    )
    parser.add_argument(
        "--spacing-csv", type=Path, default=DEFAULT_DATA_DIR / "sd_spacing_observations.csv"
    )
    parser.add_argument(
        "--hexagon-csv", type=Path, default=DEFAULT_DATA_DIR / "sd_reference_hexagon.csv"
    )
    parser.add_argument("--orientation-grid-csv", type=Path, default=DEFAULT_ORIENTATION_GRID_CSV)
    parser.add_argument("--orientation-observed-csv", type=Path, default=DEFAULT_ORIENTATION_OBSERVED_CSV)
    parser.add_argument(
        "--orientation-alpha-summary-csv",
        type=Path,
        default=DEFAULT_ORIENTATION_ALPHA_SUMMARY_CSV,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "sd_sieving")
    parser.add_argument("--theta-step-deg", type=float, default=THETA_STEP_DEG)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = calculate(
        args.spacing_csv,
        args.hexagon_csv,
        args.output_dir,
        orientation_grid_csv=args.orientation_grid_csv,
        orientation_observed_csv=args.orientation_observed_csv,
        orientation_alpha_summary_csv=args.orientation_alpha_summary_csv,
        theta_step_deg=float(args.theta_step_deg),
    )
    print(f"SD outputs: {args.output_dir.resolve()}")
    print(f"Normal fit: mu={summary['normal_fit_mu_nm']:.8f} nm, sigma={summary['normal_fit_sigma_nm']:.8f} nm")
    print(f"Pe={summary['peclet']:.12g}")
    print(f"Integrated sieving={summary['sieving_from_integrated_aeff_over_area']:.12g}")


if __name__ == "__main__":
    main()

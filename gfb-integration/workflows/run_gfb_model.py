"""Run manuscript GFB conditions through the conserved-flow serial model."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gfb_integration import SerialGFBParameters, solve_serial_gfb


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-csv", type=Path, default=PROJECT_ROOT / "data" / "manuscript_conditions.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "manuscript_conditions")
    args = parser.parse_args()

    with args.conditions_csv.open(newline="", encoding="utf-8-sig") as handle:
        conditions = list(csv.DictReader(handle))
    if not conditions:
        raise ValueError(f"No conditions found in {args.conditions_csv}")

    rows = []
    for condition in conditions:
        params = SerialGFBParameters(
            q_water_m3_s=float(condition["q_water_m3_s"]),
            albumin_diffusion_m2_s=float(condition["albumin_diffusion_m2_s"]),
            bulk1_length_nm=float(condition["bulk1_length_nm"]),
            bulk1_area_nm2=float(condition["bulk1_area_nm2"]),
            bulk2_length_nm=float(condition["bulk2_length_nm"]),
            bulk2_area_nm2=float(condition["bulk2_area_nm2"]),
            s_gbm=float(condition["s_gbm"]),
            s_sd=float(condition["s_sd"]),
        )
        result = solve_serial_gfb(params)
        legacy_product = float(condition["s_gbm"]) * float(condition["figure_bulk1_factor"]) * float(condition["figure_bulk2_factor"]) * float(condition["figure_sd_factor"])
        rows.append(
            {
                "condition": condition["condition"],
                "sample_type": condition["sample_type"],
                "exclusion_radius_nm": float(condition["exclusion_radius_nm"]),
                **result,
                "figure6_fixed_factor_product": legacy_product,
                "coupled_minus_fixed_product": result["gfb_sieving"] - legacy_product,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "gfb_serial_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    validation = {
        "n_conditions": len(rows),
        "max_abs_equation_residual": max(row["max_abs_equation_residual"] for row in rows),
        "shared_q_water_columns_equal": all(
            row["q_water_bulk1_m3_s"] == row["q_water_gbm_m3_s"] == row["q_water_bulk2_m3_s"] == row["q_water_sd_m3_s"]
            for row in rows
        ),
        "shared_q_albumin_columns_equal": all(
            row["q_albumin_bulk1_at_c0_1_m3_s"] == row["q_albumin_gbm_at_c0_1_m3_s"] == row["q_albumin_bulk2_at_c0_1_m3_s"] == row["q_albumin_sd_at_c0_1_m3_s"]
            for row in rows
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scope": "Checks the serial model equations and identical total water/albumin flow columns. GBM and SD coefficients are effective layer inputs, not independently recalculated here.",
    }
    validation["all_passed"] = bool(
        validation["max_abs_equation_residual"] < 1e-12
        and validation["shared_q_water_columns_equal"]
        and validation["shared_q_albumin_columns_equal"]
    )
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print("condition sample_type exclusion_radius_nm gfb_sieving plasma_to_gbm gbm_to_sd sd")
    for row in rows:
        print(
            row["condition"], row["sample_type"], f'{row["exclusion_radius_nm"]:.2f}',
            f'{row["gfb_sieving"]:.10f}', f'{row["plasma_to_gbm_transfer"]:.7f}',
            f'{row["gbm_to_sd_transfer"]:.7f}', f'{row["sd_transfer"]:.7f}'
        )
    print(json.dumps(validation, indent=2))
    if not validation["all_passed"]:
        raise SystemExit("GFB conservation validation failed")


if __name__ == "__main__":
    main()

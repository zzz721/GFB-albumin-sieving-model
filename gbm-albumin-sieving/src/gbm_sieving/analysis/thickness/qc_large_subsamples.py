from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AXIS_NAMES = ("X Coord", "Y Coord", "Z Coord")


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_thickness_summary(path: Path) -> dict:
    out = {
        "average_thickness_nm": np.nan,
        "thickness_along_axis_nm": np.nan,
        "theta_deg_vs_xy": np.nan,
        "used_tilt_refine": np.nan,
        "nx": np.nan,
        "ny": np.nan,
        "nz": np.nan,
    }
    if not path.exists():
        return out
    df = pd.read_excel(path)
    mapping = {
        "Average Thickness (nm)": "average_thickness_nm",
        "Thickness_along_axis_nm": "thickness_along_axis_nm",
        "Theta_deg_vs_XY": "theta_deg_vs_xy",
        "Used_tilt_refine": "used_tilt_refine",
        "nx": "nx",
        "ny": "ny",
        "nz": "nz",
    }
    for col, key in mapping.items():
        if col in df.columns and len(df):
            out[key] = pd.to_numeric(df[col].iloc[0], errors="coerce")
    return out


def _read_subsample_thickness_stats(root: Path, sample: str) -> dict:
    path = root / "results_core_subsamples" / sample / f"{sample}_sub_samples_summary.xlsx"
    out = {
        "subsample_thickness_n": 0,
        "subsample_thickness_mean_nm": np.nan,
        "subsample_thickness_median_nm": np.nan,
        "subsample_thickness_min_nm": np.nan,
        "subsample_thickness_max_nm": np.nan,
    }
    if not path.exists():
        return out
    df = pd.read_excel(path)
    col = "Thickness (nm)" if "Thickness (nm)" in df.columns else "Average Thickness (nm)" if "Average Thickness (nm)" in df.columns else None
    if col is None:
        col = "Thickness_along_axis_nm" if "Thickness_along_axis_nm" in df.columns else None
    if col is None:
        return out
    vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(float)
    if len(vals) == 0:
        return out
    out.update(
        {
            "subsample_thickness_n": int(len(vals)),
            "subsample_thickness_mean_nm": float(np.mean(vals)),
            "subsample_thickness_median_nm": float(np.median(vals)),
            "subsample_thickness_min_nm": float(np.min(vals)),
            "subsample_thickness_max_nm": float(np.max(vals)),
        }
    )
    return out


def _surface_axis(class_df: pd.DataFrame) -> tuple[int, float]:
    left = class_df[class_df["Is Surface X Left"].astype(bool)]
    right = class_df[class_df["Is Surface X Right"].astype(bool)]
    diffs = []
    for axis, col in enumerate(AXIS_NAMES):
        diffs.append(abs(float(right[col].median()) - float(left[col].median())))
    axis = int(np.nanargmax(diffs))
    return axis, float(diffs[axis])


def _plane_axes_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, ref))) > 0.92:
        ref = np.array([0.0, 1.0, 0.0])
    v = np.cross(normal, ref)
    v = v / np.linalg.norm(v)
    w = np.cross(normal, v)
    w = w / np.linalg.norm(w)
    return v, w


def _local_window_thickness(
    class_df: pd.DataFrame,
    *,
    permeation_normal: np.ndarray | None,
    label_prefix: str = "",
    bins_v: int,
    bins_w: int,
    min_surface_points: int,
    percentile_clip: float,
) -> tuple[pd.DataFrame, dict]:
    class_df = class_df.copy()
    for col in AXIS_NAMES:
        class_df[col] = pd.to_numeric(class_df[col], errors="coerce")
    class_df = class_df.dropna(subset=list(AXIS_NAMES))

    left = class_df[class_df["Is Surface X Left"].astype(bool)].copy()
    right = class_df[class_df["Is Surface X Right"].astype(bool)].copy()
    if len(left) == 0 or len(right) == 0:
        return pd.DataFrame(), {"status": "missing_left_or_right_surface", "n_left": len(left), "n_right": len(right)}

    normal = None
    if permeation_normal is not None:
        normal = np.asarray(permeation_normal, dtype=float)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) <= 0:
            normal = None

    if normal is not None:
        normal = normal / np.linalg.norm(normal)
        v_basis, w_basis = _plane_axes_from_normal(normal)
        pts = class_df.loc[:, AXIS_NAMES].to_numpy(float)
        class_df["_u_proj"] = pts @ normal
        class_df["_v_proj"] = pts @ v_basis
        class_df["_w_proj"] = pts @ w_basis
        left = class_df[class_df["Is Surface X Left"].astype(bool)].copy()
        right = class_df[class_df["Is Surface X Right"].astype(bool)].copy()
        u_col, v_col, w_col = "_u_proj", "_v_proj", "_w_proj"
        global_surface_gap = abs(float(right[u_col].median()) - float(left[u_col].median()))
        surface_axis_label = "permeation_normal_projection"
        cross_axis_v_label = "normal_plane_axis_1"
        cross_axis_w_label = "normal_plane_axis_2"
    else:
        u_axis, global_surface_gap = _surface_axis(class_df)
        v_axis, w_axis = [i for i in range(3) if i != u_axis]
        u_col, v_col, w_col = AXIS_NAMES[u_axis], AXIS_NAMES[v_axis], AXIS_NAMES[w_axis]
        surface_axis_label = u_col
        cross_axis_v_label = v_col
        cross_axis_w_label = w_col

    surf = pd.concat([left, right], ignore_index=True)
    lo_q = float(percentile_clip)
    hi_q = 100.0 - lo_q
    v0, v1 = np.nanpercentile(surf[v_col].to_numpy(float), [lo_q, hi_q])
    w0, w1 = np.nanpercentile(surf[w_col].to_numpy(float), [lo_q, hi_q])
    if not np.isfinite(v0) or not np.isfinite(v1) or not np.isfinite(w0) or not np.isfinite(w1) or v1 <= v0 or w1 <= w0:
        return pd.DataFrame(), {"status": "invalid_surface_extent", "n_left": len(left), "n_right": len(right)}

    v_edges = np.linspace(v0, v1, int(bins_v) + 1)
    w_edges = np.linspace(w0, w1, int(bins_w) + 1)
    rows: list[dict] = []
    for i in range(int(bins_v)):
        for j in range(int(bins_w)):
            vl, vh = v_edges[i], v_edges[i + 1]
            wl, wh = w_edges[j], w_edges[j + 1]
            lmask = (left[v_col] >= vl) & (left[v_col] < vh) & (left[w_col] >= wl) & (left[w_col] < wh)
            rmask = (right[v_col] >= vl) & (right[v_col] < vh) & (right[w_col] >= wl) & (right[w_col] < wh)
            lwin = left[lmask]
            rwin = right[rmask]
            if len(lwin) < min_surface_points or len(rwin) < min_surface_points:
                continue
            left_u = float(lwin[u_col].median())
            right_u = float(rwin[u_col].median())
            rows.append(
                {
                    "window_v": i,
                    "window_w": j,
                    "v_min": vl,
                    "v_max": vh,
                    "w_min": wl,
                    "w_max": wh,
                    "n_left": len(lwin),
                    "n_right": len(rwin),
                    "left_u_median": left_u,
                    "right_u_median": right_u,
                    "local_thickness_nm": abs(right_u - left_u),
                }
            )
    win = pd.DataFrame(rows)
    vals = pd.to_numeric(win.get("local_thickness_nm", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
    vals = vals[np.isfinite(vals)]
    stats = {
        "status": "ok" if len(vals) else "insufficient_matched_windows",
        "surface_axis": surface_axis_label,
        "cross_axis_v": cross_axis_v_label,
        "cross_axis_w": cross_axis_w_label,
        "n_left": int(len(left)),
        "n_right": int(len(right)),
        "n_local_windows": int(len(vals)),
        "global_surface_gap_nm": float(global_surface_gap),
        "local_median_thickness_nm": float(np.median(vals)) if len(vals) else np.nan,
        "local_mean_thickness_nm": float(np.mean(vals)) if len(vals) else np.nan,
        "local_min_thickness_nm": float(np.min(vals)) if len(vals) else np.nan,
        "local_max_thickness_nm": float(np.max(vals)) if len(vals) else np.nan,
        "local_iqr_thickness_nm": float(np.percentile(vals, 75) - np.percentile(vals, 25)) if len(vals) else np.nan,
        "local_cv": float(np.std(vals, ddof=1) / np.mean(vals)) if len(vals) > 1 and np.mean(vals) else np.nan,
    }
    if label_prefix:
        prefixed = {"status": stats["status"]}
        for key, value in stats.items():
            if key == "status":
                continue
            prefixed[f"{label_prefix}{key}"] = value
        stats = prefixed
    return win, stats


def _plot_summary(summary: pd.DataFrame, out_png: Path) -> None:
    ok = summary[summary["status"] == "ok"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), facecolor="white")
    colors = np.where(ok["type"].astype(str).str.upper() == "AS", "#D9854A", "#4C93BA")

    axes[0].scatter(ok["thickness_along_axis_nm"], ok["local_median_thickness_nm"], c=colors, s=28, alpha=0.82)
    lim = np.nanmax([ok["thickness_along_axis_nm"].max(), ok["local_median_thickness_nm"].max()])
    if np.isfinite(lim):
        axes[0].plot([0, lim], [0, lim], color="#9AA3A8", lw=0.9, ls="--")
    axes[0].set_xlabel("Along-axis thickness (nm)")
    axes[0].set_ylabel("Local median thickness (nm)")
    axes[0].set_title("Local vs global gap")

    axes[1].scatter(ok["theta_deg_vs_xy"], ok["local_cv"], c=colors, s=28, alpha=0.82)
    axes[1].axvline(45, color="#53616A", lw=0.9, ls="--")
    axes[1].set_xlabel("Global tilt angle (deg)")
    axes[1].set_ylabel("Local thickness CV")
    axes[1].set_title("Curvature / heterogeneity screen")

    have_sub = np.isfinite(ok["subsample_thickness_median_nm"])
    axes[2].scatter(
        ok.loc[have_sub, "subsample_thickness_median_nm"],
        ok.loc[have_sub, "local_median_thickness_nm"],
        c=colors[have_sub],
        s=28,
        alpha=0.82,
    )
    lim2 = np.nanmax(
        [
            ok.loc[have_sub, "subsample_thickness_median_nm"].max(),
            ok.loc[have_sub, "local_median_thickness_nm"].max(),
        ]
    )
    if np.isfinite(lim2):
        axes[2].plot([0, lim2], [0, lim2], color="#9AA3A8", lw=0.9, ls="--")
    axes[2].set_xlabel("Small-subsample median thickness (nm)")
    axes[2].set_ylabel("Local median thickness (nm)")
    axes[2].set_title("Local vs subsample thickness")

    for ax in axes:
        ax.grid(True, color="#E9EEF2", linewidth=0.65)
        for spine in ax.spines.values():
            spine.set_color("#C9D2D8")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent local thickness QC for large GBM subsamples.")
    parser.add_argument("--samples", nargs="*", default=None, help="Optional sample names, e.g. AS323 WT195.")
    parser.add_argument("--out-dir", default="figures_for_article/local_thickness_qc_large_subsamples")
    parser.add_argument("--bins-v", type=int, default=4)
    parser.add_argument("--bins-w", type=int, default=4)
    parser.add_argument("--min-surface-points", type=int, default=3)
    parser.add_argument("--percentile-clip", type=float, default=2.0)
    args = parser.parse_args()

    root = _workspace_root()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results_root = root / "results_core_large_subsamples"
    sample_dirs = sorted(p for p in results_root.iterdir() if p.is_dir())
    if args.samples:
        wanted = {s.upper() for s in args.samples}
        sample_dirs = [p for p in sample_dirs if p.name.upper() in wanted]

    summary_rows: list[dict] = []
    for sample_dir in sample_dirs:
        sample = sample_dir.name
        sample_type = "AS" if sample.upper().startswith("AS") else "WT" if sample.upper().startswith("WT") else "Other"
        sub = f"{sample}_largebox"
        sub_dir = sample_dir / sub
        class_path = sub_dir / f"{sub}_pore_classification.xlsx"
        thick_path = sub_dir / f"{sub}_thickness_summary.xlsx"
        if not class_path.exists() or not thick_path.exists():
            continue
        class_df = pd.read_excel(class_path)
        thickness = _read_thickness_summary(thick_path)
        normal_vals = np.array([thickness.get("nx", np.nan), thickness.get("ny", np.nan), thickness.get("nz", np.nan)], dtype=float)
        win_axis, local_axis_stats = _local_window_thickness(
            class_df,
            permeation_normal=None,
            label_prefix="axis_",
            bins_v=int(args.bins_v),
            bins_w=int(args.bins_w),
            min_surface_points=int(args.min_surface_points),
            percentile_clip=float(args.percentile_clip),
        )
        win_normal, local_normal_stats = _local_window_thickness(
            class_df,
            permeation_normal=normal_vals,
            label_prefix="normal_",
            bins_v=int(args.bins_v),
            bins_w=int(args.bins_w),
            min_surface_points=int(args.min_surface_points),
            percentile_clip=float(args.percentile_clip),
        )
        local_status = local_normal_stats.get("status", "missing_normal")
        if local_status != "ok":
            local_status = local_axis_stats.get("status", local_status)
        subsample_thickness = _read_subsample_thickness_stats(root, sample)
        row = {
            "sample": sample,
            "subsample": sub,
            "type": sample_type,
            **thickness,
            **subsample_thickness,
            "status": local_status,
            **local_axis_stats,
            **local_normal_stats,
        }
        row["local_median_thickness_nm"] = row.get("axis_local_median_thickness_nm", np.nan)
        row["local_mean_thickness_nm"] = row.get("axis_local_mean_thickness_nm", np.nan)
        row["local_cv"] = row.get("axis_local_cv", np.nan)
        row["n_local_windows"] = row.get("axis_n_local_windows", 0)
        row["surface_axis"] = row.get("axis_surface_axis", "")
        summary_rows.append(row)
        if not win_axis.empty:
            win_axis.insert(0, "mode", "axis_inferred")
            win_axis.insert(0, "subsample", sub)
            win_axis.insert(0, "sample", sample)
            win_axis.to_csv(out_dir / f"{sub}_local_axis_windows.csv", index=False)
        if not win_normal.empty:
            win_normal.insert(0, "mode", "normal_projection")
            win_normal.insert(0, "subsample", sub)
            win_normal.insert(0, "sample", sample)
            win_normal.to_csv(out_dir / f"{sub}_local_normal_windows.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "local_thickness_qc_summary.csv", index=False)
    if not summary.empty:
        _plot_summary(summary, out_dir / "local_thickness_qc_summary.png")
        meta = {
            "method": "Use existing left/right surface pore labels; project coordinates onto the saved permeation normal (nx, ny, nz); split the normal-plane cross-section into local windows and compute the median right-left projected gap in each window.",
            "window_width": "The 2-98 percentile clipped surface extent is divided into bins_v x bins_w windows; each window width is therefore sample-specific.",
            "surface_axis": "Prefer the saved permeation-normal projection. If a valid normal is unavailable, fall back to the original coordinate axis with the largest median separation between left and right surface pores.",
            "bins_v": int(args.bins_v),
            "bins_w": int(args.bins_w),
            "min_surface_points": int(args.min_surface_points),
            "percentile_clip": float(args.percentile_clip),
        }
        (out_dir / "local_thickness_qc_method.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[local-thickness-qc] saved: {out_dir / 'local_thickness_qc_summary.csv'}")
    print(f"[local-thickness-qc] saved: {out_dir / 'local_thickness_qc_summary.png'}")


if __name__ == "__main__":
    main()

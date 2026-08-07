"""
阶段 1.5：参数 vs 厚度 可视化检查（WT / AS 分开）

- 读取所有子样本的 summary（与 phase2 相同）
- 绘制：厚度 vs 各类结构参数 的散点图（WT、AS 分开）
- 可选：孔半径五分位「组内喉」占比（口径 A）+ 标签置换零假设（--pore-quintile-n-perm）
- 仅做可视化和探索性检验（Mann-Whitney、置换 p 等），不做分布拟合 / KS 检验
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import linregress, mannwhitneyu, pearsonr, spearmanr  # noqa: E402

# 本模块出图：子图标题与 x/y 轴标签统一加粗（总标题 suptitle 在各自函数中已设 fontweight="bold"）
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

from gbm_sieving.analysis.parameter_fitting.fit_pore_throat_parameters import (
    DENSITY_FRAC_BIN_WIDTH_NM,
    get_thickness_bin_edges_density_frac,
)
from gbm_sieving.analysis.structure.thickness_relationships import (
    collect_all_sub_samples_with_normal,
)
from gbm_sieving.analysis.structure.compare_subsamples import (
    find_sub_samples,
)
from gbm_sieving.analysis.structure import statistics as st
from gbm_sieving.paths import OUTPUT_ROOT, WORKSPACE_ROOT


def collect_summary(split_samples_dir: Path, analysis_dir: Path) -> pd.DataFrame:
    """
    复用 thickness_vs_pore_radius 的 collect_all_sub_samples_with_normal，
    得到每个子样本一行的 summary DataFrame。
    """
    rows = collect_all_sub_samples_with_normal(
        split_samples_dir, analysis_dir, min_throat_radius=None
    )
    df = pd.DataFrame(rows)
    expected_cols = [
        "sample_name",
        "sub_name",
        "thickness",
        "is_wt",
        "mean_pore_radius",
        "mean_throat_radius",
        "mean_deg",
        "frac_pore",
        "frac_throat",
        "rho_pore",
        "rho_throat",
        "mean_S",
        "angle_Qmax_normal_3d_deg",
        "angle_Qmax_normal_xy_deg",
        "cos_Qmax_normal",
    ]
    # 确保缺失列存在（填 NaN），避免后续 KeyError
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def collect_throat_radius_length(
    df: pd.DataFrame, split_samples_dir: Path
) -> dict:
    """
    收集所有子样本中每条喉的 (喉半径, 孔心距 d)，按 WT/AS 分组。
    返回 {"WT": (radii, d), "AS": (radii, d)}，每个为 np.ndarray。
    """
    split_samples_dir = Path(split_samples_dir)
    out = {"WT": [], "AS": []}
    for _, row in df.iterrows():
        sample_dir = split_samples_dir / row["sample_name"]
        sub_list = find_sub_samples(sample_dir, row["sample_name"])
        for sn, pf, tf in sub_list:
            if sn != row["sub_name"] or not pf or not pf.exists() or not tf or not tf.exists():
                continue
            try:
                (
                    pore_coords,
                    pore_radii,
                    _,
                    pore_id_to_idx,
                    throat_pore1,
                    throat_pore2,
                    throat_radii,
                ) = st.load_pores_and_throats(pf, tf)
                for i in range(len(throat_pore1)):
                    p1_id, p2_id = int(throat_pore1[i]), int(throat_pore2[i])
                    if p1_id in pore_id_to_idx and p2_id in pore_id_to_idx:
                        idx1 = pore_id_to_idx[p1_id]
                        idx2 = pore_id_to_idx[p2_id]
                        d = np.linalg.norm(pore_coords[idx1] - pore_coords[idx2])
                        key = "WT" if row["is_wt"] else "AS"
                        out[key].append((float(throat_radii[i]), float(d)))
            except Exception:
                pass
            break
    result = {}
    for k in ("WT", "AS"):
        if out[k]:
            arr = np.array(out[k])
            result[k] = (arr[:, 0], arr[:, 1])
        else:
            result[k] = (np.array([]), np.array([]))
    return result


def debug_plot_raw_throat_length_histogram(
    df: pd.DataFrame, split_samples_dir: Path, out_dir: Path
) -> None:
    """
    调试用：直接从原始几何计算喉长度 L_raw = |p1-p2| - r_p1 - r_p2，统计是否存在 L_raw <= 0。
    仅做简单直方图和计数输出，不参与后续拟合。
    """
    split_samples_dir = Path(split_samples_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lengths_wt = []
    lengths_as = []
    for _, row in df.iterrows():
        sample_dir = split_samples_dir / row["sample_name"]
        sub_list = find_sub_samples(sample_dir, row["sample_name"])
        for sn, pf, tf in sub_list:
            if sn != row["sub_name"] or not pf or not pf.exists() or not tf or not tf.exists():
                continue
            try:
                (
                    pore_coords,
                    pore_radii,
                    _,
                    pore_id_to_idx,
                    throat_pore1,
                    throat_pore2,
                    _,
                ) = st.load_pores_and_throats(pf, tf)
                for i in range(len(throat_pore1)):
                    p1_id, p2_id = int(throat_pore1[i]), int(throat_pore2[i])
                    if p1_id in pore_id_to_idx and p2_id in pore_id_to_idx:
                        idx1 = pore_id_to_idx[p1_id]
                        idx2 = pore_id_to_idx[p2_id]
                        d = np.linalg.norm(pore_coords[idx1] - pore_coords[idx2])
                        r_p1 = float(pore_radii[idx1])
                        r_p2 = float(pore_radii[idx2])
                        L_raw = d - r_p1 - r_p2
                        if row["is_wt"]:
                            lengths_wt.append(L_raw)
                        else:
                            lengths_as.append(L_raw)
            except Exception:
                pass
            break

    for name, arr in [("WT", np.array(lengths_wt, dtype=float)), ("AS", np.array(lengths_as, dtype=float))]:
        if arr.size == 0:
            continue
        n_le_zero = int(np.sum(arr <= 0))
        print(f"[phase1_5 debug] {name}: 总喉数 = {arr.size}, 其中 L_raw <= 0 的数量 = {n_le_zero}")
        fig, ax = plt.subplots(figsize=(5, 4))
        valid = arr[np.isfinite(arr)]
        bins = min(50, max(10, valid.size // 100)) if valid.size > 0 else 10
        ax.hist(valid, bins=bins, density=False, alpha=0.7, edgecolor="white")
        ax.set_xlabel("L_raw = |p1-p2| - r_p1 - r_p2 (nm)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} Raw throat length distribution (no truncation)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = out_dir / f"debug_raw_throat_length_hist_{name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[phase1_5 debug] 已保存: {plot_path}")


def debug_plot_negative_L_raw_radius_scatter(
    df: pd.DataFrame, split_samples_dir: Path, out_dir: Path
) -> None:
    """
    调试用：针对 L_raw < 0 的喉，绘制
    小端孔半径 / 大端孔半径 vs 喉半径 的散点图，WT / AS 各一组：
    - 小孔半径 r_small = min(r1, r2)
    - 大孔半径 r_large = max(r1, r2)
    """
    split_samples_dir = Path(split_samples_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {"WT": [], "AS": []}  # 每项: (r_throat, r_small, r_large)
    for _, row in df.iterrows():
        sample_dir = split_samples_dir / row["sample_name"]
        sub_list = find_sub_samples(sample_dir, row["sample_name"])
        for sn, pf, tf in sub_list:
            if sn != row["sub_name"] or not pf or not pf.exists() or not tf or not tf.exists():
                continue
            try:
                (
                    pore_coords,
                    pore_radii,
                    _,
                    pore_id_to_idx,
                    throat_pore1,
                    throat_pore2,
                    throat_radii,
                ) = st.load_pores_and_throats(pf, tf)
                for i in range(len(throat_pore1)):
                    p1_id, p2_id = int(throat_pore1[i]), int(throat_pore2[i])
                    if p1_id in pore_id_to_idx and p2_id in pore_id_to_idx:
                        idx1 = pore_id_to_idx[p1_id]
                        idx2 = pore_id_to_idx[p2_id]
                        d = np.linalg.norm(pore_coords[idx1] - pore_coords[idx2])
                        r1 = float(pore_radii[idx1])
                        r2 = float(pore_radii[idx2])
                        L_raw = d - r1 - r2
                        if L_raw < 0:
                            r_t = float(throat_radii[i])
                            r_small = min(r1, r2)
                            r_large = max(r1, r2)
                            key = "WT" if row["is_wt"] else "AS"
                            data[key].append((r_t, r_small, r_large))
            except Exception:
                pass
            break

    for key, color in [("WT", "tab:blue"), ("AS", "tab:orange")]:
        if not data[key]:
            continue
        arr = np.array(data[key], dtype=float)
        r_t = arr[:, 0]
        r_small = arr[:, 1]
        r_large = arr[:, 2]
        n = len(r_t)
        print(f"[phase1_5 debug] {key}: L_raw < 0 的喉数 = {n}")

        # 小孔半径 vs 喉半径
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(r_small, r_t, alpha=0.3, s=5, color="tab:green", edgecolor="none")
        x_min, x_max = float(np.min(r_small)), float(np.max(r_small))
        if x_max <= x_min:
            x_max = x_min + 1.0
        x_line = np.linspace(x_min, x_max, 100)
        ax.plot(x_line, x_line, "k--", linewidth=1, label="y = x")
        ax.set_xlabel("Small pore radius min(r1, r2) (nm)")
        ax.set_ylabel("Throat radius r_throat (nm)")
        ax.set_title(f"{key}: When L_raw < 0, small pore radius vs throat radius\n(n={n})")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = out_dir / f"debug_Lraw_negative_rthroat_vs_small_pore_{key}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[phase1_5 debug] 已保存: {plot_path}")

        # 大孔半径 vs 喉半径
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(r_large, r_t, alpha=0.3, s=5, color="tab:red", edgecolor="none")
        x_min, x_max = float(np.min(r_large)), float(np.max(r_large))
        if x_max <= x_min:
            x_max = x_min + 1.0
        x_line = np.linspace(x_min, x_max, 100)
        ax.plot(x_line, x_line, "k--", linewidth=1, label="y = x")
        ax.set_xlabel("Large pore radius max(r1, r2) (nm)")
        ax.set_ylabel("Throat radius r_throat (nm)")
        ax.set_title(f"{key}: When L_raw < 0, large pore radius vs throat radius\n(n={n})")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = out_dir / f"debug_Lraw_negative_rthroat_vs_large_pore_{key}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[phase1_5 debug] 已保存: {plot_path}")


def collect_pore_radius_vs_mean_throat_radius(
    df: pd.DataFrame, split_samples_dir: Path
) -> dict:
    """
    收集所有子样本中「每个孔的半径 vs 该孔上喉的平均半径」数据，按 WT/AS 分组。
    返回 {"WT": (r_pore_all, r_throat_mean_all), "AS": (...)}。
    仅统计至少有 1 条喉连接的孔。
    """
    split_samples_dir = Path(split_samples_dir)
    out = {"WT": [], "AS": []}
    for _, row in df.iterrows():
        sample_dir = split_samples_dir / row["sample_name"]
        sub_list = find_sub_samples(sample_dir, row["sample_name"])
        for sn, pf, tf in sub_list:
            if sn != row["sub_name"] or not pf or not pf.exists() or not tf or not tf.exists():
                continue
            try:
                (
                    pore_coords,
                    pore_radii,
                    pore_ids,
                    pore_id_to_idx,
                    throat_pore1,
                    throat_pore2,
                    throat_radii,
                ) = st.load_pores_and_throats(pf, tf)
                # 为每个孔收集其上喉半径
                pore_throat_r = [[] for _ in range(len(pore_ids))]
                for i in range(len(throat_pore1)):
                    pid1 = int(throat_pore1[i])
                    pid2 = int(throat_pore2[i])
                    r_t = float(throat_radii[i])
                    if pid1 in pore_id_to_idx:
                        pore_throat_r[pore_id_to_idx[pid1]].append(r_t)
                    if pid2 in pore_id_to_idx and pid2 != pid1:
                        pore_throat_r[pore_id_to_idx[pid2]].append(r_t)
                key = "WT" if row["is_wt"] else "AS"
                for pi, r_p in enumerate(pore_radii):
                    lst = pore_throat_r[pi]
                    if not lst:
                        continue
                    r_mean_t = float(np.mean(lst))
                    out[key].append((float(r_p), r_mean_t))
            except Exception:
                pass
            # 每个 summary 行对应唯一子样本，找到后即可跳出
            break

    result = {}
    for k in ("WT", "AS"):
        if out[k]:
            arr = np.array(out[k])
            result[k] = (arr[:, 0], arr[:, 1])
        else:
            result[k] = (np.array([]), np.array([]))
    return result


def collect_throat_radius_vs_mean_pore_radius(
    df: pd.DataFrame, split_samples_dir: Path
) -> dict:
    """
    收集所有子样本中每条喉的 (喉半径, 两端孔平均半径)，按 WT/AS 分组。
    返回 {"WT": (throat_radii, mean_pore_radii), "AS": (...)}。
    """
    split_samples_dir = Path(split_samples_dir)
    out = {"WT": [], "AS": []}
    for _, row in df.iterrows():
        sample_dir = split_samples_dir / row["sample_name"]
        sub_list = find_sub_samples(sample_dir, row["sample_name"])
        for sn, pf, tf in sub_list:
            if sn != row["sub_name"] or not pf or not pf.exists() or not tf or not tf.exists():
                continue
            try:
                (
                    pore_coords,
                    pore_radii,
                    pore_ids,
                    pore_id_to_idx,
                    throat_pore1,
                    throat_pore2,
                    throat_radii,
                ) = st.load_pores_and_throats(pf, tf)
                for i in range(len(throat_pore1)):
                    p1_id, p2_id = int(throat_pore1[i]), int(throat_pore2[i])
                    if p1_id not in pore_id_to_idx or p2_id not in pore_id_to_idx:
                        continue
                    idx1 = pore_id_to_idx[p1_id]
                    idx2 = pore_id_to_idx[p2_id]
                    r_p1 = float(pore_radii[idx1])
                    r_p2 = float(pore_radii[idx2])
                    r_th = float(throat_radii[i])
                    mean_pore = 0.5 * (r_p1 + r_p2)
                    key = "WT" if row["is_wt"] else "AS"
                    out[key].append((r_th, mean_pore))
            except Exception:
                pass
            break
    result = {}
    for k in ("WT", "AS"):
        if out[k]:
            arr = np.array(out[k])
            result[k] = (arr[:, 0], arr[:, 1])  # throat_radius, mean_pore_radius
        else:
            result[k] = (np.array([]), np.array([]))
    return result


def collect_pore_degree_vs_radius(
    df: pd.DataFrame, split_samples_dir: Path
) -> dict:
    """
    收集每个孔的 (度数, 半径)，按 WT/AS 分组。
    返回 {"WT": (degrees, radii), "AS": (...)}。
    仅统计至少有 1 条喉连接的孔。
    """
    split_samples_dir = Path(split_samples_dir)
    out = {"WT": [], "AS": []}
    for _, row in df.iterrows():
        sample_dir = split_samples_dir / row["sample_name"]
        sub_list = find_sub_samples(sample_dir, row["sample_name"])
        for sn, pf, tf in sub_list:
            if sn != row["sub_name"] or not pf or not pf.exists() or not tf or not tf.exists():
                continue
            try:
                (
                    pore_coords,
                    pore_radii,
                    pore_ids,
                    pore_id_to_idx,
                    throat_pore1,
                    throat_pore2,
                    _,
                ) = st.load_pores_and_throats(pf, tf)
                deg = np.zeros(len(pore_ids), dtype=int)
                for i in range(len(throat_pore1)):
                    pid1 = int(throat_pore1[i])
                    pid2 = int(throat_pore2[i])
                    if pid1 in pore_id_to_idx:
                        deg[pore_id_to_idx[pid1]] += 1
                    if pid2 in pore_id_to_idx and pid2 != pid1:
                        deg[pore_id_to_idx[pid2]] += 1
                key = "WT" if row["is_wt"] else "AS"
                for pi in range(len(pore_ids)):
                    if deg[pi] > 0:
                        out[key].append((int(deg[pi]), float(pore_radii[pi])))
            except Exception:
                pass
            break
    result = {}
    for k in ("WT", "AS"):
        if out[k]:
            arr = np.array(out[k])
            result[k] = (arr[:, 0], arr[:, 1])  # degrees, radii
        else:
            result[k] = (np.array([]), np.array([]))
    return result


def collect_connected_vs_random_radius_diff(
    df: pd.DataFrame, split_samples_dir: Path, n_random_per_sample: int = 5000
) -> dict:
    """
    收集：实际连接孔对的半径差 |r1-r2|，以及同子样本内随机孔对的半径差。
    返回 {"WT": (connected_diffs, random_diffs), "AS": (...)}。
    """
    split_samples_dir = Path(split_samples_dir)
    out_conn = {"WT": [], "AS": []}
    out_rand = {"WT": [], "AS": []}
    rng = np.random.default_rng(42)
    for _, row in df.iterrows():
        sample_dir = split_samples_dir / row["sample_name"]
        sub_list = find_sub_samples(sample_dir, row["sample_name"])
        for sn, pf, tf in sub_list:
            if sn != row["sub_name"] or not pf or not pf.exists() or not tf or not tf.exists():
                continue
            try:
                (
                    pore_coords,
                    pore_radii,
                    pore_ids,
                    pore_id_to_idx,
                    throat_pore1,
                    throat_pore2,
                    _,
                ) = st.load_pores_and_throats(pf, tf)
                n_pores = len(pore_radii)
                if n_pores < 2:
                    continue
                key = "WT" if row["is_wt"] else "AS"
                # 实际连接：每条喉两端孔的半径差
                for i in range(len(throat_pore1)):
                    p1_id, p2_id = int(throat_pore1[i]), int(throat_pore2[i])
                    if p1_id in pore_id_to_idx and p2_id in pore_id_to_idx:
                        r1 = float(pore_radii[pore_id_to_idx[p1_id]])
                        r2 = float(pore_radii[pore_id_to_idx[p2_id]])
                        out_conn[key].append(abs(r1 - r2))
                # 随机配对：同子样本内随机选孔对
                n_rand = min(n_random_per_sample, n_pores * (n_pores - 1) // 2)
                for _ in range(n_rand):
                    i, j = rng.choice(n_pores, size=2, replace=False)
                    out_rand[key].append(abs(float(pore_radii[i]) - float(pore_radii[j])))
            except Exception:
                pass
            break
    result = {}
    for k in ("WT", "AS"):
        result[k] = (
            np.array(out_conn[k], dtype=float) if out_conn[k] else np.array([]),
            np.array(out_rand[k], dtype=float) if out_rand[k] else np.array([]),
        )
    return result


def add_mean_throat_length(df: pd.DataFrame, split_samples_dir: Path) -> pd.DataFrame:
    """
    为每个子样本计算均值喉长度（这里喉长度指孔心距 d = |p1-p2| 的均值），新增列 mean_throat_length。
    仅用于几何统计，与筛过物理模型中使用的有效喉长区分。
    """
    split_samples_dir = Path(split_samples_dir)
    out = []
    for _, row in df.iterrows():
        sample_dir = split_samples_dir / row["sample_name"]
        sub_list = find_sub_samples(sample_dir, row["sample_name"])
        mean_len = np.nan
        for sn, pf, tf in sub_list:
            if sn != row["sub_name"] or not pf or not pf.exists() or not tf or not tf.exists():
                continue
            try:
                pore_coords, _, _, pore_id_to_idx, throat_pore1, throat_pore2, _ = (
                    st.load_pores_and_throats(pf, tf)
                )
                lengths = []
                for i in range(len(throat_pore1)):
                    p1_id, p2_id = int(throat_pore1[i]), int(throat_pore2[i])
                    if p1_id in pore_id_to_idx and p2_id in pore_id_to_idx:
                        idx1 = pore_id_to_idx[p1_id]
                        idx2 = pore_id_to_idx[p2_id]
                        d = np.linalg.norm(pore_coords[idx1] - pore_coords[idx2])
                        lengths.append(d)
                if lengths:
                    mean_len = float(np.mean(lengths))
            except Exception:
                pass
            break
        out.append(mean_len)
    df = df.copy()
    df["mean_throat_length"] = out
    return df


def _scatter_with_fit(
    ax,
    thickness: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: str,
    show_correlation: bool = True,
):
    """
    散点 + 线性拟合；图例格式与 AS 分段一致：仅用虚线图例「分组 (n=…, R^2=…)」，散点不进图例。
    Pearson/Spearman（可选）单独左上角小框，避免与右下角图例挤在一起。
    """
    ax.scatter(
        thickness,
        values,
        alpha=0.7,
        color=color,
        edgecolor="white",
        s=45,
        label="_nolegend_",
    )
    corr_lines: list[str] = []
    n = int(len(thickness))
    if show_correlation and n >= 3:
        try:
            r_pearson, p_pearson = pearsonr(thickness, values)
            r_spearman, p_spearman = spearmanr(thickness, values)
            corr_lines.append(f"Pearson r={r_pearson:.3f}, p={p_pearson:.3f}")
            corr_lines.append(f"Spearman r={r_spearman:.3f}, p={p_spearman:.3f}")
        except Exception:
            pass

    if n >= 2:
        try:
            slope, intercept, r_value, p_value, std_err = linregress(thickness, values)
            x_line = np.linspace(thickness.min(), thickness.max(), 100)
            y_line = slope * x_line + intercept
            r2 = float(r_value**2)
            # 与 AS 分段图例一致：mathtext + 等号两侧空格 + R^{2} 上标
            leg = rf"$\mathrm{{{label}}}\ (n = {n},\ R^{{2}} = {r2:.3f})$"
            ax.plot(
                x_line,
                y_line,
                color=color,
                linestyle="--",
                linewidth=2,
                label=leg,
            )
        except Exception:
            ax.plot(
                [],
                [],
                color=color,
                linestyle="--",
                label=rf"$\mathrm{{{label}}}\ (n = {n})$",
            )
    else:
        ax.plot([], [], color=color, linestyle="--", label=rf"$\mathrm{{{label}}}\ (n = {n})$")

    if corr_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(corr_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )


def _scatter_with_line_only(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    label: str,
    color: str,
    legend_mathtext_inner: str | None = None,
) -> None:
    """
    仅画散点和线性拟合曲线；散点不进图例，样本数与 R^2 写在虚线图例（与 WT 单段一致）。

    legend_mathtext_inner:
        若给定，图例使用 mathtext（$$ 包裹），用于 AS 分段使「<」与「≥」与正文同字号、同字体；
        应为不含外层 $ 的公式片段，例如 r"\\mathrm{AS\\ thickness} < 150\\ \\mathrm{nm}"。
    """
    ax.scatter(x, y, alpha=0.7, color=color, edgecolor="white", s=45, label="_nolegend_")
    n = len(x)

    def _legend_fit(r2: float | None) -> str:
        if legend_mathtext_inner is not None:
            if r2 is not None:
                return rf"${legend_mathtext_inner}\ (n = {n},\ R^{{2}} = {r2:.3f})$"
            return rf"${legend_mathtext_inner}\ (n = {n})$"
        if r2 is not None:
            return f"{label} (n = {n}, R^2 = {r2:.3f})"
        return f"{label} (n = {n})"

    if n >= 2:
        try:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                color=color,
                linestyle="--",
                linewidth=2,
                label=_legend_fit(float(r_value**2)),
            )
            return
        except Exception:
            pass
    ax.plot([], [], color=color, linestyle="--", label=_legend_fit(None))


# 厚度 vs 参数（散点与 density heatmap）共用 Y 轴标签；英文便于单独导出 WT/AS 插图。
_THICKNESS_VS_PARAM_YLABELS_EN: dict[str, str] = {
    "mean_pore_radius": "Mean pore radius (nm)",
    "mean_throat_radius": "Mean throat radius (nm)",
    "mean_throat_length": "Mean throat length (nm)",
    "rho_pore": "Pore number density (1/nm³)",
    "rho_throat": "Throat number density (1/nm³)",
    "frac_pore": "Pore volume fraction",
    "frac_throat": "Throat volume fraction",
    "mean_S": "Global order parameter S",
    "angle_Qmax_normal_3d_deg": (
        "Angle between Q principal axis and permeability direction (3D) (°)"
    ),
    "angle_Qmax_normal_xy_deg": (
        "Angle between Q principal axis XY projection and permeability direction (°)"
    ),
    "cos_Qmax_normal": "cos(throat principal orientation vs permeability direction)",
}


def _plot_one_param_vs_thickness_axis(
    ax,
    df: pd.DataFrame,
    col: str,
    ylabel: str,
    *,
    is_wt_flag: bool,
    name: str,
    color: str,
) -> None:
    """单个子图：厚度 vs 某一参数（WT 单段拟合 或 AS 按 150 nm 两段拟合）。"""
    sub = df[df["is_wt"] == is_wt_flag].copy()
    sub = sub[["thickness", col]].dropna()
    if len(sub) == 0:
        ax.set_title(name, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        return
    x = sub["thickness"].to_numpy(dtype=float)
    y = sub[col].to_numpy(dtype=float)

    if is_wt_flag:
        # WT：整体作为一组拟合；不显示 Pearson/Spearman 角标框（图例中已有 n 与 R^{2}）。
        _scatter_with_fit(
            ax,
            x,
            y,
            label=name,
            color=color,
            show_correlation=False,
        )
        ax.set_title(name, fontsize=12)
    else:
        # AS：按厚度 150 nm 分界，在同一张图上分两组拟合
        threshold = 150.0
        mask_low = x < threshold
        mask_high = x >= threshold
        if mask_low.sum() > 0:
            _scatter_with_line_only(
                ax,
                x[mask_low],
                y[mask_low],
                label="",
                color="tab:orange",
                legend_mathtext_inner=(
                    rf"\mathrm{{AS\ thickness}} < {threshold:.0f}\ \mathrm{{nm}}"
                ),
            )
        if mask_high.sum() > 0:
            _scatter_with_line_only(
                ax,
                x[mask_high],
                y[mask_high],
                label="",
                color="tab:red",
                legend_mathtext_inner=(
                    rf"\mathrm{{AS\ thickness}} \geq {threshold:.0f}\ \mathrm{{nm}}"
                ),
            )
        ax.axvline(threshold, color="gray", linestyle=":", linewidth=1)
        ax.set_title(name, fontsize=12)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)


def plot_param_vs_thickness(df: pd.DataFrame, out_dir: Path) -> None:
    """
    对若干关键参数画 厚度 vs 参数 的散点图，WT / AS 分开。
    每组指标保存：合并双面板图 thickness_vs_{col}.png，以及单面板
    thickness_vs_{col}_WT.png / thickness_vs_{col}_AS.png。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [(True, "WT", "tab:blue"), (False, "AS", "tab:orange")]

    for col, ylabel in _THICKNESS_VS_PARAM_YLABELS_EN.items():
        if col not in df.columns:
            continue
        suptitle = f"Thickness vs {ylabel}"

        fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")

        for ax, (is_wt_flag, name, color) in zip(axes, panels):
            _plot_one_param_vs_thickness_axis(
                ax, df, col, ylabel, is_wt_flag=is_wt_flag, name=name, color=color
            )

        axes[-1].set_xlabel("Thickness (nm)", fontsize=11)
        for ax in axes:
            if ax.has_data():
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

        plt.tight_layout()
        plot_path = out_dir / f"thickness_vs_{col}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"已保存: {plot_path}")

        # 每个子图单独成文件（便于插图只用 WT 或只用 AS）
        for is_wt_flag, name, color in panels:
            fig1, ax1 = plt.subplots(1, 1, figsize=(7, 4.5))
            fig1.suptitle(suptitle, fontsize=14, fontweight="bold")
            _plot_one_param_vs_thickness_axis(
                ax1, df, col, ylabel, is_wt_flag=is_wt_flag, name=name, color=color
            )
            ax1.set_xlabel("Thickness (nm)", fontsize=11)
            if ax1.has_data():
                h, lb = ax1.get_legend_handles_labels()
                if h:
                    ax1.legend(loc="lower right", fontsize=9, framealpha=0.95)
            plt.tight_layout()
            split_path = out_dir / f"thickness_vs_{col}_{name}.png"
            plt.savefig(split_path, dpi=300, bbox_inches="tight")
            plt.close(fig1)
            print(f"已保存: {split_path}")


def plot_throat_radius_vs_length(
    data: dict, out_dir: Path
) -> dict:
    """
    喉半径 vs 喉长度 散点图 + 线性拟合，WT/AS 各一图。
    返回 linear_fits 供 main 统一写入 phase1_5_linear_fits.json。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    fig.suptitle("Throat radius vs throat length", fontsize=14, fontweight="bold")

    # 收集线性拟合参数，供 Phase3 使用
    linear_fits: dict[str, dict] = {}

    for ax, (name, color) in zip(axes, [("WT", "tab:blue"), ("AS", "tab:orange")]):
        radii, lengths = data.get(name, (np.array([]), np.array([])))
        if len(radii) == 0:
            ax.set_title(f"{name}（无数据）")
            ax.set_ylabel("Throat length (nm)", fontsize=11)
            ax.grid(True, alpha=0.3)
            continue
        ax.scatter(radii, lengths, alpha=0.3, color=color, s=8, edgecolors="none")
        text_lines = []
        if len(radii) >= 3:
            try:
                r_pearson, p_pearson = pearsonr(radii, lengths)
                r_spearman, p_spearman = spearmanr(radii, lengths)
                text_lines.append(f"Pearson r={r_pearson:.3f}, p={p_pearson:.3e}")
                text_lines.append(f"Spearman r={r_spearman:.3f}, p={p_spearman:.3e}")
            except Exception:
                r_pearson = p_pearson = r_spearman = p_spearman = None  # type: ignore[assignment]
            try:
                slope, intercept, r_value, p_value, std_err = linregress(radii, lengths)
                x_line = np.linspace(radii.min(), radii.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=2)
                text_lines.append(f"Linear: L={slope:.3f}*r+{intercept:.2f}, R^2={r_value**2:.3f}")
                linear_fits[name] = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "pearson_r": float(r_pearson) if 'r_pearson' in locals() and r_pearson is not None else None,
                    "spearman_r": float(r_spearman) if 'r_spearman' in locals() and r_spearman is not None else None,
                    "r_squared": float(r_value ** 2),
                }
            except Exception:
                pass
        if text_lines:
            ax.text(
                0.02,
                0.98,
                "\n".join(text_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
        ax.set_ylabel("Throat length (nm)", fontsize=11)
        ax.set_title(f"{name} (n={len(radii)} throats)", fontsize=12)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Throat radius (nm)", fontsize=11)
    plt.tight_layout()
    plot_path = out_dir / "throat_radius_vs_length.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存: {plot_path}")
    return linear_fits


def plot_throat_radius_vs_mean_pore_radius(
    data: dict, out_dir: Path
) -> dict:
    """
    喉半径 vs 两端孔平均半径 散点图 + 线性拟合，WT/AS 各一子图。
    返回 linear_fits 供 main 统一写入 phase1_5_linear_fits.json。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    fig.suptitle("Throat radius vs mean pore radius (per throat)", fontsize=14, fontweight="bold")

    # 收集线性拟合参数，供 Phase3 使用
    linear_fits: dict[str, dict] = {}

    for ax, (name, color) in zip(axes, [("WT", "tab:blue"), ("AS", "tab:orange")]):
        throat_r, mean_pore_r = data.get(name, (np.array([]), np.array([])))
        if len(throat_r) == 0:
            ax.set_title(f"{name} (no data)")
            ax.set_ylabel("Throat radius (nm)", fontsize=11)
            ax.grid(True, alpha=0.3)
            continue
        x = mean_pore_r
        y = throat_r
        ax.scatter(x, y, alpha=0.3, color=color, s=8, edgecolors="none")
        text_lines = []
        if len(x) >= 3:
            try:
                r_pearson, p_pearson = pearsonr(x, y)
                r_spearman, p_spearman = spearmanr(x, y)
                text_lines.append(f"Pearson r={r_pearson:.3f}, p={p_pearson:.3e}")
                text_lines.append(f"Spearman r={r_spearman:.3f}, p={p_spearman:.3e}")
            except Exception:
                r_pearson = p_pearson = r_spearman = p_spearman = None  # type: ignore[assignment]
            try:
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=2)
                text_lines.append(f"Linear: r_throat={slope:.3f}*r_pore_mean+{intercept:.2f}, R^2={r_value**2:.3f}")
                linear_fits[name] = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "pearson_r": float(r_pearson) if 'r_pearson' in locals() and r_pearson is not None else None,
                    "spearman_r": float(r_spearman) if 'r_spearman' in locals() and r_spearman is not None else None,
                    "r_squared": float(r_value ** 2),
                }
            except Exception:
                pass
        if text_lines:
            ax.text(
                0.02,
                0.98,
                "\n".join(text_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
        ax.set_ylabel("Throat radius (nm)", fontsize=11)
        ax.set_title(f"{name} (n={len(throat_r)} throats)", fontsize=12)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Mean pore radius (nm)", fontsize=11)
    plt.tight_layout()
    plot_path = out_dir / "throat_radius_vs_mean_pore_radius.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存: {plot_path}")
    return linear_fits


def plot_pore_degree_vs_radius(data: dict, out_dir: Path) -> dict:
    """
    孔度数 vs 孔半径 散点图 + 线性拟合，WT/AS 各一子图。
    返回 linear_fits 供 main 统一写入 phase1_5_linear_fits.json。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    fig.suptitle("Pore degree vs pore radius", fontsize=14, fontweight="bold")

    linear_fits: dict[str, dict] = {}
    for ax, (name, color) in zip(axes, [("WT", "tab:blue"), ("AS", "tab:orange")]):
        deg, radii = data.get(name, (np.array([]), np.array([])))
        if len(deg) == 0:
            ax.set_title(f"{name} (no data)")
            ax.set_ylabel("Pore radius (nm)", fontsize=11)
            ax.grid(True, alpha=0.3)
            continue
        ax.scatter(deg, radii, alpha=0.3, color=color, s=8, edgecolors="none")
        text_lines = []
        r_pearson = r_spearman = None
        if len(deg) >= 3:
            try:
                r_pearson, p_pearson = pearsonr(deg, radii)
                r_spearman, p_spearman = spearmanr(deg, radii)
                text_lines.append(f"Pearson r={r_pearson:.3f}, p={p_pearson:.3e}")
                text_lines.append(f"Spearman r={r_spearman:.3f}, p={p_spearman:.3e}")
            except Exception:
                pass
            try:
                slope, intercept, r_value, p_value, std_err = linregress(deg, radii)
                x_line = np.linspace(deg.min(), deg.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=2)
                text_lines.append(f"Linear: r={slope:.3f}*deg+{intercept:.2f}, R^2={r_value**2:.3f}")
                linear_fits[name] = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "pearson_r": float(r_pearson) if r_pearson is not None else None,
                    "spearman_r": float(r_spearman) if r_spearman is not None else None,
                    "r_squared": float(r_value ** 2),
                }
            except Exception:
                pass
        if text_lines:
            ax.text(
                0.02, 0.98, "\n".join(text_lines),
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
        ax.set_ylabel("Pore radius (nm)", fontsize=11)
        ax.set_title(f"{name} (n={len(deg)} pores)", fontsize=12)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Pore degree", fontsize=11)
    plt.tight_layout()
    plot_path = out_dir / "pore_degree_vs_radius.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")
    return linear_fits


def plot_connected_vs_random_radius_diff(data: dict, out_dir: Path) -> None:
    """
    实际连接孔对 vs 随机孔对的半径差分布，纵轴为概率，WT/AS 各一子图。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    fig.suptitle("Connected vs random pore pairs: |r1-r2| distribution", fontsize=14, fontweight="bold")

    for ax, (name, color) in zip(axes, [("WT", "tab:blue"), ("AS", "tab:orange")]):
        conn_diffs, rand_diffs = data.get(name, (np.array([]), np.array([])))
        if len(conn_diffs) == 0 and len(rand_diffs) == 0:
            ax.set_title(f"{name} (no data)")
            ax.set_ylabel("Probability", fontsize=11)
            ax.grid(True, alpha=0.3)
            continue
        # 共用 bins，便于比较
        all_diffs = np.concatenate([conn_diffs, rand_diffs])
        if len(all_diffs) < 2:
            ax.set_title(f"{name} (insufficient data)")
            ax.set_ylabel("Probability", fontsize=11)
            ax.grid(True, alpha=0.3)
            continue
        x_max = max(float(np.max(all_diffs)) * 1.05, 1.0)
        bins = np.linspace(0, min(x_max, 25), 40)
        if len(conn_diffs) > 0:
            ax.hist(
                conn_diffs, bins=bins, density=True, alpha=0.6, color=color,
                label=f"Connected (n={len(conn_diffs)})", edgecolor="white"
            )
        if len(rand_diffs) > 0:
            ax.hist(
                rand_diffs, bins=bins, density=True, alpha=0.4, color="gray",
                label=f"Random (n={len(rand_diffs)})", edgecolor="white"
            )
        ax.set_ylabel("Probability", fontsize=11)
        ax.set_title(f"{name}", fontsize=12)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("|r1 - r2| (nm)", fontsize=11)
    plt.tight_layout()
    plot_path = out_dir / "connected_vs_random_radius_diff.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")


def plot_mean_throat_radius_vs_length(df: pd.DataFrame, out_dir: Path) -> None:
    """
    平均喉半径 vs 平均喉长度 散点图（每子样本一点）+ 线性拟合，WT/AS 各一。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    fig.suptitle("Mean throat radius vs mean throat length (per subsample)", fontsize=14, fontweight="bold")
    for ax, (is_wt_flag, name, color) in zip(
        axes,
        [(True, "WT", "tab:blue"), (False, "AS", "tab:orange")],
    ):
        sub = df[df["is_wt"] == is_wt_flag][["mean_throat_radius", "mean_throat_length"]].dropna()
        if len(sub) == 0:
            ax.set_title(f"{name}（无数据）")
            ax.set_ylabel("Mean throat length (nm)", fontsize=11)
            ax.grid(True, alpha=0.3)
            continue
        x = sub["mean_throat_radius"].to_numpy(dtype=float)
        y = sub["mean_throat_length"].to_numpy(dtype=float)
        ax.scatter(x, y, alpha=0.7, color=color, edgecolor="white", s=45)
        text_lines = []
        if len(x) >= 3:
            try:
                r_pearson, p_pearson = pearsonr(x, y)
                text_lines.append(f"Pearson r={r_pearson:.3f}, p={p_pearson:.3e}")
            except Exception:
                pass
            try:
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=2)
                text_lines.append(f"Linear: L={slope:.3f}*r+{intercept:.2f}, R^2={r_value**2:.3f}")
            except Exception:
                pass
        if text_lines:
            ax.text(
                0.02, 0.98, "\n".join(text_lines),
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
        ax.set_ylabel("Mean throat length (nm)", fontsize=11)
        ax.set_title(f"{name} (n={len(sub)} subsamples)", fontsize=12)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Mean throat radius (nm)", fontsize=11)
    plt.tight_layout()
    plot_path = out_dir / "mean_throat_radius_vs_mean_length.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存: {plot_path}")


def plot_pore_radius_vs_mean_throat_radius(
    data: dict, out_dir: Path
) -> None:
    """
    每个孔一点评：孔半径 vs 该孔上喉的平均半径，WT/AS 各一图。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    fig.suptitle("Pore radius vs mean throat radius (per pore)", fontsize=14, fontweight="bold")

    for ax, (name, color) in zip(axes, [("WT", "tab:blue"), ("AS", "tab:orange")]):
        r_pore, r_th_mean = data.get(name, (np.array([]), np.array([])))
        if len(r_pore) == 0:
            ax.set_title(f"{name} (no data)")
            ax.set_ylabel("Mean throat radius (nm)", fontsize=11)
            ax.grid(True, alpha=0.3)
            continue
        ax.scatter(
            r_pore,
            r_th_mean,
            alpha=0.3,
            color=color,
            s=8,
            edgecolors="none",
        )
        text_lines = []
        if len(r_pore) >= 3:
            try:
                r_pearson, p_pearson = pearsonr(r_pore, r_th_mean)
                r_spearman, p_spearman = spearmanr(r_pore, r_th_mean)
                text_lines.append(f"Pearson r={r_pearson:.3f}, p={p_pearson:.3e}")
                text_lines.append(f"Spearman r={r_spearman:.3f}, p={p_spearman:.3e}")
            except Exception:
                pass
            try:
                slope, intercept, r_value, p_value, std_err = linregress(r_pore, r_th_mean)
                x_line = np.linspace(r_pore.min(), r_pore.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=2)
                text_lines.append(
                    f"Linear: r_throat={slope:.3f}*r_pore+{intercept:.2f}, R^2={r_value**2:.3f}"
                )
            except Exception:
                pass
        if text_lines:
            ax.text(
                0.02,
                0.98,
                "\n".join(text_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
        ax.set_ylabel("Mean throat radius (nm)", fontsize=11)
        ax.set_title(f"{name} (n={len(r_pore)} pores)", fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Pore radius (nm)", fontsize=11)
    plt.tight_layout()
    plot_path = out_dir / "pore_radius_vs_mean_throat_radius.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存: {plot_path}")

def plot_mean_pore_radius_vs_mean_throat_radius(df: pd.DataFrame, out_dir: Path) -> None:
    """
    平均孔半径 vs 平均喉半径 散点图（每子样本一点）+ 线性拟合，WT/AS 各一。
    AS 不再按厚度切分，仅作为整体拟合一条直线。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    fig.suptitle("Mean pore radius vs mean throat radius (per subsample)", fontsize=14, fontweight="bold")

    for ax, (is_wt_flag, name, color) in zip(
        axes,
        [(True, "WT", "tab:blue"), (False, "AS", "tab:orange")],
    ):
        sub = df[df["is_wt"] == is_wt_flag][
            ["mean_pore_radius", "mean_throat_radius"]
        ].dropna()
        if len(sub) == 0:
            ax.set_title(f"{name}（无数据）")
            ax.set_ylabel("Mean throat radius (nm)", fontsize=11)
            ax.grid(True, alpha=0.3)
            continue

        x = sub["mean_pore_radius"].to_numpy(dtype=float)
        y = sub["mean_throat_radius"].to_numpy(dtype=float)
        ax.scatter(x, y, alpha=0.7, color=color, edgecolor="white", s=45)

        text_lines = []
        if len(x) >= 3:
            try:
                r_pearson, p_pearson = pearsonr(x, y)
                text_lines.append(f"Pearson r={r_pearson:.3f}, p={p_pearson:.3e}")
            except Exception:
                pass
            try:
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(
                    x_line,
                    y_line,
                    color=color,
                    linestyle="--",
                    linewidth=2,
                )
                text_lines.append(
                    f"Linear: r_throat={slope:.3f}*r_pore+{intercept:.2f}, R^2={r_value**2:.3f}"
                )
            except Exception:
                pass
        if text_lines:
            ax.text(
                0.02,
                0.98,
                "\n".join(text_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            )
        ax.set_ylabel("Mean throat radius (nm)", fontsize=11)
        ax.set_title(f"{name} (n={len(sub)} subsamples)", fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Mean pore radius (nm)", fontsize=11)
    plt.tight_layout()
    plot_path = out_dir / "mean_pore_radius_vs_mean_throat_radius.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存: {plot_path}")

def plot_mean_pore_radius_vs_degree(
    df: pd.DataFrame, out_dir: Path, as_split_nm: float = 150.0
) -> None:
    """
    平均孔半径 vs 平均孔度数 散点图（每子样本一点），WT/AS 各一。
    AS 在 as_split_nm 处分割（按厚度）做两组拟合。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    fig.suptitle("Mean pore radius vs mean pore degree (per subsample)", fontsize=14, fontweight="bold")

    for ax, (is_wt_flag, name, color) in zip(
        axes,
        [(True, "WT", "tab:blue"), (False, "AS", "tab:orange")],
    ):
        if is_wt_flag:
            sub = df[df["is_wt"] == is_wt_flag][
                ["mean_pore_radius", "mean_deg"]
            ].dropna()
        else:
            # AS 需要厚度列用于按厚度分段
            sub = df[df["is_wt"] == is_wt_flag][
                ["mean_pore_radius", "mean_deg", "thickness"]
            ].dropna()

        if len(sub) == 0:
            ax.set_title(f"{name} (no data)")
            ax.set_ylabel("Mean pore degree", fontsize=11)
            ax.grid(True, alpha=0.3)
            continue

        x = sub["mean_pore_radius"].to_numpy(dtype=float)
        y = sub["mean_deg"].to_numpy(dtype=float)
        ax.scatter(x, y, alpha=0.7, color=color, s=45, edgecolor="white")

        if name == "WT":
            if len(x) >= 3:
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(x, y)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(
                        x_line,
                        y_line,
                        color=color,
                        linestyle="--",
                        linewidth=2,
                        label=f"Linear fit (n={len(x)}, R^2={r_value**2:.3f})",
                    )
                except Exception:
                    pass
            ax.set_title(f"{name} (n={len(x)} subsamples)", fontsize=12)
        else:
            thickness = sub["thickness"].to_numpy(dtype=float)
            mask_low = thickness < as_split_nm
            mask_high = thickness >= as_split_nm
            n_low, n_high = int(mask_low.sum()), int(mask_high.sum())

            if n_low >= 2:
                x_l, y_l = x[mask_low], y[mask_low]
                try:
                    slope, intercept, r_value, _, _ = linregress(x_l, y_l)
                    x_line = np.linspace(x_l.min(), x_l.max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(
                        x_line,
                        y_line,
                        color="tab:orange",
                        linestyle="--",
                        linewidth=2,
                        label=f"T<{as_split_nm:.0f} nm (n={n_low}, R^2={r_value**2:.3f})",
                    )
                except Exception:
                    pass
            if n_high >= 2:
                x_h, y_h = x[mask_high], y[mask_high]
                try:
                    slope, intercept, r_value, _, _ = linregress(x_h, y_h)
                    x_line = np.linspace(x_h.min(), x_h.max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(
                        x_line,
                        y_line,
                        color="tab:red",
                        linestyle="--",
                        linewidth=2,
                        label=f"T ≥ {as_split_nm:.0f} nm (n={n_high}, R^2={r_value**2:.3f})",
                    )
                except Exception:
                    pass
            ax.set_title(
                f"{name} (total n={len(x)}, <{as_split_nm:.0f}: {n_low}, ≥{as_split_nm:.0f}: {n_high})",
                fontsize=12,
            )

        ax.set_ylabel("Mean pore degree", fontsize=11)
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("Mean pore radius (nm)", fontsize=11)
    plt.tight_layout()
    plot_path = out_dir / "mean_pore_radius_vs_mean_degree.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存: {plot_path}")


def plot_thickness_vs_pore_degree(
    df: pd.DataFrame, out_dir: Path, as_split_nm: float = 150.0
) -> None:
    """
    厚度 vs 平均孔度数 散点图（每子样本一点），WT/AS 各一。AS 在 as_split_nm 处分割为两段分别线性拟合。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    fig.suptitle("Thickness vs mean pore degree", fontsize=14, fontweight="bold")
    for ax, (is_wt_flag, name, color) in zip(
        axes,
        [(True, "WT", "tab:blue"), (False, "AS", "tab:orange")],
    ):
        sub = df[df["is_wt"] == is_wt_flag][["thickness", "mean_deg"]].dropna()
        if len(sub) == 0:
            ax.set_title(f"{name} (no data)")
            ax.set_ylabel("Mean pore degree", fontsize=11)
            ax.grid(True, alpha=0.3)
            continue
        thickness = sub["thickness"].to_numpy(dtype=float)
        degree = sub["mean_deg"].to_numpy(dtype=float)
        ax.scatter(thickness, degree, alpha=0.7, color=color, s=45, edgecolor="white")
        if name == "WT":
            if len(thickness) >= 3:
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(thickness, degree)
                    x_line = np.linspace(thickness.min(), thickness.max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=2,
                            label=f"Linear fit (n={len(thickness)}, R^2={r_value**2:.3f})")
                except Exception:
                    pass
            ax.set_title(f"{name} (n={len(thickness)} subsamples)", fontsize=12)
        else:
            # AS: 在 150nm 处分割，两段分别拟合
            mask_low = thickness < as_split_nm
            mask_high = thickness >= as_split_nm
            n_low, n_high = int(mask_low.sum()), int(mask_high.sum())
            if n_low >= 2:
                t_l, d_l = thickness[mask_low], degree[mask_low]
                try:
                    slope, intercept, r_value, _, _ = linregress(t_l, d_l)
                    x_line = np.linspace(t_l.min(), min(t_l.max(), as_split_nm), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color="tab:orange", linestyle="--", linewidth=2,
                            label=f"T<{as_split_nm:.0f} nm (n={n_low}, R^2={r_value**2:.3f})")
                except Exception:
                    pass
            if n_high >= 2:
                t_h, d_h = thickness[mask_high], degree[mask_high]
                try:
                    slope, intercept, r_value, _, _ = linregress(t_h, d_h)
                    x_line = np.linspace(max(t_h.min(), as_split_nm), t_h.max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color="tab:red", linestyle="--", linewidth=2,
                            label=f"T ≥ {as_split_nm:.0f} nm (n={n_high}, R^2={r_value**2:.3f})")
                except Exception:
                    pass
            ax.axvline(as_split_nm, color="gray", linestyle=":", linewidth=1)
            ax.set_title(f"{name} (total n={len(thickness)}, <{as_split_nm:.0f}: {n_low}, ≥{as_split_nm:.0f}: {n_high})", fontsize=12)
        ax.set_ylabel("Mean pore degree", fontsize=11)
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("Thickness (nm)", fontsize=11)
    plt.tight_layout()
    plot_path = out_dir / "thickness_vs_mean_pore_degree.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"已保存: {plot_path}")


def plot_param_vs_thickness_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """
    对与散点图相同的指标，画 厚度 vs 参数 的二维密度热图（WT / AS 各一 panel）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for col, ylabel in _THICKNESS_VS_PARAM_YLABELS_EN.items():
        if col not in df.columns:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        fig.suptitle(f"Thickness vs {ylabel} (density heatmap)", fontsize=14, fontweight="bold")

        for ax, (is_wt_flag, name) in zip(
            axes,
            [(True, "WT"), (False, "AS")],
        ):
            sub = df[df["is_wt"] == is_wt_flag][["thickness", col]].dropna()
            if len(sub) < 2:
                ax.set_title(f"{name} (no data or too few points)")
                ax.set_ylabel(ylabel, fontsize=11)
                ax.grid(True, alpha=0.3)
                continue
            x = sub["thickness"].to_numpy(dtype=float)
            y = sub[col].to_numpy(dtype=float)
            h = ax.hist2d(x, y, bins=(30, 30), cmap="viridis", cmin=1e-9)
            plt.colorbar(h[3], ax=ax, label="Sample count")
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{name} (n={len(sub)})", fontsize=12)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Thickness (nm)", fontsize=11)
        plt.tight_layout()
        plot_path = out_dir / f"thickness_vs_{col}_heatmap.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"已保存: {plot_path}")


def _pore_radius_quintile_labels(pore_radii: np.ndarray) -> np.ndarray | None:
    """
    子样本内孔半径五分位标签 0..K-1（K<=5，pd.qcut 在重复值多时会合并档）。
    孔数 < 5 或无法分档时返回 None。
    """
    r = np.asarray(pore_radii, dtype=float).ravel()
    n = int(r.size)
    if n < 5:
        return None
    s = pd.Series(r)
    try:
        q = pd.qcut(s, q=5, labels=False, duplicates="drop")
    except (ValueError, TypeError):
        return None
    q = np.asarray(q, dtype=float)
    if np.any(~np.isfinite(q)):
        return None
    return q.astype(int)


def _throat_endpoint_indices(
    throat_pore1: np.ndarray,
    throat_pore2: np.ndarray,
    pore_id_to_idx: dict,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """有效喉两端孔在 pore_radii 数组中的下标。"""
    i1_list: list[int] = []
    i2_list: list[int] = []
    for i in range(len(throat_pore1)):
        p1_id, p2_id = int(throat_pore1[i]), int(throat_pore2[i])
        if p1_id not in pore_id_to_idx or p2_id not in pore_id_to_idx:
            continue
        i1_list.append(int(pore_id_to_idx[p1_id]))
        i2_list.append(int(pore_id_to_idx[p2_id]))
    if not i1_list:
        return None
    return np.asarray(i1_list, dtype=int), np.asarray(i2_list, dtype=int)


def _frac_intra_and_diag_frac(
    q_labels: np.ndarray,
    i1_arr: np.ndarray,
    i2_arr: np.ndarray,
    n_q: int,
) -> tuple[float, np.ndarray]:
    """frac_intra 与各档对角喉占比 diag_frac[g] = #(g,g)/n_th。"""
    qa = q_labels[i1_arr]
    qb = q_labels[i2_arr]
    n_th = int(i1_arr.size)
    frac_intra = float(np.mean(qa == qb))
    diag_frac = np.zeros(n_q, dtype=float)
    for g in range(n_q):
        diag_frac[g] = float(np.mean((qa == g) & (qb == g)))
    return frac_intra, diag_frac


def _permutation_p_values_quintile_mixing(
    q_labels: np.ndarray,
    i1_arr: np.ndarray,
    i2_arr: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    """
    在固定图结构下随机置换「哪颗孔拿哪个五分位标签」（保持标签多重集不变）。
    返回：(p_frac_intra_upper, mean_frac_perm, p_diag_upper 长度 n_q, diag_frac_obs, mean_diag_frac_perm)
    p_* 为单侧上尾：观测不低于置换分布的比例（ assortativity 增强）。
    """
    n_pores = int(q_labels.size)
    n_q = int(np.max(q_labels)) + 1
    frac_obs, diag_obs = _frac_intra_and_diag_frac(q_labels, i1_arr, i2_arr, n_q)
    fr_perm = np.empty(n_perm, dtype=float)
    diag_perm_sum = np.zeros((n_perm, n_q), dtype=float)
    for b in range(n_perm):
        phi = rng.permutation(n_pores)
        qp = q_labels[phi]
        f, d = _frac_intra_and_diag_frac(qp, i1_arr, i2_arr, n_q)
        fr_perm[b] = f
        diag_perm_sum[b, :] = d
    mean_frac_perm = float(np.mean(fr_perm))
    mean_diag_perm = np.mean(diag_perm_sum, axis=0)
    # 单侧：置换中 >= 观测 的比例（+1 小样本校正）
    p_frac = (1.0 + float(np.sum(fr_perm >= frac_obs))) / float(n_perm + 1)
    p_diag = np.empty(n_q, dtype=float)
    for g in range(n_q):
        p_diag[g] = (1.0 + float(np.sum(diag_perm_sum[:, g] >= diag_obs[g]))) / float(n_perm + 1)
    return p_frac, mean_frac_perm, p_diag, diag_obs, mean_diag_perm


def analyze_pore_quintile_throat_mixing_from_paths(
    pf: Path,
    tf: Path,
    *,
    n_perm: int = 0,
    rng: np.random.Generator | None = None,
    min_throats_for_perm: int = 30,
) -> dict | None:
    """
    口径 A：给定孔/喉 xlsx，子样本内孔按半径五分位；每条喉两端同档则计为组内喉。
    供真实子样本与 synthetic_{TYPE}_run{i}_plane 网络复用。
    """
    pf, tf = Path(pf), Path(tf)
    if rng is None:
        rng = np.random.default_rng()
    if not pf.is_file() or not tf.is_file():
        return None
    try:
        _, pore_radii, _, pore_id_to_idx, throat_pore1, throat_pore2, _ = st.load_pores_and_throats(pf, tf)
        q_labels = _pore_radius_quintile_labels(pore_radii)
        if q_labels is None:
            return {
                "ok": False,
                "reason": "quintile_fail_or_too_few_pores",
                "n_pores": int(len(pore_radii)),
                "n_throats": 0,
                "n_intra": 0,
                "frac_intra": float("nan"),
                "frac_inter": float("nan"),
                "n_perm_done": 0,
            }
        pair_idx = _throat_endpoint_indices(throat_pore1, throat_pore2, pore_id_to_idx)
        if pair_idx is None:
            return {
                "ok": True,
                "reason": "no_throats",
                "n_pores": int(len(pore_radii)),
                "n_throats": 0,
                "n_intra": 0,
                "frac_intra": float("nan"),
                "frac_inter": float("nan"),
                "n_perm_done": 0,
            }
        i1_arr, i2_arr = pair_idx
        n_th = int(i1_arr.size)
        qa = q_labels[i1_arr]
        qb = q_labels[i2_arr]
        n_intra = int(np.sum(qa == qb))
        frac_intra = float(n_intra / n_th)
        n_q = int(np.max(q_labels)) + 1
        _, diag_obs = _frac_intra_and_diag_frac(q_labels, i1_arr, i2_arr, n_q)

        base: dict = {
            "ok": True,
            "reason": "",
            "n_pores": int(len(pore_radii)),
            "n_throats": n_th,
            "n_intra": n_intra,
            "frac_intra": frac_intra,
            "frac_inter": float(1.0 - frac_intra),
            "n_quintile_bins": n_q,
        }
        for g in range(5):
            base[f"diag_frac_q{g}"] = float(diag_obs[g]) if g < n_q else float("nan")
            base[f"diag_n_q{g}"] = int(np.sum((qa == g) & (qb == g))) if g < n_q else 0

        if n_perm > 0 and n_th >= min_throats_for_perm:
            p_frac, mean_frac, p_diag, _, mean_diag = _permutation_p_values_quintile_mixing(
                q_labels, i1_arr, i2_arr, n_perm, rng
            )
            base["n_perm_done"] = int(n_perm)
            base["p_perm_frac_intra_upper"] = float(p_frac)
            base["frac_intra_perm_mean"] = float(mean_frac)
            base["delta_frac_intra"] = float(frac_intra - mean_frac)
            for g in range(5):
                base[f"p_perm_diag_q{g}_upper"] = float(p_diag[g]) if g < len(p_diag) else float("nan")
                base[f"diag_frac_q{g}_perm_mean"] = float(mean_diag[g]) if g < len(mean_diag) else float("nan")
                base[f"delta_diag_frac_q{g}"] = (
                    float(diag_obs[g] - mean_diag[g]) if g < n_q and g < len(mean_diag) else float("nan")
                )
        else:
            base["n_perm_done"] = 0
            base["p_perm_frac_intra_upper"] = float("nan")
            base["frac_intra_perm_mean"] = float("nan")
            base["delta_frac_intra"] = float("nan")
            for g in range(5):
                base[f"p_perm_diag_q{g}_upper"] = float("nan")
                base[f"diag_frac_q{g}_perm_mean"] = float("nan")
                base[f"delta_diag_frac_q{g}"] = float("nan")

        return base
    except Exception:
        return None


def analyze_pore_quintile_throat_mixing_one_row(
    row: pd.Series,
    split_samples_dir: Path,
    *,
    n_perm: int = 0,
    rng: np.random.Generator | None = None,
    min_throats_for_perm: int = 30,
) -> dict | None:
    """
    口径 A：子样本内孔按半径五分位；每条喉若两端孔同档则计为组内喉。
    n_perm>0 时在固定喉连接下对孔的五分位标签做置换检验（保持各档孔数不变）。
    """
    split_samples_dir = Path(split_samples_dir)
    if rng is None:
        rng = np.random.default_rng()
    sample_dir = split_samples_dir / row["sample_name"]
    sub_list = find_sub_samples(sample_dir, row["sample_name"])
    for sn, pf, tf in sub_list:
        if sn != row["sub_name"] or not pf or not pf.exists() or not tf or not tf.exists():
            continue
        return analyze_pore_quintile_throat_mixing_from_paths(
            pf, tf, n_perm=n_perm, rng=rng, min_throats_for_perm=min_throats_for_perm
        )
    return None


def collect_pore_quintile_throat_mixing_dataframe(
    df: pd.DataFrame,
    split_samples_dir: Path,
    *,
    n_perm: int = 0,
    random_seed: int | None = None,
    min_throats_for_perm: int = 30,
) -> pd.DataFrame:
    """逐子样本统计孔五分位组内喉占比；可选置换检验。"""
    rng = np.random.default_rng(random_seed)
    rows: list[dict] = []
    for _, row in df.iterrows():
        rec = analyze_pore_quintile_throat_mixing_one_row(
            row,
            split_samples_dir,
            n_perm=n_perm,
            rng=rng,
            min_throats_for_perm=min_throats_for_perm,
        )
        if rec is None:
            continue
        rows.append(
            {
                "sample_name": row["sample_name"],
                "sub_name": row["sub_name"],
                "thickness": row.get("thickness"),
                "is_wt": bool(row["is_wt"]),
                **rec,
            }
        )
    return pd.DataFrame(rows)


def collect_synthetic_run_quintile_mixing_dataframe(
    phase3_networks_dir: Path,
    thickness_sampling_dir: Path,
    sample_types: list,
    n_runs: int,
    *,
    phase1_dir: Path | None = None,
    n_perm: int = 0,
    random_seed: int | None = None,
    min_throats_for_perm: int = 30,
) -> pd.DataFrame:
    """
    与 ``run_uniform_sieving_pipeline._collect_run_stats`` 相同的 run 枚举方式，
    对每套 ``synthetic_{TYPE}_run{i}_plane`` 孔/喉 xlsx 计算口径 A 的 frac_intra（可选置换）。
    """
    phase3_networks_dir = Path(phase3_networks_dir)
    thickness_sampling_dir = Path(thickness_sampling_dir)
    xlsx_dirs = [phase3_networks_dir]
    if phase3_networks_dir.parent.exists():
        xlsx_dirs.append(phase3_networks_dir.parent)
    search_dirs = [thickness_sampling_dir]
    if phase1_dir is not None:
        search_dirs.append(Path(phase1_dir))
    rng = np.random.default_rng(random_seed)
    rows: list[dict] = []
    for sample_type in sample_types:
        json_path = None
        for d in search_dirs:
            if not d.exists():
                continue
            p = d / f"sampled_thickness_n{n_runs}_{sample_type}.json"
            if p.exists():
                json_path = p
                break
            alt = list(d.glob(f"sampled_thickness_n*_{sample_type}.json"))
            if alt:
                json_path = alt[0]
                break
        if json_path is None:
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        thickness_list = data.get("thickness_nm", [])
        st_upper = str(sample_type).upper()
        is_wt = st_upper == "WT"
        for i in range(min(int(n_runs), len(thickness_list))):
            T = float(thickness_list[i])
            sample_name = f"synthetic_{sample_type}_run{i}_plane"
            pf: Path | None = None
            tf: Path | None = None
            for d in xlsx_dirs:
                cand_p = d / f"{sample_name}_pores.xlsx"
                cand_t = d / f"{sample_name}_throats.xlsx"
                if cand_p.is_file() and cand_t.is_file():
                    pf, tf = cand_p, cand_t
                    break
            if pf is None or tf is None:
                rec = {
                    "ok": False,
                    "reason": "missing_xlsx",
                    "n_pores": 0,
                    "n_throats": 0,
                    "n_intra": 0,
                    "frac_intra": float("nan"),
                    "frac_inter": float("nan"),
                    "n_perm_done": 0,
                }
            else:
                rec = analyze_pore_quintile_throat_mixing_from_paths(
                    pf,
                    tf,
                    n_perm=n_perm,
                    rng=rng,
                    min_throats_for_perm=min_throats_for_perm,
                )
                if rec is None:
                    rec = {
                        "ok": False,
                        "reason": "load_or_compute_failed",
                        "n_pores": 0,
                        "n_throats": 0,
                        "n_intra": 0,
                        "frac_intra": float("nan"),
                        "frac_inter": float("nan"),
                        "n_perm_done": 0,
                    }
            rows.append(
                {
                    "sample_name": sample_name,
                    "sub_name": sample_name,
                    "run": int(i),
                    "sample_type": sample_type,
                    "thickness": T,
                    "is_wt": is_wt,
                    **rec,
                }
            )
    return pd.DataFrame(rows)


def _build_permutation_summary(df_ok: pd.DataFrame) -> dict:
    """
    汇总子样本级置换检验：全局 frac_intra 与各档对角 (g,g) 喉占比。
    p_upper 小表示「观测不低于置换」的概率低，即相对标签随机更易出现组内/(g,g) 连接。
    """
    if "n_perm_done" not in df_ok.columns:
        return {}
    sub = df_ok[df_ok["n_perm_done"] > 0]
    if sub.empty:
        return {}
    n_perm = int(sub["n_perm_done"].iloc[0])

    def _one_group(sdf: pd.DataFrame) -> dict:
        if len(sdf) == 0:
            return {}
        by_q: list[dict] = []
        for g in range(5):
            col_d = f"delta_diag_frac_q{g}"
            col_p = f"p_perm_diag_q{g}_upper"
            if col_d not in sdf.columns or col_p not in sdf.columns:
                continue
            d = sdf[col_d].to_numpy(dtype=float)
            p = sdf[col_p].to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            p = p[np.isfinite(p)]
            by_q.append(
                {
                    "quintile_g": g,
                    "median_delta_diag_frac": float(np.median(d)) if len(d) else None,
                    "median_p_upper": float(np.median(p)) if len(p) else None,
                    "frac_subsamples_p_upper_lt_0.05": float(np.mean(p < 0.05)) if len(p) else None,
                }
            )
        ranked = sorted(
            [x for x in by_q if x.get("median_p_upper") is not None],
            key=lambda x: float(x["median_p_upper"]),
        )
        return {
            "n_subsamples": int(len(sdf)),
            "n_perm": n_perm,
            "frac_intra_median_delta_obs_minus_perm_mean": float(np.median(sdf["delta_frac_intra"])),
            "frac_intra_median_p_upper": float(np.median(sdf["p_perm_frac_intra_upper"])),
            "frac_intra_frac_subsamples_p_upper_lt_0.05": float(
                np.mean(sdf["p_perm_frac_intra_upper"] < 0.05)
            ),
            "by_quintile_diag": by_q,
            "quintile_g_ranked_strongest_evidence_first": [x["quintile_g"] for x in ranked],
        }

    return {
        "interpretation": (
            "固定喉连接，随机重排孔上五分位标签（保持各档孔数不变）。"
            "p_*_upper = P(置换后指标 >= 观测)，小表示观测相对随机标签更偏同档/(g,g) 连接。"
        ),
        "all": _one_group(sub),
        "WT": _one_group(sub[sub["is_wt"] == True]),  # noqa: E712
        "AS": _one_group(sub[sub["is_wt"] == False]),  # noqa: E712
    }


def plot_pore_quintile_permutation_figures(df_ok: pd.DataFrame, out_dir: Path) -> None:
    """置换检验：各档 Δdiag_frac 与 p_upper（WT/AS）。"""
    if "n_perm_done" not in df_ok.columns:
        return
    sub = df_ok[df_ok["n_perm_done"] > 0]
    if sub.empty:
        return
    out_dir = Path(out_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (is_wt, name) in zip(axes, [(True, "WT"), (False, "AS")]):
        s = sub[sub["is_wt"] == is_wt]
        meds = []
        for g in range(5):
            col = f"delta_diag_frac_q{g}"
            if col not in s.columns:
                meds.append(np.nan)
                continue
            v = s[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            meds.append(float(np.median(v)) if len(v) else np.nan)
        x = np.arange(5)
        ax.axhline(0.0, color="k", linewidth=0.9, zorder=0)
        ax.bar(x, meds, color="steelblue" if is_wt else "darkorange", alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{g}" for g in range(5)])
        ax.set_xlabel("Pore radius quintile (within sub-sample)")
        ax.set_ylabel(r"median $\Delta$ diag frac (obs $-$ perm mean)")
        ax.set_title(f"{name}: excess (g,g) throats vs label permutation null")
        ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("Which quintile shows strongest (g,g) excess vs permutation null?", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p_out = out_dir / "pore_quintile_perm_delta_diag_frac_WT_AS.png"
    plt.savefig(p_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [孔五分位-喉] 已保存: {p_out}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (is_wt, name) in zip(axes, [(True, "WT"), (False, "AS")]):
        s = sub[sub["is_wt"] == is_wt]
        med_p = []
        for g in range(5):
            col = f"p_perm_diag_q{g}_upper"
            if col not in s.columns:
                med_p.append(np.nan)
                continue
            p = s[col].to_numpy(dtype=float)
            p = p[np.isfinite(p)]
            med_p.append(float(np.median(p)) if len(p) else np.nan)
        x = np.arange(5)
        ax.bar(x, med_p, color="steelblue" if is_wt else "darkorange", alpha=0.85, edgecolor="white")
        ax.axhline(0.05, color="crimson", linestyle="--", linewidth=1.0, label="p=0.05")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{g}" for g in range(5)])
        ax.set_xlabel("Pore radius quintile")
        ax.set_ylabel("median p_upper (one-sided vs permutation)")
        ax.set_title(f"{name}: median permutation p per diagonal (g,g)")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("Smaller p_upper ⇒ less likely under random labels (stronger local signal)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p_out2 = out_dir / "pore_quintile_perm_pvalue_diag_WT_AS.png"
    plt.savefig(p_out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [孔五分位-喉] 已保存: {p_out2}")


def _safe_dataframe_to_csv(df: pd.DataFrame, path: Path, *, tag: str = "[孔五分位-喉]", **kwargs) -> Path:
    """
    写入 CSV。Windows 下若目标被 Excel 等程序锁定，则写入同目录带时间戳的备用文件，避免整段流程中断。
    """
    path = Path(path)
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        df.to_csv(alt, **kwargs)
        print(f"  {tag} 警告：无法写入（文件可能被占用）：{path}", file=sys.stderr)
        print(f"  {tag} 已改存：{alt}", file=sys.stderr)
        return alt


def _uniform_thickness_bin_edges(
    thickness_arr: np.ndarray,
    bin_width_nm: float,
) -> np.ndarray:
    """全样本厚度统一等宽分箱边界（用于 WT vs AS 同口径比较）。"""
    t = np.asarray(thickness_arr, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return np.array([0.0, bin_width_nm])
    lo = float(np.floor(np.min(t) / bin_width_nm) * bin_width_nm)
    hi = float(np.ceil(np.max(t) / bin_width_nm) * bin_width_nm)
    if hi <= lo:
        hi = lo + bin_width_nm
    edges = np.arange(lo, hi + 1e-9, bin_width_nm)
    if edges[-1] < hi + 1e-6:
        edges = np.concatenate([edges, [hi + 1e-9]])
    return edges


def analyze_and_plot_pore_quintile_throat_mixing(
    df_mix: pd.DataFrame,
    out_dir: Path,
    *,
    bin_width_nm: float = DENSITY_FRAC_BIN_WIDTH_NM,
) -> dict:
    """
    按厚度分层（Phase2 密度分层 + 统一等宽分层）汇总 frac_intra，Mann-Whitney U（WT vs AS），并出图。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = df_mix["ok"] == True  # noqa: E712
    df_ok = df_mix[ok & np.isfinite(df_mix["frac_intra"])].copy()
    summary: dict = {
        "metric": "frac_intra = fraction of throats with both endpoints in same pore-radius quintile (within sub-sample)",
        "bin_width_native_nm": float(bin_width_nm),
        "bin_width_uniform_nm": float(bin_width_nm),
        "n_subsamples_total": int(len(df_mix)),
        "n_subsamples_ok_frac": int(len(df_ok)),
    }

    if df_ok.empty:
        summary["note"] = "无有效 frac_intra 子样本"
        with open(out_dir / "pore_quintile_throat_mixing_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("  [孔五分位-喉] 无有效数据，跳过统计与作图")
        return summary

    # 全局 WT vs AS（忽略厚度）
    wt_all = df_ok[df_ok["is_wt"]]["frac_intra"].to_numpy(dtype=float)
    as_all = df_ok[~df_ok["is_wt"]]["frac_intra"].to_numpy(dtype=float)
    mw_global = None
    if wt_all.size >= 2 and as_all.size >= 2:
        try:
            u_stat, p_two = mannwhitneyu(wt_all, as_all, alternative="two-sided")
            mw_global = {"U": float(u_stat), "p_value_two_sided": float(p_two), "n_wt": int(wt_all.size), "n_as": int(as_all.size)}
        except Exception:
            mw_global = None
    summary["mann_whitney_global_wt_vs_as"] = mw_global

    # 统一等宽厚度 bin：WT vs AS 每 bin
    t_all = df_ok["thickness"].to_numpy(dtype=float)
    u_edges = _uniform_thickness_bin_edges(t_all, bin_width_nm)
    n_u = len(u_edges) - 1
    u_bin_tests: list[dict] = []
    for bi in range(n_u):
        t_lo, t_hi = float(u_edges[bi]), float(u_edges[bi + 1])
        m = df_ok[(df_ok["thickness"] >= t_lo) & (df_ok["thickness"] < t_hi)]
        wt_b = m[m["is_wt"]]["frac_intra"].to_numpy(dtype=float)
        as_b = m[~m["is_wt"]]["frac_intra"].to_numpy(dtype=float)
        rec = {
            "t_min": t_lo,
            "t_max": t_hi,
            "t_center": (t_lo + t_hi) / 2.0,
            "n_wt": int(wt_b.size),
            "n_as": int(as_b.size),
        }
        if wt_b.size >= 2 and as_b.size >= 2:
            try:
                u_stat, p_two = mannwhitneyu(wt_b, as_b, alternative="two-sided")
                rec["mann_whitney_p_two_sided"] = float(p_two)
            except Exception:
                rec["mann_whitney_p_two_sided"] = None
        else:
            rec["mann_whitney_p_two_sided"] = None
        u_bin_tests.append(rec)
    summary["uniform_thickness_bins_mann_whitney"] = u_bin_tests

    # Phase2 原生分层：WT / AS 各自 bin_edges
    native: dict = {"WT": {}, "AS": {}}
    for is_wt, name in [(True, "WT"), (False, "AS")]:
        sub = df_ok[df_ok["is_wt"] == is_wt]
        th = sub["thickness"].dropna().to_numpy(dtype=float)
        if th.size == 0:
            native[name] = {"bin_edges": [], "bins": []}
            continue
        edges = get_thickness_bin_edges_density_frac(th, is_wt, bin_width=float(bin_width_nm))
        bin_stats: list[dict] = []
        for i in range(len(edges) - 1):
            t_lo, t_hi = float(edges[i]), float(edges[i + 1])
            vals = sub[(sub["thickness"] >= t_lo) & (sub["thickness"] < t_hi)]["frac_intra"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            bin_stats.append(
                {
                    "t_min": t_lo,
                    "t_max": t_hi,
                    "t_center": (t_lo + t_hi) / 2.0,
                    "n": int(vals.size),
                    "median_frac_intra": float(np.median(vals)) if vals.size else None,
                    "mean_frac_intra": float(np.mean(vals)) if vals.size else None,
                }
            )
        native[name] = {"bin_edges": edges.tolist(), "bins": bin_stats}
    summary["native_thickness_bins"] = native

    perm_summ = _build_permutation_summary(df_ok)
    if perm_summ.get("all"):
        summary["label_permutation_null"] = perm_summ

    with open(out_dir / "pore_quintile_throat_mixing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [孔五分位-喉] 已保存: {out_dir / 'pore_quintile_throat_mixing_summary.json'}")
    if perm_summ.get("all"):
        for grp in ("all", "WT", "AS"):
            block = perm_summ.get(grp) or {}
            rk = block.get("quintile_g_ranked_strongest_evidence_first")
            if rk is not None:
                print(f"  [孔五分位-置换] {grp}: 对角(g,g)证据由强到弱（median p_upper 升序）quintile_g = {rk}")

    csv_path = _safe_dataframe_to_csv(
        df_ok,
        out_dir / "pore_quintile_throat_mixing_by_subsample.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"  [孔五分位-喉] 已保存: {csv_path}")

    # 图1：厚度 vs frac_intra 散点（WT / AS）
    fig, ax = plt.subplots(figsize=(8, 5))
    wt_df = df_ok[df_ok["is_wt"]]
    as_df = df_ok[~df_ok["is_wt"]]
    if len(wt_df):
        ax.scatter(wt_df["thickness"], wt_df["frac_intra"], alpha=0.55, s=22, label="WT", c="tab:blue")
    if len(as_df):
        ax.scatter(as_df["thickness"], as_df["frac_intra"], alpha=0.55, s=22, label="AS", c="tab:orange")
    ax.set_xlabel("Thickness (nm)")
    ax.set_ylabel(r"frac_intra (throat with both endpoints in same pore-radius quintile)")
    ax.set_title("Pore-radius quintile: intra-group throat fraction (per sub-sample)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    p1 = out_dir / "pore_quintile_frac_intra_vs_thickness_scatter.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [孔五分位-喉] 已保存: {p1}")

    # 图2：原生分层下各厚度 bin 的箱线图（WT / AS 分两行）
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    for ax_idx, (is_wt, name) in enumerate([(True, "WT"), (False, "AS")]):
        ax = axes[ax_idx]
        sub = df_ok[df_ok["is_wt"] == is_wt]
        th = sub["thickness"].dropna().to_numpy(dtype=float)
        if th.size == 0:
            ax.set_title(f"{name}: 无厚度数据")
            continue
        edges = get_thickness_bin_edges_density_frac(th, is_wt, bin_width=float(bin_width_nm))
        positions = []
        data = []
        labels = []
        for i in range(len(edges) - 1):
            t_lo, t_hi = float(edges[i]), float(edges[i + 1])
            vals = sub[(sub["thickness"] >= t_lo) & (sub["thickness"] < t_hi)]["frac_intra"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            positions.append((t_lo + t_hi) / 2.0)
            data.append(vals)
            labels.append(f"{t_lo:.0f}–{t_hi:.0f}")
        if data:
            bp = ax.boxplot(data, positions=range(len(data)), widths=0.6, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightsteelblue" if name == "WT" else "moccasin")
                patch.set_alpha(0.85)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("frac_intra")
        ax.set_title(f"{name}: frac_intra by thickness bin (Phase2 density-style edges)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(-0.02, 1.02)
    axes[-1].set_xlabel("Thickness bin (nm)")
    plt.suptitle("Intra-quintile throat fraction (native per-group thickness bins)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p2 = out_dir / "pore_quintile_frac_intra_boxplot_native_WT_AS.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [孔五分位-喉] 已保存: {p2}")

    # 图3：统一厚度 bin 下 WT vs AS 中位数（逐 bin 的 p 值见 JSON）
    fig, ax = plt.subplots(figsize=(10, 5))
    med_wt = []
    med_as = []
    for rec in u_bin_tests:
        t_lo, t_hi = rec["t_min"], rec["t_max"]
        m = df_ok[(df_ok["thickness"] >= t_lo) & (df_ok["thickness"] < t_hi)]
        wt_b = m[m["is_wt"]]["frac_intra"].to_numpy(dtype=float)
        as_b = m[~m["is_wt"]]["frac_intra"].to_numpy(dtype=float)
        med_wt.append(float(np.median(wt_b)) if wt_b.size else np.nan)
        med_as.append(float(np.median(as_b)) if as_b.size else np.nan)
    x = np.arange(len(u_bin_tests))
    w = 0.35
    ax.bar(x - w / 2, med_wt, width=w, label="WT median", color="tab:blue", alpha=0.85)
    ax.bar(x + w / 2, med_as, width=w, label="AS median", color="tab:orange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{u_bin_tests[i]['t_min']:.0f}–{u_bin_tests[i]['t_max']:.0f}" for i in range(len(u_bin_tests))],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Median frac_intra")
    ax.set_xlabel(f"Uniform thickness bins ({bin_width_nm:.0f} nm)")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.02)
    p_g = summary.get("mann_whitney_global_wt_vs_as") or {}
    p_glob = p_g.get("p_value_two_sided")
    if p_glob is not None:
        title = f"WT vs AS median frac_intra (uniform bins); global Mann-Whitney p={p_glob:.4g}"
    else:
        title = "WT vs AS median frac_intra (uniform thickness bins)"
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    p3 = out_dir / "pore_quintile_frac_intra_uniform_bins_WT_vs_AS.png"
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [孔五分位-喉] 已保存: {p3}")

    if perm_summ.get("all"):
        plot_pore_quintile_permutation_figures(df_ok, out_dir)

    return summary


def main() -> None:
    _parent_dir = WORKSPACE_ROOT
    parser = argparse.ArgumentParser(
        description="阶段 1.5：参数 vs 厚度（WT/AS 分开）可视化检查"
    )
    parser.add_argument(
        "--split-samples-dir",
        type=str,
        default=str(_parent_dir / "data_subsamples"),
        help="子样本根目录（默认：肾脏/data_subsamples）",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=str(_parent_dir / "results_core_subsamples"),
        help="分析结果根目录（默认：肾脏/results_core_subsamples）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_ROOT / "phase1_5_parameter_trends"),
        help="输出目录（默认：项目 outputs/phase1_5_parameter_trends）",
    )
    parser.add_argument(
        "--skip-pore-quintile-mixing",
        action="store_true",
        help="跳过孔半径五分位组内喉占比统计（口径 A，需逐子样本读几何）",
    )
    parser.add_argument(
        "--pore-quintile-n-perm",
        type=int,
        default=199,
        help="孔五分位标签置换检验次数（0=不做置换，仅描述统计；默认 199）",
    )
    parser.add_argument(
        "--pore-quintile-min-throats",
        type=int,
        default=30,
        help="子样本喉条数≥此值才做置换（避免小样本置换不稳定）",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="随机种子（五分位置换等）",
    )
    args = parser.parse_args()

    split_samples_dir = Path(args.split_samples_dir)
    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.output_dir)

    print("=" * 60)
    print("阶段 1.5：参数 vs 厚度 可视化（WT / AS 分开）")
    print("=" * 60)
    print(f"子样本目录: {split_samples_dir}")
    print(f"分析结果目录: {analysis_dir}")
    print(f"输出目录: {out_dir}")

    df = collect_summary(split_samples_dir, analysis_dir)
    print(f"共收集 {len(df)} 个子样本的 summary 行")

    print("计算子样本均值喉长度（与 Phase2 一致）...")
    df = add_mean_throat_length(df, split_samples_dir)
    print("喉长度列已加入。")

    # 喉半径 vs 喉长度（逐喉散点 + 线性拟合）
    print("收集喉半径与长度（逐喉）...")
    throat_rl = collect_throat_radius_length(df, split_samples_dir)
    fits_throat_length = plot_throat_radius_vs_length(throat_rl, out_dir)

    # 调试：检查几何上是否存在 L_raw <= 0 的喉，并画直方图
    print("调试：统计 L_raw = |p1-p2| - r_p1 - r_p2 的分布，以及 L_raw <= 0 的喉数量...")
    debug_plot_raw_throat_length_histogram(df, split_samples_dir, out_dir)

    # 调试：对 L_raw < 0 的喉，画喉半径 vs 两端孔平均半径散点图
    print("调试：对 L_raw < 0 的喉，绘制喉半径 vs 两端孔平均半径散点图...")
    debug_plot_negative_L_raw_radius_scatter(df, split_samples_dir, out_dir)

    # 孔半径 vs 该孔上喉平均半径（逐孔散点 + 线性拟合）
    print("收集孔半径与孔上喉平均半径（逐孔）...")
    pore_th = collect_pore_radius_vs_mean_throat_radius(df, split_samples_dir)
    plot_pore_radius_vs_mean_throat_radius(pore_th, out_dir)

    # 喉半径 vs 两端孔平均半径（每条喉）+ 线性拟合
    print("收集喉半径与两端孔平均半径（逐喉）...")
    th_meanp = collect_throat_radius_vs_mean_pore_radius(df, split_samples_dir)
    fits_throat_pore = plot_throat_radius_vs_mean_pore_radius(th_meanp, out_dir)

    # 孔度数 vs 孔半径（逐孔）
    print("收集孔度数与孔半径（逐孔）...")
    deg_rad = collect_pore_degree_vs_radius(df, split_samples_dir)
    fits_degree_radius = plot_pore_degree_vs_radius(deg_rad, out_dir)

    # 统一写入 phase1_5_linear_fits.json，供 pipeline 使用
    fits_all = {
        "throat_radius_vs_length": fits_throat_length or {},
        "throat_radius_vs_mean_pore_radius": fits_throat_pore or {},
        "pore_degree_vs_radius": fits_degree_radius or {},
    }
    fits_path = out_dir / "phase1_5_linear_fits.json"
    with fits_path.open("w", encoding="utf-8") as f:
        json.dump(fits_all, f, indent=2, ensure_ascii=False)
    print(f"已保存线性拟合: {fits_path}")

    # 实际连接 vs 随机孔对的半径差分布
    print("收集连接孔对与随机孔对的半径差...")
    conn_rand = collect_connected_vs_random_radius_diff(df, split_samples_dir)
    plot_connected_vs_random_radius_diff(conn_rand, out_dir)

    # 平均喉半径 vs 平均喉长度（每子样本一点 + 线性拟合）
    if "mean_throat_length" in df.columns and "mean_throat_radius" in df.columns:
        plot_mean_throat_radius_vs_length(df, out_dir)

    # 厚度 vs 平均孔度数（每子样本一点，AS 在 150nm 处分两段拟合）
    plot_thickness_vs_pore_degree(df, out_dir, as_split_nm=150.0)

    # 平均孔半径 vs 平均孔度数（每子样本一点，AS 在 150nm 处分两段拟合）
    plot_mean_pore_radius_vs_degree(df, out_dir, as_split_nm=150.0)

    # 平均孔半径 vs 平均喉半径（每子样本一点，WT/AS 各一，不切分 AS 厚度）
    if "mean_pore_radius" in df.columns and "mean_throat_radius" in df.columns:
        plot_mean_pore_radius_vs_mean_throat_radius(df, out_dir)

    plot_param_vs_thickness(df, out_dir)
    plot_param_vs_thickness_heatmap(df, out_dir)

    if not getattr(args, "skip_pore_quintile_mixing", False):
        print("\n孔半径五分位（口径 A）：组内喉占比 frac_intra（两端孔同五分位档）...")
        n_perm = max(0, int(getattr(args, "pore_quintile_n_perm", 199)))
        if n_perm > 0:
            print(f"  置换检验: n_perm={n_perm}, min_throats={getattr(args, 'pore_quintile_min_throats', 30)}")
        df_mix = collect_pore_quintile_throat_mixing_dataframe(
            df,
            split_samples_dir,
            n_perm=n_perm,
            random_seed=getattr(args, "random_seed", None),
            min_throats_for_perm=int(getattr(args, "pore_quintile_min_throats", 30)),
        )
        analyze_and_plot_pore_quintile_throat_mixing(df_mix, out_dir)

    print("\n" + "=" * 60)
    print("阶段 1.5 完成！")
    print("=" * 60)
    print(f"\n图像输出目录: {out_dir}")


if __name__ == "__main__":
    main()

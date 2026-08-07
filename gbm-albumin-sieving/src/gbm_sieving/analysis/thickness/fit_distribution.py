"""
阶段一：厚度分布描述（WT / AS）
- 汇总厚度数据（WT / AS）
- 厚度分布可视化与描述
- 厚度分布参数化（单峰拟合或混合分布/KDE）
"""

import argparse
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

from gbm_sieving.analysis.structure.thickness_relationships import (
    collect_all_sub_samples_with_normal,
)
from gbm_sieving.analysis.structure.compare_subsamples import (
    find_sub_samples,
)
from gbm_sieving.analysis.structure import statistics as st
from gbm_sieving.paths import OUTPUT_ROOT, WORKSPACE_ROOT

# 与网络提取模块的 TILT_REFINE_DEG 一致（θ 图参照线）
TILT_REFINE_DEG_REFERENCE = 25.0


def calculate_thickness_from_pore_model(
    pore_coords: np.ndarray,
    pore_radii: np.ndarray,
    normal_vec: np.ndarray,
    cell_size: float = 10.0,
    edge_trim_ratio: float = 0.1,
    min_pores_per_cell: int = 10,
) -> tuple[float, np.ndarray]:
    """
    从球棍模型计算厚度（基于法向量方向）。
    
    方法：在垂直于法向量的平面上打网格，计算每个网格内在法向量方向上的厚度范围。
    只采样中间区域，排除边缘部分以减少误差。
    
    Parameters:
    -----------
    pore_coords : np.ndarray
        孔坐标 (N, 3)
    pore_radii : np.ndarray
        孔半径 (N,)
    normal_vec : np.ndarray
        法向量（渗透方向）(3,)，已归一化。如果主要在XY平面，会正确处理。
    cell_size : float
        网格大小 (nm)
    edge_trim_ratio : float
        边缘排除比例（默认 0.1，即排除边缘 0.10 的区域）
    min_pores_per_cell : int
        每个网格至少需要的孔数（默认 20），少于该数的网格会被过滤掉
    
    Returns:
    --------
    mean_thickness : float
        平均厚度
    local_thicknesses : np.ndarray
        每个网格的局部厚度值（仅中间区域）
    """
    # 处理法向量：如果主要在XY平面（z分量很小），使用XY平面作为基平面
    abs_normal = np.abs(normal_vec)
    if abs_normal[2] < 0.1:  # z分量很小，主要在XY平面
        # 法向量在XY平面：normal_vec ≈ (nx, ny, 0)
        # 使用X、Y轴作为基向量（在XY平面内）
        e1 = np.array([1.0, 0.0, 0.0], dtype=float)  # X轴
        e2 = np.array([0.0, 1.0, 0.0], dtype=float)  # Y轴
        # 如果法向量不平行于X轴，需要旋转基向量使其与法向量对齐
        if abs_normal[0] < 0.9:  # 法向量不主要沿X方向
            # 构建与法向量垂直的基向量
            # e1 垂直于法向量且在XY平面内
            e1_xy = np.array([normal_vec[1], -normal_vec[0], 0.0], dtype=float)
            e1_xy_norm = np.linalg.norm(e1_xy)
            if e1_xy_norm > 1e-12:
                e1 = e1_xy / e1_xy_norm
            e2 = np.cross(normal_vec, e1)  # 确保在XY平面内
            e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    else:
        # 法向量有显著的z分量，使用通用方法
        # 选择与法向量最不共线的标准轴作为第一个基向量
        min_axis = np.argmin(abs_normal)
        e1 = np.zeros(3)
        e1[min_axis] = 1.0
        # 正交化
        e1 = e1 - np.dot(e1, normal_vec) * normal_vec
        e1 = e1 / (np.linalg.norm(e1) + 1e-12)
        e2 = np.cross(normal_vec, e1)
        e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    
    # 将孔坐标投影到法向量和平面坐标系
    coords_along_normal = np.dot(pore_coords, normal_vec)  # 沿法向量方向的坐标
    coords_u = np.dot(pore_coords, e1)  # 平面内第一个坐标
    coords_v = np.dot(pore_coords, e2)  # 平面内第二个坐标
    
    # 确定平面范围
    u_min, u_max = coords_u.min(), coords_u.max()
    v_min, v_max = coords_v.min(), coords_v.max()
    
    # 计算中间区域范围（排除边缘）
    u_range = u_max - u_min
    v_range = v_max - v_min
    u_center_min = u_min + edge_trim_ratio * u_range
    u_center_max = u_max - edge_trim_ratio * u_range
    v_center_min = v_min + edge_trim_ratio * v_range
    v_center_max = v_max - edge_trim_ratio * v_range
    
    # 构建网格字典：(gu, gv) -> {'min': val, 'max': val, 'count': int}
    grid_stats = {}
    
    for i in range(len(pore_coords)):
        u, v = coords_u[i], coords_v[i]
        val = coords_along_normal[i]
        r = pore_radii[i]
        
        # 只处理中间区域的孔（排除边缘）
        if u < u_center_min or u > u_center_max or v < v_center_min or v > v_center_max:
            continue
        
        # 物理边界（考虑孔半径）
        val_min = val - r
        val_max = val + r
        
        gu = int((u - u_min) / cell_size)
        gv = int((v - v_min) / cell_size)
        key = (gu, gv)
        
        if key not in grid_stats:
            grid_stats[key] = {'min': val_min, 'max': val_max, 'count': 1}
        else:
            if val_min < grid_stats[key]['min']:
                grid_stats[key]['min'] = val_min
            if val_max > grid_stats[key]['max']:
                grid_stats[key]['max'] = val_max
            grid_stats[key]['count'] += 1
    
    # 计算每个网格的厚度（只保留有足够孔的网格，进一步减少边缘误差）
    local_thicknesses = []
    
    for stat in grid_stats.values():
        if stat['count'] < min_pores_per_cell:
            continue  # 跳过孔数太少的网格（可能是边缘或稀疏区域）
        t = stat['max'] - stat['min']
        if t > 0:
            local_thicknesses.append(t)
    
    local_thicknesses = np.array(local_thicknesses) if local_thicknesses else np.array([])
    mean_thickness = float(np.mean(local_thicknesses)) if len(local_thicknesses) > 0 else 0.0
    
    return mean_thickness, local_thicknesses


def _load_corrected_thickness_from_summary(thickness_file: Path) -> float | None:
    """读取 sub_analyze 写入的 Average Thickness (nm)（已含 θ>阈值时 ×cosθ 倾斜修正）。失败返回 None。"""
    if not thickness_file.exists():
        return None
    try:
        tdf = pd.read_excel(thickness_file)
        if "Average Thickness (nm)" not in tdf.columns or len(tdf) < 1:
            return None
        tv = float(tdf["Average Thickness (nm)"].iloc[0])
        if np.isfinite(tv) and tv > 0:
            return tv
    except Exception:
        pass
    return None


def collect_thickness_data(
    split_samples_dir: Path,
    analysis_dir: Path,
    *,
    use_fine_scale: bool = False,
    cell_size: float = 10.0,
    edge_trim_ratio: float = 0.1,
    min_pores_per_cell: int =20,
) -> pd.DataFrame:
    """
    汇总所有子样本的厚度数据（WT / AS）。

    主列 ``thickness`` 一律优先为 **修正后厚度**：各子样本分析目录下
    ``{sub}_thickness_summary.xlsx`` 的 **Average Thickness (nm)**（与 sub_analyze 一致，含倾斜修正）。

    Parameters:
    -----------
    use_fine_scale : bool
        若 True，仍从球棍模型计算 **local_thicknesses**（网格局部厚度，供导出参考），
        但 ``thickness`` 仍以汇总表修正值为准（若文件存在）；否则回退为球棍模型均值。
    cell_size : float
        精细厚度计算的网格大小 (nm)，仅在 use_fine_scale=True 时使用
    edge_trim_ratio : float
        边缘排除比例（默认 0.1，即排除边缘 0.10 的区域），仅在 use_fine_scale=True 时使用
    min_pores_per_cell : int
        每个网格至少需要的孔数（默认 10），少于该数的网格会被过滤掉，仅在 use_fine_scale=True 时使用

    Returns:
    --------
    DataFrame with columns: sample_name, sub_name, thickness, is_wt
    如果 use_fine_scale=True，还会包含 local_thicknesses（每个网格的厚度数组，未单独做 cosθ 标量修正）
    """
    split_samples_dir = Path(split_samples_dir)
    analysis_dir = Path(analysis_dir)
    rows = []
    
    if use_fine_scale:
        # 网格局部厚度 + 主厚度仍以 thickness_summary 的 Average（修正后）为准
        print(
            "精细模式：从球棍模型计算 local_thicknesses；"
            "主厚度 thickness 优先读 analysis 目录下 *_thickness_summary.xlsx 的 Average Thickness（修正后）…"
        )
        for sample_dir in sorted(split_samples_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            sample_name = sample_dir.name
            is_wt = not sample_name.upper().startswith("AS")
            
            sub_list = find_sub_samples(sample_dir, sample_name)
            if not sub_list:
                continue
            
            for sub_name, pores_file, throats_file in sub_list:
                # 读取法向量
                meta_file = sample_dir / f"{sub_name}_metadata.json"
                normal_vec = None
                if meta_file.exists():
                    try:
                        import json
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        nv = meta.get("normal_vector")
                        if nv is not None:
                            nx = float(nv.get("nx", 0.0))
                            ny = float(nv.get("ny", 0.0))
                            nz = float(nv.get("nz", 0.0))
                            normal = np.array([nx, ny, nz], dtype=float)
                            n_norm = np.linalg.norm(normal)
                            if n_norm > 1e-12:
                                normal_vec = normal / n_norm
                    except Exception as e:
                        print(f"  跳过 {sub_name}（读取法向量失败）: {e}")
                        continue
                
                if normal_vec is None:
                    print(f"  跳过 {sub_name}（无法向量）")
                    continue
                
                # 加载孔数据
                try:
                    pore_coords, pore_radii, _, _, _, _, _ = st.load_pores_and_throats(
                        pores_file, throats_file
                    )
                except Exception as e:
                    print(f"  跳过 {sub_name}（加载孔数据失败）: {e}")
                    continue
                
                # 计算精细厚度
                try:
                    mean_thickness, local_thicknesses = calculate_thickness_from_pore_model(
                        pore_coords, pore_radii, normal_vec,
                        cell_size=cell_size,
                        edge_trim_ratio=edge_trim_ratio,
                        min_pores_per_cell=min_pores_per_cell,
                    )
                    thickness_file = (
                        analysis_dir / sample_name / sub_name / f"{sub_name}_thickness_summary.xlsx"
                    )
                    corrected = _load_corrected_thickness_from_summary(thickness_file)
                    report_thickness = (
                        float(corrected)
                        if corrected is not None
                        else float(mean_thickness)
                    )
                    row_fs = {
                        "sample_name": sample_name,
                        "sub_name": sub_name,
                        "thickness": report_thickness,
                        "is_wt": is_wt,
                        "nx": float(normal_vec[0]),
                        "ny": float(normal_vec[1]),
                        "nz": float(normal_vec[2]),
                        "local_thicknesses": local_thicknesses,
                        "n_local_samples": len(local_thicknesses),
                        "thickness_fine_grid_mean_nm": float(mean_thickness),
                    }
                    rows.append(row_fs)
                except Exception as e:
                    print(f"  跳过 {sub_name}（计算厚度失败）: {e}")
                    continue
    else:
        # collect_all_sub_samples_with_normal 内 load_thickness 已读 Average Thickness (nm)（修正后）
        print("从各子样本 thickness_summary 读取 Average Thickness (nm)（含倾斜修正）…")
        df_all = pd.DataFrame(
            collect_all_sub_samples_with_normal(split_samples_dir, analysis_dir, min_throat_radius=None)
        )
        # 若汇总表中已经带有 normal_vector 组件，则一并保留，方便后续角度统计
        base_cols = ["sample_name", "sub_name", "thickness", "is_wt"]
        extra_cols = [c for c in ("nx", "ny", "nz", "theta_deg_vs_xy") if c in df_all.columns]
        use_cols = base_cols + extra_cols
        rows = df_all[use_cols].to_dict("records")

    df = pd.DataFrame(rows)
    return df


def summarize_permeation_direction_angles(df: pd.DataFrame, out_dir: Path) -> None:
    """
    按样本汇总渗透方向与 XY 平面的夹角 θ，并输出 CSV 与柱状图（按 θ 升序）。

    优先使用子样本 thickness_summary 中读入的 theta_deg_vs_xy（与 sub_analyze 写入一致）；
    对多样本子样本取平均，标准差作为误差条（仅 n>1 时）。
    若无该列，则回退为对各子样本 nx,ny,nz 取平均后再算 θ=arcsin(|nz|)。
    """
    out_dir = Path(out_dir)
    if "sample_name" not in df.columns:
        print("  [phase1 厚度] 缺少 sample_name，跳过渗透方向夹角汇总。")
        return

    sample_names = sorted(df["sample_name"].dropna().astype(str).unique())
    has_theta = "theta_deg_vs_xy" in df.columns
    has_nxyz = {"nx", "ny", "nz"}.issubset(df.columns)

    records: list[dict] = []
    for sample_name in sample_names:
        g = df[df["sample_name"].astype(str) == str(sample_name)]
        angle_source = None
        theta_deg = np.nan
        theta_std = 0.0
        n_theta = 0
        nx_mean = ny_mean = nz_mean = np.nan
        abs_nz = np.nan

        if has_theta:
            ts = pd.to_numeric(g["theta_deg_vs_xy"], errors="coerce").dropna()
            if len(ts) > 0:
                theta_deg = float(ts.mean())
                theta_std = float(ts.std()) if len(ts) > 1 else 0.0
                if not np.isfinite(theta_std):
                    theta_std = 0.0
                n_theta = int(len(ts))
                angle_source = "thickness_summary_theta"

        if not np.isfinite(theta_deg) and has_nxyz:
            sub = g[["nx", "ny", "nz"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) > 0:
                mean_vec = sub.to_numpy(dtype=float).mean(axis=0)
                norm = float(np.linalg.norm(mean_vec))
                if np.isfinite(norm) and norm > 1e-9:
                    n_hat = mean_vec / norm
                    nx_mean, ny_mean, nz_mean = float(n_hat[0]), float(n_hat[1]), float(n_hat[2])
                    abs_nz = float(abs(n_hat[2]))
                    nz_clipped = float(np.clip(abs_nz, 0.0, 1.0))
                    theta_deg = float(np.degrees(np.arcsin(nz_clipped)))
                    theta_std = 0.0
                    n_theta = int(len(sub))
                    angle_source = "mean_nx_ny_nz"

        if not np.isfinite(theta_deg):
            continue

        if not np.isfinite(nx_mean) and has_nxyz:
            sub = g[["nx", "ny", "nz"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) > 0:
                mean_vec = sub.to_numpy(dtype=float).mean(axis=0)
                norm = float(np.linalg.norm(mean_vec))
                if np.isfinite(norm) and norm > 1e-9:
                    n_hat = mean_vec / norm
                    nx_mean, ny_mean, nz_mean = float(n_hat[0]), float(n_hat[1]), float(n_hat[2])
                    abs_nz = float(abs(n_hat[2]))

        records.append(
            {
                "sample_name": sample_name,
                "theta_deg_vs_xy": theta_deg,
                "theta_deg_vs_xy_std": theta_std,
                "n_subsamples_used": n_theta,
                "angle_source": angle_source,
                "nx_mean": nx_mean,
                "ny_mean": ny_mean,
                "nz_mean": nz_mean,
                "abs_nz": abs_nz,
            }
        )

    if not records:
        print("  [phase1 厚度] 无有效角度数据（缺 theta_deg_vs_xy 且无法从 nx/ny/nz 推算），跳过渗透方向夹角汇总。")
        return

    summ = pd.DataFrame(records)
    summ.sort_values("theta_deg_vs_xy", inplace=True, ignore_index=True)

    csv_path = out_dir / "permeation_direction_angles_by_sample.csv"
    summ.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  [phase1 厚度] 渗透方向夹角汇总已保存: {csv_path}")

    fig, ax = plt.subplots(figsize=(max(6.0, 0.45 * len(summ)), 4.2))
    x = np.arange(len(summ))
    theta = summ["theta_deg_vs_xy"].to_numpy(dtype=float)
    yerr = summ["theta_deg_vs_xy_std"].to_numpy(dtype=float)
    yerr = np.where(np.isfinite(yerr) & (yerr > 0), yerr, 0.0)
    colors = np.where(theta > TILT_REFINE_DEG_REFERENCE, "#d95f0e", "#1b9e77")
    ax.bar(
        x,
        theta,
        yerr=yerr,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
        capsize=2.0,
        error_kw={"elinewidth": 0.8, "capthick": 0.8},
    )
    ax.axhline(
        TILT_REFINE_DEG_REFERENCE,
        color="crimson",
        linestyle="--",
        linewidth=1.0,
        label=f"{TILT_REFINE_DEG_REFERENCE:.0f}° 阈值",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summ["sample_name"].astype(str).tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("渗透方向与 XY 平面夹角 θ (deg)")
    ax.set_xlabel("样本")
    n_from_file = (summ["angle_source"] == "thickness_summary_theta").sum()
    title_note = (
        "（θ 来自各子样本 thickness_summary；多样本取均值±标差）"
        if n_from_file > 0
        else "（θ 由各子样本法向均值推算）"
    )
    ax.set_title("每个样本的渗透方向与 XY 平面夹角（按 θ 升序）" + title_note)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig_path = out_dir / "permeation_direction_angles_by_sample.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [phase1 厚度] 渗透方向夹角图已保存: {fig_path}")


def visualize_thickness_distribution(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    use_fine_scale: bool = True,
) -> None:
    """
    对 WT、AS 分别画直方图 + 核密度估计（KDE）；
    报告均值、方差、分位数；
    判断 AS 是否明显多峰。

    厚度数据取 ``df['thickness']``（每子样本一条，为 thickness_summary 的修正后主厚度）。
    ``use_fine_scale`` 仅保留调用兼容，不再用局部网格池化参与分布。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 分布与统计统一用每子样本主列 thickness（修正后 Average），不用局部网格池化（避免与 cosθ 修正不一致）
    wt_thickness = df[df["is_wt"]]["thickness"].dropna().to_numpy(dtype=float)
    as_thickness = df[~df["is_wt"]]["thickness"].dropna().to_numpy(dtype=float)

    # 统计信息
    stats_dict = {}
    for name, data in [("WT", wt_thickness), ("AS", as_thickness)]:
        if len(data) == 0:
            print(f"警告：{name} 样本数为 0，跳过")
            continue
        stats_dict[name] = {
            "count": len(data),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "median": float(np.median(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        }
        print(f"\n{name} 厚度统计:")
        print(f"  样本数: {stats_dict[name]['count']}")
        print(f"  均值: {stats_dict[name]['mean']:.2f} nm")
        print(f"  标准差: {stats_dict[name]['std']:.2f} nm")
        print(f"  中位数: {stats_dict[name]['median']:.2f} nm")
        print(f"  0.25 分位数: {stats_dict[name]['q25']:.2f} nm")
        print(f"  0.75 分位数: {stats_dict[name]['q75']:.2f} nm")
        print(f"  范围: [{stats_dict[name]['min']:.2f}, {stats_dict[name]['max']:.2f}] nm")

    # 保存统计表
    stats_df = pd.DataFrame(stats_dict).T
    stats_path = out_dir / "thickness_statistics.xlsx"
    stats_df.to_excel(stats_path, index=True)
    print(f"\n统计表已保存: {stats_path}")

    # 绘制分布图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("厚度分布分析（WT vs AS）", fontsize=16, fontweight="bold")

    # WT 分布
    if len(wt_thickness) > 0:
        ax = axes[0, 0]
        ax.hist(
            wt_thickness,
            bins=30,
            density=True,
            alpha=0.6,
            color="blue",
            label="直方图",
        )
        # KDE
        kde_wt = gaussian_kde(wt_thickness)
        x_wt = np.linspace(wt_thickness.min(), wt_thickness.max(), 200)
        ax.plot(x_wt, kde_wt(x_wt), "b-", linewidth=2, label="KDE")
        ax.axvline(
            stats_dict["WT"]["mean"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"均值 = {stats_dict['WT']['mean']:.2f} nm",
        )
        ax.axvline(
            stats_dict["WT"]["median"],
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"中位数 = {stats_dict['WT']['median']:.2f} nm",
        )
        ax.set_xlabel("厚度 (nm)", fontsize=12)
        ax.set_ylabel("概率密度", fontsize=12)
        ax.set_title(f"WT 厚度分布 (n={len(wt_thickness)})", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # AS 分布
    if len(as_thickness) > 0:
        ax = axes[0, 1]
        ax.hist(
            as_thickness,
            bins=30,
            density=True,
            alpha=0.6,
            color="orange",
            label="直方图",
        )
        # KDE
        kde_as = gaussian_kde(as_thickness)
        x_as = np.linspace(as_thickness.min(), as_thickness.max(), 200)
        ax.plot(x_as, kde_as(x_as), "orange", linewidth=2, label="KDE")
        ax.axvline(
            stats_dict["AS"]["mean"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"均值 = {stats_dict['AS']['mean']:.2f} nm",
        )
        ax.axvline(
            stats_dict["AS"]["median"],
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"中位数 = {stats_dict['AS']['median']:.2f} nm",
        )
        ax.set_xlabel("厚度 (nm)", fontsize=12)
        ax.set_ylabel("概率密度", fontsize=12)
        ax.set_title(f"AS 厚度分布 (n={len(as_thickness)})", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 对比图
    ax = axes[1, 0]
    if len(wt_thickness) > 0:
        ax.hist(
            wt_thickness,
            bins=30,
            density=True,
            alpha=0.5,
            color="blue",
            label="WT",
        )
        x_wt = np.linspace(wt_thickness.min(), wt_thickness.max(), 200)
        ax.plot(x_wt, kde_wt(x_wt), "b-", linewidth=2)
    if len(as_thickness) > 0:
        ax.hist(
            as_thickness,
            bins=30,
            density=True,
            alpha=0.5,
            color="orange",
            label="AS",
        )
        x_as = np.linspace(as_thickness.min(), as_thickness.max(), 200)
        ax.plot(x_as, kde_as(x_as), "orange", linewidth=2)
    ax.set_xlabel("厚度 (nm)", fontsize=12)
    ax.set_ylabel("概率密度", fontsize=12)
    ax.set_title("WT vs AS 厚度分布对比", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 箱线图
    ax = axes[1, 1]
    data_to_plot = []
    labels = []
    if len(wt_thickness) > 0:
        data_to_plot.append(wt_thickness)
        labels.append("WT")
    if len(as_thickness) > 0:
        data_to_plot.append(as_thickness)
        labels.append("AS")
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        bp["boxes"][0].set_facecolor("lightblue")
        if len(bp["boxes"]) > 1:
            bp["boxes"][1].set_facecolor("lightcoral")
        ax.set_ylabel("厚度 (nm)", fontsize=12)
        ax.set_title("WT vs AS 厚度箱线图", fontsize=14)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = out_dir / "thickness_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"分布图已保存: {plot_path}")

    return stats_dict


def plot_corrected_thickness_boxplot_by_sample(df: pd.DataFrame, out_dir: Path) -> Path | None:
    """
    每个样本一个箱线图，展示该样本下各子样本的修正后厚度（thickness 列）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "sample_name" not in df.columns or "thickness" not in df.columns:
        print("  [phase1 厚度] 缺少 sample_name/thickness 列，跳过每样本箱线图。")
        return None

    dff = df[["sample_name", "thickness", "is_wt"]].copy()
    dff["sample_name"] = dff["sample_name"].astype(str)
    dff["thickness"] = pd.to_numeric(dff["thickness"], errors="coerce")
    dff = dff.dropna(subset=["sample_name", "thickness"])
    if dff.empty:
        print("  [phase1 厚度] 无有效厚度数据，跳过每样本箱线图。")
        return None

    sample_order = (
        dff[["sample_name", "is_wt"]]
        .drop_duplicates()
        .sort_values(["is_wt", "sample_name"], ascending=[False, True])["sample_name"]
        .tolist()
    )
    if not sample_order:
        return None

    data_to_plot = []
    kept_samples = []
    box_colors = []
    sample_counts = []
    for sn in sample_order:
        vals = dff.loc[dff["sample_name"] == sn, "thickness"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        kept_samples.append(sn)
        data_to_plot.append(vals)
        sample_counts.append(int(vals.size))
        is_wt = bool(dff.loc[dff["sample_name"] == sn, "is_wt"].iloc[0])
        box_colors.append("lightblue" if is_wt else "lightcoral")

    if not data_to_plot:
        print("  [phase1 厚度] 所有样本厚度均为空，跳过每样本箱线图。")
        return None

    fig_w = max(10.0, 0.5 * len(data_to_plot))
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    bp = ax.boxplot(data_to_plot, labels=kept_samples, patch_artist=True, showfliers=True)
    for b, c in zip(bp["boxes"], box_colors):
        b.set_facecolor(c)
        b.set_alpha(0.8)

    ax.set_ylabel("Corrected thickness (nm)", fontsize=12)
    ax.set_xlabel("Sample", fontsize=12)
    ax.set_title("Corrected thickness by sample (one box per sample)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # 在 x 轴标签后附上子样本数量，便于阅读每个箱线图的统计稳健性。
    tick_labels = [f"{s}\n(n={n})" for s, n in zip(kept_samples, sample_counts)]
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    legend_handles = [
        plt.Line2D([0], [0], color="lightblue", lw=8, label="WT"),
        plt.Line2D([0], [0], color="lightcoral", lw=8, label="AS"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "corrected_thickness_boxplot_by_sample.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"每样本修正后厚度箱线图已保存: {out_path}")
    return out_path


def detect_multimodality(data: np.ndarray, max_components: int = 3) -> dict:
    """
    检测数据是否多峰（使用高斯混合模型）。
    使用 BIC 选择最佳组件数（比 AIC 惩罚更重，倾向于更少组件）。
    返回：最佳组件数、AIC/BIC、拟合参数。
    """
    if len(data) < 10:
        return {"n_components": 1, "is_multimodal": False, "aic": None, "bic": None}

    aic_scores = []
    bic_scores = []
    best_n = 1
    best_bic = np.inf

    for n in range(1, min(max_components + 1, len(data) // 5 + 1)):
        try:
            gm = GaussianMixture(n_components=n, random_state=42, max_iter=100)
            gm.fit(data.reshape(-1, 1))
            aic = gm.aic(data.reshape(-1, 1))
            bic = gm.bic(data.reshape(-1, 1))
            aic_scores.append(aic)
            bic_scores.append(bic)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        except Exception as e:
            print(f"  警告：拟合 {n} 组件 GMM 失败: {e}")
            break

    is_multimodal = best_n > 1
    result = {
        "n_components": best_n,
        "is_multimodal": is_multimodal,
        "aic_scores": aic_scores,
        "bic_scores": bic_scores,
    }

    if is_multimodal:
        gm_best = GaussianMixture(n_components=best_n, random_state=42, max_iter=100)
        gm_best.fit(data.reshape(-1, 1))
        result["weights"] = gm_best.weights_.tolist()
        result["means"] = gm_best.means_.flatten().tolist()
        result["covariances"] = gm_best.covariances_.flatten().tolist()

    return result


def _fitted_pdf(x: np.ndarray, result: dict, data: np.ndarray) -> np.ndarray:
    """
    计算拟合分布在 x 处的 PDF 值。
    result 来自 parameterize_thickness_distribution 的 results[name]。
    data 仅用于 KDE 类型时重新拟合。
    """
    x = np.asarray(x, dtype=float)
    if result["type"] == "gmm":
        w = np.array(result["weights"])
        mu = np.array(result["means"])
        cov = np.array(result["covariances"])
        sigma = np.sqrt(cov)
        pdf = np.zeros_like(x)
        for k in range(len(w)):
            pdf += w[k] * stats.norm.pdf(x, loc=mu[k], scale=sigma[k])
        return pdf
    if result["type"] == "parametric":
        dist_name = result["distribution"]
        params = result["params"]
        if dist_name == "gamma":
            return stats.gamma.pdf(x, *params)
        if dist_name == "lognorm":
            return stats.lognorm.pdf(x, *params)
        if dist_name == "norm":
            return stats.norm.pdf(x, *params)
        if dist_name == "weibull_min":
            return stats.weibull_min.pdf(x, *params)
        return np.zeros_like(x)
    if result["type"] == "kde":
        kde = gaussian_kde(data)
        return kde(x)
    return np.zeros_like(x)


def plot_fitted_vs_data(
    results: dict,
    wt_thickness: np.ndarray,
    as_thickness: np.ndarray,
    out_dir: Path,
) -> None:
    """
    绘制拟合分布与真实数据的对比图（直方图 + 拟合曲线）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (name, data) in enumerate([("WT", wt_thickness), ("AS", as_thickness)]):
        ax = axes[idx]
        if name not in results or len(data) == 0:
            ax.set_title(f"{name}（无数据）")
            continue

        # 直方图（归一化密度）
        ax.hist(data, bins=30, density=True, alpha=0.5, color="steelblue", edgecolor="white", label="真实数据")

        result = results[name]
        x_min, x_max = data.min(), data.max()
        x_pad = max((x_max - x_min) * 0.1, 1.0)
        x_plot = np.linspace(max(0, x_min - x_pad), x_max + x_pad, 300)

        # 多组件（如 GMM）：先画各分量，再画整体
        if result["type"] == "gmm":
            w = np.array(result["weights"])
            mu = np.array(result["means"])
            cov = np.array(result["covariances"])
            sigma = np.sqrt(cov)
            colors = ["green", "purple", "orange", "brown"]  # 分量用不同颜色
            for k in range(len(w)):
                comp_pdf = w[k] * stats.norm.pdf(x_plot, loc=mu[k], scale=sigma[k])
                ax.plot(
                    x_plot, comp_pdf,
                    linestyle="--", linewidth=1.5, color=colors[k % len(colors)],
                    label=f"组件 {k+1} (w={w[k]:.2f}, mu={mu[k]:.0f})",
                )
            pdf_vals = _fitted_pdf(x_plot, result, data)
            ax.plot(x_plot, pdf_vals, "r-", linewidth=2, label="拟合整体")
        else:
            pdf_vals = _fitted_pdf(x_plot, result, data)
            ax.plot(x_plot, pdf_vals, "r-", linewidth=2, label="拟合分布")

        ax.set_xlabel("厚度 (nm)", fontsize=12)
        ax.set_ylabel("概率密度", fontsize=12)
        fit_desc = result.get("distribution", result.get("type", ""))
        if result["type"] == "gmm":
            fit_desc = f"GMM ({result['n_components']} 组件)"
        elif result["type"] == "parametric":
            fit_desc = result["distribution"]
        ax.set_title(f"{name} 厚度：拟合 vs 真实 (n={len(data)})\n拟合: {fit_desc}", fontsize=12)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plot_path = out_dir / "thickness_fitted_vs_data.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"拟合 vs 真实数据对比图已保存: {plot_path}")


def parameterize_thickness_distribution(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    use_fine_scale: bool = True,
) -> dict:
    """
    为后续"按厚度采样"准备：
    - 若单峰则拟合为参数分布（Gamma / Lognormal）
    - 若多峰则采用混合分布或分段 KDE
    返回：分布参数字典，包含采样函数。

    拟合数据为 ``df['thickness']``（修正后主厚度）。``use_fine_scale`` 仅保留调用兼容。

    单峰时：在 gamma/lognorm/norm/weibull_min 上同时计算 AIC 与 BIC；若二者最小值对应同一分布则采用，
    否则采用 BIC 最优（``parametric_selection``: ``aic_bic_agree`` / ``bic_tiebreak``）。候选全部失败则 KDE。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wt_thickness = df[df["is_wt"]]["thickness"].dropna().to_numpy(dtype=float)
    as_thickness = df[~df["is_wt"]]["thickness"].dropna().to_numpy(dtype=float)

    distributions_to_test = [
        ("gamma", stats.gamma),
        ("lognorm", stats.lognorm),
        ("norm", stats.norm),
        ("weibull_min", stats.weibull_min),
    ]

    results = {}

    for name, data in [("WT", wt_thickness), ("AS", as_thickness)]:
        if len(data) == 0:
            continue

        print(f"\n=== {name} 厚度分布参数化 ===")

        # 检测多峰性
        multimodality = detect_multimodality(data)
        print(f"多峰检测结果（BIC 选组件数）:")
        print(f"  最佳组件数: {multimodality['n_components']}")
        print(f"  是否多峰: {multimodality['is_multimodal']}")

        if multimodality["is_multimodal"]:
            # 多峰：使用混合分布或 KDE
            print(f"  检测到多峰分布，使用高斯混合模型")
            n_components = multimodality["n_components"]
            gm = GaussianMixture(n_components=n_components, random_state=42, max_iter=100)
            gm.fit(data.reshape(-1, 1))

            def sample_gmm(size=1):
                return gm.sample(size)[0].flatten()

            results[name] = {
                "type": "gmm",
                "n_components": n_components,
                "weights": gm.weights_.tolist(),
                "means": gm.means_.flatten().tolist(),
                "covariances": gm.covariances_.flatten().tolist(),
                "sample": sample_gmm,
            }

            # 也保存 KDE 作为备选
            kde = gaussian_kde(data)

            def sample_kde(size=1):
                return kde.resample(size).flatten()

            results[name]["kde_sample"] = sample_kde

        else:
            # 单峰：在候选族上同时算 AIC 与 BIC；若 AIC 最小与 BIC 最小为同一分布则采用，否则采用 BIC 最优（更惩罚复杂度）
            print(f"  单峰分布，尝试拟合参数分布（AIC+BIC 联合选型）")
            n_obs = int(len(data))
            log_n = float(np.log(n_obs)) if n_obs > 1 else 0.0

            candidates = []
            for dist_name, dist_class in distributions_to_test:
                try:
                    params = dist_class.fit(data)
                    k = len(params)
                    loglik = float(np.sum(dist_class.logpdf(data, *params)))
                    aic = -2.0 * loglik + 2.0 * k
                    bic = -2.0 * loglik + float(k) * log_n
                    candidates.append(
                        {
                            "dist_name": dist_name,
                            "dist_class": dist_class,
                            "params": params,
                            "aic": aic,
                            "bic": bic,
                            "k": k,
                        }
                    )
                except Exception as e:
                    print(f"    警告：拟合 {dist_name} 失败: {e}")
                    continue

            if len(candidates) > 0:
                best_aic_c = min(candidates, key=lambda c: c["aic"])
                best_bic_c = min(candidates, key=lambda c: c["bic"])
                if best_aic_c["dist_name"] == best_bic_c["dist_name"]:
                    chosen = best_aic_c
                    selection = "aic_bic_agree"
                    print(
                        f"  AIC 与 BIC 均指向: {chosen['dist_name']} "
                        f"(AIC={chosen['aic']:.2f}, BIC={chosen['bic']:.2f})"
                    )
                else:
                    chosen = best_bic_c
                    selection = "bic_tiebreak"
                    print(
                        f"  AIC 最佳: {best_aic_c['dist_name']} (AIC={best_aic_c['aic']:.2f}) | "
                        f"BIC 最佳: {best_bic_c['dist_name']} (BIC={best_bic_c['bic']:.2f}) → 采用 BIC"
                    )
                    print(
                        f"  选定: {chosen['dist_name']} (AIC={chosen['aic']:.2f}, BIC={chosen['bic']:.2f})"
                    )
                best_dist = chosen["dist_class"]
                best_params = chosen["params"]
                print(f"  参数: {best_params}")

                def sample_dist(size=1):
                    return best_dist.rvs(*best_params, size=size)

                results[name] = {
                    "type": "parametric",
                    "distribution": chosen["dist_name"],
                    "params": [float(p) for p in chosen["params"]],
                    "aic": float(chosen["aic"]),
                    "bic": float(chosen["bic"]),
                    "parametric_selection": selection,
                    "sample": sample_dist,
                }
            else:
                # 如果所有参数分布都失败，使用 KDE
                print(f"  参数分布拟合失败，使用 KDE")
                kde = gaussian_kde(data)

                def sample_kde(size=1):
                    return kde.resample(size).flatten()

                results[name] = {
                    "type": "kde",
                    "sample": sample_kde,
                }

        # 验证采样
        if "sample" in results[name]:
            test_samples = results[name]["sample"](1000)
            print(f"  采样验证: 均值 = {np.mean(test_samples):.2f}, 标准差 = {np.std(test_samples):.2f}")
            print(f"  原始数据: 均值 = {np.mean(data):.2f}, 标准差 = {np.std(data):.2f}")

    # 额外导出固定口径，便于下游显式切换：
    # - WT_norm / WT_kde / WT_gmm
    # - AS_kde / AS_gmm
    if len(wt_thickness) > 0:
        try:
            wt_mu, wt_sigma = stats.norm.fit(wt_thickness)

            def sample_wt_norm(size=1):
                return stats.norm.rvs(loc=wt_mu, scale=wt_sigma, size=size)

            results["WT_norm"] = {
                "type": "parametric",
                "distribution": "norm",
                "params": [float(wt_mu), float(wt_sigma)],
                "source": "forced_norm_fit_for_wt",
                "sample": sample_wt_norm,
            }
            print(f"[WT_norm] 已导出：mu={wt_mu:.4f}, sigma={wt_sigma:.4f}")
        except Exception as e:
            print(f"[WT_norm] 导出失败：{e}")

        try:
            kde_wt_fixed = gaussian_kde(wt_thickness)

            def sample_wt_kde(size=1):
                return kde_wt_fixed.resample(size).flatten()

            results["WT_kde"] = {
                "type": "kde",
                "source": "forced_kde_for_wt",
                "kde_data": [float(x) for x in wt_thickness.tolist()],
                "sample": sample_wt_kde,
            }
            print(f"[WT_kde] 已导出：n_data={len(wt_thickness)}")
        except Exception as e:
            print(f"[WT_kde] 导出失败：{e}")

        try:
            wt_mm = detect_multimodality(wt_thickness)
            wt_n_comp = int(max(1, wt_mm.get("n_components", 1)))
            gm_wt_fixed = GaussianMixture(n_components=wt_n_comp, random_state=42, max_iter=200)
            gm_wt_fixed.fit(wt_thickness.reshape(-1, 1))

            def sample_wt_gmm(size=1):
                return gm_wt_fixed.sample(size)[0].flatten()

            results["WT_gmm"] = {
                "type": "gmm",
                "source": "forced_gmm_for_wt",
                "n_components": wt_n_comp,
                "weights": gm_wt_fixed.weights_.tolist(),
                "means": gm_wt_fixed.means_.flatten().tolist(),
                "covariances": gm_wt_fixed.covariances_.flatten().tolist(),
                "sample": sample_wt_gmm,
            }
            print(f"[WT_gmm] 已导出：n_components={wt_n_comp}")
        except Exception as e:
            print(f"[WT_gmm] 导出失败：{e}")

    if len(as_thickness) > 0:
        try:
            kde_as_fixed = gaussian_kde(as_thickness)

            def sample_as_kde(size=1):
                return kde_as_fixed.resample(size).flatten()

            results["AS_kde"] = {
                "type": "kde",
                "source": "forced_kde_for_as",
                "kde_data": [float(x) for x in as_thickness.tolist()],
                "sample": sample_as_kde,
            }
            print(f"[AS_kde] 已导出：n_data={len(as_thickness)}")
        except Exception as e:
            print(f"[AS_kde] 导出失败：{e}")

        try:
            as_mm = detect_multimodality(as_thickness)
            as_n_comp = int(max(1, as_mm.get("n_components", 1)))
            gm_as_fixed = GaussianMixture(n_components=as_n_comp, random_state=42, max_iter=200)
            gm_as_fixed.fit(as_thickness.reshape(-1, 1))

            def sample_as_gmm(size=1):
                return gm_as_fixed.sample(size)[0].flatten()

            results["AS_gmm"] = {
                "type": "gmm",
                "source": "forced_gmm_for_as",
                "n_components": as_n_comp,
                "weights": gm_as_fixed.weights_.tolist(),
                "means": gm_as_fixed.means_.flatten().tolist(),
                "covariances": gm_as_fixed.covariances_.flatten().tolist(),
                "sample": sample_as_gmm,
            }
            print(f"[AS_gmm] 已导出：n_components={as_n_comp}")
        except Exception as e:
            print(f"[AS_gmm] 导出失败：{e}")

    # 拟合分布与真实数据对比图
    plot_fitted_vs_data(results, wt_thickness, as_thickness, out_dir)

    # 保存参数（JSON，不包含函数）
    params_to_save = {}
    for name, result in results.items():
        params_to_save[name] = {k: v for k, v in result.items() if k != "sample" and k != "kde_sample"}

    params_path = out_dir / "thickness_distribution_parameters.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params_to_save, f, indent=2, ensure_ascii=False)
    print(f"\n分布参数已保存: {params_path}")

    # 保存采样函数到 Python 模块（用于后续调用）
    module_path = out_dir / "thickness_sampler.py"
    with open(module_path, "w", encoding="utf-8") as f:
        f.write('"""厚度采样模块 - 自动生成"""\n\n')
        f.write("import numpy as np\n")
        f.write("from scipy import stats\n")
        f.write("from scipy.stats import gaussian_kde\n")
        f.write("from sklearn.mixture import GaussianMixture\n\n\n")

        for name, result in results.items():
            f.write(f"def sample_{name.lower()}_thickness(size=1):\n")
            if result["type"] == "gmm":
                f.write(f'    """采样 {name} 厚度（高斯混合模型，{result["n_components"]} 组件）"""\n')
                f.write(f'    gm = GaussianMixture(n_components={result["n_components"]}, random_state=42, max_iter=100)\n')
                f.write(f'    gm.weights_ = np.array({result["weights"]})\n')
                f.write(f'    gm.means_ = np.array({result["means"]}).reshape(-1, 1)\n')
                f.write(f'    gm.covariances_ = np.array({result["covariances"]}).reshape(-1, 1, 1)\n')
                f.write("    gm.precisions_cholesky_ = np.linalg.cholesky(1.0 / gm.covariances_)\n")
                f.write("    return gm.sample(size)[0].flatten()\n\n")
            elif result["type"] == "parametric":
                dist_name = result["distribution"]
                params = result["params"]
                f.write(f'    """采样 {name} 厚度（{dist_name} 分布）"""\n')
                if dist_name == "gamma":
                    f.write(f"    return stats.gamma.rvs({params[0]}, loc={params[1]}, scale={params[2]}, size=size)\n")
                elif dist_name == "lognorm":
                    f.write(f"    return stats.lognorm.rvs({params[0]}, loc={params[1]}, scale={params[2]}, size=size)\n")
                elif dist_name == "norm":
                    f.write(f"    return stats.norm.rvs(loc={params[0]}, scale={params[1]}, size=size)\n")
                elif dist_name == "weibull_min":
                    f.write(f"    return stats.weibull_min.rvs({params[0]}, loc={params[1]}, scale={params[2]}, size=size)\n")
                f.write("\n")
            elif result["type"] == "kde":
                f.write(f'    """采样 {name} 厚度（KDE）"""\n')
                kde_data = result.get("kde_data")
                if kde_data:
                    f.write(f"    _data = np.array({kde_data}, dtype=float)\n")
                    f.write("    _kde = gaussian_kde(_data)\n")
                    f.write("    return _kde.resample(size).flatten()\n\n")
                else:
                    f.write("    # 注意：KDE 采样需要原始数据，这里返回空数组（需在运行时提供数据）\n")
                    f.write("    raise NotImplementedError('KDE sampling requires original data')\n\n")

    print(f"采样模块已保存: {module_path}")

    return results


def main():
    _parent_dir = WORKSPACE_ROOT

    parser = argparse.ArgumentParser(
        description="阶段一：厚度分布描述（WT / AS）"
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
        default=str(OUTPUT_ROOT / "phase1_thickness"),
        help="输出目录（默认：项目 outputs/phase1_thickness）",
    )
    parser.add_argument(
        "--fine-scale",
        dest="use_fine_scale",
        action="store_true",
        help=(
            "额外从球棍模型计算网格局部厚度并导出；"
            "主厚度 thickness 仍优先用各子样本 thickness_summary 的 Average（含倾斜修正），"
            "与默认模式一致"
        ),
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=10.0,
        help="精细厚度计算的网格大小 (nm)，仅在精细厚度测量模式下使用（默认 10.0）",
    )
    parser.add_argument(
        "--edge-trim-ratio",
        type=float,
        default=0.1,
        help="边缘排除比例（默认 0.1，即排除边缘 0.10 的区域），仅在精细厚度测量模式下使用",
    )
    parser.add_argument(
        "--min-pores-per-cell",
        type=int,
        default=10,
        help="每个网格至少需要的孔数（默认 10），少于该数的网格会被过滤掉，仅在精细厚度测量模式下使用",
    )
    args = parser.parse_args()

    split_samples_dir = Path(args.split_samples_dir)
    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.output_dir)
    
    # 确保输出目录存在
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("阶段一：厚度分布描述（WT / AS）")
    print("=" * 60)
    print("注意：使用所有结构数据，不进行喉半径过滤（构建全模型需要完整结构）")
    if args.use_fine_scale:
        print(
            f"模式：--fine-scale（网格局部厚度 + 导出；"
            f"分布/统计用各子样本 Average Thickness 修正值；网格 = {args.cell_size} nm，"
            f"边缘排除比例 = {args.edge_trim_ratio:.2f}，最小孔数/网格 = {args.min_pores_per_cell}）"
        )
    else:
        print("模式：各子样本 thickness_summary 的 Average Thickness (nm)（含倾斜修正，默认）")

    # 1.1 汇总厚度数据
    print("\n[1.1] 汇总厚度数据（WT / AS）...")
    df = collect_thickness_data(
        split_samples_dir, analysis_dir,
        use_fine_scale=args.use_fine_scale,
        cell_size=args.cell_size,
        edge_trim_ratio=args.edge_trim_ratio,
    )
    
    if args.use_fine_scale:
        print(f"共收集 {len(df)} 个子样本的精细厚度数据")
        if "n_local_samples" in df.columns:
            print(f"  总局部厚度观测数: WT={df[df['is_wt']]['n_local_samples'].sum()}, AS={df[~df['is_wt']]['n_local_samples'].sum()}")
    else:
        print(f"共收集 {len(df)} 个子样本")
    
    print(f"  WT: {df[df['is_wt']].shape[0]} 个")
    print(f"  AS: {df[~df['is_wt']].shape[0]} 个")

    # 保存原始数据（如果使用精细尺度，需要特殊处理）
    if args.use_fine_scale and "local_thicknesses" in df.columns:
        # 保存时移除 numpy 数组列（Excel 不支持）
        df_to_save = df.drop(columns=["local_thicknesses"]).copy()
        data_path = out_dir / "thickness_data_raw.xlsx"
        df_to_save.to_excel(data_path, index=False)
        print(
            f"原始数据已保存: {data_path}（已移除 local_thicknesses 列；"
            "thickness=修正后主厚度，thickness_fine_grid_mean_nm=球棍网格均值）"
        )
        
        # 单独保存展开的局部厚度数据
        all_local_thicknesses = []
        for _, row in df.iterrows():
            if "local_thicknesses" in row and isinstance(row["local_thicknesses"], np.ndarray):
                for local_t in row["local_thicknesses"]:
                    all_local_thicknesses.append({
                        "sample_name": row["sample_name"],
                        "sub_name": row["sub_name"],
                        "is_wt": row["is_wt"],
                        "local_thickness": float(local_t),
                    })
        if all_local_thicknesses:
            df_local = pd.DataFrame(all_local_thicknesses)
            local_data_path = out_dir / "thickness_data_local_fine_scale.xlsx"
            df_local.to_excel(local_data_path, index=False)
            print(f"局部厚度数据已保存: {local_data_path} ({len(df_local)} 个观测值)")
    else:
        data_path = out_dir / "thickness_data_raw.xlsx"
        df.to_excel(data_path, index=False)
        print(f"原始数据已保存: {data_path}")

    # 1.2 渗透方向与 XY 平面的夹角汇总（按样本）
    print("\n[1.2] 每个样本的渗透方向与 XY 平面夹角...")
    summarize_permeation_direction_angles(df, out_dir)

    # 1.3 厚度分布可视化与描述
    print("\n[1.3] 厚度分布可视化与描述...")
    stats_dict = visualize_thickness_distribution(df, out_dir, use_fine_scale=args.use_fine_scale)
    plot_corrected_thickness_boxplot_by_sample(df, out_dir)

    # 1.4 厚度分布参数化
    print("\n[1.4] 厚度分布参数化...")
    dist_params = parameterize_thickness_distribution(df, out_dir, use_fine_scale=args.use_fine_scale)

    print("\n" + "=" * 60)
    print("阶段一完成！")
    print("=" * 60)
    print(f"\n输出目录: {out_dir}")
    print("  - thickness_data_raw.xlsx: 原始厚度数据")
    if args.use_fine_scale:
        print("  - thickness_data_local_fine_scale.xlsx: 局部精细厚度数据（所有网格的厚度值）")
    print("  - thickness_statistics.xlsx: 统计摘要")
    print("  - permeation_direction_angles_by_sample.(csv/png): 每样本渗透方向与 XY 平面夹角汇总")
    print("  - thickness_distribution.png: 分布图")
    print("  - corrected_thickness_boxplot_by_sample.png: 每个样本一个箱线图（修正后子样本厚度）")
    print("  - thickness_distribution_parameters.json: 分布参数")
    print("  - thickness_sampler.py: 采样模块")


if __name__ == "__main__":
    main()

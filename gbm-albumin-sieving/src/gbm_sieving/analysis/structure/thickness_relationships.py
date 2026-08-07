"""
读取 data_subsamples 下所有带法向量的子样本，绘制「厚度 - 各结构指标」散点图，并加线性拟合（r、R^2）。
- 指标：平均孔半径、平均喉半径、孔上喉数、孔/喉体积分数、秩序参数 S、Q_max/Q_min 等。
- WT 样本一张、AS 样本一张、合起来一张；同一样本同色同形；可选对数坐标。
"""

import argparse
import sys
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
    }
)
# 屏蔽 Matplotlib 关于 Unicode 负号 \u2212 的字体告警，避免终端刷屏
warnings.filterwarnings(
    "ignore",
    message=r"Font 'default' does not have a glyph for '\\u2212'",
    category=UserWarning,
)

from gbm_sieving.analysis.structure import statistics as st
from gbm_sieving.analysis.structure.compare_subsamples import find_sub_samples, run_one_sub_sample
from gbm_sieving.paths import OUTPUT_ROOT, WORKSPACE_ROOT


def load_thickness(thickness_file: Path) -> float:
    """从厚度汇总表读取 Average Thickness (nm)。"""
    df = pd.read_excel(thickness_file)
    if "Average Thickness (nm)" not in df.columns:
        raise ValueError(f"厚度表需包含列 'Average Thickness (nm)': {thickness_file}")
    return float(df["Average Thickness (nm)"].iloc[0])


def collect_all_sub_samples_with_normal(split_samples_dir: Path, analysis_dir: Path, *, min_throat_radius: float = None):
    """
    遍历 data_subsamples 下每个样本目录，收集所有带法向量的子样本的厚度与结构指标（run_one_sub_sample 的 summary 标量）。
    返回: list of dict with keys: sample_name, sub_name, thickness, is_wt, 以及 mean_deg, pore_r_mean, throat_r_mean, frac_pore, mean_S 等。
    
    min_throat_radius: 如果提供，只保留半径 >= 该值的喉，然后基于过滤后的喉集合计算统计指标。
    """
    split_samples_dir = Path(split_samples_dir)
    analysis_dir = Path(analysis_dir)
    rows = []

    for sample_dir in sorted(split_samples_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample_name = sample_dir.name
        sub_list = find_sub_samples(sample_dir, sample_name)
        if not sub_list:
            continue

        is_wt = not sample_name.upper().startswith("AS")

        for sub_name, pores_file, throats_file in sub_list:
            thickness_file = analysis_dir / sample_name / sub_name / f"{sub_name}_thickness_summary.xlsx"
            if not thickness_file.exists():
                print(f"  跳过 {sub_name}（无厚度文件）: {thickness_file}")
                continue
            try:
                thickness = load_thickness(thickness_file)
            except Exception as e:
                print(f"  跳过 {sub_name}（读取厚度失败）: {e}")
                continue
            theta_from_summary = np.nan
            try:
                tdf = pd.read_excel(thickness_file)
                if "Theta_deg_vs_XY" in tdf.columns and len(tdf) > 0:
                    tv = tdf["Theta_deg_vs_XY"].iloc[0]
                    if pd.notna(tv):
                        theta_from_summary = float(tv)
            except Exception:
                pass
            try:
                summary, _ = run_one_sub_sample(pores_file, throats_file, min_throat_radius=min_throat_radius)
            except Exception as e:
                print(f"  跳过 {sub_name}（结构统计失败）: {e}")
                continue

            # 读取该子样本的法向量（渗透方向）：优先使用分析输出目录中的 metadata（含 nz，来自表面质心估计）
            normal_vec = None
            analysis_meta_file = analysis_dir / sample_name / sub_name / f"{sub_name}_metadata.json"
            if analysis_meta_file.exists():
                try:
                    with open(analysis_meta_file, "r", encoding="utf-8") as f:
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
                    print(f"  读取分析目录法向量失败（{sub_name}）: {e}")
            if normal_vec is None:
                meta_file = split_samples_dir / sample_name / f"{sub_name}_metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        nv = meta.get("normal_vector")
                        if nv is not None:
                            nx = float(nv.get("nx", 0.0))
                            ny = float(nv.get("ny", 0.0))
                            nz = float(nv.get("nz", 0.0))
                            normal = np.array([nx, ny, nz if np.isfinite(nz) else 0.0], dtype=float)
                            n_norm = np.linalg.norm(normal)
                            if n_norm > 1e-12:
                                normal_vec = normal / n_norm
                    except Exception as e:
                        print(f"  读取法向量失败（{sub_name}）: {e}")

            # 计算 Q 主轴与渗透方向（法向）的夹角：3D 主轴夹角 + XY 投影夹角
            cos_qmax_normal = np.nan
            angle_qmax_normal_deg = np.nan
            angle_Qmax_normal_3d_deg = np.nan
            angle_Qmax_normal_xy_deg = np.nan
            if normal_vec is not None:
                try:
                    pore_coords, _, pore_ids, pore_id_to_idx, throat_pore1, throat_pore2, _ = \
                        st.load_pores_and_throats(pores_file, throats_file)
                    Q, evals, evecs = st.global_q_tensor(pore_coords, pore_id_to_idx, throat_pore1, throat_pore2)
                    e_max = evecs[:, -1]
                    # 1) 最大特征值对应特征向量(3D)与渗透方向的夹角
                    cos_3d = float(np.dot(e_max, normal_vec))
                    cos_3d_clipped = max(min(abs(cos_3d), 1.0), 0.0)
                    angle_Qmax_normal_3d_deg = float(np.degrees(np.arccos(cos_3d_clipped)))
                    # 2) 最大特征值在 XY 平面的投影与渗透方向的夹角
                    e_xy = np.array([e_max[0], e_max[1], 0.0], dtype=float)
                    e_xy_norm = np.linalg.norm(e_xy)
                    if e_xy_norm > 1e-12:
                        e_xy /= e_xy_norm
                        cos_val = float(abs(np.dot(e_xy, normal_vec)))
                        cos_val_clipped = max(min(cos_val, 1.0), -1.0)
                        angle_deg = float(np.degrees(np.arccos(cos_val_clipped)))
                        cos_qmax_normal = cos_val
                        angle_qmax_normal_deg = angle_deg
                        angle_Qmax_normal_xy_deg = angle_deg
                    else:
                        angle_Qmax_normal_xy_deg = np.nan
                except Exception as e:
                    print(f"  计算 Q 主轴与法向量夹角失败（{sub_name}）: {e}")

            row = {
                "sample_name": sample_name,
                "sub_name": sub_name,
                "thickness": thickness,
                "is_wt": is_wt,
            }
            if normal_vec is not None:
                row["nx"] = float(normal_vec[0])
                row["ny"] = float(normal_vec[1])
                row["nz"] = float(normal_vec[2])
            if np.isfinite(theta_from_summary):
                row["theta_deg_vs_xy"] = float(theta_from_summary)
            row["mean_pore_radius"] = summary.get("pore_r_mean", np.nan)
            row["mean_throat_radius"] = summary.get("throat_r_mean", np.nan)
            row["mean_deg"] = summary.get("mean_deg", np.nan)
            row["frac_pore"] = summary.get("frac_pore", np.nan)
            row["frac_throat"] = summary.get("frac_throat", np.nan)
            row["rho_pore"] = summary.get("rho_pore", np.nan)
            row["rho_throat"] = summary.get("rho_throat", np.nan)
            row["mean_S"] = summary.get("mean_S", np.nan)
            row["Q_max"] = summary.get("Q_max", np.nan)
            row["Q_min"] = summary.get("Q_min", np.nan)
            row["angle_Qmax_normal_3d_deg"] = angle_Qmax_normal_3d_deg
            row["angle_Qmax_normal_xy_deg"] = angle_Qmax_normal_xy_deg
            row["pore_r_std"] = summary.get("pore_r_std", np.nan)
            row["throat_r_std"] = summary.get("throat_r_std", np.nan)
            row["cos_Qmax_normal"] = cos_qmax_normal
            row["angle_Qmax_normal_deg"] = angle_qmax_normal_deg
            rows.append(row)

    return rows


def run_thickness_plots(df: pd.DataFrame, out_dir: Path, *, skip_all_plot: bool = False, skip_as_only_corr: bool = False) -> None:
    """
    对给定的厚度+指标 DataFrame 绘制散点图与拟合，并输出相关性汇总。
    可由本模块 main() 或 run_thickness_vs_radius_filtered 调用。
    skip_all_plot: 为 True 时只画 WT/AS 分图，不画「全部」图（用于已按类型过滤、仅一种类型时）。
    skip_as_only_corr: 为 True 时不计算「仅AS样本」的相关性（用于数据已按类型过滤、只有单一类型时）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 要画的指标：列名 -> 纵轴标签
    metrics = {
        "mean_pore_radius": "平均孔半径 (nm)",
        "mean_throat_radius": "平均喉半径 (nm)",
        "mean_deg": "平均孔上喉数",
        "frac_pore": "孔体积分数",
        "frac_throat": "喉体积分数",
        "rho_pore": "孔密度 (1/nm^3)",
        "rho_throat": "喉密度 (1/nm^3)",
        # 基于全局 Q 张量的有序参数：S_global = max(lam_Q)
        "mean_S": "全局有序参数 S (lam_Q,max)",
        "Q_max": "Q_max",
        "Q_min": "Q_min",
        "pore_r_std": "孔半径标准差 (nm)",
        "throat_r_std": "喉半径标准差 (nm)",
        "cos_Qmax_normal": "cos(喉主取向 vs 渗透方向)",
        "angle_Qmax_normal_deg": "Q 主轴-渗透方向夹角 (deg)",
    }
    n_colors = 5
    base_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    markers = ["o", "s", "^", "v", "D", "P", "*", "X", "h", "d"]
    sample_names = df["sample_name"].unique()
    color_by_name = {name: base_colors[i % n_colors] for i, name in enumerate(sample_names)}
    marker_by_name = {name: markers[i // n_colors] for i, name in enumerate(sample_names)}

    def add_fit_line(ax, data: pd.DataFrame, col: str):
        x = data["thickness"].values
        y = data[col].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid) < 2:
            return
        xv, yv = x[valid], y[valid]
        slope, intercept, r_value, p_value, _ = linregress(xv, yv)
        r2 = r_value ** 2
        x_fit = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, "k--", linewidth=1.5, alpha=0.7)
        ax.text(0.05, 0.95, f"r = {r_value:.3f}\nR^2 = {r2:.3f}\np = {p_value:.3g}", transform=ax.transAxes, fontsize=9, verticalalignment="top")

    def do_scatter(ax, data: pd.DataFrame, col: str, ylabel: str, title: str, filename: str, add_fit: bool = True, save_log: bool = False, ylim: tuple = None):
        for sample_name in data["sample_name"].unique():
            sub = data[data["sample_name"] == sample_name]
            ax.scatter(sub["thickness"], sub[col], c=[color_by_name[sample_name]], marker=marker_by_name[sample_name], label=sample_name, alpha=0.8, s=50)
        ax.set_xlabel("厚度 (nm)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        if add_fit:
            add_fit_line(ax, data, col)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=150)
        print(f"  已保存: {filename}")
        if save_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(title + " (log scale)")
            ax.set_ylim(bottom=None)
            ax.set_xlim(left=None)
            plt.savefig(out_dir / str(filename).replace(".png", "_log.png"), dpi=150)
            print(f"  已保存: {filename.replace('.png', '_log.png')}")
        plt.close()

    # cos 为 [-1,1]，夹角为 [0,180] 度，需单独设 y 轴范围
    orient_ylim = {"cos_Qmax_normal": (-1.05, 1.05), "angle_Qmax_normal_deg": (0, 180)}
    wt_df = df[df["is_wt"]]
    as_df = df[~df["is_wt"]]
    for col, ylabel in metrics.items():
        if col not in df.columns:
            continue
        prefix = f"thickness_vs_{col}"
        allow_log = col not in {"cos_Qmax_normal", "angle_Qmax_normal_deg"}
        ylim = orient_ylim.get(col)
        if not wt_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            do_scatter(ax, wt_df, col, ylabel, f"厚度 - {ylabel}（WT）", f"{prefix}_WT.png", add_fit=True, save_log=allow_log, ylim=ylim)
        if not as_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            do_scatter(ax, as_df, col, ylabel, f"厚度 - {ylabel}（AS）", f"{prefix}_AS.png", add_fit=True, save_log=allow_log, ylim=ylim)
        if not skip_all_plot:
            fig, ax = plt.subplots(figsize=(9, 6))
            for sample_name in df["sample_name"].unique():
                sub = df[df["sample_name"] == sample_name]
                ax.scatter(sub["thickness"], sub[col], c=[color_by_name[sample_name]], marker=marker_by_name[sample_name], label=sample_name, alpha=0.8, s=50)
            ax.set_xlabel("厚度 (nm)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"厚度 - {ylabel}（全部）")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            if ylim is not None:
                ax.set_ylim(ylim)
            else:
                ax.set_ylim(bottom=0)
            ax.set_xlim(left=0)
            add_fit_line(ax, df, col)
            plt.tight_layout()
            plt.savefig(out_dir / f"{prefix}_all.png", dpi=150)
            print(f"  已保存: {prefix}_all.png")
            if allow_log:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"厚度 - {ylabel}（全部）(log scale)")
                ax.set_ylim(bottom=None)
                ax.set_xlim(left=None)
                plt.savefig(out_dir / f"{prefix}_all_log.png", dpi=150)
                print(f"  已保存: {prefix}_all_log.png")
            plt.close()

    corr_summary = []
    for col, ylabel in metrics.items():
        if col not in df.columns:
            continue
        x = df["thickness"].values
        y = df[col].values
        valid = ~(np.isnan(x) | np.isnan(y))
        if np.sum(valid) < 2:
            continue
        xv, yv = x[valid], y[valid]
        slope, intercept, r_value, p_value, _ = linregress(xv, yv)
        r2 = r_value ** 2
        corr_summary.append({"指标": ylabel, "列名": col, "r": r_value, "|r|": abs(r_value), "R^2": r2, "p": p_value, "n": len(xv), "显著": "是" if p_value < 0.05 else "否"})
        # 如果数据已按类型过滤（单一类型），跳过「仅AS」的计算，避免重复
        if not skip_as_only_corr:
            as_valid = valid & (df["is_wt"] == False).values
            if np.sum(as_valid) >= 2:
                x_as = df.loc[as_valid, "thickness"].values
                y_as = df.loc[as_valid, col].values
                slope_as, _, r_as, p_as, _ = linregress(x_as, y_as)
                r2_as = r_as ** 2
                corr_summary.append({"指标": f"{ylabel} (仅AS)", "列名": col, "r": r_as, "|r|": abs(r_as), "R^2": r2_as, "p": p_as, "n": len(x_as), "显著": "是" if p_as < 0.05 else "否"})
    if corr_summary:
        df_corr = pd.DataFrame(corr_summary)
        df_corr = df_corr.sort_values("|r|", ascending=False)
        corr_excel = out_dir / "thickness_correlation_summary.xlsx"
        df_corr.to_excel(corr_excel, index=False)
        print(f"\n相关系数汇总表已保存: {corr_excel}\n")
        print("【全部样本】显著相关指标 (p < 0.05):")
        print("-" * 60)
        sig_all = df_corr[(df_corr["p"] < 0.05) & (~df_corr["指标"].str.contains("仅AS"))]
        if len(sig_all) > 0:
            for _, row in sig_all.iterrows():
                direction = "正相关" if row["r"] > 0 else "负相关"
                print(f"  {row['指标']:30s}  r={row['r']:7.3f}  R^2={row['R^2']:6.3f}  p={row['p']:8.3g}  ({direction}, n={row['n']})")
        else:
            print("  无显著相关指标")
        # 如果数据已按类型过滤（单一类型），不打印「仅AS样本」部分
        if not skip_as_only_corr:
            print("\n【仅AS样本】显著相关指标 (p < 0.05):")
            print("-" * 60)
            sig_as = df_corr[(df_corr["p"] < 0.05) & (df_corr["指标"].str.contains("仅AS"))]
            if len(sig_as) > 0:
                for _, row in sig_as.iterrows():
                    direction = "正相关" if row["r"] > 0 else "负相关"
                    print(f"  {row['指标']:30s}  r={row['r']:7.3f}  R^2={row['R^2']:6.3f}  p={row['p']:8.3g}  ({direction}, n={row['n']})")
            else:
                print("  无显著相关指标")
        print("\n【全部样本】强相关指标 (|r| >= 0.5):")
        print("-" * 60)
        strong = df_corr[(df_corr["|r|"] >= 0.5) & (~df_corr["指标"].str.contains("仅AS"))]
        if len(strong) > 0:
            for _, row in strong.iterrows():
                direction = "正相关" if row["r"] > 0 else "负相关"
                sig_mark = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else ""
                print(f"  {row['指标']:30s}  r={row['r']:7.3f}  R^2={row['R^2']:6.3f}  p={row['p']:8.3g}  ({direction}, n={row['n']}) {sig_mark}")
        else:
            print("  无强相关指标 (|r| < 0.5)")
    print("\n" + "="*60)
    print("完成。")


def main():
    parser = argparse.ArgumentParser(
        description="厚度-平均孔半径散点图（WT/AS 分图 + 合图，同一样本同色）"
    )
    parser.add_argument(
        "--split-samples-dir",
        type=str,
        default=str(WORKSPACE_ROOT / "data_subsamples"),
        help="子样本根目录（默认：肾脏/data_subsamples），其下为各样本子目录",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=str(WORKSPACE_ROOT / "results_core_subsamples"),
        help="分析结果根目录（默认：肾脏/results_core_subsamples），厚度文件路径为 <analysis-dir>/<sample_name>/<sub_name>/<sub_name>_thickness_summary.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_ROOT / "analysis_thickness_vs_structure_properties"),
        help="输出目录（默认：肾脏/results_analysis_thickness_vs_structure_properties）",
    )
    args = parser.parse_args()

    split_dir = Path(args.split_samples_dir)
    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not split_dir.exists():
        print(f"错误：子样本目录不存在: {split_dir}")
        return
    if not analysis_dir.exists():
        print(f"警告：分析结果目录不存在: {analysis_dir}，将无法读取厚度。")

    print("正在收集所有带法向量的子样本（厚度 + 各结构指标）...")
    rows = collect_all_sub_samples_with_normal(split_dir, analysis_dir)
    if not rows:
        print("未找到任何有效子样本（需有法向量且存在厚度文件），退出。")
        return

    df = pd.DataFrame(rows)
    summary_path = out_dir / "thickness_vs_metrics_summary.xlsx"
    df.to_excel(summary_path, index=False)
    print(f"汇总表已保存: {summary_path}")
    run_thickness_plots(df, out_dir)


if __name__ == "__main__":
    main()

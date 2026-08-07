"""
阶段二：孔喉模型参数确定（WT / AS 分开）
- 孔/喉数量密度与体积密度
- 孔半径、喉半径分布及与厚度的关系
- 孔半径与喉半径的相关性
- 喉方向性（Q、S）与厚度
- 喉长度分布及与厚度的关系
"""

import argparse
import json
import sys
import math
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats
from scipy.stats import linregress, ks_2samp
from scipy.stats import pearsonr, spearmanr, multivariate_normal
from sklearn.mixture import GaussianMixture

from gbm_sieving.analysis.structure.thickness_relationships import (
    collect_all_sub_samples_with_normal,
)
from gbm_sieving.analysis.structure.compare_subsamples import (
    find_sub_samples,
    run_one_sub_sample,
)
from gbm_sieving.analysis.structure import statistics as st
from gbm_sieving.paths import OUTPUT_ROOT, WORKSPACE_ROOT


def collect_all_sub_sample_data(
    split_samples_dir: Path,
    analysis_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    收集所有子样本的完整数据（包括原始半径数组）。
    注意：使用所有结构数据，不进行喉半径过滤（构建全模型需要完整结构）。
    返回：
    - df_summary: DataFrame with summary statistics
    - data_dict: dict mapping (sample_name, sub_name) -> arrays (pore_radii, throat_radii, etc.)
    """
    split_samples_dir = Path(split_samples_dir)
    analysis_dir = Path(analysis_dir)
    
    # 先获取 summary 数据（不使用喉半径过滤）
    rows = collect_all_sub_samples_with_normal(
        split_samples_dir, analysis_dir, min_throat_radius=None
    )
    df_summary = pd.DataFrame(rows)
    
    # 再收集原始数组数据（不使用喉半径过滤）
    data_dict = {}
    
    n_skip_no_meta = 0
    n_skip_load_fail = 0

    for sample_dir in sorted(split_samples_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample_name = sample_dir.name
        sub_list = find_sub_samples(sample_dir, sample_name)
        if not sub_list:
            continue
        
        for sub_name, pores_file, throats_file in sub_list:
            # 只保留“有法向量的子样本”：需要存在对应的 metadata 文件。
            # 没有 metadata 的通常是拆分时最后一块“边角料子样本”，不参与 Phase2 统计。
            meta_file = sample_dir / f"{sub_name}_metadata.json"
            if not meta_file.exists():
                # 无法向量 / 无元数据 → 跳过
                n_skip_no_meta += 1
                continue

            try:
                summary, arrays = run_one_sub_sample(
                    pores_file, throats_file, min_throat_radius=None
                )
                # 供 two-point correlation（connected 模式）使用
                arrays["pores_file"] = str(pores_file)
                arrays["throats_file"] = str(throats_file)
                data_dict[(sample_name, sub_name)] = arrays
            except Exception as e:
                # 默认只做汇总，避免终端刷屏
                n_skip_load_fail += 1
                continue

    if n_skip_no_meta > 0:
        print(f"  [数据收集] 跳过无 metadata 子样本: {n_skip_no_meta} 个")
    if n_skip_load_fail > 0:
        print(f"  [数据收集] 跳过数组加载失败子样本: {n_skip_load_fail} 个")

    return df_summary, data_dict


def analyze_density_vs_thickness(
    df: pd.DataFrame,
    out_dir: Path,
) -> dict:
    """
    2.1 孔/喉数量密度与体积密度 vs 厚度
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "rho_pore": "Pore count density (1/nm^3)",
        "rho_throat": "Throat count density (1/nm^3)",
        "frac_pore": "Pore volume fraction",
        "frac_throat": "Throat volume fraction",
    }
    
    results = {}
    
    for metric, label in metrics.items():
        print(f"\n=== {label} vs 厚度 ===")
        results[metric] = {}
        
        for is_wt_val, name in [(True, "WT"), (False, "AS")]:
            subset = df[df["is_wt"] == is_wt_val]
            thickness = subset["thickness"].dropna().values
            values = subset[metric].dropna().values
            
            if len(thickness) == 0 or len(values) == 0:
                print(f"  {name}: 无数据")
                continue
            
            # 基本统计
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            print(f"  {name}: 均值 = {mean_val:.6f}, 标准差 = {std_val:.6f}")
            
            # 与厚度的相关性
            if len(thickness) == len(values) and len(thickness) > 2:
                r_pearson, p_pearson = pearsonr(thickness, values)
                r_spearman, p_spearman = spearmanr(thickness, values)
                print(f"    Pearson r = {r_pearson:.4f}, p = {p_pearson:.4f}")
                print(f"    Spearman r = {r_spearman:.4f}, p = {p_spearman:.4f}")
                
                # 线性回归
                slope, intercept, r_value, p_value, std_err = linregress(thickness, values)
                print(f"    线性回归: y = {slope:.6e} * x + {intercept:.6f}, R^2 = {r_value**2:.4f}, p = {p_value:.4f}")
                
                results[metric][name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "correlation": {
                        "pearson_r": float(r_pearson),
                        "pearson_p": float(p_pearson),
                        "spearman_r": float(r_spearman),
                        "spearman_p": float(p_spearman),
                    },
                    "linear_regression": {
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "r_squared": float(r_value**2),
                        "p_value": float(p_value),
                        "std_err": float(std_err),
                    },
                }
            else:
                results[metric][name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "correlation": None,
                    "linear_regression": None,
                }
    
    # 保存结果
    results_path = out_dir / "density_vs_thickness_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n密度分析结果已保存: {results_path}")
    
    # 绘制散点图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Density vs thickness", fontsize=16, fontweight="bold")
    
    for idx, (metric, label) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        
        for is_wt_val, name, color in [(True, "WT", "blue"), (False, "AS", "orange")]:
            subset = df[df["is_wt"] == is_wt_val]
            thickness = subset["thickness"].dropna().values
            values = subset[metric].dropna().values
            
            if len(thickness) > 0 and len(values) > 0:
                ax.scatter(thickness, values, alpha=0.5, label=name, color=color, s=30)
                
                # 添加回归线
                if len(thickness) == len(values) and len(thickness) > 2:
                    slope, intercept, r_value, p_value, _ = linregress(thickness, values)
                    x_line = np.linspace(thickness.min(), thickness.max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=2,
                           label=f"{name} fit (R^2={r_value**2:.3f})")
        
        ax.set_xlabel("Thickness (nm)", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = out_dir / "density_vs_thickness.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"密度散点图已保存: {plot_path}")
    
    return results


def _radius_fitted_pdf(x: np.ndarray, dist_name: str, params: list) -> np.ndarray:
    """计算半径拟合分布在 x 处的 PDF（单分布）。"""
    x = np.asarray(x, dtype=float)
    if dist_name == "lognorm":
        return stats.lognorm.pdf(x, *params)
    if dist_name == "gamma":
        return stats.gamma.pdf(x, *params)
    if dist_name == "norm":
        return stats.norm.pdf(x, *params)
    return np.zeros_like(x)


def _radius_gmm_pdf(x: np.ndarray, weights: list, means: list, covariances: list) -> np.ndarray:
    """计算 GMM 在 x 处的 PDF（总和）。"""
    x = np.asarray(x, dtype=float)
    w = np.array(weights)
    mu = np.array(means)
    cov = np.array(covariances)
    sigma = np.sqrt(cov)
    pdf = np.zeros_like(x)
    for k in range(len(w)):
        pdf += w[k] * stats.norm.pdf(x, loc=mu[k], scale=sigma[k])
    return pdf


def detect_multimodality_radius(data: np.ndarray, max_components: int = 3) -> dict:
    """
    用 BIC 选择最佳组件数（与 phase1 一致）。
    返回 n_components, is_multimodal, 以及 GMM 参数（若多峰）。
    """
    if len(data) < 20:
        return {"n_components": 1, "is_multimodal": False}
    data = np.asarray(data).reshape(-1, 1)
    best_n = 1
    best_bic = np.inf
    for n in range(1, min(max_components + 1, len(data) // 10 + 1)):
        try:
            gm = GaussianMixture(n_components=n, random_state=42, max_iter=100)
            gm.fit(data)
            bic = gm.bic(data)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        except Exception:
            break
    out = {"n_components": best_n, "is_multimodal": best_n > 1}
    if out["is_multimodal"]:
        gm = GaussianMixture(n_components=best_n, random_state=42, max_iter=100)
        gm.fit(data)
        out["weights"] = gm.weights_.tolist()
        out["means"] = gm.means_.flatten().tolist()
        out["covariances"] = gm.covariances_.flatten().tolist()
    return out


def _collect_radii_by_type_and_name(
    df: pd.DataFrame,
    data_dict: dict,
) -> dict:
    """按 radius_type 和 WT/AS 收集半径数组。返回 {(radius_type, name): np.ndarray}。"""
    out = {}
    for radius_type in ["pore", "throat"]:
        radii_key = f"{radius_type}_radii"
        for is_wt_val, name in [(True, "WT"), (False, "AS")]:
            arr_list = []
            for _, row in df.iterrows():
                if row["is_wt"] != is_wt_val:
                    continue
                key = (row["sample_name"], row["sub_name"])
                if key not in data_dict:
                    continue
                arrays = data_dict[key]
                if radii_key not in arrays:
                    continue
                r = arrays[radii_key]
                if len(r) > 0:
                    arr_list.append(r)
            out[(radius_type, name)] = np.concatenate(arr_list) if arr_list else np.array([])
    return out


def plot_radius_fitted_vs_data(
    results: dict,
    df: pd.DataFrame,
    data_dict: dict,
    out_dir: Path,
) -> None:
    """
    绘制孔/喉半径的拟合分布与真实数据对比图（直方图 + 拟合 PDF 曲线）。
    2x2：孔-WT、孔-AS、喉-WT、喉-AS。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    radii_data = _collect_radii_by_type_and_name(df, data_dict)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    titles = ["WT 孔半径", "AS 孔半径", "WT 喉半径", "AS 喉半径"]
    keys = [("pore", "WT"), ("pore", "AS"), ("throat", "WT"), ("throat", "AS")]

    for idx, (ax, title, (radius_type, name)) in enumerate(zip(axes, titles, keys)):
        key = (radius_type, name)
        data = radii_data.get(key, np.array([]))
        if key[0] not in results or key[1] not in results[key[0]]:
            ax.set_title(f"{title} (no result)")
            continue
        res = results[radius_type][name]
        dist_info = res.get("distribution") or {}
        dist_name = dist_info.get("name")
        params = dist_info.get("params")

        if len(data) == 0:
            ax.set_title(f"{title} (no data)")
            continue

        # 直方图：柱高为概率（半径落入子区间的概率）
        n_bins_hist = 40
        ax.hist(
            data,
            bins=n_bins_hist,
            density=False,
            weights=np.ones_like(data) / len(data),
            alpha=0.5,
            color="steelblue",
            edgecolor="white",
            label="Real data",
        )

        fit_type = res.get("fit_type", "parametric")
        x_min, x_max = data.min(), data.max()
        x_pad = max((x_max - x_min) * 0.1, 0.5)
        x_plot = np.linspace(max(0, x_min - x_pad), x_max + x_pad, 300)
        bin_width_approx = (x_max - x_min) / n_bins_hist if n_bins_hist and x_max > x_min else 1e-6

        if fit_type == "gmm" and res.get("n_components"):
            w = res["weights"]
            mu = res["means"]
            cov = res["covariances"]
            sigma = np.sqrt(cov)
            colors = ["green", "purple", "orange", "brown"]
            for k in range(len(w)):
                # 组件 PDF 转为概率曲线：PDF * 近似 bin 宽度
                comp_pdf = w[k] * stats.norm.pdf(x_plot, loc=mu[k], scale=sigma[k]) * bin_width_approx
                ax.plot(
                    x_plot, comp_pdf,
                    linestyle="--", linewidth=1.5, color=colors[k % len(colors)],
                    label=f"Component {k+1} (w={w[k]:.2f}, mu={mu[k]:.0f})",
                )
            pdf_vals = _radius_gmm_pdf(x_plot, w, mu, cov) * bin_width_approx
            ax.plot(x_plot, pdf_vals, "r-", linewidth=2, label="Fit overall")
            fit_desc = f"GMM ({res['n_components']} components)"
        elif dist_name and params:
            pdf_vals = _radius_fitted_pdf(x_plot, dist_name, params) * bin_width_approx
            ax.plot(x_plot, pdf_vals, "r-", linewidth=2, label=f"Fit {dist_name}")
            fit_desc = dist_name
        else:
            fit_desc = dist_name or "none"

        ax.set_xlabel("Radius (nm)", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_title(f"{title} (n={len(data)})\nFit: {fit_desc}", fontsize=12)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plot_path = out_dir / "radius_fitted_vs_data.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"半径拟合 vs 真实数据对比图已保存: {plot_path}")


def analyze_radius_distributions(
    df: pd.DataFrame,
    data_dict: dict,
    out_dir: Path,
) -> dict:
    """
    2.2 孔半径、喉半径分布及与厚度的关系
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for radius_type in ["pore", "throat"]:
        print(f"\n=== {radius_type} 半径分布分析 ===")
        results[radius_type] = {}
        
        # 收集所有半径数据
        all_radii = []
        thickness_list = []
        is_wt_list = []
        
        for _, row in df.iterrows():
            key = (row["sample_name"], row["sub_name"])
            if key in data_dict:
                arrays = data_dict[key]
                radii_key = f"{radius_type}_radii"
                if radii_key in arrays:
                    radii = arrays[radii_key]
                    if len(radii) > 0:
                        all_radii.append(radii)
                        thickness_list.extend([row["thickness"]] * len(radii))
                        is_wt_list.extend([row["is_wt"]] * len(radii))
        
        if not all_radii:
            print(f"  无 {radius_type} 半径数据")
            continue
        
        all_radii_flat = np.concatenate(all_radii)
        thickness_arr = np.array(thickness_list)
        is_wt_arr = np.array(is_wt_list)
        
        # 按 WT/AS 分别分析
        for is_wt_val, name in [(True, "WT"), (False, "AS")]:
            mask = is_wt_arr == is_wt_val
            radii_subset = all_radii_flat[mask]
            thickness_subset = thickness_arr[mask]
            
            if len(radii_subset) == 0:
                print(f"  {name}: 无数据")
                continue
            
            print(f"  {name}: {len(radii_subset)} 个半径值")
            print(f"    均值 = {np.mean(radii_subset):.4f} nm, 标准差 = {np.std(radii_subset):.4f} nm")
            
            # 多峰检测（BIC）
            multimodality = detect_multimodality_radius(radii_subset)
            fit_type = "gmm" if multimodality["is_multimodal"] else "parametric"
            print(f"    多峰检测（BIC）: 组件数 = {multimodality['n_components']}, 拟合类型 = {fit_type}")
            
            # 单峰时用参数分布；多峰时用 GMM，但仍保留最佳单分布供 phase3 采样
            distributions_to_test = [
                ("lognorm", stats.lognorm),
                ("gamma", stats.gamma),
                ("norm", stats.norm),
            ]
            best_dist_name = None
            best_aic = np.inf
            best_params = None
            for dist_name, dist_class in distributions_to_test:
                try:
                    params = dist_class.fit(radii_subset)
                    aic = -2 * np.sum(dist_class.logpdf(radii_subset, *params)) + 2 * len(params)
                    if aic < best_aic:
                        best_aic = aic
                        best_dist_name = dist_name
                        best_params = params
                except Exception:
                    continue
            
            if best_dist_name:
                print(f"    最佳单分布拟合: {best_dist_name} (AIC = {best_aic:.2f})")
                print(f"    参数: {best_params}")
            
            gmm_extra = {}
            if fit_type == "gmm":
                w = multimodality["weights"]
                mu = multimodality["means"]
                cov = multimodality["covariances"]
                gmm_extra = {
                    "n_components": multimodality["n_components"],
                    "weights": w,
                    "means": mu,
                    "covariances": cov,
                }
                print(f"    GMM: 权重 = {[f'{x:.3f}' for x in w]}, 均值 = {[f'{x:.1f}' for x in mu]}")
            
            # 与厚度的相关性（每个子样本的平均半径 vs 该子样本的厚度）
            df_subset = df[df["is_wt"] == is_wt_val]
            mean_radii = []
            thicknesses = []
            
            for _, row in df_subset.iterrows():
                key = (row["sample_name"], row["sub_name"])
                if key in data_dict:
                    arrays = data_dict[key]
                    radii_key = f"{radius_type}_radii"
                    if radii_key in arrays:
                        radii = arrays[radii_key]
                        if len(radii) > 0:
                            mean_radii.append(np.mean(radii))
                            thicknesses.append(row["thickness"])
            
            correlation_info = None
            regression_info = None
            
            if len(mean_radii) > 2:
                mean_radii_arr = np.array(mean_radii)
                thicknesses_arr = np.array(thicknesses)
                
                r_pearson, p_pearson = pearsonr(thicknesses_arr, mean_radii_arr)
                r_spearman, p_spearman = spearmanr(thicknesses_arr, mean_radii_arr)
                print(f"    平均半径 vs 厚度:")
                print(f"      Pearson r = {r_pearson:.4f}, p = {p_pearson:.4f}")
                print(f"      Spearman r = {r_spearman:.4f}, p = {p_spearman:.4f}")
                
                slope, intercept, r_value, p_value, std_err = linregress(thicknesses_arr, mean_radii_arr)
                print(f"      线性回归: y = {slope:.6e} * x + {intercept:.6f}, R^2 = {r_value**2:.4f}")
                
                correlation_info = {
                    "pearson_r": float(r_pearson),
                    "pearson_p": float(p_pearson),
                    "spearman_r": float(r_spearman),
                    "spearman_p": float(p_spearman),
                }
                regression_info = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": float(r_value**2),
                    "p_value": float(p_value),
                    "std_err": float(std_err),
                }
            
            results[radius_type][name] = {
                "fit_type": fit_type,
                "n_samples": len(radii_subset),
                "mean": float(np.mean(radii_subset)),
                "std": float(np.std(radii_subset)),
                "distribution": {
                    "name": best_dist_name,
                    "params": [float(p) for p in best_params] if best_params else None,
                    "aic": float(best_aic) if best_aic != np.inf else None,
                },
                "correlation_with_thickness": correlation_info,
                "regression_with_thickness": regression_info,
                **gmm_extra,
            }
    
    # 保存结果
    results_path = out_dir / "radius_distribution_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n半径分布分析结果已保存: {results_path}")
    
    return results


AS_MERGE_THRESHOLD_NM = 150.0  # legacy AS merge threshold
AS_HIGH_MERGE_START_NM = 160.0
AS_HIGH_BIN_START_NM = AS_HIGH_MERGE_START_NM  # backward-compatible alias
AS_HIGH_BIN_MODE = "custom"  # "custom": keep requested bin width up to 160 nm, then merge >160
AS_SPARSE_SMOOTH_ENABLED = True
AS_SPARSE_SMOOTH_RANGE_NM = (120.0, 160.0)
AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM = (160.0, 200.0)
AS_RADIUS_SPARSE_SMOOTH_LEFT_SOURCE_RANGE_NM = (100.0, 120.0)
AS_RADIUS_SPARSE_SMOOTH_BANDWIDTH_NM = 50.0
AS_RADIUS_SPARSE_SMOOTH_EXTERNAL_ANCHORS_ONLY = True
AS_LOW_EDGE_SMOOTH_RANGE_NM = (20.0, 30.0)
AS_LOW_EDGE_SMOOTH_DENSITY_RANGE_NM = (20.0, 40.0)
AS_LOW_EDGE_SMOOTH_BANDWIDTH_NM = 15.0
AS_LOW_EDGE_RADIUS_SMOOTH_BANDWIDTH_NM = 25.0
AS_SPARSE_SMOOTH_BANDWIDTH_NM = 30.0
AS_SPARSE_SMOOTH_SYNTHETIC_N = 2000
AS_SPARSE_SMOOTH_SEED = 20260610


def _unique_sorted_edges(edges: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(edges, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    arr = np.asarray(sorted({round(float(v), 9) for v in arr}), dtype=float)
    if arr.size < 2:
        arr = np.array([float(arr[0]), float(arr[0]) + 1.0], dtype=float)
    return arr


def _as_custom_high_bin_edges(thickness_values: np.ndarray, bin_width: float) -> np.ndarray:
    """AS-only binning: keep caller's bin width up to 160 nm, then merge >160."""
    t = np.asarray(thickness_values, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return np.array([0.0, bin_width], dtype=float)

    high = float(AS_HIGH_MERGE_START_NM)
    if float(np.nanmax(t)) <= high:
        return get_thickness_bin_edges(t, bin_width)

    t_min = max(0.0, np.floor(float(np.nanmin(t)) / float(bin_width)) * float(bin_width))
    edges = np.arange(t_min, high + 1e-9, float(bin_width)).tolist()
    if not edges or abs(edges[-1] - high) > 1e-6:
        edges.append(high)
    edges.append(float(np.ceil(float(np.nanmax(t)))) + 1e-9)
    return _unique_sorted_edges(edges)


def get_thickness_bin_edges(thickness_values: np.ndarray, bin_width: float = 10.0) -> np.ndarray:
    """厚度分层边界：以 bin_width (nm) 为区间宽度。返回 [t0, t1, t2, ...]。"""
    t = np.asarray(thickness_values)
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return np.array([0.0, bin_width])
    t_min = max(0.0, np.floor(t.min() / bin_width) * bin_width)
    t_max = np.ceil(t.max() / bin_width) * bin_width
    return np.arange(t_min, t_max + 1e-9, bin_width)


def get_thickness_bin_edges_for_group(
    thickness_values: np.ndarray,
    bin_width: float,
    is_wt: bool,
    as_merge_threshold: float = AS_MERGE_THRESHOLD_NM,
) -> np.ndarray:
    """
    WT：正常等宽分层（bin_width）。
    AS：<150 nm 按 bin_width 分层，>=150 nm 合并成一档，减少少样本问题。
    """
    if is_wt:
        return get_thickness_bin_edges(thickness_values, bin_width)
    if AS_HIGH_BIN_MODE == "custom":
        return _as_custom_high_bin_edges(thickness_values, bin_width)
    t = np.asarray(thickness_values)
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return np.array([0.0, bin_width])
    if t.max() <= as_merge_threshold:
        return get_thickness_bin_edges(t, bin_width)
    if t.min() >= as_merge_threshold:
        return np.array([as_merge_threshold, float(np.ceil(t.max())) + 1e-9])
    t_min = max(0.0, np.floor(t.min() / bin_width) * bin_width)
    edges_below = np.arange(t_min, as_merge_threshold + 1e-9, bin_width)
    t_max = float(np.ceil(t.max())) + 1e-9
    return np.concatenate([edges_below, [t_max]])


WT_DENSITY_FRAC_MIN_N = 15  # WT 密度/体积分数分层：区间样本数 < 此值则并入相邻较少的一侧


def _merge_small_wt_bins(thickness_values: np.ndarray, bin_edges: np.ndarray, min_n: int) -> np.ndarray:
    """
    WT 专用：将样本数 < min_n 的区间并入相邻的、样本数较少的区间，直到无小区间或只剩一档。
    """
    t = np.asarray(thickness_values)
    t = t[np.isfinite(t)]
    edges = list(np.asarray(bin_edges, dtype=float))
    if len(edges) <= 2:
        return np.array(edges)
    while True:
        counts = [int(np.sum((t >= edges[i]) & (t < edges[i + 1]))) for i in range(len(edges) - 1)]
        small_idx = [i for i, c in enumerate(counts) if c < min_n]
        if not small_idx:
            break
        i = small_idx[0]
        left_n = counts[i - 1] if i > 0 else np.inf
        right_n = counts[i + 1] if i + 1 < len(counts) else np.inf
        if left_n <= right_n and i > 0:
            edges.pop(i)
        elif i + 1 < len(counts):
            edges.pop(i + 1)
        else:
            if i > 0:
                edges.pop(i)
            else:
                edges.pop(i + 1)
        if len(edges) <= 2:
            break
    return np.array(edges)


def get_thickness_bin_edges_density_frac(
    thickness_values: np.ndarray,
    is_wt: bool,
    bin_width: float = 20.0,
    as_merge_threshold: float = AS_MERGE_THRESHOLD_NM,
) -> np.ndarray:
    """
    仅用于密度/体积分数按厚度分层：20 nm 一档。
    WT：若某区间 N < 15，则并入相邻样本数较少的区间。
    AS 在 150 nm 以下若最后一档不足 20 nm，则并入倒数第二档。
    喉孔半径等仍用 get_thickness_bin_edges_for_group(..., bin_width=10)。
    """
    if is_wt:
        base_edges = get_thickness_bin_edges(thickness_values, bin_width)
        return _merge_small_wt_bins(thickness_values, base_edges, WT_DENSITY_FRAC_MIN_N)
    if AS_HIGH_BIN_MODE == "custom":
        return _as_custom_high_bin_edges(thickness_values, bin_width)
    t = np.asarray(thickness_values)
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return np.array([0.0, bin_width])
    if t.max() <= as_merge_threshold:
        return get_thickness_bin_edges(t, bin_width)
    if t.min() >= as_merge_threshold:
        return np.array([as_merge_threshold, float(np.ceil(t.max())) + 1e-9])
    t_min = max(0.0, np.floor(t.min() / bin_width) * bin_width)
    edges_below = np.arange(t_min, as_merge_threshold + 1e-9, bin_width)
    # 最后一档 [edges_below[-1], 150) 若不足 20 nm，并入倒数第二档：[..., edges_below[-2], 150]
    # 注意 np.arange 不会包含 150，故最后一档为 (edges_below[-1], 150)，宽度 = 150 - edges_below[-1]
    if len(edges_below) >= 2 and (as_merge_threshold - edges_below[-1]) < bin_width - 1e-9:
        edges_below = np.concatenate([edges_below[:-1], [as_merge_threshold]])
    elif len(edges_below) >= 1 and edges_below[-1] < as_merge_threshold - 1e-9:
        edges_below = np.concatenate([edges_below, [as_merge_threshold]])
    t_max = float(np.ceil(t.max())) + 1e-9
    return np.concatenate([edges_below, [t_max]])


SUBSAMPLE_METRICS = ["rho_pore", "rho_throat", "frac_pore", "frac_throat"]
SUBSAMPLE_METRIC_LABELS = {
    "rho_pore": "孔数量密度",
    "rho_throat": "喉数量密度",
    "frac_pore": "孔体积分数",
    "frac_throat": "喉体积分数",
}


def collect_metric_by_thickness_bin(
    df: pd.DataFrame,
    metric_col: str,
    is_wt: bool,
    bin_edges: np.ndarray,
) -> List[Tuple[float, float, float, np.ndarray]]:
    """
    按厚度区间收集子样本级指标（密度、体积分数等）。
    返回 [(t_center, t_min, t_max, values_array), ...]，每个 values_array 为一维。
    """
    out = []
    for i in range(len(bin_edges) - 1):
        t_min, t_max = float(bin_edges[i]), float(bin_edges[i + 1])
        t_center = (t_min + t_max) / 2.0
        vals = []
        for _, row in df.iterrows():
            if row["is_wt"] != is_wt:
                continue
            t = row.get("thickness")
            if pd.isna(t) or t < t_min or t >= t_max:
                continue
            x = row.get(metric_col)
            if pd.isna(x) or (isinstance(x, (int, float)) and (x <= 0 or not np.isfinite(x))):
                continue
            vals.append(float(x))
        out.append((t_center, t_min, t_max, np.array(vals)))
    return out


def _fit_metric_distribution_one_bin(
    values: np.ndarray,
    min_points: int = 5,
    metric: str = "",
) -> Optional[dict]:
    """
    对子样本级指标在单区间内拟合一元分布。
    - 对 rho_pore / rho_throat（严格正值）：在 lognorm / gamma / norm / GMM 间做 AIC/BIC/KS 选优，
      且在 BIC 接近时优先正值分布（lognorm/gamma/GMM），避免固定 norm 过度抹平偏态。
    - 其余指标（frac_pore / frac_throat）保留稳健的 norm 拟合。
    """
    values = np.asarray(values).flatten()
    values = values[np.isfinite(values)]
    if len(values) < min_points:
        return None
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values))
    std = max(std, 1e-10)  # 避免 std=0 导致采样异常
    if metric in ("rho_pore", "rho_throat"):
        vals_pos = values[values > 0]
        if len(vals_pos) >= min_points:
            candidates: list[dict] = []
            # 正值指标优先用正值分布；lognorm/gamma 固定 loc=0，减少无意义平移
            dist_specs = [
                ("lognorm", stats.lognorm, {"floc": 0.0}),
                ("gamma", stats.gamma, {"floc": 0.0}),
                ("norm", stats.norm, {}),
            ]
            for dist_name, dist_cls, fit_kw in dist_specs:
                try:
                    params = dist_cls.fit(vals_pos, **fit_kw)
                    logpdf = dist_cls.logpdf(vals_pos, *params)
                    if not np.all(np.isfinite(logpdf)):
                        continue
                    ll = float(np.sum(logpdf))
                    k = int(len(params))
                    aic = float(-2.0 * ll + 2.0 * k)
                    bic = float(-2.0 * ll + k * np.log(len(vals_pos)))
                    ks_stat, ks_p = stats.kstest(vals_pos, dist_name, args=params)
                    if not np.isfinite(aic) or not np.isfinite(bic):
                        continue
                    candidates.append(
                        {
                            "name": dist_name,
                            "params": [float(x) for x in params],
                            "aic": aic,
                            "bic": bic,
                            "ks_stat": float(ks_stat),
                            "ks_pvalue": float(ks_p),
                        }
                    )
                except Exception:
                    continue

            # 允许 GMM 参与数量分数拟合（样本足够时），以保留多峰/偏态信息。
            gmm_min_points = max(min_points, 15)
            if len(vals_pos) >= gmm_min_points:
                x = vals_pos.reshape(-1, 1)
                max_components = min(3, max(1, len(vals_pos) // 5))
                best_gmm = None
                best_gmm_meta = None
                for n_comp in range(1, max_components + 1):
                    try:
                        gm = GaussianMixture(n_components=n_comp, random_state=42, max_iter=200)
                        gm.fit(x)
                        ll = float(gm.score(x) * len(vals_pos))
                        # 1D full-covariance GMM 近似参数量: weights(K-1)+means(K)+cov(K)=3K-1
                        k_param = int(3 * n_comp - 1)
                        aic = float(-2.0 * ll + 2.0 * k_param)
                        bic = float(gm.bic(x))

                        w = np.asarray(gm.weights_, dtype=float).reshape(-1)
                        mu = np.asarray(gm.means_, dtype=float).reshape(-1)
                        cov_raw = np.asarray(gm.covariances_, dtype=float).reshape(-1)
                        sigma = np.sqrt(np.maximum(cov_raw, 1e-12))

                        def _gmm_cdf(v):
                            v_arr = np.asarray(v, dtype=float)
                            cdf = np.zeros_like(v_arr, dtype=float)
                            for wk, mk, sk in zip(w, mu, sigma):
                                cdf += wk * stats.norm.cdf(v_arr, loc=mk, scale=sk)
                            return cdf

                        ks_stat, ks_p = stats.kstest(vals_pos, _gmm_cdf)
                        if not (
                            np.isfinite(aic)
                            and np.isfinite(bic)
                            and np.isfinite(ks_stat)
                            and np.isfinite(ks_p)
                        ):
                            continue

                        rec = {
                            "name": "gmm",
                            "n_components": int(n_comp),
                            "weights": [float(v) for v in w.tolist()],
                            "means": [float(v) for v in mu.tolist()],
                            "covariances": [float(v) for v in cov_raw.tolist()],
                            "aic": aic,
                            "bic": bic,
                            "ks_stat": float(ks_stat),
                            "ks_pvalue": float(ks_p),
                        }
                        if (best_gmm is None) or ((rec["bic"], rec["ks_stat"]) < (best_gmm["bic"], best_gmm["ks_stat"])):
                            best_gmm = rec
                            best_gmm_meta = {
                                "criterion": "bic_then_ks",
                                "max_components": int(max_components),
                                "min_points_for_gmm": int(gmm_min_points),
                            }
                    except Exception:
                        continue
                if best_gmm is not None:
                    candidates.append(best_gmm)

            if candidates:
                # 先按 BIC 最优；若 norm 与正值分布 BIC 接近（<=2），优先正值分布
                candidates_sorted = sorted(candidates, key=lambda d: (d["bic"], d["ks_stat"]))
                best = candidates_sorted[0]
                if best["name"] == "norm":
                    alt = [c for c in candidates_sorted if c["name"] in ("lognorm", "gamma", "gmm")]
                    if alt:
                        alt_best = alt[0]
                        if (alt_best["bic"] - best["bic"]) <= 2.0:
                            best = alt_best
                base = {
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "selection": {
                        "criterion_primary": "bic",
                        "criterion_secondary": "ks_stat",
                        "prefer_positive_distribution_when_bic_close": True,
                        "bic_close_threshold": 2.0,
                        "gmm_enabled": True,
                        "candidates": candidates_sorted,
                    },
                }
                if best["name"] == "gmm":
                    base.update(
                        {
                            "fit_type": "gmm",
                            "n_components": int(best["n_components"]),
                            "weights": [float(v) for v in best["weights"]],
                            "means": [float(v) for v in best["means"]],
                            "covariances": [float(v) for v in best["covariances"]],
                            "distribution": {
                                "name": "gmm",
                                "params": [],
                                "aic": best["aic"],
                                "bic": best["bic"],
                                "ks_stat": best["ks_stat"],
                                "ks_pvalue": best["ks_pvalue"],
                            },
                        }
                    )
                    if best_gmm_meta is not None:
                        base["selection"]["gmm_meta"] = best_gmm_meta
                    return base

                base.update(
                    {
                        "fit_type": "parametric",
                        "distribution": {
                            "name": best["name"],
                            "params": best["params"],
                            "aic": best["aic"],
                            "bic": best["bic"],
                            "ks_stat": best["ks_stat"],
                            "ks_pvalue": best["ks_pvalue"],
                        },
                    }
                )
                return base

    # 非 rho 指标保持原有稳定策略（norm）
    entry = {
        "fit_type": "parametric",
        "n": n,
        "mean": mean,
        "std": std,
        "distribution": {
            "name": "norm",
            "params": [mean, std],
            "aic": None,
            "bic": None,
            "ks_stat": None,
            "ks_pvalue": None,
        },
    }
    return entry


DENSITY_FRAC_BIN_WIDTH_NM = 20.0  # 密度/体积分数按厚度分层用 20 nm；半径等仍用 10 nm


def analyze_density_frac_by_thickness_bins(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    bin_width: float = None,
    min_points_per_bin: int = 5,
) -> dict:
    """
    按厚度区间分析孔/喉密度与体积分数的条件分布 p(x | T in bin)。
    使用 20 nm 区间；AS 在 150 nm 以下不足 20 nm 的最后一档并入倒数第二档。
    每个区间内拟合一元分布（norm/lognorm），保存 JSON 并画拟合 vs 真实图。
    """
    if bin_width is None:
        bin_width = DENSITY_FRAC_BIN_WIDTH_NM
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for metric in SUBSAMPLE_METRICS:
        if metric not in df.columns:
            continue
        results[metric] = {}
        for is_wt_val, name in [(True, "WT"), (False, "AS")]:
            thickness_group = df[df["is_wt"] == is_wt_val]["thickness"].dropna().values
            if len(thickness_group) == 0:
                print(f"  {metric} {name}: 无厚度数据，跳过该组的按厚度分层分析")
                continue
            bin_edges = get_thickness_bin_edges_density_frac(thickness_group, is_wt_val, bin_width=bin_width)
            print(f"\n{metric} {name}: 厚度分层 (区间宽度 = {bin_width} nm)，边界 = {bin_edges.tolist()}")
            bins_data = collect_metric_by_thickness_bin(df, metric, is_wt_val, bin_edges)
            bin_results = []
            for t_center, t_min, t_max, values in bins_data:
                rec = {
                    "t_center": float(t_center),
                    "t_min": float(t_min),
                    "t_max": float(t_max),
                    "n": len(values),
                }
                if len(values) >= min_points_per_bin:
                    fit = _fit_metric_distribution_one_bin(values, min_points=min_points_per_bin, metric=metric)
                    if fit:
                        rec["fit"] = fit
                        rec["distribution"] = fit.get("distribution")
                bin_results.append(rec)
            if name == "AS":
                aux_source_bins = _make_aux_right_smoothing_bins(
                    collect_metric_by_thickness_bin(
                        df,
                        metric,
                        is_wt_val,
                        np.asarray(AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM, dtype=float),
                    ),
                    fit_builder=lambda vals, metric=metric: _fit_metric_distribution_one_bin(
                        vals, min_points=min_points_per_bin, metric=metric
                    ),
                    min_points=min_points_per_bin,
                )
                smooth_as_sparse_bin_fits(
                    bin_results,
                    fit_builder=lambda vals, metric=metric: _fit_metric_distribution_one_bin(
                        vals, min_points=min_points_per_bin, metric=metric
                    ),
                    value_kind="fraction" if metric in ("frac_pore", "frac_throat") else "density",
                    label=f"{metric}_density_frac_bin",
                    min_points=min_points_per_bin,
                    seed_offset=1000 + SUBSAMPLE_METRICS.index(metric) * 100,
                    extra_source_bins=aux_source_bins,
                )
            results[metric][name] = {
                "bin_width_nm": float(bin_width),
                "bin_edges": bin_edges.tolist(),
                "bins": bin_results,
            }
            n_with_fit = sum(1 for b in bin_results if "fit" in b)
            print(f"  {metric} {name}: {len(bin_results)} 个区间, {n_with_fit} 个区间有拟合")

    results_path = out_dir / "density_frac_by_thickness_bins_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"按厚度区间密度/体积分数分析已保存: {results_path}")

    plot_density_frac_by_thickness_bins(df, results, bin_width, min_points_per_bin, out_dir)
    return results


def analyze_pore_count_fraction_by_thickness_bin(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    bin_width: float = None,
) -> dict:
    """
    孔在不同厚度区间的数量分数分布（供 Phase3 撒点用）。
    对每个厚度区间 k：fraction_k = (该区间内子样本的 n_pore 之和) / (全部子样本的 n_pore 之和)。
    使用与密度/体积分数相同的厚度分层（get_thickness_bin_edges_density_frac）。
    """
    if bin_width is None:
        bin_width = DENSITY_FRAC_BIN_WIDTH_NM
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "n_pore" not in df.columns:
        print("  df 无 n_pore 列，跳过孔数量分数按厚度区间分析")
        return {}
    results = {}
    for is_wt_val, name in [(True, "WT"), (False, "AS")]:
        subset = df[df["is_wt"] == is_wt_val]
        thickness = subset["thickness"].dropna().values
        if len(thickness) == 0:
            continue
        bin_edges = get_thickness_bin_edges_density_frac(thickness, is_wt_val, bin_width=bin_width)
        n_bins = len(bin_edges) - 1
        count_per_bin = np.zeros(n_bins)
        for _, row in subset.iterrows():
            t = row.get("thickness")
            n_pore = row.get("n_pore")
            if pd.isna(t) or pd.isna(n_pore) or n_pore <= 0:
                continue
            k = np.searchsorted(bin_edges, t, side="right") - 1
            k = np.clip(k, 0, n_bins - 1)
            count_per_bin[k] += n_pore
        total = count_per_bin.sum()
        if total <= 0:
            fraction_per_bin = np.ones(n_bins) / n_bins
        else:
            fraction_per_bin = (count_per_bin / total).tolist()
        auxiliary_sources = []
        if name == "AS" and total > 0:
            aux_lo, aux_hi = AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM
            aux_count = 0.0
            for _, row in subset.iterrows():
                t = row.get("thickness")
                n_pore = row.get("n_pore")
                if pd.isna(t) or pd.isna(n_pore) or n_pore <= 0:
                    continue
                if float(aux_lo) <= float(t) < float(aux_hi):
                    aux_count += float(n_pore)
            if aux_count > 0:
                auxiliary_sources.append(
                    {
                        "t_min": float(aux_lo),
                        "t_max": float(aux_hi),
                        "t_center": float((aux_lo + aux_hi) / 2.0),
                        "fraction": float(aux_count / total),
                        "count": float(aux_count),
                    }
                )
        results[name] = {
            "bin_edges": bin_edges.tolist(),
            "fraction_per_bin": fraction_per_bin,
            "n_pore_total": int(total),
            "count_per_bin": count_per_bin.tolist(),
        }
        if name == "AS":
            _smooth_as_count_fraction_result(results[name], auxiliary_sources=auxiliary_sources)
        print(f"  孔数量分数 {name}: {n_bins} 个厚度区间, 总孔数 = {int(total)}")
    results_path = out_dir / "pore_count_fraction_by_thickness_bin.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"孔数量分数按厚度区间已保存: {results_path}")
    return results


def _smooth_as_count_fraction_result(result: dict, auxiliary_sources: list[dict] | None = None) -> None:
    if not AS_SPARSE_SMOOTH_ENABLED or not result:
        return
    edges = np.asarray(result.get("bin_edges", []), dtype=float)
    frac = np.asarray(result.get("fraction_per_bin", []), dtype=float)
    if edges.size < 2 or frac.size != edges.size - 1:
        return
    centers = 0.5 * (edges[:-1] + edges[1:])
    raw_frac = frac.copy()
    lo, hi = AS_SPARSE_SMOOTH_RANGE_NM
    target_idx = [
        int(i)
        for i, c in enumerate(centers)
        if np.isfinite(c) and float(lo) <= float(c) < float(hi)
    ]
    smoothed_targets: dict[int, float] = {}
    source_centers = centers.copy()
    source_values = raw_frac.copy()
    aux_meta = []
    for src in auxiliary_sources or []:
        c = float(src.get("t_center", np.nan))
        val = float(src.get("fraction", np.nan))
        if not (np.isfinite(c) and np.isfinite(val) and val >= 0):
            continue
        source_centers = np.concatenate([source_centers, np.array([c], dtype=float)])
        source_values = np.concatenate([source_values, np.array([val], dtype=float)])
        aux_meta.append(
            {
                "source_type": "auxiliary_right_source",
                "t_min": float(src.get("t_min", np.nan)),
                "t_max": float(src.get("t_max", np.nan)),
                "t_center": c,
                "fraction": val,
                "count": float(src.get("count", np.nan)),
            }
        )
    for i, c in enumerate(centers):
        if i not in target_idx:
            continue
        w = np.exp(-0.5 * ((source_centers - c) / float(AS_SPARSE_SMOOTH_BANDWIDTH_NM)) ** 2)
        w = np.where(np.isfinite(w), w, 0.0)
        if w.sum() <= 0:
            continue
        smoothed_targets[i] = float(np.sum(w * source_values) / np.sum(w))
    if smoothed_targets:
        raw_target_sum = float(np.sum(raw_frac[target_idx]))
        smooth_target_sum = float(np.sum([smoothed_targets[i] for i in target_idx if i in smoothed_targets]))
        if raw_target_sum > 0 and smooth_target_sum > 0:
            frac = raw_frac.copy()
            scale = raw_target_sum / smooth_target_sum
            for i, val in smoothed_targets.items():
                frac[i] = float(val * scale)
    result["fraction_per_bin_raw_before_smoothing"] = raw_frac.tolist()
    result["fraction_per_bin"] = frac.tolist()
    result["smoothing"] = {
        "enabled": True,
        "method": "gaussian_neighborhood_weighted_fraction",
        "preserve_target_range_total_fraction": True,
        "smooth_range_nm": [float(lo), float(hi)],
        "auxiliary_right_source_range_nm": [
            float(AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM[0]),
            float(AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM[1]),
        ],
        "bandwidth_nm": float(AS_SPARSE_SMOOTH_BANDWIDTH_NM),
        "target_bins": target_idx,
        "auxiliary_sources": aux_meta,
    }


def _metric_fitted_pdf(x: np.ndarray, dist_name: str, params: list) -> np.ndarray:
    """子样本指标拟合 PDF（norm / lognorm / gamma / beta）。"""
    x = np.asarray(x, dtype=float)
    if dist_name == "norm":
        return stats.norm.pdf(x, *params)
    if dist_name == "lognorm":
        return stats.lognorm.pdf(x, *params)
    if dist_name == "gamma":
        return stats.gamma.pdf(x, *params)
    if dist_name == "beta":
        return stats.beta.pdf(x, *params)
    return np.zeros_like(x)


def plot_density_frac_by_thickness_bins(
    df: pd.DataFrame,
    results: dict,
    bin_width: float,
    min_points_per_bin: int,
    out_dir: Path,
) -> None:
    """每个厚度区间一个子图：直方图 + 拟合曲线。"""
    out_dir = Path(out_dir)
    for metric in SUBSAMPLE_METRICS:
        if metric not in results:
            continue
        label = SUBSAMPLE_METRIC_LABELS.get(metric, metric)
        for is_wt_val, name in [(True, "WT"), (False, "AS")]:
            res_name = results.get(metric, {}).get(name, {})
            bin_edges = np.array(res_name.get("bin_edges", []))
            if len(bin_edges) < 2:
                continue
            bins_data = collect_metric_by_thickness_bin(df, metric, is_wt_val, bin_edges)
            panels = [(t_center, t_min, t_max, values) for t_center, t_min, t_max, values in bins_data
                      if len(values) >= min_points_per_bin]
            if not panels:
                continue
            n_panels = len(panels)
            n_cols = 4
            n_rows = (n_panels + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes_flat = axes.flatten()
            res_name = results.get(metric, {}).get(name, {})
            bins_list = res_name.get("bins", [])

            for idx, (t_center, t_min, t_max, values) in enumerate(panels):
                ax = axes_flat[idx]
                n_bins = min(15, max(5, len(values) // 2))
                ax.hist(values, bins=n_bins, density=False, weights=np.ones_like(values) / len(values),
                        alpha=0.5, color="steelblue", edgecolor="white", label="Real data")
                bin_width_approx = (values.max() - values.min()) / n_bins if n_bins and values.max() > values.min() else 1.0
                bin_rec = next((b for b in bins_list if abs(b["t_center"] - t_center) < 1e-6 and b.get("fit")), None)
                if bin_rec and bin_rec.get("fit"):
                    fit = bin_rec["fit"]
                    x_min, x_max = values.min(), values.max()
                    x_pad = max((x_max - x_min) * 0.1, 1e-12)
                    x_plot = np.linspace(max(1e-12, x_min - x_pad), x_max + x_pad, 200)
                    if metric in ("frac_pore", "frac_throat"):
                        x_plot = np.clip(x_plot, 0, 1)
                    if fit.get("fit_type") == "gmm" and fit.get("n_components"):
                        w, mu, cov = fit["weights"], fit["means"], fit["covariances"]
                        sigma = np.sqrt(cov)
                        colors = ["green", "purple", "orange", "brown"]
                        for k in range(len(w)):
                            comp = w[k] * stats.norm.pdf(x_plot, loc=mu[k], scale=sigma[k]) * bin_width_approx
                            ax.plot(x_plot, comp, linestyle="--", linewidth=1.0, color=colors[k % len(colors)])
                        pdf_vals = _radius_gmm_pdf(x_plot, w, mu, cov) * bin_width_approx
                        ax.plot(x_plot, pdf_vals, "r-", linewidth=1.5, label="Fit GMM")
                    else:
                        dist_name = fit.get("distribution", {}).get("name")
                        params = fit.get("distribution", {}).get("params")
                        if dist_name and params:
                            pdf_vals = _metric_fitted_pdf(x_plot, dist_name, params)
                            ax.plot(x_plot, pdf_vals * bin_width_approx, "r-", linewidth=1.5, label=f"Fit {dist_name}")
                ax.set_xlabel(label)
                ax.set_ylabel("Probability")
                ax.set_title(f"T in [{t_min:.0f},{t_max:.0f}) nm (n={len(values)})")
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)
            for idx in range(len(panels), len(axes_flat)):
                axes_flat[idx].set_visible(False)
            plt.suptitle(f"{name} {label} by thickness bin ({bin_width:.0f} nm per bin)", fontsize=12, fontweight="bold")
            plt.tight_layout()
            plot_path = out_dir / f"density_frac_bins_{metric}_{name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  已保存: {plot_path}")


def compare_metric_distributions_across_thickness_bins(
    df: pd.DataFrame,
    results: dict,
    out_dir: Path,
    *,
    min_points_per_bin: int = 5,
    p_value_threshold: float = 0.05,
) -> None:
    """
    对不同厚度区间的密度/体积分数分布做两两 KS 检验，找出「近乎相同」的区间。
    热图使用完整厚度轴：样本不足的区间不参与比较，对应行列显示为空白（NaN），便于看出 AS 等组在哪些厚度缺数据。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in SUBSAMPLE_METRICS:
        if metric not in results:
            continue
        label = SUBSAMPLE_METRIC_LABELS.get(metric, metric)
        for is_wt_val, name in [(True, "WT"), (False, "AS")]:
            res = results.get(metric, {}).get(name, {})
            bin_edges = np.array(res.get("bin_edges", []))
            if len(bin_edges) < 2:
                continue
            bins_data = collect_metric_by_thickness_bin(df, metric, is_wt_val, bin_edges)
            n_all = len(bins_data)
            # 轴标签带上每个区间的样本数 n
            all_labels = []
            for i in range(n_all):
                _, t_min, t_max, values = bins_data[i]
                all_labels.append(f"[{t_min:.0f},{t_max:.0f}) (n={len(values)})")
            # 有效区间：样本量 >= 阈值
            valid_indices = [i for i in range(n_all) if len(bins_data[i][3]) >= min_points_per_bin]
            if len(valid_indices) < 2:
                print(f"  {metric} {name}: 有效区间数 < 2，跳过比较")
                continue

            values_by_idx = {i: bins_data[i][3] for i in valid_indices}
            valid_labels = [all_labels[i] for i in valid_indices]
            n_valid = len(valid_indices)

            # 仅对有效区间两两做 KS
            p_full = np.full((n_all, n_all), np.nan)
            ks_full = np.full((n_all, n_all), np.nan)
            for ii, i in enumerate(valid_indices):
                for jj, j in enumerate(valid_indices):
                    if i == j:
                        p_full[i, j] = 1.0
                        ks_full[i, j] = 0.0
                        continue
                    if i > j:
                        continue
                    stat, pval = ks_2samp(values_by_idx[i], values_by_idx[j])
                    p_full[i, j] = p_full[j, i] = float(pval)
                    ks_full[i, j] = ks_full[j, i] = float(stat)

            # 相似组（只在有效区间内算）
            p_valid = np.ones((n_valid, n_valid))
            for ii in range(n_valid):
                for jj in range(ii + 1, n_valid):
                    p_valid[ii, jj] = p_valid[jj, ii] = p_full[valid_indices[ii], valid_indices[jj]]
            similar_edges = [
                (ii, jj) for ii in range(n_valid) for jj in range(ii + 1, n_valid)
                if p_valid[ii, jj] > p_value_threshold
            ]
            groups = _union_find_groups(n_valid, similar_edges)
            similar_groups = [[valid_labels[ii] for ii in g] for g in groups]

            def _nan_to_none(x):
                if isinstance(x, np.ndarray):
                    return _nan_to_none(x.tolist())
                if isinstance(x, list):
                    return [_nan_to_none(v) for v in x]
                if isinstance(x, float) and np.isnan(x):
                    return None
                return x

            summary = {
                "metric": metric,
                "name": name,
                "p_value_threshold": p_value_threshold,
                "min_points_per_bin": min_points_per_bin,
                "bin_labels_full": all_labels,
                "valid_bin_indices": valid_indices,
                "valid_bin_labels": valid_labels,
                "n_bins_total": n_all,
                "n_bins_with_sufficient_data": n_valid,
                "p_value_matrix_full": _nan_to_none(p_full),
                "ks_statistic_matrix_full": _nan_to_none(ks_full),
                "similar_groups": similar_groups,
                "note": "p > threshold 表示两区间分布无法拒绝相同。仅样本量>=min_points_per_bin的区间参与比较；缺数据区间在矩阵中为null。",
            }
            json_path = out_dir / f"density_frac_bins_compare_{metric}_{name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            excel_path = out_dir / f"density_frac_bins_compare_{metric}_{name}.xlsx"
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                pd.DataFrame(p_full, index=all_labels, columns=all_labels).to_excel(
                    writer, sheet_name="p_value"
                )
                pd.DataFrame(ks_full, index=all_labels, columns=all_labels).to_excel(
                    writer, sheet_name="KS_statistic"
                )
                pd.DataFrame({"similar_group": [str(g) for g in similar_groups]}).to_excel(
                    writer, sheet_name="similar_groups", index=False
                )

            # 热图：完整厚度轴，NaN 用灰色显示（用 masked array 让 imshow 把 NaN 画成灰）
            fig, ax = plt.subplots(figsize=(max(6, n_all * 0.5), max(5, n_all * 0.5)))
            p_plot = np.ma.array(p_full, mask=np.isnan(p_full))
            im = ax.imshow(p_plot, cmap="RdYlGn", vmin=0, vmax=1)
            im.cmap.set_bad(color="lightgray")
            ax.set_xticks(range(n_all))
            ax.set_yticks(range(n_all))
            ax.set_xticklabels(all_labels, rotation=45, ha="right")
            ax.set_yticklabels(all_labels)
            for i in range(n_all):
                for j in range(n_all):
                    if np.isfinite(p_full[i, j]):
                        txt = f"{p_full[i, j]:.2f}"
                        ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                                color="black" if 0.3 < p_full[i, j] < 0.7 else "white")
                    else:
                        ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
            ax.set_xlabel("Thickness bin (nm)")
            ax.set_ylabel("Thickness bin (nm)")
            ax.set_title(
                f"{name} {label}: KS test p-value between bins\n"
                f"(p > {p_value_threshold} similar; gray = n<{min_points_per_bin})"
            )
            plt.colorbar(im, ax=ax, label="p-value")
            plt.tight_layout()
            plot_path = out_dir / f"density_frac_bins_compare_{metric}_{name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"  {metric} {name}: 总区间数 {n_all}，有效区间数 {n_valid}，比较矩阵与相似组已保存 ({len(similar_groups)} 组)")
            for idx, g in enumerate(similar_groups):
                if len(g) >= 2:
                    print(f"    相似组 {idx + 1}: {g}")


def collect_radii_by_thickness_bin(
    df: pd.DataFrame,
    data_dict: dict,
    radius_type: str,
    is_wt: bool,
    bin_edges: np.ndarray,
) -> List[Tuple[float, float, float, np.ndarray]]:
    """
    按厚度区间收集半径。返回 [(t_center, t_min, t_max, radii_array), ...]。
    """
    radii_key = f"{radius_type}_radii"
    out = []
    for i in range(len(bin_edges) - 1):
        t_min, t_max = float(bin_edges[i]), float(bin_edges[i + 1])
        t_center = (t_min + t_max) / 2.0
        arr_list = []
        for _, row in df.iterrows():
            if row["is_wt"] != is_wt:
                continue
            t = row.get("thickness")
            if pd.isna(t) or t < t_min or t >= t_max:
                continue
            key = (row["sample_name"], row["sub_name"])
            if key not in data_dict:
                continue
            arrays = data_dict[key]
            if radii_key not in arrays:
                continue
            r = arrays[radii_key]
            if len(r) > 0:
                arr_list.append(r)
        radii = np.concatenate(arr_list) if arr_list else np.array([])
        out.append((t_center, t_min, t_max, radii))
    return out


def collect_degree_by_thickness_bin(
    df: pd.DataFrame,
    data_dict: dict,
    is_wt: bool,
    bin_edges: np.ndarray,
) -> List[Tuple[float, float, float, np.ndarray]]:
    """
    按厚度区间收集孔度数。返回 [(t_center, t_min, t_max, degree_array), ...]。
    与孔半径使用相同的厚度分层方式。
    """
    out = []
    for i in range(len(bin_edges) - 1):
        t_min, t_max = float(bin_edges[i]), float(bin_edges[i + 1])
        t_center = (t_min + t_max) / 2.0
        arr_list = []
        for _, row in df.iterrows():
            if row["is_wt"] != is_wt:
                continue
            t = row.get("thickness")
            if pd.isna(t) or t < t_min or t >= t_max:
                continue
            key = (row["sample_name"], row["sub_name"])
            if key not in data_dict:
                continue
            arrays = data_dict[key]
            if "deg" not in arrays:
                continue
            d = arrays["deg"]
            if len(d) > 0:
                arr_list.append(np.asarray(d, dtype=float))
        degrees = np.concatenate(arr_list) if arr_list else np.array([])
        out.append((t_center, t_min, t_max, degrees))
    return out


def _empirical_degree_cdf_one_bin(
    degrees: np.ndarray,
    min_points: int = 30,
) -> dict | None:
    """
    从柱状图直接推出经验 CDF，不做参数拟合。
    返回可 JSON 序列化的 dict：fit_type="empirical", degree_values, counts, cdf。
    Phase3 用逆变换采样从 cdf 采样。
    """
    if len(degrees) < min_points:
        return None
    degrees = np.asarray(degrees, dtype=float)
    degrees = degrees[~np.isnan(degrees)]
    degrees = np.round(degrees).astype(int)
    degrees = np.maximum(degrees, 0)
    if len(degrees) == 0:
        return None
    n_sample = len(degrees)
    deg_min, deg_max = int(degrees.min()), int(degrees.max())
    # 柱状图：每个度数 0..deg_max 的计数
    degree_values = list(range(0, deg_max + 1))
    counts = [int(np.sum(degrees == d)) for d in degree_values]
    total = sum(counts)
    if total <= 0:
        return None
    # CDF: cdf[i] = P(deg <= degree_values[i])
    cdf = np.cumsum(counts).astype(float) / total
    cdf = cdf.tolist()
    entry = {
        "fit_type": "empirical",
        "n": n_sample,
        "mean": float(np.mean(degrees)),
        "std": float(np.std(degrees)),
        "degree_values": degree_values,
        "counts": counts,
        "cdf": cdf,
    }
    return entry


def _fit_radius_distribution_one_bin(
    radii: np.ndarray,
    min_points: int = 30,
) -> dict | None:
    """对一组半径做分布拟合（BIC 选 GMM 或单分布）。返回可 JSON 序列化的 dict 或 None。"""
    if len(radii) < min_points:
        return None
    multimodality = detect_multimodality_radius(radii)
    fit_type = "gmm" if multimodality["is_multimodal"] else "parametric"
    distributions_to_test = [
        ("lognorm", stats.lognorm),
        ("gamma", stats.gamma),
        ("norm", stats.norm),
    ]
    n = len(radii)
    best_dist_name = None
    best_bic = np.inf
    best_params = None
    for dist_name, dist_class in distributions_to_test:
        try:
            params = dist_class.fit(radii)
            loglike = np.sum(dist_class.logpdf(radii, *params))
            bic = -2 * loglike + len(params) * np.log(n)
            if bic < best_bic:
                best_bic = bic
                best_dist_name = dist_name
                best_params = params
        except Exception:
            continue
    entry = {
        "fit_type": fit_type,
        "n": n,
        "mean": float(np.mean(radii)),
        "std": float(np.std(radii)),
        "distribution": {
            "name": best_dist_name,
            "params": [float(p) for p in best_params] if best_params else None,
            "bic": float(best_bic) if best_bic != np.inf else None,
        },
    }
    if fit_type == "gmm":
        entry["n_components"] = multimodality["n_components"]
        entry["weights"] = multimodality["weights"]
        entry["means"] = multimodality["means"]
        entry["covariances"] = multimodality["covariances"]
    return entry


def _radius_pdf_from_fit_dict(x: np.ndarray, fit: dict) -> np.ndarray:
    """
    根据单个区间的拟合结果（_fit_radius_distribution_one_bin 的返回值）计算半径的 PDF。
    支持 lognorm / gamma / norm 以及 GMM（混合正态）。
    """
    x = np.asarray(x, dtype=float)
    if not fit:
        return np.zeros_like(x)
    dist_info = fit.get("distribution") or {}
    name = dist_info.get("name", "")
    params = dist_info.get("params") or []
    params = [float(p) for p in params]

    # GMM：weights / means / covariances
    if fit.get("fit_type") == "gmm" and all(k in fit for k in ("weights", "means", "covariances")):
        w = np.asarray(fit["weights"], dtype=float)
        mu = np.asarray(fit["means"], dtype=float)
        cov = np.asarray(fit["covariances"], dtype=float)
        sigma = np.sqrt(np.maximum(cov, 1e-12))
        pdf = np.zeros_like(x)
        for wi, mi, si in zip(w, mu, sigma):
            pdf += wi * stats.norm.pdf(x, loc=mi, scale=si)
        return np.maximum(pdf, 0.0)

    # 单分布：直接用 scipy.stats 的 pdf
    try:
        if name == "lognorm" and len(params) >= 3:
            return stats.lognorm.pdf(x, params[0], loc=params[1], scale=params[2])
        if name == "gamma" and len(params) >= 3:
            return stats.gamma.pdf(x, params[0], loc=params[1], scale=params[2])
        if name == "norm" and len(params) >= 2:
            return stats.norm.pdf(x, loc=params[0], scale=params[1])
    except Exception:
        pass
    return np.zeros_like(x)


def _bin_center_from_record(rec: dict) -> float:
    if "t_center" in rec:
        try:
            return float(rec.get("t_center"))
        except Exception:
            pass
    try:
        return 0.5 * (float(rec.get("t_min")) + float(rec.get("t_max")))
    except Exception:
        return float("nan")


def _is_as_sparse_smooth_target(rec: dict) -> bool:
    t_center = _bin_center_from_record(rec)
    lo, hi = AS_SPARSE_SMOOTH_RANGE_NM
    return np.isfinite(t_center) and (float(lo) <= float(t_center) < float(hi))


def _is_bin_center_in_range(rec: dict, smooth_range: tuple[float, float]) -> bool:
    t_center = _bin_center_from_record(rec)
    lo, hi = smooth_range
    return np.isfinite(t_center) and (float(lo) <= float(t_center) < float(hi))


def _low_edge_smooth_range_for_bin(rec: dict) -> tuple[float, float]:
    try:
        width = float(rec.get("t_max")) - float(rec.get("t_min"))
    except Exception:
        width = float("nan")
    if np.isfinite(width) and width >= 15.0:
        return AS_LOW_EDGE_SMOOTH_DENSITY_RANGE_NM
    return AS_LOW_EDGE_SMOOTH_RANGE_NM


def _as_smoothing_config_for_target(rec: dict) -> dict | None:
    low_range = _low_edge_smooth_range_for_bin(rec)
    if _is_bin_center_in_range(rec, low_range):
        return {
            "region": "low_edge_sparse",
            "smooth_range": low_range,
            "bandwidth_nm": float(AS_LOW_EDGE_SMOOTH_BANDWIDTH_NM),
        }
    if _is_as_sparse_smooth_target(rec):
        return {
            "region": "mid_high_sparse",
            "smooth_range": AS_SPARSE_SMOOTH_RANGE_NM,
            "bandwidth_nm": float(AS_SPARSE_SMOOTH_BANDWIDTH_NM),
        }
    return None


def _smoothing_bandwidth_for_value_kind(smoothing_config: dict, value_kind: str) -> float:
    region = str(smoothing_config.get("region", ""))
    if value_kind == "radius" and region == "low_edge_sparse":
        return float(AS_LOW_EDGE_RADIUS_SMOOTH_BANDWIDTH_NM)
    if value_kind == "radius" and region == "mid_high_sparse":
        return float(AS_RADIUS_SPARSE_SMOOTH_BANDWIDTH_NM)
    return float(smoothing_config["bandwidth_nm"])


def _source_allowed_for_smoothing(
    *,
    target_idx: int,
    source_idx: int,
    source_type: str,
    source_min: float,
    source_max: float,
    smoothing_config: dict,
    value_kind: str,
) -> bool:
    region = str(smoothing_config.get("region", ""))
    smooth_lo, smooth_hi = (float(v) for v in smoothing_config["smooth_range"])

    if int(source_idx) == int(target_idx):
        return True
    if str(source_type) != "output_bin":
        return True

    if (
        value_kind == "radius"
        and region == "mid_high_sparse"
        and bool(AS_RADIUS_SPARSE_SMOOTH_EXTERNAL_ANCHORS_ONLY)
    ):
        left_lo, left_hi = AS_RADIUS_SPARSE_SMOOTH_LEFT_SOURCE_RANGE_NM
        return (
            float(source_min) >= float(left_lo) - 1e-6
            and float(source_max) <= float(left_hi) + 1e-6
        )

    if float(source_max) <= smooth_lo + 1e-6:
        return True
    if (
        float(source_min) >= smooth_hi - 1e-6
        and float(source_max) <= float(AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM[1]) + 1e-6
    ):
        return True
    return False


def _effective_source_center_for_smoothing(
    *,
    target_rec: dict,
    source_center: float,
    source_type: str,
    source_min: float,
    source_max: float,
    smoothing_config: dict,
    value_kind: str,
) -> float:
    """Compress excluded internal thickness span when weighting external radius anchors."""
    center = float(source_center)
    region = str(smoothing_config.get("region", ""))
    if not (
        value_kind == "radius"
        and region == "mid_high_sparse"
        and bool(AS_RADIUS_SPARSE_SMOOTH_EXTERNAL_ANCHORS_ONLY)
    ):
        return center

    try:
        target_min = float(target_rec.get("t_min"))
        target_max = float(target_rec.get("t_max"))
    except Exception:
        return center
    if not (np.isfinite(target_min) and np.isfinite(target_max)):
        return center

    smooth_lo, smooth_hi = (float(v) for v in smoothing_config["smooth_range"])
    source_min = float(source_min)
    source_max = float(source_max)
    if str(source_type) == "output_bin" and source_max <= smooth_lo + 1e-6:
        return float(target_min - (smooth_lo - center))
    if source_min >= smooth_hi - 1e-6 or str(source_type) != "output_bin":
        return float(target_max + (center - smooth_hi))
    return center


def _sample_continuous_fit(
    fit: dict,
    n: int,
    rng: np.random.Generator,
    *,
    value_kind: str,
) -> np.ndarray:
    if not fit or n <= 0:
        return np.array([], dtype=float)
    vals: np.ndarray
    if fit.get("fit_type") == "gmm" and all(k in fit for k in ("weights", "means", "covariances")):
        w = np.asarray(fit.get("weights", []), dtype=float)
        mu = np.asarray(fit.get("means", []), dtype=float)
        cov = np.asarray(fit.get("covariances", []), dtype=float)
        if w.size == 0 or mu.size == 0 or cov.size == 0:
            return np.array([], dtype=float)
        w = np.maximum(w, 0.0)
        if w.sum() <= 0:
            w = np.ones_like(w) / float(w.size)
        else:
            w = w / w.sum()
        comp = rng.choice(np.arange(w.size), size=int(n), p=w)
        sigma = np.sqrt(np.maximum(cov, 1e-12))
        vals = rng.normal(loc=mu[comp], scale=sigma[comp])
    else:
        dist = fit.get("distribution", {}) or {}
        name = dist.get("name")
        params = dist.get("params") or []
        try:
            if name == "lognorm" and len(params) >= 3:
                vals = stats.lognorm.rvs(params[0], loc=params[1], scale=params[2], size=int(n), random_state=rng)
            elif name == "gamma" and len(params) >= 3:
                vals = stats.gamma.rvs(params[0], loc=params[1], scale=params[2], size=int(n), random_state=rng)
            elif name == "beta" and len(params) >= 4:
                vals = stats.beta.rvs(params[0], params[1], loc=params[2], scale=params[3], size=int(n), random_state=rng)
            elif name == "norm" and len(params) >= 2:
                vals = rng.normal(loc=float(params[0]), scale=max(float(params[1]), 1e-12), size=int(n))
            else:
                mean = float(fit.get("mean", 0.0))
                std = max(float(fit.get("std", 0.0)), 1e-12)
                vals = rng.normal(loc=mean, scale=std, size=int(n))
        except Exception:
            mean = float(fit.get("mean", 0.0))
            std = max(float(fit.get("std", 0.0)), 1e-12)
            vals = rng.normal(loc=mean, scale=std, size=int(n))

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if value_kind in {"positive", "radius", "length", "ratio", "density"}:
        vals = vals[vals > 0]
        vals = np.maximum(vals, 1e-12)
    elif value_kind == "fraction":
        vals = np.clip(vals, 0.0, 1.0)
    return vals


def _sample_degree_fit(fit: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    if not fit or int(n) <= 0:
        return np.array([], dtype=float)
    degree_values = np.asarray(fit.get("degree_values", []), dtype=int)
    counts = np.asarray(fit.get("counts", []), dtype=float)
    if degree_values.size == 0 or counts.size != degree_values.size or counts.sum() <= 0:
        return np.array([], dtype=float)
    p = np.maximum(counts, 0.0)
    p = p / p.sum()
    return rng.choice(degree_values, size=int(n), p=p).astype(float)


def _install_smoothed_fit(rec: dict, fit: dict, smoothing_meta: dict) -> None:
    raw_fit = rec.get("fit")
    if raw_fit is not None and "fit_raw_before_smoothing" not in rec:
        rec["fit_raw_before_smoothing"] = raw_fit
    rec["fit"] = fit
    rec["distribution"] = fit.get("distribution")
    for key in ("n_components", "weights", "means", "covariances"):
        rec.pop(key, None)
    if fit.get("fit_type") == "gmm":
        for key in ("n_components", "weights", "means", "covariances"):
            if key in fit:
                rec[key] = fit[key]
    rec["smoothing"] = smoothing_meta
    fit["smoothing"] = smoothing_meta


def _make_aux_right_smoothing_bins(
    bins_data: list[tuple[float, float, float, np.ndarray]],
    *,
    fit_builder: Callable[[np.ndarray], Optional[dict]],
    min_points: int,
    n_key: str = "n",
) -> list[dict]:
    """Build an AS high-thickness source-only bin for smoothing without changing output bins."""
    lo, hi = AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM
    collected: list[np.ndarray] = []
    for t_center, t_min, t_max, values in bins_data:
        t_min_f = float(t_min)
        t_max_f = float(t_max)
        if t_min_f < float(lo) - 1e-6 or t_max_f > float(hi) + 1e-6:
            continue
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            collected.append(values)
    if not collected:
        return []

    merged_values = np.concatenate(collected)
    rec = {
        "t_center": float((lo + hi) / 2.0),
        "t_min": float(lo),
        "t_max": float(hi),
        n_key: int(len(merged_values)),
        "smoothing_source_only": True,
    }
    if len(merged_values) >= int(min_points):
        fit = fit_builder(merged_values)
        if fit:
            rec["fit"] = fit
            rec["distribution"] = fit.get("distribution")
    return [rec] if rec.get("fit") else []


def smooth_as_sparse_bin_fits(
    bins: list[dict],
    *,
    fit_builder: Callable[[np.ndarray], Optional[dict]],
    value_kind: str,
    label: str,
    min_points: int,
    n_key: str = "n",
    seed_offset: int = 0,
    extra_source_bins: list[dict] | None = None,
) -> None:
    """Replace configured sparse AS thickness-bin fits with Gaussian-neighborhood refits."""
    if not AS_SPARSE_SMOOTH_ENABLED or not bins:
        return
    sources: list[tuple[int, float, dict, int, str, float, float]] = []
    source_records = [(idx, rec, "output_bin") for idx, rec in enumerate(bins)]
    if extra_source_bins:
        source_records.extend(
            [(len(bins) + idx, rec, "auxiliary_right_source") for idx, rec in enumerate(extra_source_bins)]
        )
    for idx, rec, source_type in source_records:
        fit = rec.get("fit")
        if not fit:
            continue
        center = _bin_center_from_record(rec)
        if not np.isfinite(center):
            continue
        n_raw = rec.get(n_key, fit.get("n", min_points))
        try:
            n_raw = int(n_raw)
        except Exception:
            n_raw = int(min_points)
        sources.append(
            (
                idx,
                center,
                fit,
                max(n_raw, int(min_points)),
                str(source_type),
                float(rec.get("t_min", np.nan)),
                float(rec.get("t_max", np.nan)),
            )
        )
    if len(sources) < 2:
        return

    for idx, rec in enumerate(bins):
        smoothing_config = _as_smoothing_config_for_target(rec)
        if smoothing_config is None:
            continue
        center = _bin_center_from_record(rec)
        target_sources = [
            s
            for s in sources
            if _source_allowed_for_smoothing(
                target_idx=idx,
                source_idx=int(s[0]),
                source_type=str(s[4]),
                source_min=float(s[5]),
                source_max=float(s[6]),
                smoothing_config=smoothing_config,
                value_kind=value_kind,
            )
        ]
        if len(target_sources) < 2:
            continue
        effective_source_centers = np.asarray(
            [
                _effective_source_center_for_smoothing(
                    target_rec=rec,
                    source_center=float(s[1]),
                    source_type=str(s[4]),
                    source_min=float(s[5]),
                    source_max=float(s[6]),
                    smoothing_config=smoothing_config,
                    value_kind=value_kind,
                )
                for s in target_sources
            ],
            dtype=float,
        )
        bandwidth_nm = _smoothing_bandwidth_for_value_kind(smoothing_config, value_kind)
        weights = np.exp(-0.5 * ((effective_source_centers - center) / bandwidth_nm) ** 2)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        if weights.sum() <= 0:
            continue
        weights = weights / weights.sum()

        target_raw_n = rec.get(n_key, rec.get("n", min_points))
        try:
            target_raw_n = int(target_raw_n)
        except Exception:
            target_raw_n = int(min_points)
        n_synth = max(int(AS_SPARSE_SMOOTH_SYNTHETIC_N), target_raw_n, int(min_points))
        rng = np.random.default_rng(int(AS_SPARSE_SMOOTH_SEED) + int(seed_offset) + int(idx) * 7919)
        draw_counts = rng.multinomial(int(n_synth), weights)
        samples: list[np.ndarray] = []
        source_meta: list[dict] = []
        for src_pos, (count, (src_idx, src_center, fit, src_n, source_type, src_min, src_max)) in enumerate(zip(draw_counts, target_sources)):
            if int(count) <= 0:
                continue
            vals = (
                _sample_degree_fit(fit, int(count), rng)
                if value_kind == "degree"
                else _sample_continuous_fit(fit, int(count), rng, value_kind=value_kind)
            )
            if vals.size > 0:
                samples.append(vals)
            source_meta.append(
                {
                    "bin_index": int(src_idx),
                    "source_type": str(source_type),
                    "t_center": float(src_center),
                    "effective_t_center": float(effective_source_centers[src_pos]),
                    "effective_distance_nm": float(effective_source_centers[src_pos] - center),
                    "t_min": float(src_min),
                    "t_max": float(src_max),
                    "weight": float(weights[src_pos]),
                    "n_raw": int(src_n),
                    "n_draw": int(count),
                }
            )
        if not samples:
            continue
        sampled = np.concatenate(samples)
        sampled = sampled[np.isfinite(sampled)]
        if sampled.size < int(min_points):
            continue
        fit_new = fit_builder(sampled)
        if not fit_new:
            continue
        smoothing_meta = {
            "enabled": True,
            "method": "gaussian_neighborhood_refit_from_fitted_distributions",
            "region": str(smoothing_config["region"]),
            "label": str(label),
            "target_t_center_nm": float(center),
            "target_t_min_nm": float(rec.get("t_min", np.nan)),
            "target_t_max_nm": float(rec.get("t_max", np.nan)),
            "smooth_range_nm": [
                float(smoothing_config["smooth_range"][0]),
                float(smoothing_config["smooth_range"][1]),
            ],
            "auxiliary_right_source_range_nm": [
                float(AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM[0]),
                float(AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM[1]),
            ],
            "bandwidth_nm": bandwidth_nm,
            "source_filter": (
                "radius_target_bin_plus_external_anchors_with_compressed_distance"
                if value_kind == "radius"
                and str(smoothing_config.get("region")) == "mid_high_sparse"
                and bool(AS_RADIUS_SPARSE_SMOOTH_EXTERNAL_ANCHORS_ONLY)
                else "target_bin_plus_outside_smooth_range"
            ),
            "radius_left_source_range_nm": [
                float(AS_RADIUS_SPARSE_SMOOTH_LEFT_SOURCE_RANGE_NM[0]),
                float(AS_RADIUS_SPARSE_SMOOTH_LEFT_SOURCE_RANGE_NM[1]),
            ] if value_kind == "radius" else None,
            "n_synthetic": int(sampled.size),
            "n_raw_in_bin": int(target_raw_n),
            "sources": source_meta,
        }
        _install_smoothed_fit(rec, fit_new, smoothing_meta)


def _fit_2d_gmm_joint(
    x: np.ndarray,
    y: np.ndarray,
    min_components: int = 1,
    max_components: int = 4,
    max_samples: int = 200000,
    random_state: int = 0,
) -> dict | None:
    """
    对 (x, y) 做 2D GMM 拟合，使用 BIC 选择分量数。返回可 JSON 序列化的 dict。
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    X = np.column_stack([x[mask], y[mask]])
    if X.shape[0] < 10:
        return None
    # 大样本时随机下采样，加快拟合
    rng = np.random.default_rng(random_state)
    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X_fit = X[idx]
    else:
        X_fit = X

    best_bic = np.inf
    best_gmm: GaussianMixture | None = None
    for n_comp in range(min_components, max_components + 1):
        try:
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type="full",
                random_state=random_state,
            )
            gmm.fit(X_fit)
            bic = gmm.bic(X_fit)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except Exception:
            continue
    if best_gmm is None:
        return None
    return {
        "n_components": int(best_gmm.n_components),
        "weights": best_gmm.weights_.tolist(),
        "means": best_gmm.means_.tolist(),  # shape (K, 2)
        "covariances": best_gmm.covariances_.tolist(),  # shape (K, 2, 2)
    }


def _gmm2d_pdf_grid(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    gmm_dict: dict,
) -> np.ndarray:
    """
    在 (x_grid, y_grid) 网格上计算 2D GMM 的联合 PDF p(x, y)。
    gmm_dict 由 _fit_2d_gmm_joint 返回。
    """
    if not gmm_dict:
        return np.zeros((len(x_grid), len(y_grid)), dtype=float)
    weights = np.asarray(gmm_dict.get("weights", []), dtype=float)
    means = np.asarray(gmm_dict.get("means", []), dtype=float)
    covs = np.asarray(gmm_dict.get("covariances", []), dtype=float)
    if (
        weights.ndim != 1
        or means.ndim != 2
        or means.shape[1] != 2
        or covs.ndim != 3
        or covs.shape[1:] != (2, 2)
    ):
        return np.zeros((len(x_grid), len(y_grid)), dtype=float)

    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    pdf = np.zeros(points.shape[0], dtype=float)
    for w, mu, cov in zip(weights, means, covs):
        try:
            pdf += float(w) * multivariate_normal.pdf(points, mean=mu, cov=cov, allow_singular=True)
        except Exception:
            continue
    pdf = pdf.reshape(X.shape)
    return np.maximum(pdf, 0.0)


def analyze_radius_by_thickness_bins(
    df: pd.DataFrame,
    data_dict: dict,
    out_dir: Path,
    *,
    bin_width: float = 10.0,
    min_points_per_bin: int = 30,
) -> dict:
    """
    按厚度区间（默认 10 nm）分析孔/喉半径的条件分布 p(x | T in bin)。
    每个区间内拟合一元分布（BIC 选 GMM 或参数分布），保存 JSON 并画拟合 vs 真实图。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for radius_type in ["pore", "throat"]:
        results[radius_type] = {}
        for is_wt_val, name in [(True, "WT"), (False, "AS")]:
            thickness_group = df[df["is_wt"] == is_wt_val]["thickness"].dropna().values
            if len(thickness_group) == 0:
                print(f"  {radius_type} {name}: 无厚度数据，跳过该组的按厚度分层分析")
                continue
            bin_edges = get_thickness_bin_edges_for_group(thickness_group, bin_width, is_wt_val)
            print(f"\n{radius_type} {name}: 厚度分层 (区间宽度 = {bin_width} nm)，边界 = {bin_edges.tolist()}")
            bins_data = collect_radii_by_thickness_bin(
                df, data_dict, radius_type, is_wt_val, bin_edges
            )
            bin_results = []
            for t_center, t_min, t_max, radii in bins_data:
                rec = {
                    "t_center": float(t_center),
                    "t_min": float(t_min),
                    "t_max": float(t_max),
                    "n": len(radii),
                }
                if len(radii) >= min_points_per_bin:
                    fit = _fit_radius_distribution_one_bin(radii, min_points=min_points_per_bin)
                    if fit:
                        rec["fit"] = fit
                        rec["distribution"] = fit.get("distribution")
                        if fit.get("fit_type") == "gmm":
                            rec["n_components"] = fit["n_components"]
                            rec["weights"] = fit["weights"]
                            rec["means"] = fit["means"]
                            rec["covariances"] = fit["covariances"]
                bin_results.append(rec)
            if name == "AS":
                seed_offset = 2000 if radius_type == "pore" else 2100
                aux_source_bins = _make_aux_right_smoothing_bins(
                    collect_radii_by_thickness_bin(
                        df,
                        data_dict,
                        radius_type,
                        is_wt_val,
                        np.asarray(AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM, dtype=float),
                    ),
                    fit_builder=lambda vals: _fit_radius_distribution_one_bin(
                        vals, min_points=min_points_per_bin
                    ),
                    min_points=min_points_per_bin,
                )
                smooth_as_sparse_bin_fits(
                    bin_results,
                    fit_builder=lambda vals: _fit_radius_distribution_one_bin(
                        vals, min_points=min_points_per_bin
                    ),
                    value_kind="radius",
                    label=f"{radius_type}_radius_by_thickness",
                    min_points=min_points_per_bin,
                    seed_offset=seed_offset,
                    extra_source_bins=aux_source_bins,
                )
            results[radius_type][name] = {
                "bin_width_nm": float(bin_width),
                "bin_edges": bin_edges.tolist(),
                "bins": bin_results,
            }
            n_with_fit = sum(1 for b in bin_results if "fit" in b)
            print(f"  {radius_type} {name}: {len(bin_results)} 个区间, {n_with_fit} 个区间有拟合")

    results_path = out_dir / "radius_by_thickness_bins_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n按厚度区间半径分析已保存: {results_path}")

    plot_radius_by_thickness_bins(
        df, data_dict, results, bin_width, min_points_per_bin, out_dir
    )
    return results


def plot_radius_by_thickness_bins(
    df: pd.DataFrame,
    data_dict: dict,
    results: dict,
    bin_width: float,
    min_points_per_bin: int,
    out_dir: Path,
    *,
    thickness_bin_title_suffix: str | None = None,
) -> None:
    """每个厚度区间一个子图：直方图 + 拟合曲线（与 phase1 风格一致）。

    thickness_bin_title_suffix:
        若给定，则总标题括号内为该字符串（用于自定义分箱示例）；否则为「{bin_width:.0f} nm per bin」。
    """
    out_dir = Path(out_dir)
    for radius_type in ["pore", "throat"]:
        for is_wt_val, name in [(True, "WT"), (False, "AS")]:
            res_name = results.get(radius_type, {}).get(name, {})
            bin_edges = np.array(res_name.get("bin_edges", []))
            if len(bin_edges) < 2:
                continue
            bins_data = collect_radii_by_thickness_bin(
                df, data_dict, radius_type, is_wt_val, bin_edges
            )
            # 只画有拟合的区间
            panels = [(t_center, t_min, t_max, radii) for t_center, t_min, t_max, radii in bins_data
                      if len(radii) >= min_points_per_bin]
            if not panels:
                continue
            n_panels = len(panels)
            n_cols = 4
            n_rows = (n_panels + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes_flat = axes.flatten()
            bins_list = res_name.get("bins", [])

            for idx, (t_center, t_min, t_max, radii) in enumerate(panels):
                ax = axes_flat[idx]
                n_bins_hist = 30
                # 孔半径：各子图横轴统一为 WT [0,15] nm、AS [0,20] nm，便于对比；喉半径仍按数据范围。
                if radius_type == "pore":
                    x_hi = 15.0 if is_wt_val else 20.0
                    bin_edges_hist = np.linspace(0.0, x_hi, n_bins_hist + 1)
                    bin_width_approx = x_hi / float(n_bins_hist)
                    x_plot = np.linspace(0.0, x_hi, 200)
                    ax.hist(
                        radii,
                        bins=bin_edges_hist,
                        density=False,
                        weights=np.ones_like(radii) / len(radii),
                        alpha=0.5,
                        color="steelblue",
                        edgecolor="white",
                        label="Real data",
                    )
                else:
                    x_hi = None
                    x_min, x_max = radii.min(), radii.max()
                    x_pad = max((x_max - x_min) * 0.1, 0.5)
                    x_plot = np.linspace(max(0, x_min - x_pad), x_max + x_pad, 200)
                    bin_width_approx = (
                        (x_max - x_min) / n_bins_hist if n_bins_hist and x_max > x_min else 1e-6
                    )
                    ax.hist(
                        radii,
                        bins=n_bins_hist,
                        density=False,
                        weights=np.ones_like(radii) / len(radii),
                        alpha=0.5,
                        color="steelblue",
                        edgecolor="white",
                        label="Real data",
                    )
                # Find fit for this bin
                bin_rec = next((b for b in bins_list if b["t_center"] == t_center and b.get("fit")), None)
                if bin_rec and bin_rec.get("fit"):
                    fit = bin_rec["fit"]
                    if fit.get("fit_type") == "gmm":
                        w, mu, cov = fit["weights"], fit["means"], fit["covariances"]
                        sigma = np.sqrt(cov)
                        colors = ["green", "purple", "orange", "brown"]
                        for k in range(len(w)):
                            comp = w[k] * stats.norm.pdf(x_plot, loc=mu[k], scale=sigma[k]) * bin_width_approx
                            ax.plot(x_plot, comp, linestyle="--", linewidth=1.2, color=colors[k % len(colors)],
                                    label=f"Component {k+1}" if k < 2 else None)
                        pdf_vals = _radius_gmm_pdf(x_plot, w, mu, cov) * bin_width_approx
                        ax.plot(x_plot, pdf_vals, "r-", linewidth=1.5, label="Fit overall")
                    else:
                        dist_name = fit.get("distribution", {}).get("name")
                        params = fit.get("distribution", {}).get("params")
                        if dist_name and params:
                            pdf_vals = _radius_fitted_pdf(x_plot, dist_name, params) * bin_width_approx
                            ax.plot(x_plot, pdf_vals, "r-", linewidth=1.5, label=f"Fit {dist_name}")
                ax.set_xlabel("Radius (nm)")
                ax.set_ylabel("Probability")
                ax.set_title(f"T in [{t_min:.0f},{t_max:.0f}) nm (n={len(radii)})")
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0)
                if radius_type == "pore":
                    ax.set_xlim(0.0, x_hi)
            for idx in range(len(panels), len(axes_flat)):
                axes_flat[idx].set_visible(False)
            cap = (
                thickness_bin_title_suffix
                if thickness_bin_title_suffix is not None
                else f"{bin_width:.0f} nm per bin"
            )
            plt.suptitle(
                f"{name} {radius_type} radius by thickness bin ({cap})",
                fontsize=12,
                fontweight="bold",
            )
            plt.tight_layout()
            plot_path = out_dir / f"radius_by_thickness_bins_{radius_type}_{name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  已保存: {plot_path}")


def analyze_degree_by_thickness_bins(
    df: pd.DataFrame,
    data_dict: dict,
    out_dir: Path,
    *,
    bin_width: float = 10.0,
    min_points_per_bin: int = 30,
) -> dict:
    """
    按厚度区间（与孔半径相同分层）分析孔度数的条件分布 p(deg | T in bin)。
    每个区间内拟合离散分布（Poisson / nbinom），保存 JSON 并画拟合 vs 真实图。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for is_wt_val, name in [(True, "WT"), (False, "AS")]:
        thickness_group = df[df["is_wt"] == is_wt_val]["thickness"].dropna().values
        if len(thickness_group) == 0:
            print(f"  孔度数 {name}: 无厚度数据，跳过")
            continue
        bin_edges = get_thickness_bin_edges_for_group(thickness_group, bin_width, is_wt_val)
        print(f"\n孔度数 {name}: 厚度分层 (区间宽度 = {bin_width} nm)，边界 = {bin_edges.tolist()}")
        bins_data = collect_degree_by_thickness_bin(df, data_dict, is_wt_val, bin_edges)
        bin_results = []
        for t_center, t_min, t_max, degrees in bins_data:
            rec = {
                "t_center": float(t_center),
                "t_min": float(t_min),
                "t_max": float(t_max),
                "n": len(degrees),
            }
            if len(degrees) >= min_points_per_bin:
                emp = _empirical_degree_cdf_one_bin(degrees, min_points=min_points_per_bin)
                if emp:
                    rec["fit"] = emp
            bin_results.append(rec)
        if name == "AS":
            aux_source_bins = _make_aux_right_smoothing_bins(
                collect_degree_by_thickness_bin(
                    df,
                    data_dict,
                    is_wt_val,
                    np.asarray(AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM, dtype=float),
                ),
                fit_builder=lambda vals: _empirical_degree_cdf_one_bin(
                    vals, min_points=min_points_per_bin
                ),
                min_points=min_points_per_bin,
            )
            smooth_as_sparse_bin_fits(
                bin_results,
                fit_builder=lambda vals: _empirical_degree_cdf_one_bin(
                    vals, min_points=min_points_per_bin
                ),
                value_kind="degree",
                label="degree_by_thickness",
                min_points=min_points_per_bin,
                seed_offset=3000,
                extra_source_bins=aux_source_bins,
            )
        results[name] = {
            "bin_width_nm": float(bin_width),
            "bin_edges": bin_edges.tolist(),
            "bins": bin_results,
        }
        n_with_fit = sum(1 for b in bin_results if "fit" in b)
        print(f"  孔度数 {name}: {len(bin_results)} 个区间, {n_with_fit} 个区间有经验 CDF")

    results_path = out_dir / "degree_by_thickness_bins_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n按厚度区间孔度数分析已保存: {results_path}")

    plot_degree_by_thickness_bins(
        df, data_dict, results, bin_width, min_points_per_bin, out_dir
    )
    return results


def _degree_pmf(x: np.ndarray, dist_name: str, params: list) -> np.ndarray:
    """离散分布 PMF，用于绘图。"""
    x = np.asarray(x, dtype=int)
    x = np.maximum(x, 0)
    if dist_name == "poisson" and len(params) >= 1:
        return stats.poisson.pmf(x, params[0])
    if dist_name == "nbinom" and len(params) >= 2:
        return stats.nbinom.pmf(x, params[0], params[1])
    return np.zeros_like(x, dtype=float)


def plot_degree_by_thickness_bins(
    df: pd.DataFrame,
    data_dict: dict,
    results: dict,
    bin_width: float,
    min_points_per_bin: int,
    out_dir: Path,
) -> None:
    """每个厚度区间一个子图：直方图（概率）+ 经验 PMF（柱状图导出，不拟合）。"""
    out_dir = Path(out_dir)
    for is_wt_val, name in [(True, "WT"), (False, "AS")]:
        res_name = results.get(name, {})
        bin_edges = np.array(res_name.get("bin_edges", []))
        if len(bin_edges) < 2:
            continue
        bins_data = collect_degree_by_thickness_bin(df, data_dict, is_wt_val, bin_edges)
        panels = [(t_center, t_min, t_max, deg) for t_center, t_min, t_max, deg in bins_data
                  if len(deg) >= min_points_per_bin]
        if not panels:
            continue
        n_panels = len(panels)
        n_cols = 4
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        bins_list = res_name.get("bins", [])

        for idx, (t_center, t_min, t_max, degrees) in enumerate(panels):
            ax = axes_flat[idx]
            degrees = np.asarray(degrees, dtype=int)
            degrees = np.maximum(degrees, 0)
            deg_max = int(degrees.max()) if len(degrees) > 0 else 5
            bins_hist = np.arange(-0.5, deg_max + 2, 1)
            counts, _, _ = ax.hist(
                degrees, bins=bins_hist, density=True, alpha=0.5,
                color="steelblue", edgecolor="white", label="Real data"
            )
            bin_rec = next((b for b in bins_list if b["t_center"] == t_center and b.get("fit")), None)
            if bin_rec and bin_rec.get("fit"):
                fit = bin_rec["fit"]
                if fit.get("fit_type") == "empirical":
                    dv = fit.get("degree_values", [])
                    cnt = fit.get("counts", [])
                    total = sum(cnt)
                    if total > 0 and dv:
                        pmf_vals = [c / total for c in cnt]
                        ax.plot(dv, pmf_vals, "ro-", markersize=4, linewidth=1.5, label="Empirical PMF")
            ax.set_xlabel("Pore degree")
            ax.set_ylabel("Probability")
            ax.set_title(f"T in [{t_min:.0f},{t_max:.0f}) nm (n={len(degrees)})")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
        for idx in range(len(panels), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.suptitle(f"{name} Pore degree by thickness bin ({bin_width:.0f} nm per bin)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plot_path = out_dir / f"degree_by_thickness_bins_{name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {plot_path}")


def compare_degree_distributions_across_thickness_bins(
    df: pd.DataFrame,
    data_dict: dict,
    results: dict,
    out_dir: Path,
    *,
    min_points_per_bin: int = 30,
    p_value_threshold: float = 0.05,
) -> None:
    """
    对不同厚度区间的孔度数分布做两两 KS 检验，绘制 p-value 热图。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for is_wt_val, name in [(True, "WT"), (False, "AS")]:
        res = results.get(name, {})
        bin_edges = np.array(res.get("bin_edges", []))
        if len(bin_edges) < 2:
            continue
        bins_data = collect_degree_by_thickness_bin(df, data_dict, is_wt_val, bin_edges)
        n_all = len(bins_data)
        all_labels = []
        for i in range(n_all):
            _, t_min, t_max, deg = bins_data[i]
            all_labels.append(f"[{t_min:.0f},{t_max:.0f}) (n={len(deg)})")
        valid_indices = [i for i in range(n_all) if len(bins_data[i][3]) >= min_points_per_bin]
        if len(valid_indices) < 2:
            print(f"  孔度数 {name}: 有效区间数 < 2，跳过比较")
            continue

        degrees_by_idx = {i: bins_data[i][3] for i in valid_indices}
        valid_labels = [all_labels[i] for i in valid_indices]
        n_valid = len(valid_indices)

        p_full = np.full((n_all, n_all), np.nan)
        for ii, i in enumerate(valid_indices):
            for jj, j in enumerate(valid_indices):
                if i == j:
                    p_full[i, j] = 1.0
                    continue
                if i > j:
                    continue
                stat, pval = ks_2samp(degrees_by_idx[i], degrees_by_idx[j])
                p_full[i, j] = p_full[j, i] = float(pval)

        similar_edges = [
            (ii, jj) for ii in range(n_valid) for jj in range(ii + 1, n_valid)
            if p_full[valid_indices[ii], valid_indices[jj]] > p_value_threshold
        ]
        groups = _union_find_groups(n_valid, similar_edges)
        similar_groups = [[valid_labels[ii] for ii in g] for g in groups]

        summary = {
            "name": name,
            "p_value_threshold": p_value_threshold,
            "min_points_per_bin": min_points_per_bin,
            "bin_labels_full": all_labels,
            "valid_bin_indices": valid_indices,
            "p_value_matrix_full": _nan_to_none_for_json(p_full),
            "similar_groups": similar_groups,
        }
        json_path = out_dir / f"degree_bins_compare_{name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        excel_path = out_dir / f"degree_bins_compare_{name}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pd.DataFrame(p_full, index=all_labels, columns=all_labels).to_excel(
                writer, sheet_name="p_value"
            )
            pd.DataFrame({"similar_group": [str(g) for g in similar_groups]}).to_excel(
                writer, sheet_name="similar_groups", index=False
            )

        fig, ax = plt.subplots(figsize=(max(6, n_all * 0.5), max(5, n_all * 0.5)))
        p_plot = np.ma.array(p_full, mask=np.isnan(p_full))
        im = ax.imshow(p_plot, cmap="RdYlGn", vmin=0, vmax=1)
        im.cmap.set_bad(color="lightgray")
        ax.set_xticks(range(n_all))
        ax.set_yticks(range(n_all))
        ax.set_xticklabels(all_labels, rotation=45, ha="right")
        ax.set_yticklabels(all_labels)
        for i in range(n_all):
            for j in range(n_all):
                if np.isfinite(p_full[i, j]):
                    txt = f"{p_full[i, j]:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                            color="black" if 0.3 < p_full[i, j] < 0.7 else "white")
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
        ax.set_xlabel("Thickness bin (nm)")
        ax.set_ylabel("Thickness bin (nm)")
        ax.set_title(
            f"{name} Pore degree: KS test p-value between bins\n"
            f"(p > {p_value_threshold} similar; gray = n<{min_points_per_bin})"
        )
        plt.colorbar(im, ax=ax, label="p-value")
        plt.tight_layout()
        plot_path = out_dir / f"degree_bins_compare_{name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {plot_path}")


def _union_find_groups(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """给定节点数 n 和边列表 (i,j)，返回连通分量（每组为节点下标列表）。"""
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j in edges:
        union(i, j)
    comp: dict = {}
    for i in range(n):
        root = find(i)
        comp.setdefault(root, []).append(i)
    return list(comp.values())


def _nan_to_none_for_json(x):
    """递归把 NaN 转为 None 以便 JSON 序列化。"""
    if isinstance(x, np.ndarray):
        return _nan_to_none_for_json(x.tolist())
    if isinstance(x, list):
        return [_nan_to_none_for_json(v) for v in x]
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def compare_distributions_across_thickness_bins(
    df: pd.DataFrame,
    data_dict: dict,
    results: dict,
    out_dir: Path,
    *,
    min_points_per_bin: int = 30,
    p_value_threshold: float = 0.05,
) -> None:
    """
    对不同厚度区间的半径分布做两两 KS 检验，找出「近乎相同」的区间。
    热图使用完整厚度轴：样本不足的区间不参与比较，对应行列显示为灰格，便于看出在哪些厚度缺数据。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for radius_type in ["pore", "throat"]:
        for is_wt_val, name in [(True, "WT"), (False, "AS")]:
            res = results.get(radius_type, {}).get(name, {})
            bin_edges = np.array(res.get("bin_edges", []))
            if len(bin_edges) < 2:
                continue
            bins_data = collect_radii_by_thickness_bin(
                df, data_dict, radius_type, is_wt_val, bin_edges
            )
            n_all = len(bins_data)
            # 轴标签带上每个区间内的半径样本数 n
            all_labels = []
            for i in range(n_all):
                _, t_min, t_max, radii = bins_data[i]
                all_labels.append(f"[{t_min:.0f},{t_max:.0f}) (n={len(radii)})")
            valid_indices = [i for i in range(n_all) if len(bins_data[i][3]) >= min_points_per_bin]
            if len(valid_indices) < 2:
                print(f"  {radius_type} {name}: 有效区间数 < 2，跳过比较")
                continue

            radii_by_idx = {i: bins_data[i][3] for i in valid_indices}
            valid_labels = [all_labels[i] for i in valid_indices]
            n_valid = len(valid_indices)

            p_full = np.full((n_all, n_all), np.nan)
            ks_full = np.full((n_all, n_all), np.nan)
            for ii, i in enumerate(valid_indices):
                for jj, j in enumerate(valid_indices):
                    if i == j:
                        p_full[i, j] = 1.0
                        ks_full[i, j] = 0.0
                        continue
                    if i > j:
                        continue
                    stat, pval = ks_2samp(radii_by_idx[i], radii_by_idx[j])
                    p_full[i, j] = p_full[j, i] = float(pval)
                    ks_full[i, j] = ks_full[j, i] = float(stat)

            p_valid = np.ones((n_valid, n_valid))
            for ii in range(n_valid):
                for jj in range(ii + 1, n_valid):
                    p_valid[ii, jj] = p_valid[jj, ii] = p_full[valid_indices[ii], valid_indices[jj]]
            similar_edges = [
                (ii, jj) for ii in range(n_valid) for jj in range(ii + 1, n_valid)
                if p_valid[ii, jj] > p_value_threshold
            ]
            groups = _union_find_groups(n_valid, similar_edges)
            similar_groups = [[valid_labels[ii] for ii in g] for g in groups]

            summary = {
                "radius_type": radius_type,
                "name": name,
                "p_value_threshold": p_value_threshold,
                "min_points_per_bin": min_points_per_bin,
                "bin_labels_full": all_labels,
                "valid_bin_indices": valid_indices,
                "valid_bin_labels": valid_labels,
                "n_bins_total": n_all,
                "n_bins_with_sufficient_data": n_valid,
                "p_value_matrix_full": _nan_to_none_for_json(p_full),
                "ks_statistic_matrix_full": _nan_to_none_for_json(ks_full),
                "similar_groups": similar_groups,
                "note": "p > threshold 表示两区间分布无法拒绝相同。仅样本量>=min_points_per_bin的区间参与比较；缺数据区间在矩阵中为null。",
            }
            json_path = out_dir / f"radius_bins_compare_{radius_type}_{name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            excel_path = out_dir / f"radius_bins_compare_{radius_type}_{name}.xlsx"
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                pd.DataFrame(p_full, index=all_labels, columns=all_labels).to_excel(
                    writer, sheet_name="p_value"
                )
                pd.DataFrame(ks_full, index=all_labels, columns=all_labels).to_excel(
                    writer, sheet_name="KS_statistic"
                )
                pd.DataFrame({"similar_group": [str(g) for g in similar_groups]}).to_excel(
                    writer, sheet_name="similar_groups", index=False
                )

            fig, ax = plt.subplots(figsize=(max(6, n_all * 0.5), max(5, n_all * 0.5)))
            p_plot = np.ma.array(p_full, mask=np.isnan(p_full))
            im = ax.imshow(p_plot, cmap="RdYlGn", vmin=0, vmax=1)
            im.cmap.set_bad(color="lightgray")
            ax.set_xticks(range(n_all))
            ax.set_yticks(range(n_all))
            ax.set_xticklabels(all_labels, rotation=45, ha="right")
            ax.set_yticklabels(all_labels)
            for i in range(n_all):
                for j in range(n_all):
                    if np.isfinite(p_full[i, j]):
                        txt = f"{p_full[i, j]:.2f}"
                        ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                                color="black" if 0.3 < p_full[i, j] < 0.7 else "white")
                    else:
                        ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
            ax.set_xlabel("Thickness bin (nm)")
            ax.set_ylabel("Thickness bin (nm)")
            ax.set_title(
                f"{name} {radius_type} radius: KS test p-value between bins\n"
                f"(p > {p_value_threshold} similar; gray = n<{min_points_per_bin})"
            )
            plt.colorbar(im, ax=ax, label="p-value")
            plt.tight_layout()
            plot_path = out_dir / f"radius_bins_compare_{radius_type}_{name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"  {radius_type} {name}: 总区间数 {n_all}，有效区间数 {n_valid}，比较矩阵与相似组已保存 ({len(similar_groups)} 组)")
            for idx, g in enumerate(similar_groups):
                if len(g) >= 2:
                    print(f"    相似组 {idx + 1}: {g}")


def analyze_pore_throat_radius_correlation(
    df: pd.DataFrame,
    data_dict: dict,
    split_samples_dir: Path,
    out_dir: Path,
) -> dict:
    """
    2.3 孔半径与喉半径的相关性
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== 孔半径与喉半径相关性分析 ===")
    
    results = {}
    
    # 收集每条喉对应的孔半径和喉半径
    for is_wt_val, name in [(True, "WT"), (False, "AS")]:
        df_subset = df[df["is_wt"] == is_wt_val]
        
        pore_radii_list = []
        throat_radii_list = []
        pore1_radii_list = []
        pore2_radii_list = []
        
        for _, row in df_subset.iterrows():
            key = (row["sample_name"], row["sub_name"])
            if key not in data_dict:
                continue
            
            arrays = data_dict[key]
            if "pore_radii" not in arrays or "throat_radii" not in arrays:
                continue
            
            # 需要加载孔喉连接关系
            sample_dir = split_samples_dir / row["sample_name"]
            pores_file = sample_dir / f"{row['sub_name']}_pores.xlsx"
            throats_file = sample_dir / f"{row['sub_name']}_throats.xlsx"
            
            try:
                pore_coords, pore_radii_arr, pore_ids, pore_id_to_idx, throat_pore1, throat_pore2, throat_radii_arr = \
                    st.load_pores_and_throats(pores_file, throats_file)
                
                # 对每条喉，获取两端孔的半径
                for i in range(len(throat_pore1)):
                    p1_id = int(throat_pore1[i])
                    p2_id = int(throat_pore2[i])
                    if p1_id in pore_id_to_idx and p2_id in pore_id_to_idx:
                        p1_r = pore_radii_arr[pore_id_to_idx[p1_id]]
                        p2_r = pore_radii_arr[pore_id_to_idx[p2_id]]
                        t_r = throat_radii_arr[i]
                        
                        pore1_radii_list.append(p1_r)
                        pore2_radii_list.append(p2_r)
                        throat_radii_list.append(t_r)
                        pore_radii_list.append((p1_r + p2_r) / 2.0)  # 平均孔半径
            except Exception as e:
                continue
        
        if len(throat_radii_list) == 0:
            print(f"  {name}: 无数据")
            continue
        
        throat_radii_arr = np.array(throat_radii_list, dtype=float)
        pore_radii_arr = np.array(pore_radii_list, dtype=float)  # 平均孔半径
        pore1_radii_arr = np.array(pore1_radii_list, dtype=float)
        pore2_radii_arr = np.array(pore2_radii_list, dtype=float)
        
        print(f"  {name}: {len(throat_radii_arr)} 条喉")
        
        # 喉半径 vs 平均孔半径
        r_pearson, p_pearson = pearsonr(pore_radii_arr, throat_radii_arr)
        r_spearman, p_spearman = spearmanr(pore_radii_arr, throat_radii_arr)
        print(f"    喉半径 vs 平均孔半径:")
        print(f"      Pearson r = {r_pearson:.4f}, p = {p_pearson:.4f}")
        print(f"      Spearman r = {r_spearman:.4f}, p = {p_spearman:.4f}")
        
        # 喉半径 vs 孔1半径
        r1_pearson, p1_pearson = pearsonr(pore1_radii_arr, throat_radii_arr)
        r1_spearman, p1_spearman = spearmanr(pore1_radii_arr, throat_radii_arr)
        print(f"    喉半径 vs 孔1半径:")
        print(f"      Pearson r = {r1_pearson:.4f}, p = {p1_pearson:.4f}")
        print(f"      Spearman r = {r1_spearman:.4f}, p = {p1_spearman:.4f}")
        
        # 喉半径 vs 孔2半径
        r2_pearson, p2_pearson = pearsonr(pore2_radii_arr, throat_radii_arr)
        r2_spearman, p2_spearman = spearmanr(pore2_radii_arr, throat_radii_arr)
        print(f"    喉半径 vs 孔2半径:")
        print(f"      Pearson r = {r2_pearson:.4f}, p = {p2_pearson:.4f}")
        print(f"      Spearman r = {r2_spearman:.4f}, p = {p2_spearman:.4f}")
        
        # 1) 相关性与线性回归结果（保持原有结构）
        results[name] = {
            "n_throats": len(throat_radii_arr),
            "throat_vs_mean_pore": {
                "pearson_r": float(r_pearson),
                "pearson_p": float(p_pearson),
                "spearman_r": float(r_spearman),
                "spearman_p": float(p_spearman),
            },
            "throat_vs_pore1": {
                "pearson_r": float(r1_pearson),
                "pearson_p": float(p1_pearson),
                "spearman_r": float(r1_spearman),
                "spearman_p": float(p1_spearman),
            },
            "throat_vs_pore2": {
                "pearson_r": float(r2_pearson),
                "pearson_p": float(p2_pearson),
                "spearman_r": float(r2_spearman),
                "spearman_p": float(p2_spearman),
            },
        }

        # 2) 喉半径在不同“平均孔半径”区间下的条件分布 p(r | mean_pore in bin)
        #    生成联合分布分析结果（供 Phase3 使用）并画 2D 联合分布图。
        try:
            n_bins_mean_pore = 10
            p_min = float(np.min(pore_radii_arr))
            p_max = float(np.max(pore_radii_arr))
            if np.isfinite(p_min) and np.isfinite(p_max) and p_max > p_min:
                bin_edges = np.linspace(p_min, p_max + 1e-6, n_bins_mean_pore + 1)
                bins_stats: list[dict] = []
                min_points_per_bin = 50
                for i_bin in range(len(bin_edges) - 1):
                    mp_min = float(bin_edges[i_bin])
                    mp_max = float(bin_edges[i_bin + 1])
                    mp_center = 0.5 * (mp_min + mp_max)
                    mask_bin = (pore_radii_arr >= mp_min) & (pore_radii_arr < mp_max)
                    r_bin = throat_radii_arr[mask_bin]
                    rec: dict = {
                        "mean_pore_center": mp_center,
                        "mean_pore_min": mp_min,
                        "mean_pore_max": mp_max,
                        "n": int(len(r_bin)),
                    }
                    if len(r_bin) >= min_points_per_bin:
                        fit = _fit_radius_distribution_one_bin(
                            r_bin, min_points=min_points_per_bin
                        )
                        if fit:
                            rec["fit"] = fit
                            rec["distribution"] = fit.get("distribution")
                            if fit.get("fit_type") == "gmm":
                                rec["n_components"] = fit["n_components"]
                                rec["weights"] = fit["weights"]
                                rec["means"] = fit["means"]
                                rec["covariances"] = fit["covariances"]
                            print(
                                f"    mean_pore∈[{mp_min:.2f},{mp_max:.2f}) nm: n={len(r_bin)}, 拟合={fit.get('fit_type','')}"
                            )
                    bins_stats.append(rec)

                results[name]["throat_radius_by_mean_pore_bins"] = {
                    "bin_edges_mean_pore": bin_edges.tolist(),
                    "bins": bins_stats,
                }

                # 使用 GMM 拟合 (mean_pore, throat_radius) 的 2D 联合分布
                gmm_joint = _fit_2d_gmm_joint(
                    pore_radii_arr,
                    throat_radii_arr,
                    min_components=1,
                    max_components=4,
                    random_state=0,
                )
                if gmm_joint:
                    results[name]["joint_gmm_mean_pore_throat"] = gmm_joint

                # 2D 联合分布可视化：hexbin 显示 (mean_pore, throat_radius)
                plt.figure(figsize=(6, 5))
                hb = plt.hexbin(
                    pore_radii_arr,
                    throat_radii_arr,
                    gridsize=60,
                    cmap="viridis",
                    mincnt=1,
                )
                plt.colorbar(hb, label="Count")
                plt.xlabel("Mean pore radius (nm)")
                plt.ylabel("Throat radius (nm)")
                plt.title(f"Throat radius vs mean pore radius joint ({name})")
                out_png = out_dir / f"throat_radius_vs_mean_pore_joint_{name}.png"
                plt.tight_layout()
                plt.savefig(out_png, dpi=150)
                plt.close()
                print(f"  {name}: 喉半径 vs 平均孔半径联合分布图已保存: {out_png}")

                # 拟合的联合分布曲面：X=mean_pore, Y=radius, Z=p(mean_pore, r)
                if gmm_joint:
                    y_min = max(0.1, float(np.min(throat_radii_arr)))
                    y_max = float(np.max(throat_radii_arr))
                    y_grid = np.linspace(y_min, y_max, 80)
                    x_grid = np.linspace(p_min, p_max, 80)
                    Z = _gmm2d_pdf_grid(x_grid, y_grid, gmm_joint)
                fig = plt.figure(figsize=(7, 5))
                ax3d = fig.add_subplot(111, projection="3d")
                if gmm_joint:
                    X_surf, Y_surf = np.meshgrid(x_grid, y_grid, indexing="ij")
                    surf = ax3d.plot_surface(
                        X_surf,
                        Y_surf,
                        Z,
                        cmap="viridis",
                        linewidth=0,
                        antialiased=True,
                    )
                    fig.colorbar(surf, shrink=0.6, aspect=12, label="Joint PDF p(mean_pore, r)")
                ax3d.set_xlabel("Mean pore radius (nm)")
                ax3d.set_ylabel("Throat radius r (nm)")
                ax3d.set_zlabel("Joint PDF")
                ax3d.set_title(f"Fitted joint PDF surface: p(mean_pore, r) ({name})")
                out_surf = out_dir / f"throat_radius_vs_mean_pore_surface_{name}.png"
                plt.tight_layout()
                plt.savefig(out_surf, dpi=150)
                plt.close()
                print(f"  {name}: 喉半径 vs 平均孔半径拟合曲面图已保存: {out_surf}")
        except Exception as e:
            print(f"  {name}: 生成 mean_pore×喉半径联合分布时出错: {e}")
    
    # 保存结果
    results_path = out_dir / "pore_throat_radius_correlation.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n孔喉半径相关性分析结果已保存: {results_path}")
    
    return results


def analyze_directionality_vs_thickness(
    df: pd.DataFrame,
    out_dir: Path,
) -> dict:
    """
    2.4 喉方向性（Q、S）与厚度
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "mean_S": "全局有序参数 S",
        "Q_max": "Q_max",
        "Q_min": "Q_min",
        "cos_Qmax_normal": "cos(喉主取向 vs 渗透方向)",
    }
    
    results = {}
    
    for metric, label in metrics.items():
        print(f"\n=== {label} vs 厚度 ===")
        results[metric] = {}
        
        for is_wt_val, name in [(True, "WT"), (False, "AS")]:
            subset = df[df["is_wt"] == is_wt_val]
            thickness = subset["thickness"].dropna().values
            values = subset[metric].dropna().values
            
            if len(thickness) == 0 or len(values) == 0:
                print(f"  {name}: 无数据")
                continue
            
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            print(f"  {name}: 均值 = {mean_val:.6f}, 标准差 = {std_val:.6f}")
            
            if len(thickness) == len(values) and len(thickness) > 2:
                r_pearson, p_pearson = pearsonr(thickness, values)
                r_spearman, p_spearman = spearmanr(thickness, values)
                print(f"    Pearson r = {r_pearson:.4f}, p = {p_pearson:.4f}")
                print(f"    Spearman r = {r_spearman:.4f}, p = {p_spearman:.4f}")
                
                slope, intercept, r_value, p_value, std_err = linregress(thickness, values)
                print(f"    线性回归: y = {slope:.6e} * x + {intercept:.6f}, R^2 = {r_value**2:.4f}")
                
                results[metric][name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "correlation": {
                        "pearson_r": float(r_pearson),
                        "pearson_p": float(p_pearson),
                        "spearman_r": float(r_spearman),
                        "spearman_p": float(p_spearman),
                    },
                    "linear_regression": {
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "r_squared": float(r_value**2),
                        "p_value": float(p_value),
                        "std_err": float(std_err),
                    },
                }
            else:
                results[metric][name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "correlation": None,
                    "linear_regression": None,
                }
    
    # 保存结果
    results_path = out_dir / "directionality_vs_thickness_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n方向性分析结果已保存: {results_path}")
    
    return results


THROAT_LENGTH_BIN_WIDTH_NM = 10.0  # 喉长度按厚度区间分层宽度（与喉半径一致，此处喉长度指孔心距 d）

# 与 phase3_synthetic_network.TOL_NM_OVERLAP 一致：overlap 判定与交界面圆 R_cap
OVERLAP_TOL_NM = 0.2
# 拟合时对 ratio 的上限裁剪（与经验直方图展示一致，减轻长尾对参数拟合的影响）
OVERLAP_RATIO_CLIP_MAX_FIT = 10.0
# 2.5e：cross-face 非 overlap 喉密度统计的默认参数
NONOVERLAP_CROSS_CUT_QUANTILES = (1.0 / 3.0, 2.0 / 3.0)  # 每个子样本两次切割
NONOVERLAP_CROSS_SIDE_EPS_NM = 1e-6
NONOVERLAP_CROSS_MIN_AREA_NM2 = 1e-9


def _intersection_circle_radius_nm_overlap(d: float, R1: float, R2: float, tol: float = OVERLAP_TOL_NM) -> float:
    """
    两球心距 d，孔半径 R1、R2（nm）。返回两球交线圆半径 R_cap；无交线圆时 nan。
    与 gbm_full_model.phase3_synthetic_network._intersection_circle_radius_nm 定义一致。
    """
    if not (np.isfinite(d) and np.isfinite(R1) and np.isfinite(R2)):
        return float("nan")
    if R1 <= 0 or R2 <= 0 or d <= 0:
        return float("nan")
    if d >= R1 + R2 - tol:
        return float("nan")
    if d <= abs(R1 - R2) + tol:
        return float("nan")
    a1 = (d * d + R1 * R1 - R2 * R2) / (2.0 * d)
    sq = R1 * R1 - a1 * a1
    if sq <= 0:
        return float("nan")
    return float(np.sqrt(sq))


def _get_throat_lengths_from_row(row, data_dict, split_samples_dir, st_module):
    """从 data_dict 或文件加载获取子样本的喉长度数组。优先 data_dict，避免重复读取。
    此处“喉长度”指孔心距 d = |p1-p2|，不减去孔半径。
    """
    key = (row["sample_name"], row["sub_name"])
    if data_dict and key in data_dict and "throat_lengths" in data_dict[key]:
        return data_dict[key]["throat_lengths"]
    # 回退：从文件加载（兼容无 data_dict 的调用）
    from gbm_sieving.analysis.structure.compare_subsamples import find_sub_samples
    sample_dir = split_samples_dir / row["sample_name"]
    sub_list = find_sub_samples(sample_dir, row["sample_name"])
    for sn, pf, tf in sub_list:
        if sn == row["sub_name"] and pf and pf.exists() and tf and tf.exists():
            try:
                pore_coords, _, _, pore_id_to_idx, throat_pore1, throat_pore2, throat_radii = \
                    st_module.load_pores_and_throats(pf, tf)
                lengths = []
                for i in range(len(throat_pore1)):
                    p1_id, p2_id = int(throat_pore1[i]), int(throat_pore2[i])
                    if p1_id in pore_id_to_idx and p2_id in pore_id_to_idx:
                        idx1 = pore_id_to_idx[p1_id]
                        idx2 = pore_id_to_idx[p2_id]
                        d = np.linalg.norm(pore_coords[idx1] - pore_coords[idx2])
                        lengths.append(d)
                return np.array(lengths) if lengths else np.array([], dtype=float)
            except Exception:
                pass
            break
    return None


def _get_throat_effective_lengths_from_row(row, split_samples_dir, st_module):
    """
    从文件重新计算子样本的“有效喉长”数组：
    L_raw = |p1-p2| - r_pore1 - r_pore2（允许出现 <=0 的值，不做截断）。
    仅用于 Phase2 中的真实 L_raw 分布可视化。
    """
    from gbm_sieving.analysis.structure.compare_subsamples import find_sub_samples

    sample_dir = split_samples_dir / row["sample_name"]
    sub_list = find_sub_samples(sample_dir, row["sample_name"])
    for sn, pf, tf in sub_list:
        if sn == row["sub_name"] and pf and pf.exists() and tf and tf.exists():
            try:
                pore_coords, pore_radii, _, pore_id_to_idx, throat_pore1, throat_pore2, throat_radii = \
                    st_module.load_pores_and_throats(pf, tf)
                lengths = []
                for i in range(len(throat_pore1)):
                    p1_id, p2_id = int(throat_pore1[i]), int(throat_pore2[i])
                    if p1_id in pore_id_to_idx and p2_id in pore_id_to_idx:
                        idx1 = pore_id_to_idx[p1_id]
                        idx2 = pore_id_to_idx[p2_id]
                        d = np.linalg.norm(pore_coords[idx1] - pore_coords[idx2])
                        r_p1 = float(pore_radii[idx1])
                        r_p2 = float(pore_radii[idx2])
                        L_raw = d - r_p1 - r_p2
                        lengths.append(L_raw)
                return np.array(lengths, dtype=float) if lengths else np.array([], dtype=float)
            except Exception:
                pass
            break
    return None


def _get_overlap_R_ratios_from_row(row, split_samples_dir: Path, st_module) -> np.ndarray | None:
    """
    对子样本中每条 overlap 喉（d < R1+R2-tol）计算 ratio = r_throat / R_cap，返回一维数组。
    与 Phase3 中 overlap 喉的几何定义一致。
    """
    from gbm_sieving.analysis.structure.compare_subsamples import find_sub_samples

    sample_dir = split_samples_dir / row["sample_name"]
    sub_list = find_sub_samples(sample_dir, row["sample_name"])
    tol = OVERLAP_TOL_NM
    ratios: list[float] = []
    for sn, pf, tf in sub_list:
        if sn != row["sub_name"] or not pf or not pf.exists() or not tf or not tf.exists():
            continue
        try:
            pore_coords, pore_radii, _, pore_id_to_idx, throat_pore1, throat_pore2, throat_radii = (
                st_module.load_pores_and_throats(pf, tf)
            )
            for i in range(len(throat_pore1)):
                p1_id, p2_id = int(throat_pore1[i]), int(throat_pore2[i])
                if p1_id not in pore_id_to_idx or p2_id not in pore_id_to_idx:
                    continue
                idx1 = pore_id_to_idx[p1_id]
                idx2 = pore_id_to_idx[p2_id]
                d = float(np.linalg.norm(pore_coords[idx1] - pore_coords[idx2]))
                R1 = float(pore_radii[idx1])
                R2 = float(pore_radii[idx2])
                rt = float(throat_radii[i])
                if not (R1 > 0 and R2 > 0 and d > 0):
                    continue
                if d >= R1 + R2 - tol:
                    continue
                rcap = _intersection_circle_radius_nm_overlap(d, R1, R2, tol)
                if not (np.isfinite(rcap) and rcap > 0):
                    continue
                ratio = rt / rcap
                if np.isfinite(ratio) and ratio > 0:
                    ratios.append(ratio)
        except Exception:
            pass
        break
    if not ratios:
        return np.array([], dtype=float)
    return np.asarray(ratios, dtype=float)


def _get_throat_gt_min_pore_fractions_from_row(
    row,
    split_samples_dir: Path,
    st_module,
    *,
    tol_nm: float = OVERLAP_TOL_NM,
) -> tuple[float, int, float, int]:
    """
    单个子样本：overlap / 非 overlap 喉中，r_throat > min(两端孔半径) 的比例各一条。

    返回 (frac_overlap, n_overlap_throats, frac_nonoverlap, n_nonoverlap_throats)。
    若某侧喉条数为 0，对应 frac 为 nan。
    overlap 判定：d < R1+R2-tol。
    「超过小孔」：rt > min(R1,R2)（严格大于）。
    """
    from gbm_sieving.analysis.structure.compare_subsamples import find_sub_samples

    sample_dir = split_samples_dir / row["sample_name"]
    sub_list = find_sub_samples(sample_dir, row["sample_name"])
    n_ov = n_ov_gt = n_non = n_non_gt = 0

    for sn, pf, tf in sub_list:
        if sn != row["sub_name"] or not pf or not pf.exists() or not tf or not tf.exists():
            continue
        try:
            pore_coords, pore_radii, _, pore_id_to_idx, throat_pore1, throat_pore2, throat_radii = (
                st_module.load_pores_and_throats(pf, tf)
            )
            for i in range(len(throat_pore1)):
                p1_id, p2_id = int(throat_pore1[i]), int(throat_pore2[i])
                if p1_id not in pore_id_to_idx or p2_id not in pore_id_to_idx:
                    continue
                idx1 = pore_id_to_idx[p1_id]
                idx2 = pore_id_to_idx[p2_id]
                d = float(np.linalg.norm(pore_coords[idx1] - pore_coords[idx2]))
                R1 = float(pore_radii[idx1])
                R2 = float(pore_radii[idx2])
                rt = float(throat_radii[i])
                if not (np.isfinite(d) and np.isfinite(R1) and np.isfinite(R2) and np.isfinite(rt)):
                    continue
                if d <= 0 or R1 <= 0 or R2 <= 0 or rt <= 0:
                    continue
                rmin = R1 if R1 <= R2 else R2
                if rmin <= 0:
                    continue
                is_ov = d < (R1 + R2 - tol_nm)
                gt = rt > rmin
                if is_ov:
                    n_ov += 1
                    if gt:
                        n_ov_gt += 1
                else:
                    n_non += 1
                    if gt:
                        n_non_gt += 1
        except Exception:
            pass
        break

    frac_ov = float(n_ov_gt / n_ov) if n_ov > 0 else float("nan")
    frac_non = float(n_non_gt / n_non) if n_non > 0 else float("nan")
    return frac_ov, n_ov, frac_non, n_non


def collect_throat_gt_min_pore_frac_per_subsample_by_thickness_bin(
    df: pd.DataFrame,
    split_samples_dir: Path,
    is_wt: bool,
    bin_edges: np.ndarray,
    st_module,
    *,
    kind: str,
) -> List[Tuple[float, float, float, np.ndarray]]:
    """
    按厚度区间收集「子样本级」比例：该子样本内 overlap 或 非 overlap 喉中 r_throat > min(孔) 的比例。
    kind: \"overlap\" | \"nonoverlap\"
    返回 [(t_center, t_min, t_max, fracs_array), ...]，fracs 为每个子样本一个标量（[0,1]），
    仅包含该侧喉数 > 0 的子样本。
    """
    assert kind in ("overlap", "nonoverlap")
    out: List[Tuple[float, float, float, np.ndarray]] = []
    df_sub = df[df["is_wt"] == is_wt]
    for i in range(len(bin_edges) - 1):
        t_min, t_max = float(bin_edges[i]), float(bin_edges[i + 1])
        t_center = (t_min + t_max) / 2.0
        fracs: list[float] = []
        for _, row in df_sub.iterrows():
            t = row.get("thickness")
            if pd.isna(t) or t < t_min or t >= t_max:
                continue
            frac_ov, n_ov, frac_non, n_non = _get_throat_gt_min_pore_fractions_from_row(
                row, split_samples_dir, st_module
            )
            if kind == "overlap":
                if n_ov > 0 and np.isfinite(frac_ov):
                    fracs.append(float(frac_ov))
            else:
                if n_non > 0 and np.isfinite(frac_non):
                    fracs.append(float(frac_non))
        arr = np.asarray(fracs, dtype=float)
        out.append((t_center, t_min, t_max, arr))
    return out


def analyze_throat_gt_min_pore_fraction_per_subsample_by_thickness_bins(
    df: pd.DataFrame,
    split_samples_dir: Path,
    out_dir: Path,
    *,
    bin_width: float | None = None,
    min_points_per_bin: int = 5,
) -> dict:
    """
    按厚度区间：对每个子样本计算 overlap / 非 overlap 喉中「r_throat > min(两端孔半径)」的比例；
    再在各区间内对该 **比例** 的分布（跨子样本）做直方图与 norm 拟合（与 density_frac 子样本指标一致）。
    """
    if bin_width is None:
        bin_width = DENSITY_FRAC_BIN_WIDTH_NM
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_samples_dir = Path(split_samples_dir)

    print("\n=== 子样本级：喉 r_throat > min(两端孔半径) 比例 分布 vs 厚度（overlap / 非overlap 分开）===")

    results: dict = {
        "overlap_tol_nm": OVERLAP_TOL_NM,
        "metric": "per sub-sample fraction of throats with r_throat > min(r_pore1,r_pore2)",
        "overlap": {},
        "nonoverlap": {},
    }

    for kind in ("overlap", "nonoverlap"):
        results[kind] = {}
        for is_wt_val, name in [(True, "WT"), (False, "AS")]:
            df_subset = df[df["is_wt"] == is_wt_val]
            thickness_arr = df_subset["thickness"].dropna().values
            if len(thickness_arr) == 0:
                print(f"  [{kind}] {name}: 无厚度数据，跳过")
                results[kind][name] = {"bin_edges": [], "bins": []}
                continue

            bin_edges = get_thickness_bin_edges_density_frac(thickness_arr, is_wt_val, bin_width=float(bin_width))
            bins_data = collect_throat_gt_min_pore_frac_per_subsample_by_thickness_bin(
                df, split_samples_dir, is_wt_val, bin_edges, st, kind=kind
            )
            print(f"  [{kind}] {name}: 厚度分层边界 = {bin_edges.tolist()}")

            all_f = np.concatenate([b[3] for b in bins_data if len(b[3]) > 0]) if bins_data else np.array([])
            if len(all_f) == 0:
                print(f"  [{kind}] {name}: 无子样本比例数据")
                results[kind][name] = {
                    "n_subsamples": 0,
                    "bin_width_nm": float(bin_width),
                    "bin_edges": bin_edges.tolist(),
                    "bins": [],
                }
                continue

            print(
                f"  [{kind}] {name}: 共 {len(all_f)} 个子样本比例点，"
                f"均值 = {float(np.mean(all_f)):.4f}, 标准差 = {float(np.std(all_f)):.4f}"
            )

            bin_results: list[dict] = []
            for t_center, t_min, t_max, fracs in bins_data:
                rec: dict = {
                    "t_center": float(t_center),
                    "t_min": float(t_min),
                    "t_max": float(t_max),
                    "n_subsamples": int(len(fracs)),
                }
                if len(fracs) >= min_points_per_bin:
                    fit = _fit_metric_distribution_one_bin(fracs, min_points=min_points_per_bin, metric="frac_gt_min_pore")
                    if fit:
                        rec["fit"] = fit
                        rec["distribution"] = fit.get("distribution")
                        print(f"    区间 [{t_min:.0f},{t_max:.0f}) nm: n_sub={len(fracs)}, 拟合=norm")
                bin_results.append(rec)

            if name == "AS":
                aux_source_bins = _make_aux_right_smoothing_bins(
                    collect_throat_gt_min_pore_frac_per_subsample_by_thickness_bin(
                        df,
                        split_samples_dir,
                        is_wt_val,
                        np.asarray(AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM, dtype=float),
                        st,
                        kind=kind,
                    ),
                    fit_builder=lambda vals: _fit_metric_distribution_one_bin(
                        vals, min_points=min_points_per_bin, metric="frac_gt_min_pore"
                    ),
                    min_points=min_points_per_bin,
                    n_key="n_subsamples",
                )
                smooth_as_sparse_bin_fits(
                    bin_results,
                    fit_builder=lambda vals: _fit_metric_distribution_one_bin(
                        vals, min_points=min_points_per_bin, metric="frac_gt_min_pore"
                    ),
                    value_kind="fraction",
                    label=f"throat_gt_min_pore_fraction_{kind}",
                    min_points=min_points_per_bin,
                    n_key="n_subsamples",
                    seed_offset=6000 if kind == "overlap" else 6100,
                    extra_source_bins=aux_source_bins,
                )

            n_fit = sum(1 for b in bin_results if "fit" in b)
            print(f"  [{kind}] {name}: {len(bin_results)} 个区间, {n_fit} 个区间有拟合")

            results[kind][name] = {
                "mean": float(np.mean(all_f)),
                "std": float(np.std(all_f)),
                "median": float(np.median(all_f)),
                "min": float(np.min(all_f)),
                "max": float(np.max(all_f)),
                "n_subsamples": int(len(all_f)),
                "bin_width_nm": float(bin_width),
                "bin_edges": bin_edges.tolist(),
                "bins": bin_results,
                "description": (
                    f"Across sub-samples: fraction of {kind} throats with r_throat > min(r_pore1,r_pore2)"
                ),
            }

    results_path = out_dir / "throat_gt_min_pore_frac_per_subsample_by_thickness_bin.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n分析结果已保存: {results_path}")

    plot_throat_gt_min_pore_fraction_per_subsample_by_thickness_bins(
        df,
        results,
        split_samples_dir,
        out_dir,
        min_points_per_bin=min_points_per_bin,
    )
    return results


def plot_throat_gt_min_pore_fraction_per_subsample_by_thickness_bins(
    df: pd.DataFrame,
    results: dict,
    split_samples_dir: Path,
    out_dir: Path,
    *,
    min_points_per_bin: int = 5,
) -> None:
    """overlap / 非overlap 各一张图：每厚度区间为「子样本比例」直方图 + norm 拟合。"""
    out_dir = Path(out_dir)
    split_samples_dir = Path(split_samples_dir)

    for kind in ("overlap", "nonoverlap"):
        kind_results = results.get(kind) or {}
        for name in ["WT", "AS"]:
            res_name = kind_results.get(name, {})
            bin_edges = np.array(res_name.get("bin_edges", []), dtype=float)
            if len(bin_edges) < 2:
                continue
            is_wt = name == "WT"
            bins_data = collect_throat_gt_min_pore_frac_per_subsample_by_thickness_bin(
                df, split_samples_dir, is_wt, bin_edges, st, kind=kind
            )
            panels = [
                (t_center, t_min, t_max, fracs)
                for t_center, t_min, t_max, fracs in bins_data
                if len(fracs) >= min_points_per_bin
            ]
            if not panels:
                print(f"  [{kind}] {name}: 无足够子样本的子图 (n_sub>={min_points_per_bin})，跳过作图")
                continue

            n_panels = len(panels)
            n_cols = 4
            n_rows = (n_panels + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
            if n_rows == 1:
                axes = np.atleast_2d(axes)
            axes_flat = axes.flatten()
            bins_list = res_name.get("bins", [])

            for idx, (t_center, t_min, t_max, fracs) in enumerate(panels):
                ax = axes_flat[idx]
                n_bins_hist = min(20, max(5, len(fracs) // 2))
                ax.hist(
                    fracs,
                    bins=n_bins_hist,
                    range=(0.0, 1.0),
                    density=False,
                    weights=np.ones_like(fracs) / len(fracs),
                    alpha=0.55,
                    color="steelblue",
                    edgecolor="white",
                    label="Real data",
                )
                bin_rec = next(
                    (
                        b
                        for b in bins_list
                        if abs(float(b.get("t_min", 0)) - float(t_min)) < 1e-6
                        and abs(float(b.get("t_max", 0)) - float(t_max)) < 1e-6
                        and b.get("fit")
                    ),
                    None,
                )
                if bin_rec and bin_rec.get("fit"):
                    fit = bin_rec["fit"]
                    x_min, x_max = float(np.min(fracs)), float(np.max(fracs))
                    x_pad = max((x_max - x_min) * 0.1, 0.02)
                    x_plot = np.linspace(max(0.0, x_min - x_pad), min(1.0, x_max + x_pad), 200)
                    bin_width_approx = 1.0 / n_bins_hist if n_bins_hist else 1e-6
                    dist_name = fit.get("distribution", {}).get("name")
                    params = fit.get("distribution", {}).get("params")
                    if dist_name and params:
                        pdf_vals = _metric_fitted_pdf(x_plot, dist_name, params)
                        ax.plot(x_plot, pdf_vals * bin_width_approx, "r-", linewidth=2, label=f"Fit {dist_name}")

                ax.set_xlabel(r"Fraction ($r_{\mathrm{throat}} > \min(r_{\mathrm{pore1}},r_{\mathrm{pore2}})$)", fontsize=9)
                ax.set_ylabel("Probability", fontsize=10)
                ax.set_title(f"T in [{t_min:.0f},{t_max:.0f}) nm (n_sub={len(fracs)})", fontsize=10)
                ax.set_xlim(0, 1)
                ax.legend(loc="upper right", fontsize=7)
                ax.grid(True, alpha=0.3)

            for idx in range(len(panels), len(axes_flat)):
                axes_flat[idx].set_visible(False)

            kind_title = "overlap" if kind == "overlap" else "non-overlap"
            plt.suptitle(
                f"Sub-sample fraction of {kind_title} throats exceeding min pore radius by thickness ({name})",
                fontsize=12,
                fontweight="bold",
            )
            plt.tight_layout()
            out_path = out_dir / f"throat_gt_min_pore_frac_per_subsample_thickness_bins_{kind}_{name}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  图已保存: {out_path}")


def _xy_direction_from_row_or_metadata(row, split_samples_dir: Path) -> np.ndarray | None:
    """
    获取“初版渗透方向”在 XY 平面的单位向量 u0。
    优先使用 summary 的 nx/ny；若缺失则回退 metadata.normal_vector。
    """
    nx = pd.to_numeric(row.get("nx"), errors="coerce")
    ny = pd.to_numeric(row.get("ny"), errors="coerce")
    if not (np.isfinite(nx) and np.isfinite(ny)):
        sample_dir = Path(split_samples_dir) / str(row.get("sample_name", ""))
        meta_file = sample_dir / f"{row.get('sub_name', '')}_metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                nv = meta.get("normal_vector", {}) if isinstance(meta, dict) else {}
                nx = pd.to_numeric(nv.get("nx"), errors="coerce")
                ny = pd.to_numeric(nv.get("ny"), errors="coerce")
            except Exception:
                nx, ny = np.nan, np.nan
    if not (np.isfinite(nx) and np.isfinite(ny)):
        return None
    v = np.array([float(nx), float(ny), 0.0], dtype=float)
    norm_v = float(np.linalg.norm(v))
    if norm_v <= 1e-12:
        return None
    return v / norm_v


def _get_nonoverlap_cross_density_two_cuts_from_row(
    row,
    split_samples_dir: Path,
    st_module,
    *,
    cut_quantiles: tuple[float, float] = NONOVERLAP_CROSS_CUT_QUANTILES,
    overlap_tol_nm: float = OVERLAP_TOL_NM,
    side_eps_nm: float = NONOVERLAP_CROSS_SIDE_EPS_NM,
) -> np.ndarray:
    """
    单个子样本的两次切割统计：
      rho_nonoverlap_cross = N_nonoverlap_cross / A_cut
    其中：
      - cut plane 由 (u0, z) 张成，plane normal 为 n_cut = u0 × z
      - N_nonoverlap_cross：两端孔位于平面两侧且非 overlap 的喉数量
      - A_cut：在 (u0, z) 坐标中的截面包围盒面积（固定口径）
    """
    from gbm_sieving.analysis.structure.compare_subsamples import find_sub_samples

    u0 = _xy_direction_from_row_or_metadata(row, split_samples_dir)
    if u0 is None:
        return np.array([], dtype=float)

    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    n_cut = np.cross(u0, z_axis)
    norm_n = float(np.linalg.norm(n_cut))
    if norm_n <= 1e-12:
        return np.array([], dtype=float)
    n_cut = n_cut / norm_n

    sample_dir = split_samples_dir / str(row.get("sample_name", ""))
    sub_list = find_sub_samples(sample_dir, str(row.get("sample_name", "")))
    for sn, pf, tf in sub_list:
        if sn != row.get("sub_name") or not pf or not pf.exists() or not tf or not tf.exists():
            continue
        try:
            pore_coords, pore_radii, _, pore_id_to_idx, throat_pore1, throat_pore2, _ = (
                st_module.load_pores_and_throats(pf, tf)
            )
        except Exception:
            return np.array([], dtype=float)

        if pore_coords is None or len(pore_coords) < 2:
            return np.array([], dtype=float)

        pore_coords = np.asarray(pore_coords, dtype=float)
        pore_radii = np.asarray(pore_radii, dtype=float)
        proj_n = pore_coords @ n_cut
        if not np.all(np.isfinite(proj_n)):
            return np.array([], dtype=float)

        # 固定口径：截面包围盒面积取 (u0, z) 投影后的 bbox 面积
        proj_u0 = pore_coords @ u0
        proj_z = pore_coords @ z_axis
        span_u0 = float(np.nanmax(proj_u0) - np.nanmin(proj_u0))
        span_z = float(np.nanmax(proj_z) - np.nanmin(proj_z))
        area_cut_nm2 = span_u0 * span_z
        if not (np.isfinite(area_cut_nm2) and area_cut_nm2 > NONOVERLAP_CROSS_MIN_AREA_NM2):
            return np.array([], dtype=float)

        cut_levels = np.quantile(proj_n, np.asarray(cut_quantiles, dtype=float))
        out_vals: list[float] = []
        for c in cut_levels:
            side = proj_n - float(c)
            n_cross = 0
            for i in range(len(throat_pore1)):
                p1_id, p2_id = int(throat_pore1[i]), int(throat_pore2[i])
                if p1_id not in pore_id_to_idx or p2_id not in pore_id_to_idx:
                    continue
                idx1 = pore_id_to_idx[p1_id]
                idx2 = pore_id_to_idx[p2_id]
                s1 = float(side[idx1])
                s2 = float(side[idx2])
                if abs(s1) <= side_eps_nm or abs(s2) <= side_eps_nm:
                    continue
                if s1 * s2 >= 0.0:
                    continue

                # 口径一致性：非 overlap 喉
                d = float(np.linalg.norm(pore_coords[idx1] - pore_coords[idx2]))
                R1 = float(pore_radii[idx1])
                R2 = float(pore_radii[idx2])
                if not (np.isfinite(d) and np.isfinite(R1) and np.isfinite(R2)):
                    continue
                if d <= 0.0 or R1 <= 0.0 or R2 <= 0.0:
                    continue
                if d < (R1 + R2 - overlap_tol_nm):
                    continue

                n_cross += 1
            out_vals.append(float(n_cross / area_cut_nm2))

        out_arr = np.asarray(out_vals, dtype=float)
        return out_arr[np.isfinite(out_arr) & (out_arr >= 0.0)]
    return np.array([], dtype=float)


def collect_nonoverlap_cross_density_by_thickness_bin(
    df: pd.DataFrame,
    split_samples_dir: Path,
    is_wt: bool,
    bin_edges: np.ndarray,
    st_module,
) -> List[Tuple[float, float, float, np.ndarray]]:
    """
    按厚度区间收集 cross-face 非 overlap 喉密度样本值（每个子样本两次切割）。
    返回 [(t_center, t_min, t_max, values_array), ...]。
    """
    out: List[Tuple[float, float, float, np.ndarray]] = []
    df_sub = df[df["is_wt"] == is_wt]
    for i in range(len(bin_edges) - 1):
        t_min, t_max = float(bin_edges[i]), float(bin_edges[i + 1])
        t_center = (t_min + t_max) / 2.0
        vals_all: list[np.ndarray] = []
        for _, row in df_sub.iterrows():
            t = row.get("thickness")
            if pd.isna(t) or t < t_min or t >= t_max:
                continue
            arr = _get_nonoverlap_cross_density_two_cuts_from_row(
                row,
                split_samples_dir,
                st_module,
            )
            if arr is not None and len(arr) > 0:
                vals_all.append(np.asarray(arr, dtype=float))
        merged = np.concatenate(vals_all) if vals_all else np.array([], dtype=float)
        out.append((t_center, t_min, t_max, merged))
    return out


def analyze_nonoverlap_cross_density_by_thickness_bin(
    df: pd.DataFrame,
    split_samples_dir: Path,
    out_dir: Path,
    *,
    bin_width: float = None,
    min_points_per_bin: int = 5,
) -> dict:
    """
    2.5e 新统计：
      rho_nonoverlap_cross = N_nonoverlap_cross / A_cut
    按厚度区间统计分布并拟合，输出 JSON + WT/AS 分面图（柱状 + 红色拟合曲线）。
    """
    if bin_width is None:
        bin_width = DENSITY_FRAC_BIN_WIDTH_NM
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_samples_dir = Path(split_samples_dir)

    results = {
        "meta": {
            "metric": "rho_nonoverlap_cross",
            "definition": "N_nonoverlap_cross / A_cut",
            "cut_count_per_subsample": 2,
            "cut_plane_basis": "u0(xy_initial_normal), z_axis",
            "crossing_criterion": "throat endpoints on opposite plane sides",
            "nonoverlap_criterion": f"d >= R1 + R2 - {float(OVERLAP_TOL_NM)}",
            "area_definition": "bounding-box area on cut-plane coordinates (u0, z)",
            "thickness_bin_width_nm": float(bin_width),
            "fit_min_points_per_bin": int(min_points_per_bin),
        },
        "WT": {},
        "AS": {},
    }

    for is_wt_val, name in ((True, "WT"), (False, "AS")):
        thickness_arr = pd.to_numeric(
            df.loc[df["is_wt"] == is_wt_val, "thickness"], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        if len(thickness_arr) == 0:
            results[name] = {"bin_edges": [], "bins": [], "n_values": 0}
            continue

        bin_edges = get_thickness_bin_edges_density_frac(thickness_arr, is_wt_val, bin_width=float(bin_width))
        bins_data = collect_nonoverlap_cross_density_by_thickness_bin(
            df,
            split_samples_dir,
            is_wt_val,
            bin_edges,
            st,
        )
        print(f"  [2.5e] {name}: 厚度分层边界 = {bin_edges.tolist()}")

        bins_out = []
        n_total = 0
        for _, t_min, t_max, vals in bins_data:
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals) & (vals >= 0)]
            n_vals = int(len(vals))
            n_total += n_vals
            fit = _fit_metric_distribution_one_bin(vals, min_points=min_points_per_bin, metric="rho_nonoverlap_cross")
            bins_out.append(
                {
                    "t_min": float(t_min),
                    "t_max": float(t_max),
                    "n_values": n_vals,
                    "raw_values": vals.tolist(),
                    "mean": float(np.mean(vals)) if n_vals > 0 else None,
                    "std": float(np.std(vals)) if n_vals > 1 else 0.0 if n_vals == 1 else None,
                    "fit": fit,
                }
            )
            print(
                f"    区间 [{t_min:.0f},{t_max:.0f}) nm: n={n_vals}, "
                f"拟合={fit.get('fit_type', '') if fit else 'none'}"
            )

        if name == "AS":
            aux_source_bins = _make_aux_right_smoothing_bins(
                collect_nonoverlap_cross_density_by_thickness_bin(
                    df,
                    split_samples_dir,
                    is_wt_val,
                    np.asarray(AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM, dtype=float),
                    st,
                ),
                fit_builder=lambda vals: _fit_metric_distribution_one_bin(
                    vals, min_points=min_points_per_bin, metric="rho_nonoverlap_cross"
                ),
                min_points=min_points_per_bin,
                n_key="n_values",
            )
            smooth_as_sparse_bin_fits(
                bins_out,
                fit_builder=lambda vals: _fit_metric_distribution_one_bin(
                    vals, min_points=min_points_per_bin, metric="rho_nonoverlap_cross"
                ),
                value_kind="density",
                label="nonoverlap_cross_density_by_thickness",
                min_points=min_points_per_bin,
                n_key="n_values",
                seed_offset=7000,
                extra_source_bins=aux_source_bins,
            )

        results[name] = {
            "bin_edges": bin_edges.tolist(),
            "n_values": int(n_total),
            "bins": bins_out,
        }

    results_path = out_dir / "nonoverlap_cross_density_by_thickness_bin.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nnon-overlap cross-face 喉密度分析结果已保存: {results_path}")

    plot_nonoverlap_cross_density_by_thickness_bins(
        results,
        out_dir,
        min_points_per_bin=min_points_per_bin,
    )
    return results


def plot_nonoverlap_cross_density_by_thickness_bins(
    results: dict,
    out_dir: Path,
    *,
    min_points_per_bin: int = 5,
) -> None:
    """
    WT/AS 各一张图：每厚度区间一个子图，展示 rho_nonoverlap_cross 的直方图和拟合曲线（红线）。
    """
    out_dir = Path(out_dir)
    for name in ("WT", "AS"):
        res_name = results.get(name, {})
        bins_list = res_name.get("bins", [])
        if not bins_list:
            continue
        panels = [b for b in bins_list if int(b.get("n_values", 0)) > 0]
        if not panels:
            continue
        n_panels = len(panels)
        n_cols = 4
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = np.atleast_2d(axes)
        axes_flat = axes.flatten()

        for idx, rec in enumerate(panels):
            t_min = float(rec.get("t_min", np.nan))
            t_max = float(rec.get("t_max", np.nan))
            vals = np.asarray(rec.get("raw_values", []), dtype=float)
            vals = vals[np.isfinite(vals) & (vals >= 0)]
            ax = axes_flat[idx]
            if len(vals) == 0:
                ax.set_visible(False)
                continue
            n_bins_hist = min(24, max(6, len(vals) // 2))
            ax.hist(
                vals,
                bins=n_bins_hist,
                density=True,
                alpha=0.55,
                color="steelblue",
                edgecolor="white",
                label="Real data",
            )
            fit = rec.get("fit")
            if fit and int(fit.get("n", 0)) >= int(min_points_per_bin):
                x_min = float(np.min(vals))
                x_max = float(np.max(vals))
                if x_max - x_min < 1e-12:
                    x_pad = max(0.1 * max(x_max, 1e-9), 1e-9)
                else:
                    x_pad = 0.10 * (x_max - x_min)
                x_grid = np.linspace(max(0.0, x_min - x_pad), x_max + x_pad, 240)
                dist_name = fit.get("distribution", {}).get("name")
                params = fit.get("distribution", {}).get("params")
                if dist_name and params:
                    y_fit = _metric_fitted_pdf(x_grid, dist_name, params)
                    ax.plot(x_grid, y_fit, "r-", linewidth=2.0, label=f"Fit {dist_name}")

            ax.set_xlabel(r"$\rho_{\mathrm{nonoverlap\_cross}}$ (1/nm$^2$)", fontsize=9)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(f"T in [{t_min:.0f},{t_max:.0f}) nm (n={len(vals)})", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=7)

        for idx in range(len(panels), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.suptitle(
            f"Non-overlap cross-face throat density by thickness ({name})",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        out_png = out_dir / f"nonoverlap_cross_density_by_thickness_bins_{name}.png"
        plt.savefig(out_png, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"  [2.5e] 图已保存: {out_png}")


def collect_overlap_R_ratios_by_thickness_bin(
    df: pd.DataFrame,
    split_samples_dir: Path,
    is_wt: bool,
    bin_edges: np.ndarray,
    st_module,
    ratio_clip_max: float | None = OVERLAP_RATIO_CLIP_MAX_FIT,
) -> List[Tuple[float, float, float, np.ndarray]]:
    """
    按厚度区间收集 overlap 喉的 R_throat / R_cap（拟合前可对 ratio 做上限裁剪）。
    返回 [(t_center, t_min, t_max, ratios_array), ...]。
    """
    out: List[Tuple[float, float, float, np.ndarray]] = []
    df_sub = df[df["is_wt"] == is_wt]
    for i in range(len(bin_edges) - 1):
        t_min, t_max = float(bin_edges[i]), float(bin_edges[i + 1])
        t_center = (t_min + t_max) / 2.0
        arr_list: list[np.ndarray] = []
        for _, row in df_sub.iterrows():
            t = row.get("thickness")
            if pd.isna(t) or t < t_min or t >= t_max:
                continue
            ratios = _get_overlap_R_ratios_from_row(row, split_samples_dir, st_module)
            if ratios is not None and len(ratios) > 0:
                if ratio_clip_max is not None:
                    ratios = np.clip(ratios, 1e-9, float(ratio_clip_max))
                arr_list.append(ratios)
        concat = np.concatenate(arr_list) if arr_list else np.array([], dtype=float)
        out.append((t_center, t_min, t_max, concat))
    return out


def analyze_overlap_throat_R_ratio_vs_thickness(
    df: pd.DataFrame,
    split_samples_dir: Path,
    out_dir: Path,
    bin_width: float = THROAT_LENGTH_BIN_WIDTH_NM,
    min_points_per_bin: int = 30,
    ratio_clip_max_for_fit: float = OVERLAP_RATIO_CLIP_MAX_FIT,
) -> dict:
    """
    overlap 喉：R_throat / R_cap（交界面圆半径）按厚度分箱统计与拟合，供 Phase3 对几何 R_cap 乘系数抽样。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_samples_dir = Path(split_samples_dir)

    print("\n=== overlap 喉 R_throat / R_cap vs 厚度（按厚度区间分布 + 拟合）===")

    results: dict = {}
    for is_wt_val, name in [(True, "WT"), (False, "AS")]:
        df_subset = df[df["is_wt"] == is_wt_val]
        thickness_arr = df_subset["thickness"].dropna().values
        if len(thickness_arr) == 0:
            print(f"  {name}: 无厚度数据，跳过")
            results[name] = {"n_ratios": 0, "bin_edges": [], "bins": []}
            continue

        bin_edges = get_thickness_bin_edges_for_group(thickness_arr, bin_width, is_wt_val)
        bins_data = collect_overlap_R_ratios_by_thickness_bin(
            df, split_samples_dir, is_wt_val, bin_edges, st, ratio_clip_max=ratio_clip_max_for_fit
        )
        print(f"  {name}: 厚度分层边界 = {bin_edges.tolist()}")

        all_r = np.concatenate([b[3] for b in bins_data if len(b[3]) > 0]) if bins_data else np.array([])
        if len(all_r) == 0:
            print(f"  {name}: 无 overlap ratio 数据")
            results[name] = {
                "n_ratios": 0,
                "bin_width_nm": float(bin_width),
                "bin_edges": bin_edges.tolist(),
                "bins": [],
                "overlap_tol_nm": OVERLAP_TOL_NM,
                "ratio_clip_max_for_fit": float(ratio_clip_max_for_fit),
            }
            continue

        print(
            f"  {name}: 共 {len(all_r)} 个 ratio（拟合前 clip 至 {ratio_clip_max_for_fit}），"
            f"全局均值 = {float(np.mean(all_r)):.3f}, 标准差 = {float(np.std(all_r)):.3f}"
        )

        bin_results: list[dict] = []
        for t_center, t_min, t_max, ratios in bins_data:
            rec: dict = {
                "t_center": float(t_center),
                "t_min": float(t_min),
                "t_max": float(t_max),
                "n": len(ratios),
            }
            if len(ratios) >= min_points_per_bin:
                fit = _fit_radius_distribution_one_bin(ratios, min_points=min_points_per_bin)
                if fit:
                    rec["fit"] = fit
                    rec["distribution"] = fit.get("distribution")
                    if fit.get("fit_type") == "gmm":
                        rec["n_components"] = fit["n_components"]
                        rec["weights"] = fit["weights"]
                        rec["means"] = fit["means"]
                        rec["covariances"] = fit["covariances"]
                    print(f"    区间 [{t_min:.0f},{t_max:.0f}) nm: n={len(ratios)}, 拟合={fit.get('fit_type', '')}")
            bin_results.append(rec)

        if name == "AS":
            aux_source_bins = _make_aux_right_smoothing_bins(
                collect_overlap_R_ratios_by_thickness_bin(
                    df,
                    split_samples_dir,
                    is_wt_val,
                    np.asarray(AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM, dtype=float),
                    st,
                    ratio_clip_max=ratio_clip_max_for_fit,
                ),
                fit_builder=lambda vals: _fit_radius_distribution_one_bin(
                    vals, min_points=min_points_per_bin
                ),
                min_points=min_points_per_bin,
            )
            smooth_as_sparse_bin_fits(
                bin_results,
                fit_builder=lambda vals: _fit_radius_distribution_one_bin(
                    vals, min_points=min_points_per_bin
                ),
                value_kind="ratio",
                label="overlap_throat_R_ratio_by_thickness",
                min_points=min_points_per_bin,
                seed_offset=4000,
                extra_source_bins=aux_source_bins,
            )

        n_fit = sum(1 for b in bin_results if "fit" in b)
        print(f"  {name}: {len(bin_results)} 个区间, {n_fit} 个区间有拟合")

        results[name] = {
            "mean": float(np.mean(all_r)),
            "std": float(np.std(all_r)),
            "median": float(np.median(all_r)),
            "min": float(np.min(all_r)),
            "max": float(np.max(all_r)),
            "n_ratios": int(len(all_r)),
            "bin_width_nm": float(bin_width),
            "bin_edges": bin_edges.tolist(),
            "bins": bin_results,
            "overlap_tol_nm": OVERLAP_TOL_NM,
            "ratio_clip_max_for_fit": float(ratio_clip_max_for_fit),
            "description": "R_throat / R_cap for overlap throats only (d < R1+R2-tol), R_cap = intersection circle radius",
        }

    results_path = out_dir / "overlap_throat_R_ratio_by_thickness_bin.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\noverlap 喉 R_throat/R_cap 分析结果已保存: {results_path}")

    plot_overlap_throat_R_ratio_by_thickness_bins(
        df,
        results,
        bin_width,
        min_points_per_bin,
        out_dir,
        split_samples_dir,
        ratio_clip_max_for_fit=ratio_clip_max_for_fit,
    )
    return results


def plot_overlap_throat_R_ratio_by_thickness_bins(
    df: pd.DataFrame,
    results: dict,
    bin_width: float,
    min_points_per_bin: int,
    out_dir: Path,
    split_samples_dir: Path,
    ratio_clip_max_for_fit: float = OVERLAP_RATIO_CLIP_MAX_FIT,
) -> None:
    """
    每个厚度区间一个子图：overlap 喉 R_throat/R_cap 柱状分布（与拟合同口径的 clip）+ Phase2 拟合曲线。
    WT / AS 各一张图。
    """
    out_dir = Path(out_dir)
    split_samples_dir = Path(split_samples_dir)
    for name in ["WT", "AS"]:
        res_name = results.get(name, {})
        bin_edges = np.array(res_name.get("bin_edges", []))
        if len(bin_edges) < 2:
            continue
        is_wt = name == "WT"
        bins_data = collect_overlap_R_ratios_by_thickness_bin(
            df, split_samples_dir, is_wt, bin_edges, st, ratio_clip_max=ratio_clip_max_for_fit
        )
        panels = [
            (t_center, t_min, t_max, ratios)
            for t_center, t_min, t_max, ratios in bins_data
            if len(ratios) >= min_points_per_bin
        ]
        if not panels:
            continue
        n_panels = len(panels)
        n_cols = 4
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        bins_list = res_name.get("bins", [])

        for idx, (t_center, t_min, t_max, ratios) in enumerate(panels):
            ax = axes_flat[idx]
            n_bins_hist = 30
            ax.hist(
                ratios,
                bins=n_bins_hist,
                density=False,
                weights=np.ones_like(ratios) / len(ratios),
                alpha=0.5,
                color="steelblue",
                edgecolor="white",
                label="Real data",
            )
            bin_rec = next((b for b in bins_list if b["t_center"] == t_center and b.get("fit")), None)
            if bin_rec and bin_rec.get("fit"):
                fit = bin_rec["fit"]
                x_min, x_max = float(ratios.min()), float(ratios.max())
                x_pad = max((x_max - x_min) * 0.1, 0.05)
                x_plot = np.linspace(max(1e-6, x_min - x_pad), x_max + x_pad, 200)
                bin_width_approx = (x_max - x_min) / n_bins_hist if n_bins_hist and x_max > x_min else 1e-6
                if fit.get("fit_type") == "gmm":
                    w, mu, cov = fit["weights"], fit["means"], fit["covariances"]
                    sigma = np.sqrt(cov)
                    colors = ["green", "purple", "orange", "brown"]
                    for k in range(len(w)):
                        comp = w[k] * stats.norm.pdf(x_plot, loc=mu[k], scale=sigma[k]) * bin_width_approx
                        ax.plot(
                            x_plot,
                            comp,
                            linestyle="--",
                            linewidth=1.2,
                            color=colors[k % len(colors)],
                            label=f"Component {k+1}" if k < 2 else None,
                        )
                    pdf_vals = _radius_gmm_pdf(x_plot, w, mu, cov) * bin_width_approx
                    ax.plot(x_plot, pdf_vals, "r-", linewidth=1.5, label="Fit overall")
                else:
                    dist_name = fit.get("distribution", {}).get("name")
                    params = fit.get("distribution", {}).get("params")
                    if dist_name and params:
                        pdf_vals = _radius_fitted_pdf(x_plot, dist_name, params) * bin_width_approx
                        ax.plot(x_plot, pdf_vals, "r-", linewidth=1.5, label=f"Fit {dist_name}")
            ax.set_xlabel(r"$R_{\mathrm{throat}} / R_{\mathrm{cap}}$ (overlap, clipped for fit)")
            ax.set_ylabel("Probability")
            ax.set_title(f"T in [{t_min:.0f},{t_max:.0f}) nm (n={len(ratios)})")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.0, 1.0)
        for idx in range(len(panels), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.suptitle(
            f"{name} Overlap throat $R_{{\\mathrm{{throat}}}}/R_{{\\mathrm{{cap}}}}$ by thickness bin "
            f"({bin_width:.0f} nm per bin)",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        plot_path = out_dir / f"overlap_R_ratio_by_thickness_bins_throat_{name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  overlap R_throat/R_cap 直方图+拟合已保存: {plot_path}")


def collect_throat_lengths_by_thickness_bin(
    df: pd.DataFrame,
    data_dict: dict,
    split_samples_dir: Path,
    is_wt: bool,
    bin_edges: np.ndarray,
) -> List[Tuple[float, float, float, np.ndarray]]:
    """
    按厚度区间收集喉长度（所有喉的长度拼成一条数组，与喉半径一致）。
    返回 [(t_center, t_min, t_max, lengths_array), ...]。
    """
    out = []
    df_sub = df[df["is_wt"] == is_wt]
    for i in range(len(bin_edges) - 1):
        t_min, t_max = float(bin_edges[i]), float(bin_edges[i + 1])
        t_center = (t_min + t_max) / 2.0
        arr_list = []
        for _, row in df_sub.iterrows():
            t = row.get("thickness")
            if pd.isna(t) or t < t_min or t >= t_max:
                continue
            lengths = _get_throat_lengths_from_row(row, data_dict, split_samples_dir, st)
            if lengths is not None and len(lengths) > 0:
                arr_list.append(lengths)
        lengths_concat = np.concatenate(arr_list) if arr_list else np.array([])
        out.append((t_center, t_min, t_max, lengths_concat))
    return out


def collect_throat_effective_lengths_by_thickness_bin(
    df: pd.DataFrame,
    split_samples_dir: Path,
    is_wt: bool,
    bin_edges: np.ndarray,
) -> List[Tuple[float, float, float, np.ndarray]]:
    """
    按厚度区间收集“有效喉长” L_raw = |p1-p2| - r1 - r2（允许为负）。
    返回 [(t_center, t_min, t_max, lengths_array), ...]。
    """
    from gbm_sieving.analysis.structure import statistics as st_module

    out = []
    df_sub = df[df["is_wt"] == is_wt]
    for i in range(len(bin_edges) - 1):
        t_min, t_max = float(bin_edges[i]), float(bin_edges[i + 1])
        t_center = (t_min + t_max) / 2.0
        arr_list = []
        for _, row in df_sub.iterrows():
            t = row.get("thickness")
            if pd.isna(t) or t < t_min or t >= t_max:
                continue
            lengths = _get_throat_effective_lengths_from_row(row, split_samples_dir, st_module)
            if lengths is not None and len(lengths) > 0:
                arr_list.append(lengths)
        lengths_concat = np.concatenate(arr_list) if arr_list else np.array([], dtype=float)
        out.append((t_center, t_min, t_max, lengths_concat))
    return out


def analyze_throat_length_vs_thickness(
    df: pd.DataFrame,
    split_samples_dir: Path,
    out_dir: Path,
    data_dict: Optional[dict] = None,
    bin_width: float = THROAT_LENGTH_BIN_WIDTH_NM,
    min_points_per_bin: int = 30,
) -> dict:
    """
    2.5 喉长度分布及与厚度的关系（与喉半径分析方式一致）
    此处喉长度指孔心距 d = |p1-p2|。按厚度区间收集该区间内所有喉的 d，
    每个区间画柱状分布图并用 GMM（或 BIC 选的参数分布）拟合。WT / AS 各一张图。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_samples_dir = Path(split_samples_dir)
    data_dict = data_dict or {}

    print("\n=== 喉长度 vs 厚度（按厚度区间分布 + GMM 拟合）===")

    results = {}

    for is_wt_val, name in [(True, "WT"), (False, "AS")]:
        df_subset = df[df["is_wt"] == is_wt_val]
        thickness_arr = df_subset["thickness"].dropna().values
        if len(thickness_arr) == 0:
            print(f"  {name}: 无厚度数据，跳过")
            results[name] = {"mean_nm": None, "std_nm": None, "n_throats": 0, "bin_edges": [], "bins": []}
            continue

        bin_edges = get_thickness_bin_edges_for_group(thickness_arr, bin_width, is_wt_val)
        bins_data = collect_throat_lengths_by_thickness_bin(
            df, data_dict, split_samples_dir, is_wt_val, bin_edges
        )
        print(f"  {name}: 厚度分层边界 = {bin_edges.tolist()}")

        all_lengths = np.concatenate([b[3] for b in bins_data if len(b[3]) > 0]) if bins_data else np.array([])
        if len(all_lengths) == 0:
            print(f"  {name}: 无喉长度数据")
            results[name] = {"mean_nm": None, "std_nm": None, "n_throats": 0, "bin_edges": bin_edges.tolist(), "bins": []}
            continue

        mean_global = float(np.mean(all_lengths))
        std_global = float(np.std(all_lengths))
        print(f"  {name}: 共 {len(all_lengths)} 条喉，全局均值 = {mean_global:.2f} nm, 标准差 = {std_global:.2f} nm")

        bin_results = []
        for t_center, t_min, t_max, lengths in bins_data:
            rec = {
                "t_center": float(t_center),
                "t_min": float(t_min),
                "t_max": float(t_max),
                "n": len(lengths),
            }
            if len(lengths) >= min_points_per_bin:
                fit = _fit_radius_distribution_one_bin(lengths, min_points=min_points_per_bin)
                if fit:
                    rec["fit"] = fit
                    rec["distribution"] = fit.get("distribution")
                    if fit.get("fit_type") == "gmm":
                        rec["n_components"] = fit["n_components"]
                        rec["weights"] = fit["weights"]
                        rec["means"] = fit["means"]
                        rec["covariances"] = fit["covariances"]
                    print(f"    区间 [{t_min:.0f},{t_max:.0f}) nm: n={len(lengths)}, 拟合={fit.get('fit_type', '')}")
            bin_results.append(rec)

        if name == "AS":
            aux_source_bins = _make_aux_right_smoothing_bins(
                collect_throat_lengths_by_thickness_bin(
                    df,
                    data_dict,
                    split_samples_dir,
                    is_wt_val,
                    np.asarray(AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM, dtype=float),
                ),
                fit_builder=lambda vals: _fit_radius_distribution_one_bin(
                    vals, min_points=min_points_per_bin
                ),
                min_points=min_points_per_bin,
            )
            smooth_as_sparse_bin_fits(
                bin_results,
                fit_builder=lambda vals: _fit_radius_distribution_one_bin(
                    vals, min_points=min_points_per_bin
                ),
                value_kind="length",
                label="throat_length_by_thickness",
                min_points=min_points_per_bin,
                seed_offset=5000,
                extra_source_bins=aux_source_bins,
            )

        n_with_fit = sum(1 for b in bin_results if "fit" in b)
        print(f"  {name}: {len(bin_results)} 个区间, {n_with_fit} 个区间有拟合")

        results[name] = {
            "mean_nm": mean_global,
            "std_nm": std_global,
            "median_nm": float(np.median(all_lengths)),
            "min_nm": float(np.min(all_lengths)),
            "max_nm": float(np.max(all_lengths)),
            "n_throats": int(len(all_lengths)),
            "bin_width_nm": float(bin_width),
            "bin_edges": bin_edges.tolist(),
            "bins": bin_results,
        }

    results_path = out_dir / "throat_length_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n喉长度分析结果已保存: {results_path}")

    plot_throat_length_by_thickness_bins(
        df, data_dict, results, bin_width, min_points_per_bin, out_dir, split_samples_dir
    )
    # 额外：真实有效喉长 L_raw 的分布（允许为负），拟合并保存，图上叠加拟合曲线
    print("\n=== 有效喉长 L_raw = d - r1 - r2 vs 厚度（允许为负，拟合并绘图）===")
    effective_results: dict = {}
    for is_wt_val, name in [(True, "WT"), (False, "AS")]:
        df_subset = df[df["is_wt"] == is_wt_val]
        thickness_arr = df_subset["thickness"].dropna().values
        if len(thickness_arr) == 0:
            continue
        bin_edges = get_thickness_bin_edges_for_group(thickness_arr, bin_width, is_wt_val)
        bins_data_Lraw = collect_throat_effective_lengths_by_thickness_bin(
            df, split_samples_dir, is_wt_val, bin_edges
        )
        if not bins_data_Lraw:
            continue
        bin_results_Lraw: list[dict] = []
        for t_center, t_min, t_max, lengths in bins_data_Lraw:
            rec = {"t_center": float(t_center), "t_min": float(t_min), "t_max": float(t_max), "n": len(lengths)}
            if len(lengths) >= min_points_per_bin:
                fit = _fit_radius_distribution_one_bin(np.asarray(lengths, dtype=float), min_points=min_points_per_bin)
                if fit:
                    rec["fit"] = fit
                    rec["distribution"] = fit.get("distribution")
                    if fit.get("fit_type") == "gmm":
                        rec["n_components"] = fit["n_components"]
                        rec["weights"] = fit["weights"]
                        rec["means"] = fit["means"]
                        rec["covariances"] = fit["covariances"]
                    print(f"    L_raw 区间 [{t_min:.0f},{t_max:.0f}) nm: n={len(lengths)}, 拟合={fit.get('fit_type', '')}")
            bin_results_Lraw.append(rec)
        if name == "AS":
            aux_source_bins = _make_aux_right_smoothing_bins(
                collect_throat_effective_lengths_by_thickness_bin(
                    df,
                    split_samples_dir,
                    is_wt_val,
                    np.asarray(AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM, dtype=float),
                ),
                fit_builder=lambda vals: _fit_radius_distribution_one_bin(
                    np.asarray(vals, dtype=float), min_points=min_points_per_bin
                ),
                min_points=min_points_per_bin,
            )
            smooth_as_sparse_bin_fits(
                bin_results_Lraw,
                fit_builder=lambda vals: _fit_radius_distribution_one_bin(
                    vals, min_points=min_points_per_bin
                ),
                value_kind="real",
                label="effective_throat_length_by_thickness",
                min_points=min_points_per_bin,
                seed_offset=5100,
                extra_source_bins=aux_source_bins,
            )
        effective_results[name] = {
            "bin_edges": bin_edges.tolist(),
            "bins": bin_results_Lraw,
        }
        panels = [(t_center, t_min, t_max, lengths) for t_center, t_min, t_max, lengths in bins_data_Lraw
                  if len(lengths) >= min_points_per_bin]
        if not panels:
            continue
        n_panels = len(panels)
        n_cols = 4
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
        axes = np.atleast_1d(axes).reshape(n_rows, n_cols)
        axes_flat = axes.flatten()
        for idx_ax, (t_center, t_min, t_max, lengths) in enumerate(panels):
            ax = axes_flat[idx_ax]
            vals = np.asarray(lengths, dtype=float)
            if len(vals) == 0:
                ax.set_visible(False)
                continue
            n_bins_hist = min(40, max(10, len(vals) // 50))
            ax.hist(
                vals,
                bins=n_bins_hist,
                density=False,
                weights=np.ones_like(vals) / len(vals),
                alpha=0.6,
                color="steelblue",
                edgecolor="white",
                label="Real data",
            )
            bin_rec = next((b for b in bin_results_Lraw if b["t_center"] == t_center and b.get("fit")), None)
            if bin_rec and bin_rec.get("fit"):
                fit = bin_rec["fit"]
                x_min, x_max = float(vals.min()), float(vals.max())
                x_pad = max((x_max - x_min) * 0.1, 0.5)
                x_plot = np.linspace(x_min - x_pad, x_max + x_pad, 200)
                bin_width_approx = (x_max - x_min) / n_bins_hist if n_bins_hist and x_max > x_min else 1e-6
                if fit.get("fit_type") == "gmm":
                    w, mu, cov = fit["weights"], fit["means"], fit["covariances"]
                    sigma = np.sqrt(cov)
                    pdf_vals = _radius_gmm_pdf(x_plot, w, mu, cov) * bin_width_approx
                    ax.plot(x_plot, pdf_vals, "r-", linewidth=1.5, label="Fit")
                else:
                    dist_name = fit.get("distribution", {}).get("name")
                    params = fit.get("distribution", {}).get("params")
                    if dist_name and params:
                        pdf_vals = _radius_fitted_pdf(x_plot, dist_name, params) * bin_width_approx
                        ax.plot(x_plot, pdf_vals, "r-", linewidth=1.5, label=f"Fit {dist_name}")
            ax.set_xlabel("Effective throat length L_raw = d - r1 - r2 (nm)")
            ax.set_ylabel("Probability")
            ax.set_title(f"T in [{t_min:.0f},{t_max:.0f}) nm (n={len(vals)})")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.0, 1.0)
        for idx_ax in range(len(panels), len(axes_flat)):
            axes_flat[idx_ax].set_visible(False)
        plt.suptitle(f"{name} Effective throat length L_raw by thickness bin ({bin_width:.0f} nm per bin)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plot_path = out_dir / f"effective_length_by_thickness_bins_throat_{name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {plot_path}")

    if effective_results:
        eff_path = out_dir / "effective_throat_length_analysis.json"
        with open(eff_path, "w", encoding="utf-8") as f:
            json.dump(effective_results, f, indent=2, ensure_ascii=False)
        print(f"  有效喉长 L_raw 拟合结果已保存: {eff_path}")

    return results


def plot_throat_length_by_thickness_bins(
    df: pd.DataFrame,
    data_dict: dict,
    results: dict,
    bin_width: float,
    min_points_per_bin: int,
    out_dir: Path,
    split_samples_dir: Path,
) -> None:
    """每个厚度区间一个子图：喉长度柱状分布 + GMM/参数拟合曲线（与喉半径图一致）。WT / AS 各一张图。"""
    out_dir = Path(out_dir)
    split_samples_dir = Path(split_samples_dir)
    for name in ["WT", "AS"]:
        res_name = results.get(name, {})
        bin_edges = np.array(res_name.get("bin_edges", []))
        if len(bin_edges) < 2:
            continue
        is_wt = name == "WT"
        bins_data = collect_throat_lengths_by_thickness_bin(
            df, data_dict, split_samples_dir, is_wt, bin_edges
        )
        panels = [(t_center, t_min, t_max, lengths) for t_center, t_min, t_max, lengths in bins_data
                  if len(lengths) >= min_points_per_bin]
        if not panels:
            continue
        n_panels = len(panels)
        n_cols = 4
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        bins_list = res_name.get("bins", [])

        for idx, (t_center, t_min, t_max, lengths) in enumerate(panels):
            ax = axes_flat[idx]
            n_bins_hist = 30
            # 直方图：柱高为概率
            ax.hist(
                lengths,
                bins=n_bins_hist,
                density=False,
                weights=np.ones_like(lengths) / len(lengths),
                alpha=0.5,
                color="steelblue",
                edgecolor="white",
                label="Real data",
            )
            bin_rec = next((b for b in bins_list if b["t_center"] == t_center and b.get("fit")), None)
            if bin_rec and bin_rec.get("fit"):
                fit = bin_rec["fit"]
                x_min, x_max = lengths.min(), lengths.max()
                x_pad = max((x_max - x_min) * 0.1, 0.5)
                x_plot = np.linspace(max(0, x_min - x_pad), x_max + x_pad, 200)
                bin_width_approx = (x_max - x_min) / n_bins_hist if n_bins_hist and x_max > x_min else 1e-6
                if fit.get("fit_type") == "gmm":
                    w, mu, cov = fit["weights"], fit["means"], fit["covariances"]
                    sigma = np.sqrt(cov)
                    colors = ["green", "purple", "orange", "brown"]
                    for k in range(len(w)):
                        comp = w[k] * stats.norm.pdf(x_plot, loc=mu[k], scale=sigma[k]) * bin_width_approx
                        ax.plot(x_plot, comp, linestyle="--", linewidth=1.2, color=colors[k % len(colors)],
                                label=f"Component {k+1}" if k < 2 else None)
                    pdf_vals = _radius_gmm_pdf(x_plot, w, mu, cov) * bin_width_approx
                    ax.plot(x_plot, pdf_vals, "r-", linewidth=1.5, label="Fit overall")
                else:
                    dist_name = fit.get("distribution", {}).get("name")
                    params = fit.get("distribution", {}).get("params")
                    if dist_name and params:
                        pdf_vals = _radius_fitted_pdf(x_plot, dist_name, params) * bin_width_approx
                        ax.plot(x_plot, pdf_vals, "r-", linewidth=1.5, label=f"Fit {dist_name}")
            ax.set_xlabel("Throat length (nm)")
            ax.set_ylabel("Probability")
            ax.set_title(f"T in [{t_min:.0f},{t_max:.0f}) nm (n={len(lengths)})")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.0, 1.0)
        for idx in range(len(panels), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.suptitle(f"{name} Throat length by thickness bin ({bin_width:.0f} nm per bin)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plot_path = out_dir / f"length_by_thickness_bins_throat_{name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {plot_path}")


def _collect_throat_radius_and_length_by_group(
    df: pd.DataFrame,
    data_dict: dict,
    is_wt: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    为 WT / AS 收集所有喉的 (length, radius) 数组。
    返回 (lengths_all, radii_all)，长度和半径一一对应。
    """
    lengths_list: list[np.ndarray] = []
    radii_list: list[np.ndarray] = []
    df_sub = df[df["is_wt"] == is_wt]
    for _, row in df_sub.iterrows():
        key = (row["sample_name"], row["sub_name"])
        if key not in data_dict:
            continue
        arrays = data_dict[key]
        if "throat_lengths" not in arrays or "throat_radii" not in arrays:
            continue
        L = np.asarray(arrays["throat_lengths"], dtype=float)
        R = np.asarray(arrays["throat_radii"], dtype=float)
        if len(L) == 0 or len(R) == 0:
            continue
        n = min(len(L), len(R))
        if n <= 0:
            continue
        lengths_list.append(L[:n])
        radii_list.append(R[:n])
    if not lengths_list:
        return np.array([], dtype=float), np.array([], dtype=float)
    lengths_all = np.concatenate(lengths_list)
    radii_all = np.concatenate(radii_list)
    return lengths_all, radii_all


def analyze_throat_radius_by_length_bins(
    df: pd.DataFrame,
    data_dict: dict,
    out_dir: Path,
    *,
    length_bin_width_nm: float = 10.0,
    length_merge_threshold_nm: float = 60.0,
    min_points_per_bin: int = 30,
) -> dict:
    """
    2.x 喉半径在不同喉长度区间下的分布及拟合（WT / AS）。
    此处喉长度仍指孔心距 d。
    - 以喉长度 d 为自变量，10nm 为一个长度区间；
    - 所有 > length_merge_threshold_nm 的喉长度归入一个末尾区间；
    - 每个长度区间上分析喉半径的条件分布 p(r | d in bin)，并做 GMM/参数拟合。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== 喉半径 vs 喉长度区间（条件分布）===")

    results: dict = {}

    for is_wt_val, name in [(True, "WT"), (False, "AS")]:
        lengths_all, radii_all = _collect_throat_radius_and_length_by_group(
            df, data_dict, is_wt_val
        )
        if len(lengths_all) == 0 or len(radii_all) == 0:
            print(f"  {name}: 无喉数据，跳过按长度区间的喉半径分析")
            results[name] = {
                "n_throats": 0,
                "bin_edges_nm": [],
                "bins": [],
            }
            continue

        print(
            f"  {name}: 共 {len(lengths_all)} 条喉，长度范围 [{lengths_all.min():.2f}, {lengths_all.max():.2f}] nm"
        )

        # 构造长度分层边界：0,10,20,...,length_merge_threshold_nm, max(L)+1
        max_len = float(np.max(lengths_all))
        upper = max(max_len, length_merge_threshold_nm) + 1.0
        base_edges = np.arange(0.0, length_merge_threshold_nm, length_bin_width_nm)
        bin_edges = np.concatenate([base_edges, [length_merge_threshold_nm, upper]])
        print(f"  {name}: 喉长度分层边界 (nm) = {bin_edges.tolist()}")

        bin_results: list[dict] = []
        for i in range(len(bin_edges) - 1):
            L_min = float(bin_edges[i])
            L_max = float(bin_edges[i + 1])
            L_center = 0.5 * (L_min + L_max)
            mask = (lengths_all >= L_min) & (lengths_all < L_max)
            radii_bin = radii_all[mask]
            rec: dict = {
                "L_center_nm": L_center,
                "L_min_nm": L_min,
                "L_max_nm": L_max,
                "n": int(len(radii_bin)),
            }
            if len(radii_bin) >= min_points_per_bin:
                fit = _fit_radius_distribution_one_bin(
                    radii_bin, min_points=min_points_per_bin
                )
                if fit:
                    rec["fit"] = fit
                    rec["distribution"] = fit.get("distribution")
                    if fit.get("fit_type") == "gmm":
                        rec["n_components"] = fit["n_components"]
                        rec["weights"] = fit["weights"]
                        rec["means"] = fit["means"]
                        rec["covariances"] = fit["covariances"]
                    print(
                        f"    L∈[{L_min:.0f},{L_max:.0f}) nm: n={len(radii_bin)}, 拟合={fit.get('fit_type', '')}"
                    )
            bin_results.append(rec)

        n_with_fit = sum(1 for b in bin_results if "fit" in b)
        print(
            f"  {name}: 共 {len(bin_results)} 个长度区间，其中 {n_with_fit} 个区间有半径分布拟合"
        )

        results[name] = {
            "n_throats": int(len(lengths_all)),
            "bin_width_nm": float(length_bin_width_nm),
            "length_merge_threshold_nm": float(length_merge_threshold_nm),
            "bin_edges_nm": bin_edges.tolist(),
            "bins": bin_results,
        }

        # 额外：记录真实样本中 r²L 的 log10 分布直方图（供 Phase3 直接对比）
        try:
            r2L = np.pi * np.asarray(radii_all, dtype=float) ** 2 * np.asarray(lengths_all, dtype=float)
            vals = r2L[np.isfinite(r2L) & (r2L > 0)]
            if len(vals) > 0:
                log_vals = np.log10(vals)
                n_bins = 80
                hist, bin_edges_r2L = np.histogram(log_vals, bins=n_bins)
                total = hist.sum()
                if total > 0:
                    prob = (hist / float(total)).tolist()
                    results[name]["r2L_log10_hist"] = {
                        "bin_edges": bin_edges_r2L.tolist(),
                        "prob": prob,
                    }
        except Exception:
            pass

    results_path = out_dir / "radius_by_length_bins_throat_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n喉半径按喉长度区间的分析结果已保存: {results_path}")

    # 额外：喉长度×喉半径的联合分布及拟合曲面可视化，WT / AS 各一张图
    try:
        for is_wt_val, name in [(True, "WT"), (False, "AS")]:
            lengths_all, radii_all = _collect_throat_radius_and_length_by_group(
                df, data_dict, is_wt_val
            )
            if len(lengths_all) == 0 or len(radii_all) == 0:
                continue
            # 使用 GMM 拟合 (L, r) 的 2D 联合分布
            gmm_joint = _fit_2d_gmm_joint(
                lengths_all,
                radii_all,
                min_components=1,
                max_components=4,
                random_state=0,
            )
            if gmm_joint:
                results[name]["joint_gmm_length_throat"] = gmm_joint
            # 2D hexbin
            plt.figure(figsize=(6, 5))
            hb = plt.hexbin(
                lengths_all,
                radii_all,
                gridsize=60,
                cmap="viridis",
                mincnt=1,
            )
            plt.colorbar(hb, label="Count")
            plt.xlabel("Throat length L (nm)")
            plt.ylabel("Throat radius r (nm)")
            plt.title(f"Throat length vs throat radius joint ({name})")
            joint_png = out_dir / f"throat_length_radius_joint_{name}.png"
            plt.tight_layout()
            plt.savefig(joint_png, dpi=150)
            plt.close()
            print(f"  {name}: 喉长度 vs 喉半径联合分布图已保存: {joint_png}")

            # 拟合条件分布曲面：X=L, Y=r, Z=p(r | L)
            res_name = results.get(name, {})
            bin_edges_L = np.array(res_name.get("bin_edges_nm", []), dtype=float)
            bins_L = res_name.get("bins", [])
            if len(bin_edges_L) < 2 or not bins_L:
                continue
            y_min = max(0.1, float(np.min(radii_all)))
            y_max = float(np.max(radii_all))
            y_grid = np.linspace(y_min, y_max, 80)
            x_grid = np.linspace(float(bin_edges_L[0]), float(bin_edges_L[-1]), 80)
            if gmm_joint:
                Z = _gmm2d_pdf_grid(x_grid, y_grid, gmm_joint)
            fig = plt.figure(figsize=(7, 5))
            ax3d = fig.add_subplot(111, projection="3d")
            if gmm_joint:
                X_surf, Y_surf = np.meshgrid(x_grid, y_grid, indexing="ij")
                surf = ax3d.plot_surface(
                    X_surf,
                    Y_surf,
                    Z,
                    cmap="viridis",
                    linewidth=0,
                    antialiased=True,
                )
                fig.colorbar(surf, shrink=0.6, aspect=12, label="Joint PDF p(L, r)")
            ax3d.set_xlabel("Throat length L (nm)")
            ax3d.set_ylabel("Throat radius r (nm)")
            ax3d.set_zlabel("Joint PDF")
            ax3d.set_title(f"Fitted joint PDF surface: p(L, r) ({name})")
            surf_png = out_dir / f"throat_length_radius_surface_{name}.png"
            plt.tight_layout()
            plt.savefig(surf_png, dpi=150)
            plt.close()
            print(f"  {name}: 喉长度 vs 喉半径拟合曲面图已保存: {surf_png}")
    except Exception as e:
        print(f"  生成喉长度×喉半径联合分布图时出错: {e}")

    plot_throat_radius_by_length_bins(
        df,
        data_dict,
        results,
        length_bin_width_nm,
        length_merge_threshold_nm,
        min_points_per_bin,
        out_dir,
    )
    return results


def plot_throat_radius_by_length_bins(
    df: pd.DataFrame,
    data_dict: dict,
    results: dict,
    length_bin_width_nm: float,
    length_merge_threshold_nm: float,
    min_points_per_bin: int,
    out_dir: Path,
) -> None:
    """
    每个喉长度区间一个子图：喉半径直方图 + 拟合曲线，WT / AS 各一张图。
    """
    out_dir = Path(out_dir)
    for is_wt_val, name in [(True, "WT"), (False, "AS")]:
        res_name = results.get(name, {})
        bin_edges = np.array(res_name.get("bin_edges_nm", []), dtype=float)
        if len(bin_edges) < 2:
            continue

        lengths_all, radii_all = _collect_throat_radius_and_length_by_group(
            df, data_dict, is_wt_val
        )
        if len(lengths_all) == 0 or len(radii_all) == 0:
            continue

        bins_list = res_name.get("bins", [])
        panels = []
        for i in range(len(bin_edges) - 1):
            L_min = float(bin_edges[i])
            L_max = float(bin_edges[i + 1])
            mask = (lengths_all >= L_min) & (lengths_all < L_max)
            radii_bin = radii_all[mask]
            if len(radii_bin) >= min_points_per_bin:
                panels.append((L_min, L_max, radii_bin))

        if not panels:
            continue

        n_panels = len(panels)
        n_cols = 4
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows)
        )
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()

        for idx, (L_min, L_max, radii_bin) in enumerate(panels):
            ax = axes_flat[idx]
            n_bins_hist = 30
            # 直方图：柱高为概率
            ax.hist(
                radii_bin,
                bins=n_bins_hist,
                density=False,
                weights=np.ones_like(radii_bin) / len(radii_bin),
                alpha=0.5,
                color="steelblue",
                edgecolor="white",
                label="Real data",
            )
            # Find fit for this length bin
            bin_rec = next(
                (
                    b
                    for b in bins_list
                    if float(b.get("L_min_nm", 0.0)) == L_min
                    and float(b.get("L_max_nm", 0.0)) == L_max
                    and b.get("fit")
                ),
                None,
            )
            if bin_rec and bin_rec.get("fit"):
                fit = bin_rec["fit"]
                x_min, x_max = radii_bin.min(), radii_bin.max()
                x_pad = max((x_max - x_min) * 0.1, 0.5)
                x_plot = np.linspace(max(0, x_min - x_pad), x_max + x_pad, 200)
                bin_width_approx = (x_max - x_min) / n_bins_hist if n_bins_hist and x_max > x_min else 1e-6
                if fit.get("fit_type") == "gmm":
                    w, mu, cov = (
                        fit["weights"],
                        fit["means"],
                        fit["covariances"],
                    )
                    sigma = np.sqrt(cov)
                    colors = ["green", "purple", "orange", "brown"]
                    for k in range(len(w)):
                        comp = w[k] * stats.norm.pdf(
                            x_plot, loc=mu[k], scale=sigma[k]
                        ) * bin_width_approx
                        ax.plot(
                            x_plot,
                            comp,
                            linestyle="--",
                            linewidth=1.2,
                            color=colors[k % len(colors)],
                            label=f"Component {k+1}" if k < 2 else None,
                        )
                    pdf_vals = _radius_gmm_pdf(x_plot, w, mu, cov) * bin_width_approx
                    ax.plot(
                        x_plot,
                        pdf_vals,
                        "r-",
                        linewidth=1.5,
                        label="Fit overall",
                    )
                else:
                    dist_name = fit.get("distribution", {}).get("name")
                    params = fit.get("distribution", {}).get("params")
                    if dist_name and params:
                        pdf_vals = _radius_fitted_pdf(x_plot, dist_name, params) * bin_width_approx
                        ax.plot(
                            x_plot,
                            pdf_vals,
                            "r-",
                            linewidth=1.5,
                            label=f"Fit {dist_name}",
                        )
            ax.set_xlabel("Throat radius (nm)")
            ax.set_ylabel("Probability")
            if L_max <= length_merge_threshold_nm:
                title = f"L in [{L_min:.0f},{L_max:.0f}) nm"
            else:
                title = f"L >= {length_merge_threshold_nm:.0f} nm"
            ax.set_title(f"{title} (n={len(radii_bin)})")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        for idx in range(len(panels), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.suptitle(
            f"{name} Throat radius by length bin (dL={length_bin_width_nm:.0f} nm, L>{length_merge_threshold_nm:.0f} nm merged)",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        plot_path = out_dir / f"radius_by_length_bins_throat_{name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  已保存: {plot_path}")


def compare_length_distributions_across_thickness_bins(
    df: pd.DataFrame,
    data_dict: dict,
    results: dict,
    out_dir: Path,
    split_samples_dir: Path,
    *,
    min_points_per_bin: int = 30,
    p_value_threshold: float = 0.05,
) -> None:
    """
    对不同厚度区间的喉长度分布做两两 KS 检验，热图与喉半径一致。
    输出：length_bins_compare_throat_{WT|AS}.json / .xlsx / .png
    """
    out_dir = Path(out_dir)
    split_samples_dir = Path(split_samples_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ["WT", "AS"]:
        res = results.get(name, {})
        bin_edges = np.array(res.get("bin_edges", []))
        if len(bin_edges) < 2:
            continue
        is_wt = name == "WT"
        bins_data = collect_throat_lengths_by_thickness_bin(
            df, data_dict, split_samples_dir, is_wt, bin_edges
        )
        n_all = len(bins_data)
        all_labels = []
        for i in range(n_all):
            _, t_min, t_max, lengths = bins_data[i]
            all_labels.append(f"[{t_min:.0f},{t_max:.0f}) (n={len(lengths)})")
        valid_indices = [i for i in range(n_all) if len(bins_data[i][3]) >= min_points_per_bin]
        if len(valid_indices) < 2:
            print(f"  喉长度 {name}: 有效区间数 < 2，跳过比较")
            continue

        lengths_by_idx = {i: bins_data[i][3] for i in valid_indices}
        valid_labels = [all_labels[i] for i in valid_indices]
        n_valid = len(valid_indices)

        p_full = np.full((n_all, n_all), np.nan)
        ks_full = np.full((n_all, n_all), np.nan)
        for ii, i in enumerate(valid_indices):
            for jj, j in enumerate(valid_indices):
                if i == j:
                    p_full[i, j] = 1.0
                    ks_full[i, j] = 0.0
                    continue
                if i > j:
                    continue
                stat, pval = ks_2samp(lengths_by_idx[i], lengths_by_idx[j])
                p_full[i, j] = p_full[j, i] = float(pval)
                ks_full[i, j] = ks_full[j, i] = float(stat)

        p_valid = np.ones((n_valid, n_valid))
        for ii in range(n_valid):
            for jj in range(ii + 1, n_valid):
                p_valid[ii, jj] = p_valid[jj, ii] = p_full[valid_indices[ii], valid_indices[jj]]
        similar_edges = [
            (ii, jj) for ii in range(n_valid) for jj in range(ii + 1, n_valid)
            if p_valid[ii, jj] > p_value_threshold
        ]
        groups = _union_find_groups(n_valid, similar_edges)
        similar_groups = [[valid_labels[ii] for ii in g] for g in groups]

        summary = {
            "name": name,
            "p_value_threshold": p_value_threshold,
            "min_points_per_bin": min_points_per_bin,
            "bin_labels_full": all_labels,
            "valid_bin_indices": valid_indices,
            "valid_bin_labels": valid_labels,
            "n_bins_total": n_all,
            "n_bins_with_sufficient_data": n_valid,
            "p_value_matrix_full": _nan_to_none_for_json(p_full),
            "ks_statistic_matrix_full": _nan_to_none_for_json(ks_full),
            "similar_groups": similar_groups,
            "note": "p > threshold 表示两区间分布无法拒绝相同。仅样本量>=min_points_per_bin的区间参与比较；缺数据区间在矩阵中为null。",
        }
        json_path = out_dir / f"length_bins_compare_throat_{name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        excel_path = out_dir / f"length_bins_compare_throat_{name}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pd.DataFrame(p_full, index=all_labels, columns=all_labels).to_excel(
                writer, sheet_name="p_value"
            )
            pd.DataFrame(ks_full, index=all_labels, columns=all_labels).to_excel(
                writer, sheet_name="KS_statistic"
            )
            pd.DataFrame({"similar_group": [str(g) for g in similar_groups]}).to_excel(
                writer, sheet_name="similar_groups", index=False
            )

        fig, ax = plt.subplots(figsize=(max(6, n_all * 0.5), max(5, n_all * 0.5)))
        p_plot = np.ma.array(p_full, mask=np.isnan(p_full))
        im = ax.imshow(p_plot, cmap="RdYlGn", vmin=0, vmax=1)
        im.cmap.set_bad(color="lightgray")
        ax.set_xticks(range(n_all))
        ax.set_yticks(range(n_all))
        ax.set_xticklabels(all_labels, rotation=45, ha="right")
        ax.set_yticklabels(all_labels)
        for i in range(n_all):
            for j in range(n_all):
                if np.isfinite(p_full[i, j]):
                    txt = f"{p_full[i, j]:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                            color="black" if 0.3 < p_full[i, j] < 0.7 else "white")
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
        ax.set_xlabel("Thickness bin (nm)")
        ax.set_ylabel("Thickness bin (nm)")
        ax.set_title(
            f"{name} Throat length: KS test p-value between bins\n"
            f"(p > {p_value_threshold} similar; gray = n<{min_points_per_bin})"
        )
        plt.colorbar(im, ax=ax, label="p-value")
        plt.tight_layout()
        plot_path = out_dir / f"length_bins_compare_throat_{name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  喉长度 {name}: 总区间数 {n_all}，有效区间数 {n_valid}，比较矩阵与相似组已保存 ({len(similar_groups)} 组)")
        for idx, g in enumerate(similar_groups):
            if len(g) >= 2:
                print(f"    相似组 {idx + 1}: {g}")


def _pair_corr_sums_spatial_all_pairs(
    coords: np.ndarray,
    radii: np.ndarray,
    dist_edges: np.ndarray,
    mu: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """空间全孔对 two-point：按距离分箱累计 raw/norm/count。"""
    coords = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float)
    nb = len(dist_edges) - 1
    if coords.ndim != 2 or coords.shape[1] < 3 or len(coords) < 2 or len(radii) != len(coords):
        return np.zeros(nb), np.zeros(nb), np.zeros(nb, dtype=np.int64)

    n = len(radii)

    if not np.isfinite(sigma) or sigma <= 1e-12:
        return np.zeros(nb), np.zeros(nb), np.zeros(nb, dtype=np.int64)
    delta = (radii - mu) / sigma

    sum_raw = np.zeros(nb, dtype=float)
    sum_norm = np.zeros(nb, dtype=float)
    count = np.zeros(nb, dtype=np.int64)

    for i in range(n - 1):
        dxyz = coords[i + 1 :, :3] - coords[i, :3]
        d = np.linalg.norm(dxyz, axis=1)
        b = np.searchsorted(dist_edges, d, side="right") - 1
        valid = (b >= 0) & (b < nb) & np.isfinite(d)
        if not np.any(valid):
            continue
        bb = b[valid]
        raw_ij = radii[i] * radii[i + 1 :][valid]
        norm_ij = delta[i] * delta[i + 1 :][valid]
        sum_raw += np.bincount(bb, weights=raw_ij, minlength=nb)[:nb]
        sum_norm += np.bincount(bb, weights=norm_ij, minlength=nb)[:nb]
        count += np.bincount(bb, minlength=nb)[:nb].astype(np.int64)

    return sum_raw, sum_norm, count


def _pair_corr_sums_connected_pairs(
    pores_file: Path,
    throats_file: Path,
    dist_edges: np.ndarray,
    mu: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """喉连接孔对 two-point：按连接长度分箱累计 raw/norm/count。"""
    nb = len(dist_edges) - 1
    try:
        (
            pore_coords,
            pore_radii,
            _pore_ids,
            pore_id_to_idx,
            throat_pore1,
            throat_pore2,
            _throat_radii,
        ) = st.load_pores_and_throats(Path(pores_file), Path(throats_file))
    except Exception:
        return np.zeros(nb), np.zeros(nb), np.zeros(nb, dtype=np.int64)

    pore_coords = np.asarray(pore_coords, dtype=float)
    pore_radii = np.asarray(pore_radii, dtype=float)
    if len(pore_radii) < 2:
        return np.zeros(nb), np.zeros(nb), np.zeros(nb, dtype=np.int64)
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return np.zeros(nb), np.zeros(nb), np.zeros(nb, dtype=np.int64)
    delta = (pore_radii - mu) / sigma

    sum_raw = np.zeros(nb, dtype=float)
    sum_norm = np.zeros(nb, dtype=float)
    count = np.zeros(nb, dtype=np.int64)

    for i in range(len(throat_pore1)):
        p1_id = int(throat_pore1[i])
        p2_id = int(throat_pore2[i])
        idx1 = pore_id_to_idx.get(p1_id, None)
        idx2 = pore_id_to_idx.get(p2_id, None)
        if idx1 is None or idx2 is None:
            continue
        d = float(np.linalg.norm(pore_coords[idx1, :3] - pore_coords[idx2, :3]))
        if not np.isfinite(d):
            continue
        b = int(np.searchsorted(dist_edges, d, side="right") - 1)
        if b < 0 or b >= nb:
            continue
        sum_raw[b] += float(pore_radii[idx1] * pore_radii[idx2])
        sum_norm[b] += float(delta[idx1] * delta[idx2])
        count[b] += 1

    return sum_raw, sum_norm, count


def analyze_pore_radius_two_point_correlation(
    df_summary: pd.DataFrame,
    data_dict: dict,
    out_dir: Path,
    *,
    thickness_bin_width: float = 20.0,
    distance_bin_width: float = 5.0,
    max_distance_nm: float = 400.0,
) -> dict:
    """2.4b 孔半径 two-point（空间全孔对 + 喉连接孔对），按厚度区间每箱一个子图。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(df_summary) == 0:
        return {}

    tvals = pd.to_numeric(df_summary["thickness"], errors="coerce").to_numpy(dtype=float)
    tvals = tvals[np.isfinite(tvals)]
    if len(tvals) == 0:
        print("  [2.4b] 无有效厚度，跳过 two-point correlation。")
        return {}

    t_min = float(np.min(tvals))
    t_max = float(np.max(tvals))
    t0 = math.floor(t_min / thickness_bin_width) * thickness_bin_width
    t1 = math.ceil(t_max / thickness_bin_width) * thickness_bin_width
    thick_edges = np.arange(t0, t1 + thickness_bin_width, thickness_bin_width, dtype=float)
    if len(thick_edges) < 2:
        thick_edges = np.array([t0, t0 + thickness_bin_width], dtype=float)

    dist_edges = np.arange(0.0, max_distance_nm + distance_bin_width, distance_bin_width, dtype=float)
    if len(dist_edges) < 2:
        dist_edges = np.array([0.0, max(1.0, float(max_distance_nm))], dtype=float)
    dist_centers = 0.5 * (dist_edges[:-1] + dist_edges[1:])
    nb = len(dist_centers)

    n_bins = len(thick_edges) - 1
    agg_spatial = {}
    agg_connected = {}
    for bi in range(n_bins):
        agg_spatial[bi] = {
            "WT": {"sum_raw": np.zeros(nb), "sum_norm": np.zeros(nb), "count": np.zeros(nb, dtype=np.int64), "n_sub": 0},
            "AS": {"sum_raw": np.zeros(nb), "sum_norm": np.zeros(nb), "count": np.zeros(nb, dtype=np.int64), "n_sub": 0},
        }
        agg_connected[bi] = {
            "WT": {"sum_raw": np.zeros(nb), "sum_norm": np.zeros(nb), "count": np.zeros(nb, dtype=np.int64), "n_sub": 0},
            "AS": {"sum_raw": np.zeros(nb), "sum_norm": np.zeros(nb), "count": np.zeros(nb, dtype=np.int64), "n_sub": 0},
        }

    for _, row in df_summary.iterrows():
        sample_name = row.get("sample_name", "")
        sub_name = row.get("sub_name", "")
        thickness = pd.to_numeric(row.get("thickness"), errors="coerce")
        if not np.isfinite(thickness):
            continue
        bi = int(np.searchsorted(thick_edges, float(thickness), side="right") - 1)
        if bi < 0 or bi >= n_bins:
            continue
        sample_type = "AS" if str(sample_name).upper().startswith("AS") else "WT"

        key = (sample_name, sub_name)
        data = data_dict.get(key, None)
        if data is None:
            continue
        coords = data.get("pore_coords", np.array([]))
        radii = np.asarray(data.get("pore_radii", np.array([])), dtype=float)
        if radii.size > 1:
            mu = float(np.mean(radii))
            sigma = float(np.std(radii))
            s_raw, s_norm, s_cnt = _pair_corr_sums_spatial_all_pairs(
                coords,
                radii,
                dist_edges,
                mu,
                sigma,
            )
            if int(s_cnt.sum()) > 0:
                agg_spatial[bi][sample_type]["sum_raw"] += s_raw
                agg_spatial[bi][sample_type]["sum_norm"] += s_norm
                agg_spatial[bi][sample_type]["count"] += s_cnt
                agg_spatial[bi][sample_type]["n_sub"] += 1

        pores_file = data.get("pores_file", "")
        throats_file = data.get("throats_file", "")
        if pores_file and throats_file and radii.size > 1:
            mu = float(np.mean(radii))
            sigma = float(np.std(radii))
            c_raw, c_norm, c_cnt = _pair_corr_sums_connected_pairs(
                Path(pores_file),
                Path(throats_file),
                dist_edges,
                mu,
                sigma,
            )
            if int(c_cnt.sum()) > 0:
                agg_connected[bi][sample_type]["sum_raw"] += c_raw
                agg_connected[bi][sample_type]["sum_norm"] += c_norm
                agg_connected[bi][sample_type]["count"] += c_cnt
                agg_connected[bi][sample_type]["n_sub"] += 1

    rows = []
    for pair_mode, agg in (("spatial_all_pairs", agg_spatial), ("connected_pairs", agg_connected)):
        for bi in range(n_bins):
            tlo, thi = float(thick_edges[bi]), float(thick_edges[bi + 1])
            tlabel = f"[{tlo:.0f}, {thi:.0f})"
            for tp in ("WT", "AS"):
                rec = agg[bi][tp]
                cnt = rec["count"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    g_raw = np.where(cnt > 0, rec["sum_raw"] / cnt, np.nan)
                    c_norm = np.where(cnt > 0, rec["sum_norm"] / cnt, np.nan)
                for k, xc in enumerate(dist_centers):
                    rows.append(
                        {
                            "pair_mode": pair_mode,
                            "thickness_bin": tlabel,
                            "type": tp,
                            "distance_nm": float(xc),
                            "pair_count": int(cnt[k]),
                            "G_raw_rirj": float(g_raw[k]) if np.isfinite(g_raw[k]) else np.nan,
                            "C_norm_dridrj": float(c_norm[k]) if np.isfinite(c_norm[k]) else np.nan,
                            "n_subsamples": int(rec["n_sub"]),
                        }
                    )
    out_csv = out_dir / "pore_radius_two_point_correlation_by_thickness.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[2.4b] two-point 数据已保存: {out_csv}")

    ncol = 3 if n_bins >= 3 else max(1, n_bins)
    nrow = int(math.ceil(n_bins / ncol))

    def _plot_grid(
        agg: dict,
        metric_key: str,
        y_label: str,
        out_name: str,
        suptitle: str,
        *,
        draw_zero_line: bool = False,
    ) -> None:
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)
        axes = axes.flatten()
        for bi in range(n_bins):
            ax = axes[bi]
            tlo, thi = float(thick_edges[bi]), float(thick_edges[bi + 1])
            for tp, color in (("WT", "tab:blue"), ("AS", "tab:orange")):
                rec = agg[bi][tp]
                cnt = rec["count"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    yy = np.where(cnt > 0, rec[metric_key] / cnt, np.nan)
                valid = np.isfinite(yy)
                if np.any(valid):
                    ax.plot(
                        dist_centers[valid],
                        yy[valid],
                        marker="o",
                        markersize=3.0,
                        linewidth=1.3,
                        color=color,
                        label=f"{tp} (n_sub={rec['n_sub']})",
                    )
            if draw_zero_line:
                ax.axhline(0.0, color="0.45", linestyle="--", linewidth=1.0, alpha=0.75)
            ax.set_title(f"厚度 [{tlo:.0f}, {thi:.0f}) nm", fontsize=10)
            ax.set_xlabel("distance x (nm)")
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")
        for j in range(n_bins, len(axes)):
            axes[j].axis("off")
        fig.suptitle(suptitle, fontsize=13)
        plt.tight_layout()
        out_png = out_dir / out_name
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"[2.4b] 图已保存: {out_png}")

    _plot_grid(
        agg_spatial,
        "sum_raw",
        r"$G_r(x)=\langle r_i r_j \rangle$",
        "pore_radius_two_point_spatial_raw_by_thickness.png",
        "孔半径 two-point（空间全孔对）原始相关",
    )
    _plot_grid(
        agg_spatial,
        "sum_norm",
        r"$C_r(x)=\langle \delta r_i \delta r_j \rangle$",
        "pore_radius_two_point_spatial_dimensionless_by_thickness.png",
        "孔半径 two-point（空间全孔对）无量纲相关",
        draw_zero_line=True,
    )
    _plot_grid(
        agg_connected,
        "sum_raw",
        r"$G_r(x)=\langle r_i r_j \rangle$",
        "pore_radius_two_point_connected_raw_by_thickness.png",
        "孔半径 two-point（喉连接孔对）原始相关",
    )
    _plot_grid(
        agg_connected,
        "sum_norm",
        r"$C_r(x)=\langle \delta r_i \delta r_j \rangle$",
        "pore_radius_two_point_connected_dimensionless_by_thickness.png",
        "孔半径 two-point（喉连接孔对）无量纲相关",
        draw_zero_line=True,
    )

    # 额外一组：仅连接喉孔对，厚度分箱按“喉孔密度同款”规则，WT/AS 分开作图
    bin_edges_by_type = {}
    for tp, is_wt_val in (("WT", True), ("AS", False)):
        thick_tp = pd.to_numeric(
            df_summary.loc[df_summary["is_wt"] == is_wt_val, "thickness"], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        if len(thick_tp) == 0:
            continue
        edges_tp = get_thickness_bin_edges_density_frac(
            thick_tp, is_wt_val, bin_width=float(DENSITY_FRAC_BIN_WIDTH_NM)
        )
        if len(edges_tp) >= 2:
            bin_edges_by_type[tp] = np.asarray(edges_tp, dtype=float)

    agg_conn_density = {}
    for tp in ("WT", "AS"):
        edges = bin_edges_by_type.get(tp, None)
        if edges is None:
            continue
        nbt = len(edges) - 1
        agg_conn_density[tp] = {
            "edges": edges,
            "bins": [
                {"sum_raw": np.zeros(nb), "sum_norm": np.zeros(nb), "count": np.zeros(nb, dtype=np.int64), "n_sub": 0}
                for _ in range(nbt)
            ],
        }

    for _, row in df_summary.iterrows():
        sample_name = row.get("sample_name", "")
        sub_name = row.get("sub_name", "")
        thickness = pd.to_numeric(row.get("thickness"), errors="coerce")
        if not np.isfinite(thickness):
            continue
        tp = "AS" if str(sample_name).upper().startswith("AS") else "WT"
        if tp not in agg_conn_density:
            continue
        edges = agg_conn_density[tp]["edges"]
        bi = int(np.searchsorted(edges, float(thickness), side="right") - 1)
        if bi < 0 or bi >= len(edges) - 1:
            continue
        data = data_dict.get((sample_name, sub_name), None)
        if data is None:
            continue
        pores_file = data.get("pores_file", "")
        throats_file = data.get("throats_file", "")
        if not pores_file or not throats_file:
            continue
        pore_radii = np.asarray(data.get("pore_radii", np.array([])), dtype=float)
        if pore_radii.size < 2:
            continue
        mu = float(np.mean(pore_radii))
        sigma = float(np.std(pore_radii))
        c_raw, c_norm, c_cnt = _pair_corr_sums_connected_pairs(
            Path(pores_file),
            Path(throats_file),
            dist_edges,
            mu,
            sigma,
        )
        if int(c_cnt.sum()) == 0:
            continue
        rec = agg_conn_density[tp]["bins"][bi]
        rec["sum_raw"] += c_raw
        rec["sum_norm"] += c_norm
        rec["count"] += c_cnt
        rec["n_sub"] += 1

    def _plot_connected_density_split(tp: str, metric_key: str, y_label: str, out_name: str, *, draw_zero_line: bool = False):
        pkg = agg_conn_density.get(tp, None)
        if pkg is None:
            return
        edges = pkg["edges"]
        bins = pkg["bins"]
        nbt = len(bins)
        ncol_t = 3 if nbt >= 3 else max(1, nbt)
        nrow_t = int(math.ceil(nbt / ncol_t))
        fig, axes = plt.subplots(nrow_t, ncol_t, figsize=(5.2 * ncol_t, 3.8 * nrow_t), squeeze=False)
        axes = axes.flatten()
        color = "tab:blue" if tp == "WT" else "tab:orange"
        for bi in range(nbt):
            ax = axes[bi]
            tlo, thi = float(edges[bi]), float(edges[bi + 1])
            rec = bins[bi]
            cnt = rec["count"]
            with np.errstate(divide="ignore", invalid="ignore"):
                yy = np.where(cnt > 0, rec[metric_key] / cnt, np.nan)
            valid = np.isfinite(yy)
            if np.any(valid):
                ax.plot(
                    dist_centers[valid],
                    yy[valid],
                    marker="o",
                    markersize=3.2,
                    linewidth=1.4,
                    color=color,
                    label=f"{tp} (n_sub={rec['n_sub']})",
                )
            if draw_zero_line:
                ax.axhline(0.0, color="0.45", linestyle="--", linewidth=1.0, alpha=0.75)
            ax.set_title(f"{tp} 厚度 [{tlo:.0f}, {thi:.0f}) nm", fontsize=10)
            ax.set_xlabel("distance x (nm)")
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")
        for j in range(nbt, len(axes)):
            axes[j].axis("off")
        fig.suptitle(
            f"孔半径 two-point（喉连接孔对，密度分箱）{tp}",
            fontsize=13,
        )
        plt.tight_layout()
        out_png = out_dir / out_name
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"[2.4b] 图已保存: {out_png}")

    _plot_connected_density_split(
        "WT",
        "sum_raw",
        r"$G_r(x)=\langle r_i r_j \rangle$",
        "pore_radius_two_point_connected_densitybins_WT_raw.png",
        draw_zero_line=False,
    )
    _plot_connected_density_split(
        "WT",
        "sum_norm",
        r"$C_r(x)=\langle \delta r_i \delta r_j \rangle$",
        "pore_radius_two_point_connected_densitybins_WT_dimensionless.png",
        draw_zero_line=True,
    )
    _plot_connected_density_split(
        "AS",
        "sum_raw",
        r"$G_r(x)=\langle r_i r_j \rangle$",
        "pore_radius_two_point_connected_densitybins_AS_raw.png",
        draw_zero_line=False,
    )
    _plot_connected_density_split(
        "AS",
        "sum_norm",
        r"$C_r(x)=\langle \delta r_i \delta r_j \rangle$",
        "pore_radius_two_point_connected_densitybins_AS_dimensionless.png",
        draw_zero_line=True,
    )

    meta = {
        "pair_modes": ["spatial_all_pairs", "connected_pairs"],
        "thickness_bin_width_nm": float(thickness_bin_width),
        "distance_bin_width_nm": float(distance_bin_width),
        "max_distance_nm": float(max_distance_nm),
        "max_pores_per_subsample": None,
        "n_thickness_bins": int(n_bins),
        "connected_density_bins_mode": {
            "enabled": True,
            "bin_width_nm": float(DENSITY_FRAC_BIN_WIDTH_NM),
            "split_by_type": True,
        },
    }
    meta_path = out_dir / "pore_radius_two_point_config.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[2.4b] 配置已保存: {meta_path}")
    return meta


def main():
    _parent_dir = WORKSPACE_ROOT

    parser = argparse.ArgumentParser(
        description="阶段二：孔喉模型参数确定（WT / AS 分开）"
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
        default=str(OUTPUT_ROOT / "phase2_parameters"),
        help="输出目录（默认：项目 outputs/phase2_parameters）",
    )
    parser.add_argument(
        "--thickness-bin-width",
        type=float,
        default=10.0,
        help="厚度分层区间宽度 (nm)，用于按厚度区间的 x 分布分析（默认 10）",
    )
    parser.add_argument(
        "--min-points-per-bin",
        type=int,
        default=30,
        help="厚度区间内至少多少个半径点才做分布拟合（默认 30）",
    )
    parser.add_argument(
        "--p-value-threshold",
        type=float,
        default=0.05,
        help="区间分布比较时，p > 该值视为两区间分布近似相同（默认 0.05）",
    )
    parser.add_argument(
        "--min-points-per-bin-metric",
        type=int,
        default=5,
        help="密度/体积分数按厚度区间拟合时，区间内至少多少个子样本才拟合（默认 5）",
    )
    parser.add_argument(
        "--corr-thickness-bin-width",
        type=float,
        default=20.0,
        help="2.4b two-point correlation 的厚度分箱宽度 (nm)，默认 20",
    )
    parser.add_argument(
        "--corr-distance-bin-width",
        type=float,
        default=5.0,
        help="2.4b two-point correlation 的距离分箱宽度 (nm)，默认 5",
    )
    parser.add_argument(
        "--corr-max-distance",
        type=float,
        default=300.0,
        help="2.4b two-point correlation 最大距离 x (nm)，默认 300",
    )
    parser.add_argument(
        "--as-high-bin-mode",
        choices=("custom", "legacy"),
        default="custom",
        help=(
            "AS thickness binning mode. custom keeps each analysis' bin width up to "
            "160 nm, then merges >160; legacy uses the old >=150 merge."
        ),
    )
    parser.add_argument(
        "--smooth-as-sparse-bins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Smooth AS sparse thickness-bin fits in the configured 120-160 nm range.",
    )
    parser.add_argument(
        "--as-smooth-bandwidth-nm",
        type=float,
        default=30.0,
        help="Gaussian bandwidth for AS 120-160 nm sparse-bin smoothing.",
    )
    parser.add_argument(
        "--as-radius-smooth-bandwidth-nm",
        type=float,
        default=50.0,
        help="Gaussian bandwidth for AS radius smoothing in the 120-160 nm sparse range.",
    )
    parser.add_argument(
        "--as-radius-left-source-min-nm",
        type=float,
        default=100.0,
        help="Lower bound of the AS left anchor range used for radius smoothing.",
    )
    parser.add_argument(
        "--as-radius-left-source-max-nm",
        type=float,
        default=120.0,
        help="Upper bound of the AS left anchor range used for radius smoothing.",
    )
    parser.add_argument(
        "--as-radius-external-anchors-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For AS radius smoothing in 120-160 nm, use only the target bin plus "
            "external left/right anchor ranges, excluding other 120-160 internal bins."
        ),
    )
    parser.add_argument(
        "--as-smooth-right-source-min-nm",
        type=float,
        default=160.0,
        help="Lower bound of the AS high-thickness source-only smoothing range.",
    )
    parser.add_argument(
        "--as-smooth-right-source-max-nm",
        type=float,
        default=200.0,
        help="Upper bound of the AS high-thickness source-only smoothing range.",
    )
    parser.add_argument(
        "--as-low-edge-smooth-bandwidth-nm",
        type=float,
        default=15.0,
        help="Gaussian bandwidth for AS low-edge sparse-bin smoothing.",
    )
    parser.add_argument(
        "--as-low-edge-radius-smooth-bandwidth-nm",
        type=float,
        default=25.0,
        help="Gaussian bandwidth for AS low-edge radius smoothing.",
    )
    parser.add_argument(
        "--as-smooth-synthetic-n",
        type=int,
        default=2000,
        help="Minimum synthetic draws used to refit each smoothed AS sparse bin.",
    )
    args = parser.parse_args()

    global AS_HIGH_BIN_MODE
    global AS_SPARSE_SMOOTH_ENABLED
    global AS_SPARSE_SMOOTH_BANDWIDTH_NM
    global AS_RADIUS_SPARSE_SMOOTH_BANDWIDTH_NM
    global AS_RADIUS_SPARSE_SMOOTH_LEFT_SOURCE_RANGE_NM
    global AS_RADIUS_SPARSE_SMOOTH_EXTERNAL_ANCHORS_ONLY
    global AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM
    global AS_LOW_EDGE_SMOOTH_BANDWIDTH_NM
    global AS_LOW_EDGE_RADIUS_SMOOTH_BANDWIDTH_NM
    global AS_SPARSE_SMOOTH_SYNTHETIC_N
    AS_HIGH_BIN_MODE = str(args.as_high_bin_mode)
    AS_SPARSE_SMOOTH_ENABLED = bool(args.smooth_as_sparse_bins)
    AS_SPARSE_SMOOTH_BANDWIDTH_NM = float(args.as_smooth_bandwidth_nm)
    AS_RADIUS_SPARSE_SMOOTH_BANDWIDTH_NM = float(args.as_radius_smooth_bandwidth_nm)
    AS_RADIUS_SPARSE_SMOOTH_LEFT_SOURCE_RANGE_NM = (
        float(args.as_radius_left_source_min_nm),
        float(args.as_radius_left_source_max_nm),
    )
    AS_RADIUS_SPARSE_SMOOTH_EXTERNAL_ANCHORS_ONLY = bool(args.as_radius_external_anchors_only)
    AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM = (
        float(args.as_smooth_right_source_min_nm),
        float(args.as_smooth_right_source_max_nm),
    )
    if AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM[1] <= AS_SPARSE_SMOOTH_RIGHT_SOURCE_RANGE_NM[0]:
        raise ValueError("--as-smooth-right-source-max-nm must be greater than --as-smooth-right-source-min-nm")
    if AS_RADIUS_SPARSE_SMOOTH_LEFT_SOURCE_RANGE_NM[1] <= AS_RADIUS_SPARSE_SMOOTH_LEFT_SOURCE_RANGE_NM[0]:
        raise ValueError("--as-radius-left-source-max-nm must be greater than --as-radius-left-source-min-nm")
    AS_LOW_EDGE_SMOOTH_BANDWIDTH_NM = float(args.as_low_edge_smooth_bandwidth_nm)
    AS_LOW_EDGE_RADIUS_SMOOTH_BANDWIDTH_NM = float(args.as_low_edge_radius_smooth_bandwidth_nm)
    AS_SPARSE_SMOOTH_SYNTHETIC_N = int(args.as_smooth_synthetic_n)

    split_samples_dir = Path(args.split_samples_dir)
    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.output_dir)

    print("=" * 60)
    print("阶段二：孔喉模型参数确定（WT / AS 分开）")
    print("=" * 60)
    print("注意：使用所有结构数据，不进行喉半径过滤（构建全模型需要完整结构）")

    # 收集数据
    print("\n收集所有子样本数据...")
    df_summary, data_dict = collect_all_sub_sample_data(
        split_samples_dir, analysis_dir
    )
    print(f"共收集 {len(df_summary)} 个子样本的 summary 数据")
    print(f"共收集 {len(data_dict)} 个子样本的数组数据")
    # 为 df_summary 补全 n_pore（供孔数量分数按厚度区间用）
    n_pore_list = []
    for _, row in df_summary.iterrows():
        key = (row["sample_name"], row["sub_name"])
        if key in data_dict and "pore_radii" in data_dict[key]:
            n_pore_list.append(len(data_dict[key]["pore_radii"]))
        else:
            n_pore_list.append(np.nan)
    df_summary["n_pore"] = n_pore_list

    # 2.1 孔/喉数量密度与体积密度
    print("\n[2.1] 孔/喉数量密度与体积密度 vs 厚度...")
    density_results = analyze_density_vs_thickness(df_summary, out_dir)

    # 2.2 孔半径、喉半径分布及与厚度的关系
    print("\n[2.2] 孔半径、喉半径分布及与厚度的关系...")
    radius_results = analyze_radius_distributions(df_summary, data_dict, out_dir)
    plot_radius_fitted_vs_data(radius_results, df_summary, data_dict, out_dir)

    # 2.2b 按厚度区间（默认 10 nm）的条件分布 p(x | T in bin)
    print("\n[2.2b] 按厚度区间的孔/喉半径分布（条件分布）...")
    radius_bins_results = analyze_radius_by_thickness_bins(
        df_summary, data_dict, out_dir,
        bin_width=args.thickness_bin_width,
        min_points_per_bin=args.min_points_per_bin,
    )
    # 2.2c 区间间分布比较（KS 检验，找出近乎相同的区间）
    print("\n[2.2c] 厚度区间间分布比较（KS 检验）...")
    compare_distributions_across_thickness_bins(
        df_summary, data_dict, radius_bins_results, out_dir,
        min_points_per_bin=args.min_points_per_bin,
        p_value_threshold=args.p_value_threshold,
    )

    # 2.2c2 孔度数按厚度区间的分布与拟合 + 区间间 KS 检验热图
    print("\n[2.2c2] 孔度数按厚度区间的分布（与孔半径相同分层）...")
    degree_bins_results = analyze_degree_by_thickness_bins(
        df_summary, data_dict, out_dir,
        bin_width=args.thickness_bin_width,
        min_points_per_bin=args.min_points_per_bin,
    )
    print("\n[2.2c2b] 孔度数厚度区间间分布比较（KS 检验）...")
    compare_degree_distributions_across_thickness_bins(
        df_summary, data_dict, degree_bins_results, out_dir,
        min_points_per_bin=args.min_points_per_bin,
        p_value_threshold=args.p_value_threshold,
    )

    # 2.2d 密度与体积分数按厚度区间的条件分布 + 区间间比较
    print("\n[2.2d] 孔/喉密度与体积分数按厚度区间的分布及区间间比较...")
    density_frac_bins_results = analyze_density_frac_by_thickness_bins(
        df_summary, out_dir,
        min_points_per_bin=args.min_points_per_bin_metric,
    )
    if density_frac_bins_results:
        compare_metric_distributions_across_thickness_bins(
            df_summary, density_frac_bins_results, out_dir,
            min_points_per_bin=args.min_points_per_bin_metric,
            p_value_threshold=args.p_value_threshold,
        )

    # 2.2e 孔在不同厚度区间的数量分数（供 Phase3 撒点分布）
    print("\n[2.2e] 孔在不同厚度区间的数量分数...")
    analyze_pore_count_fraction_by_thickness_bin(df_summary, out_dir)

    # 2.3 孔半径与喉半径的相关性
    print("\n[2.3] 孔半径与喉半径的相关性...")
    correlation_results = analyze_pore_throat_radius_correlation(df_summary, data_dict, split_samples_dir, out_dir)

    # 2.4 喉方向性（Q、S）与厚度
    print("\n[2.4] 喉方向性（Q、S）与厚度...")
    directionality_results = analyze_directionality_vs_thickness(df_summary, out_dir)

    # 2.4b 孔半径 two-point correlation（空间全孔对 + 喉连接孔对）
    print("\n[2.4b] 孔半径 two-point correlation（按厚度区间子图）...")
    _ = analyze_pore_radius_two_point_correlation(
        df_summary,
        data_dict,
        out_dir,
        thickness_bin_width=float(args.corr_thickness_bin_width),
        distance_bin_width=float(args.corr_distance_bin_width),
        max_distance_nm=float(args.corr_max_distance),
    )

    # 2.5 喉长度分布及与厚度的关系
    print("\n[2.5] 喉长度 vs 厚度...")
    throat_length_results = analyze_throat_length_vs_thickness(
        df_summary, split_samples_dir, out_dir,
        data_dict=data_dict,
        min_points_per_bin=args.min_points_per_bin,
    )
    # 2.5b 喉长度区间间分布比较（KS 检验热图）
    print("\n[2.5b] 喉长度厚度区间间分布比较（KS 检验）...")
    compare_length_distributions_across_thickness_bins(
        df_summary, data_dict, throat_length_results, out_dir, split_samples_dir,
        min_points_per_bin=args.min_points_per_bin,
        p_value_threshold=args.p_value_threshold,
    )

    # 2.5c overlap 喉：R_throat / R_cap（交界面圆半径）按厚度分箱拟合（供 Phase3 乘系数）
    print("\n[2.5c] overlap 喉 R_throat vs R_cap（按厚度）...")
    analyze_overlap_throat_R_ratio_vs_thickness(
        df_summary, split_samples_dir, out_dir,
        bin_width=THROAT_LENGTH_BIN_WIDTH_NM,
        min_points_per_bin=args.min_points_per_bin,
    )

    # 2.5d 子样本级：overlap/非 overlap 喉中「r_throat > min(两端孔半径)」比例 的分布（按厚度分箱 + 拟合）
    print("\n[2.5d] 子样本级：喉超过小孔半径比例 vs 厚度（overlap / 非overlap 分开）...")
    analyze_throat_gt_min_pore_fraction_per_subsample_by_thickness_bins(
        df_summary,
        split_samples_dir,
        out_dir,
        min_points_per_bin=args.min_points_per_bin_metric,
    )

    # 2.5e cross-face 非 overlap 喉密度：N_nonoverlap_cross / A_cut（每子样本两次切割）
    print("\n[2.5e] cross-face 非 overlap 喉密度 vs 厚度...")
    analyze_nonoverlap_cross_density_by_thickness_bin(
        df_summary,
        split_samples_dir,
        out_dir,
        bin_width=DENSITY_FRAC_BIN_WIDTH_NM,
        min_points_per_bin=args.min_points_per_bin_metric,
    )

    # 2.6 喉半径在不同喉长度区间下的分布及拟合
    print("\n[2.6] 喉半径 vs 喉长度区间...")
    analyze_throat_radius_by_length_bins(
        df_summary,
        data_dict,
        out_dir,
        length_bin_width_nm=10.0,
        length_merge_threshold_nm=60.0,
        min_points_per_bin=args.min_points_per_bin,
    )

    print("\n" + "=" * 60)
    print("阶段二完成！")
    print("=" * 60)
    print(f"\n输出目录: {out_dir}")


if __name__ == "__main__":
    main()

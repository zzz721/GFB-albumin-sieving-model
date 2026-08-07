"""
对某一样本的所有子样本做结构统计学分析，并把各子样本的可视化结果画在一张图上进行比较。
- 若存在总样本（throats_and_pores_xlsx 下 <sample_name>_pores.xlsx / _throats.xlsx），则一并参与对比；
  小提琴图上对每个子样本做"子样本 vs 总样本"的 Mann-Whitney U 检验并标注 p 值（是否服从同一分布）。
- 仅考虑有法向量的子样本（排除无 metadata / normal_vector 的边角料子样本）
- 发现子样本：<sample_name>_sub*_pores.xlsx / _throats.xlsx 且存在 *_metadata.json 且含 normal_vector
- 输出：总样本+各子样本汇总表（Excel）+ 多张对比图（点线图+小提琴图，小提琴上标注 p 值）
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gbm_sieving.paths import OUTPUT_ROOT, WORKSPACE_ROOT

# 兼容中文
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

# 复用单样本分析中的函数
from gbm_sieving.analysis.structure import statistics as st


def find_sub_samples(sub_sample_dir: Path, sample_name: str):
    """返回 [(sub_name, pores_file, throats_file), ...] 按 sub_name 排序。仅包含有法向量的子样本（排除边角料）。"""
    sub_sample_dir = Path(sub_sample_dir)
    pattern = f"{sample_name}_sub*_pores.xlsx"
    pore_files = sorted(sub_sample_dir.glob(pattern))
    if not pore_files:
        inner = sub_sample_dir / sample_name
        if inner.exists():
            pore_files = sorted(inner.glob(pattern))
            if pore_files:
                sub_sample_dir = inner
    out = []
    for pf in pore_files:
        sub_name = pf.stem.replace("_pores", "")
        tf = sub_sample_dir / f"{sub_name}_throats.xlsx"
        meta_file = sub_sample_dir / f"{sub_name}_metadata.json"
        if not tf.exists():
            continue
        # 只保留有法向量的子样本（无 metadata 或无 normal_vector 的边角料跳过）
        has_normal = False
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    if meta.get("normal_vector") is not None:
                        has_normal = True
            except Exception:
                pass
        if not has_normal:
            print(f"  跳过无法向量的子样本（边角料）: {sub_name}")
            continue
        out.append((sub_name, pf, tf))
    return out


def run_one_sub_sample(pores_file: Path, throats_file: Path, *, min_throat_radius: float = None):
    """
    对单个子样本做结构统计，返回用于汇总的标量字典和用于画图的数组。
    
    min_throat_radius: 如果提供，只保留半径 >= 该值的喉，然后基于过滤后的喉集合计算统计指标。
    """
    pore_coords, pore_radii, pore_ids, pore_id_to_idx, throat_pore1, throat_pore2, throat_radii = \
        st.load_pores_and_throats(pores_file, throats_file)
    
    # 按喉半径过滤（如果指定了阈值）
    if min_throat_radius is not None:
        mask = throat_radii >= min_throat_radius
        throat_pore1 = throat_pore1[mask]
        throat_pore2 = throat_pore2[mask]
        throat_radii = throat_radii[mask]
    
    n_pore = len(pore_ids)
    n_throat = len(throat_pore1)

    vol = st.volume_expanded_convex_hull(pore_coords, pore_radii)
    V_pore = st.pore_occupied_volume(pore_radii)
    V_throat = st.throat_occupied_volume(
        pore_coords, pore_radii, pore_id_to_idx, throat_pore1, throat_pore2, throat_radii
    )
    rho_pore = n_pore / vol if vol > 0 else 0.0
    rho_throat = n_throat / vol if vol > 0 else 0.0
    frac_pore = V_pore / vol if vol > 0 else 0.0
    frac_throat = V_throat / vol if vol > 0 else 0.0

    deg = st.degree_per_pore(pore_ids, throat_pore1, throat_pore2)
    mean_deg = float(np.mean(deg)) if len(deg) else 0.0

    # 全局 Q 张量（基于该子样本所有喉方向），用于定义全局有序参数 S_global
    _, evals_q, _ = st.global_q_tensor(pore_coords, pore_id_to_idx, throat_pore1, throat_pore2)
    q_max = float(evals_q[-1])
    q_min = float(evals_q[0])
    # 在我们当前的 Q 定义下，理想单轴时 lam_max = S
    S_global = q_max

    # 局部按孔的 S 仅用于后续 S 分布图（不再写入 summary 的 mean_S_local）
    S_pore = st.per_pore_throat_order(pore_ids, pore_coords, pore_id_to_idx, throat_pore1, throat_pore2)
    S_valid = S_pore[~np.isnan(S_pore)]

    # 喉长度（几何长度）= 孔心距 d = |p1-p2|（不减去孔半径）
    idx1 = np.array([pore_id_to_idx.get(int(p), -1) for p in throat_pore1])
    idx2 = np.array([pore_id_to_idx.get(int(p), -1) for p in throat_pore2])
    mask = (idx1 >= 0) & (idx2 >= 0)
    d_center = np.linalg.norm(pore_coords[idx1[mask]] - pore_coords[idx2[mask]], axis=1)
    throat_lengths = d_center.copy()

    r_center, g_r, _ = st.compute_rdf(pore_coords)
    # 孔心距采样（用于 RDF 小提琴图）
    n_p = len(pore_coords)
    n_pairs_max = 5000
    if n_p * (n_p - 1) // 2 <= n_pairs_max:
        from scipy.spatial.distance import pdist
        pair_distances = pdist(pore_coords)
    else:
        pair_distances = []
        for _ in range(n_pairs_max):
            i, j = np.random.randint(0, n_p, 2)
            if i != j:
                pair_distances.append(np.linalg.norm(pore_coords[i] - pore_coords[j]))
        pair_distances = np.array(pair_distances)

    summary = {
        "n_pore": n_pore,
        "n_throat": n_throat,
        "vol_nm3": vol,
        "rho_pore": rho_pore,
        "rho_throat": rho_throat,
        "frac_pore": frac_pore,
        "frac_throat": frac_throat,
        "mean_deg": mean_deg,
        "pore_r_mean": float(np.mean(pore_radii)),
        "pore_r_std": float(np.std(pore_radii)),
        "throat_r_mean": float(np.mean(throat_radii)),
        "throat_r_std": float(np.std(throat_radii)),
        # 基于全局 Q 张量的有序参数（用于 thickness/结构分析）
        "Q_max": q_max,
        "Q_min": q_min,
        "mean_S": S_global,
    }
    arrays = {
        "deg": deg,
        "pore_radii": pore_radii,
        "throat_radii": throat_radii,
        "throat_lengths": throat_lengths,
        "S_valid": S_valid,
        "r_center": r_center,
        "g_r": g_r,
        "pair_distances": pair_distances,
        "pore_coords": pore_coords,
    }
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(
        description="对某样本的所有子样本做结构统计，并在一张图上对比各子样本的可视化结果"
    )
    parser.add_argument("--sample-name", type=str, required=True, help="样本名，如 AS307")
    parser.add_argument(
        "--sub-sample-dir",
        type=str,
        default=str(WORKSPACE_ROOT / "data_subsamples"),
        help="子样本顶层目录（默认：肾脏/data_subsamples），其下或有 <sample_name> 子目录，存放 *_sub*_pores.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_ROOT / "analysis_structure_properties_compare_subsample"),
        help="输出目录（默认：肾脏/results_analysis_structure_properties_compare_subsample）；对比图与汇总表写在 <output-dir>/<sample_name>/ 下",
    )
    parser.add_argument(
        "--total-sample-dir",
        type=str,
        default=str(WORKSPACE_ROOT / "throats_and_pores_xlsx"),
        help='总样本孔喉 xlsx 所在目录（默认：肾脏/throats_and_pores_xlsx）；若存在则参与对比并在小提琴图上做"子样本 vs 总样本"分布检验',
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["mannwhitney", "epps_singleton"],
        default="epps_singleton",
        help='两样本同分布检验：mannwhitney=Mann-Whitney U；epps_singleton=Epps-Singleton（基于经验特征函数，对"同分布"检验通常更有功效；需每组至少 5 个观测，不足时自动用 Mann-Whitney）',
    )
    args = parser.parse_args()

    sample_name = args.sample_name
    sub_sample_dir = Path(args.sub_sample_dir)
    total_sample_dir = Path(args.total_sample_dir)
    out_dir = Path(args.output_dir) / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sub_list = find_sub_samples(sub_sample_dir, sample_name)
    if not sub_list:
        print(f"错误：未找到任何子样本（在 {sub_sample_dir} 下匹配 {sample_name}_sub*_pores.xlsx）")
        return

    # 尝试加载总样本（throats_and_pores_xlsx 下的 <sample_name>_pores.xlsx / _throats.xlsx）
    total_summary = None
    total_arrays = None
    total_pores_file = total_sample_dir / f"{sample_name}_pores.xlsx"
    total_throats_file = total_sample_dir / f"{sample_name}_throats.xlsx"
    if total_pores_file.exists() and total_throats_file.exists():
        print(f"加载总样本: {total_pores_file.name} / {total_throats_file.name}")
        try:
            total_summary, total_arrays = run_one_sub_sample(total_pores_file, total_throats_file)
            total_summary["子样本"] = "总样本"
        except Exception as e:
            print(f"  跳过总样本: {e}")
            total_summary, total_arrays = None, None
    else:
        print(f"未找到总样本文件（{total_pores_file} 或 {total_throats_file}），仅对比子样本。")

    print(f"找到 {len(sub_list)} 个子样本，开始逐个分析并收集数据...")
    all_summaries = []
    all_arrays = []
    sub_names = []

    for sub_name, pores_file, throats_file in sub_list:
        print(f"  分析: {sub_name}")
        try:
            summary, arrays = run_one_sub_sample(pores_file, throats_file)
            summary["子样本"] = sub_name
            all_summaries.append(summary)
            all_arrays.append(arrays)
            sub_names.append(sub_name)
        except Exception as e:
            print(f"    跳过 {sub_name}: {e}")
            continue

    if not all_summaries:
        print("没有成功分析的子样本，退出。")
        return

    # 若有总样本，插入到列表最前（汇总表与图中均为第一列/第一条）
    if total_summary is not None and total_arrays is not None:
        all_summaries = [total_summary] + all_summaries
        all_arrays = [total_arrays] + all_arrays
        sub_names = ["总样本"] + sub_names

    # ---------- 汇总表（总样本 + 每个子样本一行） ----------
    summary_df = pd.DataFrame(all_summaries)
    cols_order = ["子样本", "n_pore", "n_throat", "vol_nm3", "rho_pore", "rho_throat", "frac_pore", "frac_throat",
                  "mean_deg", "pore_r_mean", "pore_r_std", "throat_r_mean", "throat_r_std", "Q_max", "Q_min", "mean_S"]
    summary_df = summary_df[[c for c in cols_order if c in summary_df.columns]]
    summary_excel = out_dir / f"{sample_name}_sub_samples_summary.xlsx"
    summary_df.to_excel(summary_excel, index=False)
    print(f"汇总表已保存: {summary_excel}")

    # ---------- 统一 RDF 的 r_max，便于公平比较 ----------
    r_max_global = None
    n_bins_rdf = 80
    if len(all_arrays) > 0 and "r_center" in all_arrays[0] and len(all_arrays[0]["r_center"]) > 1:
        dr = all_arrays[0]["r_center"][1] - all_arrays[0]["r_center"][0]
        r_max_per_sub = [arr["r_center"][-1] + dr / 2.0 for arr in all_arrays]
        r_max_global = max(r_max_per_sub)
        print(f"  统一 RDF r_max = {r_max_global:.2f} nm（原各子样本 r_max 范围: [{min(r_max_per_sub):.2f}, {max(r_max_per_sub):.2f}]）")
        for arr in all_arrays:
            r_center, g_r, _ = st.compute_rdf(arr["pore_coords"], r_max=r_max_global, n_bins=n_bins_rdf)
            arr["r_center"] = r_center
            arr["g_r"] = g_r

    # ---------- 对比图：总样本（若有）+ 子样本，小提琴图上标注"子样本 vs 总样本"检验 p 值 ----------
    n_sub = len(sub_names)
    has_total = total_arrays is not None and n_sub > 0 and sub_names[0] == "总样本"
    if has_total:
        test_name = "Epps-Singleton" if args.test == "epps_singleton" else "Mann-Whitney U"
        print(f"  子样本 vs 总样本 同分布检验: {test_name}")
    if has_total:
        colors = ['#7f7f7f'] + list(plt.cm.tab10(np.linspace(0, 1, max(n_sub - 1, 1))))
        if n_sub <= 5:
            colors = ['#7f7f7f'] + list(plt.cm.Set1(np.linspace(0, 1, max(n_sub - 1, 1))))
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_sub, 1)))
        if n_sub <= 4:
            colors = plt.cm.Set1(np.linspace(0, 1, max(n_sub, 1)))

    def sub_vs_total_pvalues(total_data, list_of_sub_arrays):
        """各子样本 vs 总样本的两样本同分布检验，返回与 list_of_sub_arrays 等长的 p 值列表。"""
        total_data = np.asarray(total_data)
        if len(total_data) == 0:
            return [np.nan] * len(list_of_sub_arrays)
        use_epps = args.test == "epps_singleton"
        out = []
        for sub_data in list_of_sub_arrays:
            sub_data = np.asarray(sub_data)
            if len(sub_data) == 0:
                out.append(np.nan)
                continue
            try:
                if use_epps and len(total_data) >= 5 and len(sub_data) >= 5:
                    _, p = stats.epps_singleton_2samp(total_data, sub_data)
                else:
                    _, p = stats.mannwhitneyu(total_data, sub_data, alternative="two-sided")
                out.append(float(p))
            except Exception:
                out.append(np.nan)
        return out

    # 统一用点线图：先做直方统计，再 plot(bin_center, count, 'o-')
    def hist_to_line(data, bins):
        h, _ = np.histogram(data, bins=bins)
        bc = (bins[:-1] + bins[1:]) / 2.0
        return bc, h

    def plot_violin(ax, data_per_sub, labels, colors, ylabel, sub_vs_total_pvalues=None):
        """在 ax 上画小提琴图；若有 sub_vs_total_pvalues（子样本 vs 总样本的 p 值），在对应小提琴上方标注。"""
        data_per_sub = [np.asarray(d) for d in data_per_sub if len(np.asarray(d)) > 0]
        if not data_per_sub:
            return
        n = len(data_per_sub)
        labels = list(labels)[:n] if len(labels) >= n else [str(i) for i in range(n)]
        pos = range(n)
        parts = ax.violinplot(data_per_sub, positions=pos, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        if sub_vs_total_pvalues is not None and len(sub_vs_total_pvalues) == n - 1 and n >= 2:
            ymax = ax.get_ylim()[1]
            for i, p in enumerate(sub_vs_total_pvalues):
                pos_i = i + 1
                if np.isnan(p):
                    txt = "—"
                elif p >= 0.05:
                    txt = f"p={p:.2f}\n(n.s.)"
                else:
                    txt = f"p={p:.3g}\n*"
                ax.text(pos_i, ymax * 0.97, txt, ha="center", va="top", fontsize=8)

    # 1) 孔上喉数：左折线图，右小提琴图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    deg_max_global = max(np.max(arr["deg"]) for arr in all_arrays)
    bins_deg = np.arange(0, min(deg_max_global + 2, 35), 1)
    for i, (name, arr) in enumerate(zip(sub_names, all_arrays)):
        bc, cnt = hist_to_line(arr["deg"], bins_deg)
        ax1.plot(bc, cnt, "o-", label=name, color=colors[i % len(colors)], linewidth=1.5, markersize=5)
    ax1.set_xlabel("孔上喉数")
    ax1.set_ylabel("孔个数")
    ax1.set_title("孔上喉数分布（折线）")
    ax1.legend()
    ax1.set_xlim(left=-0.5)
    ax1.set_ylim(bottom=0)
    deg_pvals = sub_vs_total_pvalues(all_arrays[0]["deg"], [all_arrays[i]["deg"] for i in range(1, len(all_arrays))]) if has_total else None
    plot_violin(ax2, [arr["deg"] for arr in all_arrays], sub_names, colors, "孔上喉数", sub_vs_total_pvalues=deg_pvals)
    ax2.set_title("孔上喉数分布（小提琴）")
    plt.suptitle(f"{sample_name} 子样本对比：孔上喉数分布")
    plt.tight_layout()
    plt.savefig(out_dir / f"{sample_name}_compare_degree.png", dpi=150)
    plt.close()
    print(f"  对比图: {sample_name}_compare_degree.png")

    # 2) 孔半径：左折线图，右小提琴图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rp_all = np.concatenate([arr["pore_radii"] for arr in all_arrays])
    bins_rp = np.linspace(rp_all.min(), rp_all.max(), 41)
    for i, (name, arr) in enumerate(zip(sub_names, all_arrays)):
        bc, cnt = hist_to_line(arr["pore_radii"], bins_rp)
        ax1.plot(bc, cnt, "o-", label=name, color=colors[i % len(colors)], linewidth=1.5, markersize=4)
    ax1.set_xlabel("孔半径 (nm)")
    ax1.set_ylabel("孔个数")
    ax1.set_title("孔半径分布（折线）")
    ax1.legend()
    ax1.set_ylim(bottom=0)
    rp_pvals = sub_vs_total_pvalues(all_arrays[0]["pore_radii"], [all_arrays[i]["pore_radii"] for i in range(1, len(all_arrays))]) if has_total else None
    plot_violin(ax2, [arr["pore_radii"] for arr in all_arrays], sub_names, colors, "孔半径 (nm)", sub_vs_total_pvalues=rp_pvals)
    ax2.set_title("孔半径分布（小提琴）")
    plt.suptitle(f"{sample_name} 子样本对比：孔半径分布")
    plt.tight_layout()
    plt.savefig(out_dir / f"{sample_name}_compare_pore_radius.png", dpi=150)
    plt.close()
    print(f"  对比图: {sample_name}_compare_pore_radius.png")

    # 3) 喉半径：左折线图，右小提琴图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rt_all = np.concatenate([arr["throat_radii"] for arr in all_arrays])
    bins_rt = np.linspace(rt_all.min(), rt_all.max(), 41)
    for i, (name, arr) in enumerate(zip(sub_names, all_arrays)):
        bc, cnt = hist_to_line(arr["throat_radii"], bins_rt)
        ax1.plot(bc, cnt, "o-", label=name, color=colors[i % len(colors)], linewidth=1.5, markersize=4)
    ax1.set_xlabel("喉半径 (nm)")
    ax1.set_ylabel("喉个数")
    ax1.set_title("喉半径分布（折线）")
    ax1.legend()
    ax1.set_ylim(bottom=0)
    rt_pvals = sub_vs_total_pvalues(all_arrays[0]["throat_radii"], [all_arrays[i]["throat_radii"] for i in range(1, len(all_arrays))]) if has_total else None
    plot_violin(ax2, [arr["throat_radii"] for arr in all_arrays], sub_names, colors, "喉半径 (nm)", sub_vs_total_pvalues=rt_pvals)
    ax2.set_title("喉半径分布（小提琴）")
    plt.suptitle(f"{sample_name} 子样本对比：喉半径分布")
    plt.tight_layout()
    plt.savefig(out_dir / f"{sample_name}_compare_throat_radius.png", dpi=150)
    plt.close()
    print(f"  对比图: {sample_name}_compare_throat_radius.png")

    # 4) 孔上喉 S：左折线图，右小提琴图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    S_all = np.concatenate([arr["S_valid"] for arr in all_arrays if len(arr["S_valid"]) > 0])
    if len(S_all) > 0:
        bins_S = np.linspace(S_all.min(), S_all.max(), 41)
        for i, (name, arr) in enumerate(zip(sub_names, all_arrays)):
            sv = arr["S_valid"]
            if len(sv) == 0:
                continue
            bc, cnt = hist_to_line(sv, bins_S)
            ax1.plot(bc, cnt, "o-", label=name, color=colors[i % len(colors)], linewidth=1.5, markersize=4)
    ax1.set_xlabel("秩序参数 S (孔上喉方向一致性)")
    ax1.set_ylabel("孔个数")
    ax1.set_title("孔上喉取向 S 分布（折线）")
    ax1.legend()
    ax1.set_ylim(bottom=0)
    S_list = [arr["S_valid"] for arr in all_arrays if len(arr["S_valid"]) > 0]
    sub_S = [sub_names[i] for i, arr in enumerate(all_arrays) if len(arr["S_valid"]) > 0]
    S_pvals = sub_vs_total_pvalues(S_list[0], S_list[1:]) if has_total and len(S_list) > 1 else None
    if S_list:
        plot_violin(ax2, S_list, sub_S, colors, "秩序参数 S", sub_vs_total_pvalues=S_pvals)
    ax2.set_title("孔上喉取向 S 分布（小提琴）")
    plt.suptitle(f"{sample_name} 子样本对比：孔上喉取向倾向 S 分布")
    plt.tight_layout()
    plt.savefig(out_dir / f"{sample_name}_compare_pore_throat_S.png", dpi=150)
    plt.close()
    print(f"  对比图: {sample_name}_compare_pore_throat_S.png")

    # 5) RDF 分两张图：归一化前（孔心距直方 + 小提琴）、归一化后（g(r) 折线 + g(r) 值小提琴）
    if r_max_global is not None:
        bins_rdf = np.linspace(0.0, r_max_global, n_bins_rdf + 1)
        r_center_rdf = (bins_rdf[:-1] + bins_rdf[1:]) / 2.0

        # 5a) 归一化前：左 孔心距直方（count vs r），右 孔心距小提琴
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for i, (name, arr) in enumerate(zip(sub_names, all_arrays)):
            pair_count, _ = np.histogram(arr["pair_distances"], bins=bins_rdf)
            ax1.plot(r_center_rdf, pair_count, "o-", label=name, color=colors[i % len(colors)], linewidth=1.5, markersize=3)
        ax1.set_xlabel("r (nm)")
        ax1.set_ylabel("孔心对数量")
        ax1.set_title("孔心距直方（折线，未归一化）")
        ax1.set_ylim(bottom=0)
        ax1.legend()
        pd_pvals = sub_vs_total_pvalues(all_arrays[0]["pair_distances"], [all_arrays[i]["pair_distances"] for i in range(1, len(all_arrays))]) if has_total else None
        plot_violin(ax2, [arr["pair_distances"] for arr in all_arrays], sub_names, colors, "孔心距 (nm)", sub_vs_total_pvalues=pd_pvals)
        ax2.set_title("孔心距分布（小提琴）")
        plt.suptitle(f"{sample_name} 子样本对比：孔心距（归一化前）")
        plt.tight_layout()
        plt.savefig(out_dir / f"{sample_name}_compare_RDF_before_norm.png", dpi=150)
        plt.close()
        print(f"  对比图: {sample_name}_compare_RDF_before_norm.png")

        # 5b) 归一化后：左 g(r) 折线，右 g(r) 值小提琴
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for i, (name, arr) in enumerate(zip(sub_names, all_arrays)):
            r, g = arr["r_center"], arr["g_r"]
            ax1.plot(r, g, "o-", label=name, color=colors[i % len(colors)], linewidth=1.5, markersize=3)
        ax1.set_xlabel("r (nm)")
        ax1.set_ylabel("g(r)")
        ax1.set_title("孔心径向分布函数 RDF（折线，体积归一化）")
        ax1.set_ylim(bottom=0)
        ax1.legend()
        gr_pvals = sub_vs_total_pvalues(all_arrays[0]["g_r"], [all_arrays[i]["g_r"] for i in range(1, len(all_arrays))]) if has_total else None
        plot_violin(ax2, [arr["g_r"] for arr in all_arrays], sub_names, colors, "g(r)", sub_vs_total_pvalues=gr_pvals)
        ax2.set_title("g(r) 值分布（小提琴）")
        plt.suptitle(f"{sample_name} 子样本对比：RDF g(r)（归一化后）")
        plt.tight_layout()
        plt.savefig(out_dir / f"{sample_name}_compare_RDF_after_norm.png", dpi=150)
        plt.close()
        print(f"  对比图: {sample_name}_compare_RDF_after_norm.png")

    print("子样本对比分析完成。")


if __name__ == "__main__":
    main()

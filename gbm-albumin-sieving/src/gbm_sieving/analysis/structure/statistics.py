"""
孔喉网络结构统计学分析
- 体积：凸包体积（非规则长方体）
- 密度：① 数量/总体积  ② 占体积/总体积（孔球体+喉圆柱）
- 孔上喉数：平均值 + 分布图
- 孔/喉半径：统计量 + 分布图
- 喉取向：全局 Q-tensor；孔上喉的局部取向倾向（局部 Q，秩序参数 S 的分布）
- 孔 RDF：径向分布函数
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import ConvexHull
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


def _get_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return df[c].values
    return default


def load_pores_and_throats(pores_file: Path, throats_file: Path):
    """加载孔、喉表，兼容 Pore ID, X Coord/Y/Z, EqRadius 或 Radius 等列名。"""
    df_p = pd.read_excel(pores_file)
    df_t = pd.read_excel(throats_file)

    # 孔
    pore_ids = _get_col(df_p, ['Pore ID', 'PoreID'])
    x = _get_col(df_p, ['X Coord', 'X', 'x'])
    y = _get_col(df_p, ['Y Coord', 'Y', 'y'])
    z = _get_col(df_p, ['Z Coord', 'Z', 'z'])
    if pore_ids is None or x is None or y is None or z is None:
        raise ValueError("孔表需包含 Pore ID 及 X,Y,Z 坐标列")
    pore_coords = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)])
    pore_radii = _get_col(df_p, ['EqRadius', 'Radius', 'Radius (nm)', 'Pore Radius'])
    if pore_radii is None:
        raise ValueError("孔表需包含半径列（EqRadius 或 Radius）")
    pore_radii = np.asarray(pore_radii, dtype=float)
    pore_id_to_idx = {int(pid): i for i, pid in enumerate(pore_ids)}

    # 喉
    p1 = _get_col(df_t, ['Pore ID #1', 'Pore ID 1', 'Pore1'])
    p2 = _get_col(df_t, ['Pore ID #2', 'Pore ID 2', 'Pore2'])
    if p1 is None or p2 is None:
        raise ValueError("喉表需包含两端孔 ID 列")
    throat_pore1 = np.asarray(p1, dtype=int)
    throat_pore2 = np.asarray(p2, dtype=int)
    throat_radii = _get_col(df_t, ['EqRadius', 'Radius', 'Radius (nm)', 'Throat Radius'])
    if throat_radii is None:
        raise ValueError("喉表需包含半径列")
    throat_radii = np.asarray(throat_radii, dtype=float)

    return pore_coords, pore_radii, pore_ids, pore_id_to_idx, throat_pore1, throat_pore2, throat_radii


def volume_convex_hull(pore_coords: np.ndarray) -> float:
    """凸包体积 (nm^3)，仅孔心。"""
    if len(pore_coords) < 4:
        # 退化为包围盒
        v = np.max(pore_coords, axis=0) - np.min(pore_coords, axis=0)
        return float(np.prod(np.maximum(v, 1e-6)))
    hull = ConvexHull(pore_coords)
    return float(hull.volume)


def _get_six_main_face_normals(face_normals: np.ndarray) -> np.ndarray:
    """
    将凸包面法向聚类为 6 个主方向（类似平行六面体的 6 个面）。
    用 PCA 得到 3 个主轴，6 个主方向 = ± 每个轴。
    返回 (6, 3) 单位向量。
    """
    n = face_normals.shape[0]
    if n < 3:
        # 退化为 ±X, ±Y, ±Z
        return np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ], dtype=float)
    mean_n = np.mean(face_normals, axis=0)
    C = (face_normals - mean_n).T @ (face_normals - mean_n) / n
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    axes = evecs[:, idx[:3]]
    six_dirs = np.vstack([axes, -axes])
    norms = np.linalg.norm(six_dirs, axis=1, keepdims=True)
    six_dirs = six_dirs / np.maximum(norms, 1e-12)
    return six_dirs


def volume_expanded_convex_hull(
    pore_coords: np.ndarray,
    pore_radii: np.ndarray,
) -> float:
    """
    扩展凸包体积：孔心凸包 + 最外层孔半径。
    面法向法：将凸包面聚类为 6 个主面，每个顶点按其相邻主面的法向平均方向向外扩展自身半径。
    返回 (nm^3)。
    """
    pore_coords = np.asarray(pore_coords, dtype=float)
    pore_radii = np.asarray(pore_radii, dtype=float)
    if len(pore_coords) < 4:
        v = np.max(pore_coords, axis=0) - np.min(pore_coords, axis=0)
        return float(np.prod(np.maximum(v, 1e-6)))
    if len(pore_radii) != len(pore_coords):
        pore_radii = np.full(len(pore_coords), np.mean(pore_radii) if len(pore_radii) else 0.0)

    hull = ConvexHull(pore_coords)
    n_faces = len(hull.simplices)
    face_normals = hull.equations[:, :3]
    fn_norm = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(fn_norm, 1e-12)

    six_main = _get_six_main_face_normals(face_normals)
    face_to_main = np.argmax(face_normals @ six_main.T, axis=1)

    expanded = []
    for v_idx in hull.vertices:
        v = pore_coords[v_idx]
        r = float(np.maximum(pore_radii[v_idx], 0.0))
        adj_faces = np.where(np.any(hull.simplices == v_idx, axis=1))[0]
        main_ids = face_to_main[adj_faces]
        n_avg = np.mean(six_main[main_ids], axis=0)
        n_norm = np.linalg.norm(n_avg)
        if n_norm < 1e-12:
            centroid = np.mean(pore_coords, axis=0)
            n_avg = v - centroid
            n_norm = np.linalg.norm(n_avg)
        if n_norm >= 1e-12:
            n_avg = n_avg / n_norm
        v_exp = v + r * n_avg
        expanded.append(v_exp)

    expanded = np.array(expanded)
    if len(expanded) < 4:
        v = np.max(expanded, axis=0) - np.min(expanded, axis=0)
        return float(np.prod(np.maximum(v, 1e-6)))
    hull_exp = ConvexHull(expanded)
    return float(hull_exp.volume)


def pore_occupied_volume(pore_radii: np.ndarray) -> float:
    """孔占据体积（球体之和）nm^3。"""
    return float(np.sum((4.0 / 3.0) * np.pi * np.maximum(pore_radii, 0) ** 3))


def throat_occupied_volume(pore_coords: np.ndarray, pore_radii: np.ndarray, pore_id_to_idx: dict,
                           throat_pore1: np.ndarray, throat_pore2: np.ndarray, throat_radii: np.ndarray) -> float:
    """喉占据体积（圆柱：π R^2 L，L = 孔心距 d）。nm^3。"""
    vol = 0.0
    for i in range(len(throat_pore1)):
        pid1, pid2 = int(throat_pore1[i]), int(throat_pore2[i])
        if pid1 not in pore_id_to_idx or pid2 not in pore_id_to_idx:
            continue
        c1 = pore_coords[pore_id_to_idx[pid1]]
        c2 = pore_coords[pore_id_to_idx[pid2]]
        # 使用孔心距 d 作为喉圆柱长度，与 Phase3/运行流程中导出的 Length 保持一致
        L = max(np.linalg.norm(c2 - c1), 1e-6)
        R = max(throat_radii[i], 0)
        vol += np.pi * R * R * L
    return vol


def degree_per_pore(pore_ids: np.ndarray, throat_pore1: np.ndarray, throat_pore2: np.ndarray) -> np.ndarray:
    """每个孔上的喉数（度数）。"""
    pore_set = set(np.asarray(pore_ids, dtype=int))
    deg = np.zeros(len(pore_ids), dtype=int)
    pid_to_i = {int(pid): i for i, pid in enumerate(pore_ids)}
    for a, b in zip(throat_pore1, throat_pore2):
        a, b = int(a), int(b)
        if a in pid_to_i:
            deg[pid_to_i[a]] += 1
        if b in pid_to_i and b != a:
            deg[pid_to_i[b]] += 1
    return deg


def global_q_tensor(pore_coords: np.ndarray, pore_id_to_idx: dict,
                    throat_pore1: np.ndarray, throat_pore2: np.ndarray) -> tuple:
    """
    基于所有喉方向单位向量 n 计算全局取向：
    - 先计算二阶取向张量 S_ij = <n_i n_j>
    - 再构造迹为零的 Q 张量 Q_ij = 3/2 (S_ij - δ_ij/3)

    返回:
    - Q: 3x3 traceless Q 张量
    - evals: Q 的特征值（升序排列）
    - evecs: Q 的特征向量（列向量，对应 evals）
    """
    n_list = []
    for i in range(len(throat_pore1)):
        pid1, pid2 = int(throat_pore1[i]), int(throat_pore2[i])
        if pid1 not in pore_id_to_idx or pid2 not in pore_id_to_idx:
            continue
        d = pore_coords[pore_id_to_idx[pid2]] - pore_coords[pore_id_to_idx[pid1]]
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            continue
        n_list.append(d / norm)

    I = np.eye(3, dtype=float)
    if not n_list:
        # 各向同性极限：S = I/3, Q = 0
        S = I / 3.0
    else:
        n_arr = np.array(n_list)
        # 二阶取向张量 S_ij = <n_i n_j>
        S = np.einsum('ki,kj->ij', n_arr, n_arr) / len(n_arr)
    # 迹为零的 Q 张量
    Q = 1.5 * (S - I / 3.0)
    evals, evecs = np.linalg.eigh(Q)
    return Q, evals, evecs


def per_pore_throat_order(pore_ids: np.ndarray, pore_coords: np.ndarray, pore_id_to_idx: dict,
                          throat_pore1: np.ndarray, throat_pore2: np.ndarray) -> np.ndarray:
    """
    每个孔上喉方向的局部取向：局部 Q = mean(n_i n_j)，秩序参数 S = (3*lambda_max - 1)/2（3D 向列）。
    返回每个孔一个 S（无喉的孔为 np.nan）。
    """
    pore_ids_int = np.asarray(pore_ids, dtype=int)
    S_per_pore = np.full(len(pore_ids), np.nan, dtype=float)
    # 按孔收集喉方向（从孔指向邻居）
    from collections import defaultdict
    pore_neighbors = defaultdict(list)
    for i in range(len(throat_pore1)):
        a, b = int(throat_pore1[i]), int(throat_pore2[i])
        if a not in pore_id_to_idx or b not in pore_id_to_idx:
            continue
        ca = pore_coords[pore_id_to_idx[a]]
        cb = pore_coords[pore_id_to_idx[b]]
        d = cb - ca
        n = d / (np.linalg.norm(d) + 1e-15)
        pore_neighbors[a].append(n)
        pore_neighbors[b].append(-n)
    for idx, pid in enumerate(pore_ids_int):
        vecs = pore_neighbors.get(pid, [])
        if len(vecs) < 2:
            continue
        n_arr = np.array(vecs)
        Q = np.einsum('ki,kj->ij', n_arr, n_arr) / len(n_arr)
        evals = np.linalg.eigvalsh(Q)
        lam_max = evals[-1]
        S_per_pore[idx] = (3.0 * lam_max - 1.0) / 2.0  # 向列秩序参数
    return S_per_pore


def compute_rdf(pore_coords: np.ndarray, r_max: float = None, n_bins: int = 80, sample_size: int = 50000) -> tuple:
    """
    孔心径向分布函数 g(r)。若孔数过多则对孔对采样以加速。
    返回 r_center, g_r, pair_count 数组。
    """
    n = len(pore_coords)
    if r_max is None:
        L = np.max(pore_coords, axis=0) - np.min(pore_coords, axis=0)
        r_max = min(np.max(L) * 0.5, np.linalg.norm(L) * 0.3)
    r_max = max(r_max, 1.0)
    bins = np.linspace(0.0, r_max, n_bins + 1)
    dr = bins[1] - bins[0]
    r_center = (bins[:-1] + bins[1:]) / 2.0
    pair_count = np.zeros(n_bins)

    if n * (n - 1) // 2 <= sample_size:
        from scipy.spatial.distance import pdist
        d = pdist(pore_coords)
        d = d[d > 1e-10]
        pair_count, _ = np.histogram(d, bins=bins)
    else:
        # 随机采样孔对
        for _ in range(sample_size):
            i, j = np.random.randint(0, n, 2)
            if i == j:
                continue
            r = np.linalg.norm(pore_coords[i] - pore_coords[j])
            if r < 1e-10:
                continue
            k = int(r / dr)
            if 0 <= k < n_bins:
                pair_count[k] += 1
        # 按体积壳层归一化到全对数
        scale = (n * (n - 1) / 2.0) / sample_size if sample_size > 0 else 1.0
        pair_count = pair_count * scale

    vol = volume_convex_hull(pore_coords)
    rho = n / vol if vol > 0 else 0.0
    shell_vol = 4.0 * np.pi * r_center ** 2 * dr
    shell_vol[r_center <= 0] = 1.0
    g_r = pair_count / (shell_vol * rho * n) if (rho * n > 0) else np.zeros_like(r_center)
    return r_center, g_r, pair_count


def main():
    parser = argparse.ArgumentParser(description="孔喉网络结构统计学分析（凸包体积、两种密度、度分布、半径分布、Q-tensor、孔上喉取向、RDF）")
    parser.add_argument("--sample-name", type=str, required=True, help="样本名，用于定位 <sample-name>_pores.xlsx / _throats.xlsx 及输出子目录")
    parser.add_argument("--input-dir", type=str, default=str(WORKSPACE_ROOT / "throats_and_pores_xlsx"), help="孔喉 xlsx 所在目录，其下应有 <sample-name>_pores.xlsx 和 <sample-name>_throats.xlsx")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT / "analysis_structure_properties_overall"), help="输出目录；结果写在 <output-dir>/<sample-name>/ 下")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    sample_name = args.sample_name
    pores_file = input_dir / f"{sample_name}_pores.xlsx"
    throats_file = input_dir / f"{sample_name}_throats.xlsx"
    if not pores_file.exists():
        raise FileNotFoundError(f"孔表不存在: {pores_file}")
    if not throats_file.exists():
        raise FileNotFoundError(f"喉表不存在: {throats_file}")

    out_dir = Path(args.output_dir) / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("加载孔、喉数据...")
    pore_coords, pore_radii, pore_ids, pore_id_to_idx, throat_pore1, throat_pore2, throat_radii = load_pores_and_throats(
        pores_file, throats_file
    )
    n_pore = len(pore_ids)
    n_throat = len(throat_pore1)
    print(f"  孔数: {n_pore}, 喉数: {n_throat}")

    # ---------- 体积与两种密度 ----------
    vol = volume_expanded_convex_hull(pore_coords, pore_radii)
    print(f"  扩展凸包体积: {vol:.4e} nm^3")
    V_pore = pore_occupied_volume(pore_radii)
    V_throat = throat_occupied_volume(pore_coords, pore_radii, pore_id_to_idx,
                                       throat_pore1, throat_pore2, throat_radii)
    rho_pore_count = n_pore / vol if vol > 0 else 0.0
    rho_throat_count = n_throat / vol if vol > 0 else 0.0
    frac_pore = V_pore / vol if vol > 0 else 0.0
    frac_throat = V_throat / vol if vol > 0 else 0.0
    summary_rows = [
        {"指标": "扩展凸包体积 (nm^3)", "值": vol},
        {"指标": "孔数量密度 (1/nm^3)", "值": rho_pore_count},
        {"指标": "喉数量密度 (1/nm^3)", "值": rho_throat_count},
        {"指标": "孔占体积比 (占体积/总体积)", "值": frac_pore},
        {"指标": "喉占体积比 (占体积/总体积)", "值": frac_throat},
    ]

    # ---------- 孔上喉数：平均值 + 分布图 ----------
    deg = degree_per_pore(pore_ids, throat_pore1, throat_pore2)
    deg_valid = deg[deg > 0]
    mean_deg = float(np.mean(deg)) if len(deg) else 0.0
    summary_rows.append({"指标": "孔上平均喉数", "值": mean_deg})
    # 分布图
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(deg, bins=min(50, int(deg.max()) - int(deg.min()) + 1 if deg.max() > deg.min() else 2),
            range=(0, max(deg.max(), 1)), color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel("孔上喉数")
    ax.set_ylabel("孔个数")
    ax.set_title(f"孔上喉数分布（平均={mean_deg:.3f}）")
    ax.axvline(mean_deg, color='red', linestyle='--', label=f'平均={mean_deg:.3f}')
    ax.legend()
    plt.tight_layout()
    deg_fig = out_dir / f"{sample_name}_degree_distribution.png"
    plt.savefig(deg_fig, dpi=150)
    plt.close()
    print(f"  孔上平均喉数: {mean_deg:.3f}，分布图: {deg_fig}")

    # ---------- 孔半径：统计 + 分布图 ----------
    r_p = pore_radii
    summary_rows.extend([
        {"指标": "孔半径均值 (nm)", "值": float(np.mean(r_p))},
        {"指标": "孔半径标准差 (nm)", "值": float(np.std(r_p))},
        {"指标": "孔半径中位数 (nm)", "值": float(np.median(r_p))},
    ])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r_p, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel("孔半径 (nm)")
    ax.set_ylabel("孔个数")
    ax.set_title("孔半径分布")
    plt.tight_layout()
    pore_rad_fig = out_dir / f"{sample_name}_pore_radius_distribution.png"
    plt.savefig(pore_rad_fig, dpi=150)
    plt.close()
    print(f"  孔半径分布图: {pore_rad_fig}")

    # ---------- 喉半径：统计 + 分布图 ----------
    r_t = throat_radii
    summary_rows.extend([
        {"指标": "喉半径均值 (nm)", "值": float(np.mean(r_t))},
        {"指标": "喉半径标准差 (nm)", "值": float(np.std(r_t))},
        {"指标": "喉半径中位数 (nm)", "值": float(np.median(r_t))},
    ])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r_t, bins=50, color='seagreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel("喉半径 (nm)")
    ax.set_ylabel("喉个数")
    ax.set_title("喉半径分布")
    plt.tight_layout()
    throat_rad_fig = out_dir / f"{sample_name}_throat_radius_distribution.png"
    plt.savefig(throat_rad_fig, dpi=150)
    plt.close()
    print(f"  喉半径分布图: {throat_rad_fig}")

    # ---------- 喉全局 Q-tensor ----------
    Q_glob, evals, evecs = global_q_tensor(pore_coords, pore_id_to_idx, throat_pore1, throat_pore2)
    summary_rows.append({"指标": "喉取向张量 Q 最大特征值", "值": float(evals[-1])})
    summary_rows.append({"指标": "喉取向张量 Q 最小特征值", "值": float(evals[0])})
    Q_df = pd.DataFrame(Q_glob, index=["x", "y", "z"], columns=["x", "y", "z"])
    Q_df.to_excel(out_dir / f"{sample_name}_throat_Q_tensor.xlsx")
    print(f"  喉全局 Q-tensor 已保存: {out_dir / f'{sample_name}_throat_Q_tensor.xlsx'}")

    # ---------- 孔上喉的取向倾向（局部秩序参数 S） ----------
    S_pore = per_pore_throat_order(pore_ids, pore_coords, pore_id_to_idx, throat_pore1, throat_pore2)
    S_valid = S_pore[~np.isnan(S_pore)]
    if len(S_valid) > 0:
        mean_S = float(np.nanmean(S_pore))
        summary_rows.append({"指标": "孔上喉取向秩序参数 S 均值", "值": mean_S})
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(S_valid, bins=40, color='purple', edgecolor='black', alpha=0.7)
        ax.set_xlabel("秩序参数 S (孔上喉方向一致性)")
        ax.set_ylabel("孔个数")
        ax.set_title(f"孔上喉取向倾向分布（S 均值={mean_S:.3f}）")
        ax.axvline(mean_S, color='red', linestyle='--', label=f'平均={mean_S:.3f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{sample_name}_pore_throat_orientation_S.png", dpi=150)
        plt.close()
        print(f"  孔上喉取向 S 分布图: {out_dir / f'{sample_name}_pore_throat_orientation_S.png'}")

    # ---------- RDF ----------
    r_center, g_r, pair_count = compute_rdf(pore_coords)
    rdf_df = pd.DataFrame({"r_center_nm": r_center, "g_r": g_r, "pair_count": pair_count})
    rdf_df.to_excel(out_dir / f"{sample_name}_pore_RDF.xlsx", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_center, g_r, 'b-', linewidth=1.5)
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("g(r)")
    ax.set_title("孔心径向分布函数 RDF")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_dir / f"{sample_name}_pore_RDF.png", dpi=150)
    plt.close()
    print(f"  孔 RDF 已保存: {out_dir / f'{sample_name}_pore_RDF.xlsx'} 与 PNG")

    # ---------- 汇总表 ----------
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(out_dir / f"{sample_name}_structure_summary.xlsx", index=False)
    print(f"  汇总表: {out_dir / f'{sample_name}_structure_summary.xlsx'}")

    # 半径直方图数据也存表（可选）
    hist_p, _ = np.histogram(r_p, bins=50)
    bin_edges = np.linspace(r_p.min(), r_p.max(), 51)
    pd.DataFrame({"bin_left": bin_edges[:-1], "bin_right": bin_edges[1:], "count": hist_p}).to_excel(
        out_dir / f"{sample_name}_pore_radius_histogram.xlsx", index=False
    )
    hist_t, _ = np.histogram(r_t, bins=50)
    bin_edges_t = np.linspace(r_t.min(), r_t.max(), 51)
    pd.DataFrame({"bin_left": bin_edges_t[:-1], "bin_right": bin_edges_t[1:], "count": hist_t}).to_excel(
        out_dir / f"{sample_name}_throat_radius_histogram.xlsx", index=False
    )
    print("结构统计学分析完成。")


if __name__ == "__main__":
    main()

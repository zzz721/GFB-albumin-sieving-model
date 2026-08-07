"""
子样本版本：分析孔隙和喉道数据（使用 Alpha Shape 方法识别边界孔）。
与整体版本区分，由 pipeline/run 直接调用本脚本（sub_ 前缀）。
1. 判断孔的体积和半径是否协调（体积 < (4/3)*pi*r^3 标记为外部孔）
2. 使用 Alpha Shape 识别X轴方向的外部孔（表面孔）
3. 过滤无法通过溶质的喉（半径 < 3.55nm），找到可以联通的喉
4. 创建可交互的三维可视化
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from pathlib import Path
import argparse
import json
from collections import defaultdict

try:
    import alphashape
except ImportError:
    print("警告：需要安装 alphashape 库: pip install alphashape")
    print("  正在尝试使用凸包作为 fallback...")
    alphashape = None

from scipy.spatial import cKDTree, ConvexHull

# 溶质的水合半径（单位：nm）
SOLUTE_HYDRATED_RADIUS = 3.55
# 与 new J 筛分脚本通路筛选一致：可通行喉须 R_T > r_s
SOLUTE_PASSAGE_GEOMETRY_MIN_NM = SOLUTE_HYDRATED_RADIUS * 1.0

# 射线投射：横截面内半径 R（nm）、相对半径的同心环、每环角向条数。
# RING_FRACTIONS=() 表示仅中心单条射线（无周向采样）；非空则 1 + len(环)×角向 条/孔/面。
RAY_CAST_CYLINDER_RADIUS_NM = 0.25
RAY_CAST_CYLINDER_RING_FRACTIONS = ()
RAY_CAST_CYLINDER_N_ANGULAR = 8

# ---------------- 参数解析：样本名 / 输入文件夹 / 输出文件夹 ----------------
parser = argparse.ArgumentParser(
    description="Analyze pore/throat data and identify left/right surfaces (Alpha Shape + plane fitting)."
)
parser.add_argument(
    "--sample-name",
    type=str,
    default="118",
    help="样本名字，将用于输入/输出文件名前缀，例如 118 -> 118_pores.xlsx, 118_throat_classification.xlsx 等。",
)
parser.add_argument(
    "--sample-dir",
    type=str,
    default="118example",
    help="样本所在文件夹，里面应包含 <sample-name>_pores.xlsx 和 <sample-name>_throats.xlsx。",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="results_core_subsamples",
    help="输出结果文件夹（子样本基础分析结果）。",
)
parser.add_argument(
    "--thickness-file",
    type=str,
    default=None,
    help="包含样本厚度信息的 Excel 文件路径，用于自动校准渗透方向。",
)
parser.add_argument(
    "--radius-delta",
    type=float,
    default=0.0,
    help="孔和喉半径的调整值（单位：nm），将应用到所有孔和喉的半径上。默认值为0.0（不调整）。",
)
parser.add_argument(
    "--normal-vector",
    nargs=2,
    type=float,
    default=None,
    metavar=("NX", "NY"),
    help="法向量方向（用于子样本分析），传入两个数 NX NY，例如: --normal-vector -0.33 0.94。如果提供，将使用此方向作为渗透方向。",
)

_args, _unknown = parser.parse_known_args()
SAMPLE_NAME = _args.sample_name
SAMPLE_DIR = Path(_args.sample_dir)
output_folder_base = Path(_args.output_dir)
RADIUS_DELTA = _args.radius_delta
# 现在 normal-vector 是长度为 2 的 float 列表或 None
NORMAL_VECTOR_VALS = _args.normal_vector

# 文件路径（根据样本名和样本文件夹自动拼接）
pores_file = SAMPLE_DIR / f"{SAMPLE_NAME}_pores.xlsx"
throats_file = SAMPLE_DIR / f"{SAMPLE_NAME}_throats.xlsx"

# 确保输出文件夹存在（创建以样本名命名的子文件夹）
output_folder = output_folder_base / SAMPLE_NAME
output_folder.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("孔隙和喉道分析（Alpha Shape 方法）")
print("=" * 70)
if RADIUS_DELTA != 0.0:
    print(f"半径调整值: {RADIUS_DELTA:.3f} nm (所有孔和喉的半径将调整)")
else:
    print(f"半径调整值: 0.0 nm (不调整)")

# ====================================================================
# 步骤1: 加载数据
# ====================================================================
print("\n步骤1: 加载数据...")

# 加载孔隙数据
df_pores = pd.read_excel(pores_file)
print(f"  孔隙数据: {len(df_pores)} 个孔隙")
print(f"  列: {df_pores.columns.tolist()}")

# 加载喉道数据
df_throats = pd.read_excel(throats_file)
print(f"  喉道数据: {len(df_throats)} 个喉道")
print(f"  列: {df_throats.columns.tolist()}")

# 提取关键数据
pore_ids = df_pores['Pore ID'].values
pore_volumes = df_pores['Volume'].values
pore_radii = df_pores['EqRadius'].values
# 应用半径调整值
if RADIUS_DELTA != 0.0:
    pore_radii = pore_radii + RADIUS_DELTA
    # 确保半径不为负
    pore_radii = np.maximum(pore_radii, 1e-6)  # 最小半径1e-6 nm
pore_coords = df_pores[['X Coord', 'Y Coord', 'Z Coord']].values

throat_ids = df_throats['Throat ID'].values
throat_radii = df_throats['EqRadius'].values
# 应用半径调整值
if RADIUS_DELTA != 0.0:
    throat_radii = throat_radii + RADIUS_DELTA
    # 确保半径不为负
    throat_radii = np.maximum(throat_radii, 1e-6)  # 最小半径1e-6 nm
throat_pore1 = df_throats['Pore ID #1'].values
throat_pore2 = df_throats['Pore ID #2'].values

print(f"  孔隙坐标范围:")
print(f"    X: [{pore_coords[:, 0].min():.2f}, {pore_coords[:, 0].max():.2f}]")
print(f"    Y: [{pore_coords[:, 1].min():.2f}, {pore_coords[:, 1].max():.2f}]")
print(f"    Z: [{pore_coords[:, 2].min():.2f}, {pore_coords[:, 2].max():.2f}]")
print(f"  孔隙半径范围: [{pore_radii.min():.2f}, {pore_radii.max():.2f}] nm")
print(f"  喉道半径范围: [{throat_radii.min():.2f}, {throat_radii.max():.2f}] nm")
if RADIUS_DELTA != 0.0:
    print(f"  半径调整值: {RADIUS_DELTA:.3f} nm (已应用到所有孔和喉)")
else:
    print(f"  半径调整值: 0.0 nm (未调整)")

# 创建孔ID到索引的映射
pore_id_to_idx = {pid: idx for idx, pid in enumerate(pore_ids)}

# ====================================================================
# 辅助函数：几何计算与表面识别 (Alpha Shape + Ray Casting)
# ====================================================================

def ray_triangle_intersection(ray_origin, ray_dir, triangle):
    """
    检测射线与三角形的相交 (Möller–Trumbore 算法)
    
    参数:
        ray_origin: 射线起点 (3,)
        ray_dir: 射线方向向量 (3,)，需要归一化
        triangle: 三角形的三个顶点 (3, 3)
    
    返回:
        t: 相交参数（如果相交），None 表示不相交
        point: 相交点坐标（如果相交）
    """
    v0, v1, v2 = triangle[0], triangle[1], triangle[2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)
    
    if abs(a) < 1e-8:
        return None, None  # 射线与三角形平行
    
    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)
    
    if u < 0.0 or u > 1.0:
        return None, None
    
    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)
    
    if v < 0.0 or u + v > 1.0:
        return None, None
    
    t = f * np.dot(edge2, q)
    
    if t > 1e-8:  # 射线起点在三角形后面
        point = ray_origin + t * ray_dir
        return t, point
    
    return None, None


def estimate_alpha_from_neighbors(pore_coords: np.ndarray) -> float:
    """
    用最近邻距离估算一个合适的 alpha。
    """
    print("  [Alpha] 使用最近邻距离估算平均键长以推断 alpha ...")
    tree = cKDTree(pore_coords)
    distances, _ = tree.query(pore_coords, k=2)  # 自身+最近邻
    nn_dist = distances[:, 1]

    # 去掉异常值
    q1, q2, q3 = np.percentile(nn_dist, [25, 50, 75])
    iqr = q3 - q1
    lower = q2 - 1.5 * iqr
    upper = q2 + 1.5 * iqr
    valid = nn_dist[(nn_dist >= lower) & (nn_dist <= upper)]
    if len(valid) == 0:
        avg_len = float(np.median(nn_dist))
        print(f"    使用中位数近似平均键长: {avg_len:.2f}")
    else:
        avg_len = float(np.mean(valid))
        print(f"    估算的平均键长: {avg_len:.2f}")

    # R ~ 3 * avg_len (用户调整: 增大滚动球半径，使其更宽松), alpha = 1 / R
    alpha = 1.0 / (2.0 * avg_len)
    print(f"    推断 alpha = {alpha:.6f} (R ≈ {1.0/alpha:.2f})")
    return alpha


def identify_surface_by_alphashape(pore_coords: np.ndarray,
                                   alpha: float | None = None):
    """
    使用 alphashape 或凸包识别整体表面孔：
    返回 boundary_mask (N,), 实际使用的 alpha (可能为 None), 和 alpha_mesh (可能为 None)
    """
    n = pore_coords.shape[0]

    def _fallback_convex_hull(reason: str):
        print(f"    {reason}，退回凸包")
        hull = ConvexHull(pore_coords)
        mask = np.zeros(n, dtype=bool)
        mask[hull.vertices] = True
        return mask, None, None

    def _extract_alpha_vertices(mesh_obj):
        """
        从不同 alphashape 返回类型中提取几何顶点坐标。
        支持 trimesh 风格网格、shapely Polygon/LineString/Multi*。
        """
        if mesh_obj is None:
            return None
        if hasattr(mesh_obj, "vertices"):
            verts = np.asarray(mesh_obj.vertices, dtype=float)
            if verts.ndim == 2 and verts.shape[0] > 0:
                return verts
        # shapely Polygon
        if hasattr(mesh_obj, "exterior") and getattr(mesh_obj, "exterior") is not None:
            try:
                coords = np.asarray(mesh_obj.exterior.coords, dtype=float)
                if coords.ndim == 2 and coords.shape[0] > 0:
                    return coords
            except Exception:
                pass
        # shapely LineString / LinearRing
        if hasattr(mesh_obj, "coords"):
            try:
                coords = np.asarray(mesh_obj.coords, dtype=float)
                if coords.ndim == 2 and coords.shape[0] > 0:
                    return coords
            except Exception:
                pass
        # shapely MultiLineString / MultiPolygon / GeometryCollection
        geoms = getattr(mesh_obj, "geoms", None)
        if geoms is not None:
            all_coords = []
            for g in geoms:
                g_coords = _extract_alpha_vertices(g)
                if g_coords is not None and g_coords.shape[0] > 0:
                    all_coords.append(g_coords)
            if all_coords:
                return np.vstack(all_coords)
        return None

    if alphashape is None:
        print("  [Alpha] alphashape 不可用，使用 ConvexHull 作为替代")
        hull = ConvexHull(pore_coords)
        boundary_mask = np.zeros(n, dtype=bool)
        boundary_mask[hull.vertices] = True
        return boundary_mask, None, None

    if alpha is None:
        alpha = estimate_alpha_from_neighbors(pore_coords)
    else:
        print(f"  [Alpha] 使用用户指定 alpha = {alpha:.6f} (R = {1.0/alpha:.2f})")

    print("  [Alpha] 生成 Alpha Shape ...")
    try:
        alpha_mesh = alphashape.alphashape(pore_coords, alpha)
    except Exception as e:
        return _fallback_convex_hull(f"Alpha Shape 失败: {e}")

    print("    Alpha Shape 成功，开始映射表面顶点到原始孔索引 ...")
    alpha_vertices = _extract_alpha_vertices(alpha_mesh)
    if alpha_vertices is None or alpha_vertices.shape[0] == 0:
        return _fallback_convex_hull("Alpha Shape 返回类型不支持顶点提取")
    if alpha_vertices.shape[1] != pore_coords.shape[1]:
        # 2D 几何（如 LineString）补齐到 3D，避免后续距离计算维度不一致
        if alpha_vertices.shape[1] < pore_coords.shape[1]:
            pad_width = pore_coords.shape[1] - alpha_vertices.shape[1]
            alpha_vertices = np.pad(alpha_vertices, ((0, 0), (0, pad_width)), mode="constant")
        else:
            alpha_vertices = alpha_vertices[:, : pore_coords.shape[1]]

    precision = 6
    coord_map = {}
    for i, coord in enumerate(pore_coords):
        key = tuple(np.round(coord, precision))
        if key not in coord_map:
            coord_map[key] = []
        coord_map[key].append(i)

    boundary_indices = set()
    for v in alpha_vertices:
        key = tuple(np.round(v, precision))
        if key in coord_map:
            boundary_indices.update(coord_map[key])
        else:
            # 保险起见再找一次最近邻
            dists = np.linalg.norm(pore_coords - v, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] < 1e-3:
                boundary_indices.add(idx)

    boundary_indices = sorted(boundary_indices)
    print(f"    找到表面孔数量: {len(boundary_indices)}")
    boundary_mask = np.zeros(n, dtype=bool)
    boundary_mask[boundary_indices] = True
    return boundary_mask, alpha, alpha_mesh


def calculate_average_thickness(pore_coords: np.ndarray, pore_radii: np.ndarray, axis: int, cell_size: float = 10.0) -> float:
    """
    计算指定轴向的平均厚度。
    方法：在垂直于该轴的平面上打网格，计算每个网格内该轴坐标范围 (Max - Min)，然后平均。
    *包含半径补偿*：使用 (coord + r) 和 (coord - r) 来计算物理边界。
    """
    # 确定平面轴
    plane_axes = [i for i in range(3) if i != axis]
    u_idx, v_idx = plane_axes
    
    # 获取坐标
    coords_axis = pore_coords[:, axis]
    coords_u = pore_coords[:, u_idx]
    coords_v = pore_coords[:, v_idx]
    
    # 确定平面范围
    u_min, u_max = coords_u.min(), coords_u.max()
    v_min, v_max = coords_v.min(), coords_v.max()
    
    # 构建网格字典：(gu, gv) -> {'min': val, 'max': val}
    grid_stats = {}
    
    n_points = len(pore_coords)
    for i in range(n_points):
        u, v = coords_u[i], coords_v[i]
        val = coords_axis[i]
        r = pore_radii[i]
        
        # 物理边界
        val_min = val - r
        val_max = val + r
        
        gu = int((u - u_min) / cell_size)
        gv = int((v - v_min) / cell_size)
        key = (gu, gv)
        
        if key not in grid_stats:
            grid_stats[key] = {'min': val_min, 'max': val_max}
        else:
            if val_min < grid_stats[key]['min']:
                grid_stats[key]['min'] = val_min
            if val_max > grid_stats[key]['max']:
                grid_stats[key]['max'] = val_max
    
    # 计算所有有效网格的厚度并平均
    thicknesses = []
    for stat in grid_stats.values():
        t = stat['max'] - stat['min']
        if t > 0: 
            thicknesses.append(t)
            
    if not thicknesses:
        return 0.0
        
    return float(np.mean(thicknesses))


def calculate_average_thickness_along_unit(
    pore_coords: np.ndarray,
    pore_radii: np.ndarray,
    axis_unit: np.ndarray,
    cell_size: float = 10.0,
) -> float:
    """
    沿任意单位向量 axis_unit 的网格平均厚度（与 calculate_average_thickness 同构；半径沿该方向做 ±r 补偿）。
    在垂直于 axis_unit 的平面上用 cell_size 划分网格。
    """
    n = np.asarray(axis_unit, dtype=float).reshape(3,)
    norm = np.linalg.norm(n)
    if norm <= 1e-12:
        return 0.0
    n = n / norm
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    e1 = ref - np.dot(ref, n) * n
    e1 = e1 / (np.linalg.norm(e1) + 1e-15)
    e2 = np.cross(n, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-15)

    coords_s = pore_coords @ n
    coords_u2 = pore_coords @ e1
    coords_v2 = pore_coords @ e2
    u_min, u_max = coords_u2.min(), coords_u2.max()
    v_min, v_max = coords_v2.min(), coords_v2.max()

    grid_stats = {}
    for i in range(len(pore_coords)):
        u2, v2 = coords_u2[i], coords_v2[i]
        s = coords_s[i]
        r = pore_radii[i]
        s_min = s - r
        s_max = s + r
        gu = int((u2 - u_min) / cell_size)
        gv = int((v2 - v_min) / cell_size)
        key = (gu, gv)
        if key not in grid_stats:
            grid_stats[key] = {"min": s_min, "max": s_max}
        else:
            if s_min < grid_stats[key]["min"]:
                grid_stats[key]["min"] = s_min
            if s_max > grid_stats[key]["max"]:
                grid_stats[key]["max"] = s_max

    thicknesses = []
    for stat in grid_stats.values():
        t = stat["max"] - stat["min"]
        if t > 0:
            thicknesses.append(t)
    if not thicknesses:
        return 0.0
    return float(np.mean(thicknesses))


def pca_least_squares_plane(points: np.ndarray):
    """
    过质心、使 sum_i d(p_i, 平面)^2 最小的平面；法向为协方差最小特征值对应特征向量。
    返回 (center, n, e1, e2)，e1、e2 为面内正交基；点数 < 3 时返回 None。
    """
    if points is None or len(points) < 3:
        return None
    center = np.mean(points, axis=0)
    X = points - center
    cov = (X.T @ X) / max(len(points) - 1, 1)
    _evals, evecs = np.linalg.eigh(cov)
    n = np.asarray(evecs[:, 0], dtype=float).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-15)
    e1 = np.asarray(evecs[:, 1], dtype=float).reshape(3)
    e2 = np.asarray(evecs[:, 2], dtype=float).reshape(3)
    return center, n, e1, e2


def merged_permeation_normal_surface(
    pts_left: np.ndarray,
    pts_right: np.ndarray,
) -> np.ndarray | None:
    """
    与 HTML 可视化中「修正渗透方向」一致（表面识别坐标系）：
    两侧各 ≥3 点时两面 PCA 法向融合；否则（≥2 点）回退为两侧质心连线方向。
    返回单位向量；无法定义时返回 None。
    """
    if pts_left is None or pts_right is None:
        return None
    if len(pts_left) < 2 or len(pts_right) < 2:
        return None
    if len(pts_left) >= 3 and len(pts_right) >= 3:
        pl = pca_least_squares_plane(pts_left)
        pr = pca_least_squares_plane(pts_right)
        if pl is not None and pr is not None:
            c_l, n_l, e1l, e2l = pl
            c_r, n_r, e1r, e2r = pr
            if float(np.dot(n_l, c_r - c_l)) < 0:
                n_l = -n_l
            if float(np.dot(n_r, c_l - c_r)) < 0:
                n_r = -n_r
            v_merge = n_l + (-n_r)
            nv = float(np.linalg.norm(v_merge))
            if nv > 1e-9:
                return (v_merge / nv).astype(float)
    c_l = np.mean(pts_left, axis=0)
    c_r = np.mean(pts_right, axis=0)
    d_thick = c_r - c_l
    dn = float(np.linalg.norm(d_thick))
    if dn > 1e-9:
        return (d_thick / dn).astype(float)
    return None


def write_surface_pores_real_radius_html(
    surface_coords: np.ndarray,
    face_labels: np.ndarray,
    pore_radii_surface: np.ndarray,
    sample_name: str,
    output_dir: Path,
    main_axis: int,
    E_basis: np.ndarray,
    n_corrected_lab: np.ndarray,
    theta_deg_vs_xy=None,
) -> Path:
    """
    与 surface_coords 同一坐标系（表面识别用坐标）：
    - 半透明端面：对每侧 u_min / u_max 点分别做 PCA，使该侧所有孔到平面的**距离平方和最小**（最小二乘拟合面）；矩形为点在该面内投影范围
    - 绿色箭头：拟合渗透方向（沿射线轴）
    - 橙色箭头：修正渗透方向（两面 PCA 垂线融合或质心连线，与厚度倾斜修正使用同一 merged 方向）
    """
    fig = go.Figure()
    unique_labels = np.unique(face_labels)
    radii_subset = pore_radii_surface

    for label in unique_labels:
        mask = face_labels == label
        if not np.any(mask):
            continue
        if label == "none":
            color = "lightgray"
            name = "未分配 (内部/边缘)"
            opacity = 0.1
        elif label == "u_min":
            color = "red"
            name = "左表面 (u_min)"
            opacity = 0.9
        elif label == "u_max":
            color = "blue"
            name = "右表面 (u_max)"
            opacity = 0.9
        else:
            color = "purple"
            name = f"边/角 ({label})"
            opacity = 0.9
        fig.add_trace(
            go.Scatter3d(
                x=surface_coords[mask, 0],
                y=surface_coords[mask, 1],
                z=surface_coords[mask, 2],
                mode="markers",
                marker=dict(
                    size=radii_subset[mask] * 1.0,
                    color=color,
                    opacity=opacity,
                    line=dict(width=0),
                    sizemode="diameter",
                    sizeref=0.5,
                ),
                name=name,
                text=[f"R={r:.2f}" for r in radii_subset[mask]],
                hoverinfo="text+name+x+y+z",
            )
        )

    def _plane_axes_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = np.asarray(n, dtype=float).reshape(3)
        n = n / (np.linalg.norm(n) + 1e-15)
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(n, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = ref - np.dot(ref, n) * n
        e1 = e1 / (np.linalg.norm(e1) + 1e-15)
        e2 = np.cross(n, e1)
        e2 = e2 / (np.linalg.norm(e2) + 1e-15)
        return e1, e2

    def _quad_on_plane(
        center: np.ndarray,
        n_hat: np.ndarray,
        points: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
        pad_ratio: float = 0.06,
    ):
        """过 center、法向 n_hat，用 points 在面内投影范围构造 4 顶点。"""
        if points is None or len(points) < 2:
            return None
        n_hat = np.asarray(n_hat, dtype=float).reshape(3)
        n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-15)
        center = np.asarray(center, dtype=float).reshape(3)
        s1s = []
        s2s = []
        for p in points:
            v = p - center - n_hat * float(np.dot(p - center, n_hat))
            s1s.append(float(np.dot(v, e1)))
            s2s.append(float(np.dot(v, e2)))
        s1_lo, s1_hi = np.min(s1s), np.max(s1s)
        s2_lo, s2_hi = np.min(s2s), np.max(s2s)
        pad1 = pad_ratio * (s1_hi - s1_lo + 1e-9)
        pad2 = pad_ratio * (s2_hi - s2_lo + 1e-9)
        s1_lo -= pad1
        s1_hi += pad1
        s2_lo -= pad2
        s2_hi += pad2
        return np.array(
            [
                center + s1_lo * e1 + s2_lo * e2,
                center + s1_hi * e1 + s2_lo * e2,
                center + s1_hi * e1 + s2_hi * e2,
                center + s1_lo * e1 + s2_hi * e2,
            ],
            dtype=float,
        )

    mask_min = face_labels == "u_min"
    mask_max = face_labels == "u_max"
    pts_left = surface_coords[mask_min] if np.any(mask_min) else None
    pts_right = surface_coords[mask_max] if np.any(mask_max) else None
    n_corr_surface_from_planes = merged_permeation_normal_surface(pts_left, pts_right)

    rgba_plane = "rgba(0, 180, 80, 0.35)"
    if (
        pts_left is not None
        and pts_right is not None
        and len(pts_left) >= 3
        and len(pts_right) >= 3
    ):
        # 每侧独立 PCA：最小化该侧点到本侧平面的距离平方和
        pl = pca_least_squares_plane(pts_left)
        pr = pca_least_squares_plane(pts_right)
        if pl is not None and pr is not None:
            c_l, n_l, e1l, e2l = pl
            c_r, n_r, e1r, e2r = pr
            # 法向指向对侧质心，便于与「厚度方向」直觉一致
            if float(np.dot(n_l, c_r - c_l)) < 0:
                n_l = -n_l
            if float(np.dot(n_r, c_l - c_r)) < 0:
                n_r = -n_r
            quad_l = _quad_on_plane(c_l, n_l, pts_left, e1l, e2l)
            quad_r = _quad_on_plane(c_r, n_r, pts_right, e1r, e2r)
            for pts, pname, col in [
                (quad_l, "PCA 最小二乘端面 (u_min)", rgba_plane),
                (quad_r, "PCA 最小二乘端面 (u_max)", "rgba(0, 140, 160, 0.35)"),
            ]:
                if pts is None:
                    continue
                fig.add_trace(
                    go.Mesh3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        i=[0, 0],
                        j=[1, 2],
                        k=[2, 3],
                        color=col,
                        opacity=0.34,
                        name=pname,
                        flatshading=True,
                        showlegend=True,
                    )
                )
        else:
            print("  [提示] PCA 拟合面失败，跳过端面")
    elif (
        pts_left is not None
        and pts_right is not None
        and len(pts_left) >= 2
        and len(pts_right) >= 2
    ):
        # 点太少无法 PCA：退化为两质心连线为法向的平行端面
        c_l = np.mean(pts_left, axis=0)
        c_r = np.mean(pts_right, axis=0)
        d_thick = c_r - c_l
        dn = float(np.linalg.norm(d_thick))
        if dn > 1e-9:
            n_thick = d_thick / dn
            e1p, e2p = _plane_axes_from_normal(n_thick)
            quad_l = _quad_on_plane(c_l, n_thick, pts_left, e1p, e2p)
            quad_r = _quad_on_plane(c_r, n_thick, pts_right, e1p, e2p)
            for pts, pname, col in [
                (quad_l, "质心连线端面(点<3 回退) u_min", rgba_plane),
                (quad_r, "质心连线端面(点<3 回退) u_max", "rgba(0, 140, 160, 0.35)"),
            ]:
                if pts is None:
                    continue
                fig.add_trace(
                    go.Mesh3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        i=[0, 0],
                        j=[1, 2],
                        k=[2, 3],
                        color=col,
                        opacity=0.34,
                        name=pname,
                        flatshading=True,
                        showlegend=True,
                    )
                )
        else:
            print("  [提示] 两侧质心重合，跳过端面")
    else:
        # 回退：沿射线轴分位切片（与旧版一致）
        plane_axes = [i for i in range(3) if i != main_axis]
        va, wa = plane_axes[0], plane_axes[1]
        u = surface_coords[:, main_axis]
        u_lo, u_hi = np.percentile(u, [5, 95])
        v_lo, v_hi = np.percentile(surface_coords[:, va], [2, 98])
        w_lo, w_hi = np.percentile(surface_coords[:, wa], [2, 98])
        pad = 0.05 * (max(v_hi - v_lo, w_hi - w_lo) + 1e-9)
        v_lo -= pad
        v_hi += pad
        w_lo -= pad
        w_hi += pad

        def _quad_pts_axis(u0: float) -> np.ndarray:
            pts = np.zeros((4, 3))
            pts[:, main_axis] = u0
            pts[0, va] = v_lo
            pts[0, wa] = w_lo
            pts[1, va] = v_hi
            pts[1, wa] = w_lo
            pts[2, va] = v_hi
            pts[2, wa] = w_hi
            pts[3, va] = v_lo
            pts[3, wa] = w_hi
            return pts

        for u0, pname in [(u_lo, "拟合端面(回退) u≈P5"), (u_hi, "拟合端面(回退) u≈P95")]:
            pts = _quad_pts_axis(float(u0))
            fig.add_trace(
                go.Mesh3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    i=[0, 0],
                    j=[1, 2],
                    k=[2, 3],
                    color=rgba_plane,
                    opacity=0.32,
                    name=pname,
                    flatshading=True,
                    showlegend=True,
                )
            )

    centroid = np.mean(surface_coords, axis=0)
    span = np.max(surface_coords, axis=0) - np.min(surface_coords, axis=0)
    L = 0.22 * float(np.linalg.norm(span) + 1e-9)

    n_fit = np.zeros(3, dtype=float)
    n_fit[main_axis] = 1.0
    n_fit = n_fit / (np.linalg.norm(n_fit) + 1e-15)
    tip_fit = centroid + n_fit * L
    fig.add_trace(
        go.Scatter3d(
            x=[centroid[0], tip_fit[0]],
            y=[centroid[1], tip_fit[1]],
            z=[centroid[2], tip_fit[2]],
            mode="lines+markers",
            line=dict(color="green", width=10),
            marker=dict(size=[2, 8], color="green"),
            name="拟合渗透方向（射线轴）",
        )
    )

    if n_corr_surface_from_planes is not None:
        n_corr_s = np.asarray(n_corr_surface_from_planes, dtype=float).reshape(3)
        n_corr_s = n_corr_s / (np.linalg.norm(n_corr_s) + 1e-15)
        orange_name = "修正渗透方向（两面垂线融合）"
    else:
        n_corr_lab = np.asarray(n_corrected_lab, dtype=float).reshape(3)
        n_corr_lab = n_corr_lab / (np.linalg.norm(n_corr_lab) + 1e-15)
        n_corr_s = n_corr_lab @ E_basis
        n_corr_s = n_corr_s / (np.linalg.norm(n_corr_s) + 1e-15)
        orange_name = "修正渗透方向（质心连线，可视化回退）"
    tip_c = centroid + n_corr_s * L
    fig.add_trace(
        go.Scatter3d(
            x=[centroid[0], tip_c[0]],
            y=[centroid[1], tip_c[1]],
            z=[centroid[2], tip_c[2]],
            mode="lines+markers",
            line=dict(color="darkorange", width=10),
            marker=dict(size=[2, 8], color="darkorange"),
            name=orange_name,
        )
    )

    sub = (
        f"{sample_name} 表面孔 — 半径可视化 | 绿/青面 = 各侧 PCA 最小二乘拟合面 | "
        f"绿=射线轴 | 橙=两面垂线融合（或回退质心连线）"
    )
    if theta_deg_vs_xy is not None and np.isfinite(theta_deg_vs_xy):
        sub += f" | θ(vs XY)={theta_deg_vs_xy:.2f}°"

    fig.update_layout(
        title=sub,
        scene=dict(
            xaxis_title="X (表面坐标)",
            yaxis_title="Y (表面坐标)",
            zaxis_title="Z (表面坐标)",
            aspectmode="data",
        ),
        width=1000,
        height=800,
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    radius_html = output_dir / f"{sample_name}_surface_pores_real_radius.html"
    fig.write_html(str(radius_html))
    print(f"  [射线投射] 真实半径可视化（含拟合平面与方向）已保存: {radius_html}")
    return radius_html


def identify_faces_by_ray_casting(pore_coords: np.ndarray,
                                  pore_radii: np.ndarray,
                                  boundary_mask: np.ndarray,
                                  alpha_mesh,
                                  sample_name: str,
                                  output_dir: Path,
                                  main_axis: int = 0):
    """
    通过射线投射（模拟圆柱）识别渗透方向的两个面（u_min/u_max），并处理边和角。
    """
    if alpha_mesh is None:
        print("  [射线投射] alpha_mesh 不可用，跳过射线投射方法")
        return None
    
    _n_ray = 1 + len(RAY_CAST_CYLINDER_RING_FRACTIONS) * RAY_CAST_CYLINDER_N_ANGULAR
    print(f"  [射线投射] 准备开始，渗透方向轴索引: {main_axis} ({['X','Y','Z'][main_axis]})")
    if len(RAY_CAST_CYLINDER_RING_FRACTIONS) == 0:
        print("    射线模式: 仅中心单条（无周向圆柱采样）；二次几何筛选见后续步骤")
    else:
        print(
            f"    圆柱射线: R={RAY_CAST_CYLINDER_RADIUS_NM} nm, 环(相对半径)={RAY_CAST_CYLINDER_RING_FRACTIONS}, "
            f"每环角向={RAY_CAST_CYLINDER_N_ANGULAR} → 每表面孔约 {_n_ray} 条射线"
        )
    
    # 定义轴的映射
    plane_axes = [i for i in range(3) if i != main_axis]
    u_axis_idx, v_axis_idx = plane_axes
    print(f"    平面投影轴: {['X','Y','Z'][u_axis_idx]} - {['X','Y','Z'][v_axis_idx]}")
    
    # 提取三角网格的顶点和面
    try:
        mesh_vertices = None
        mesh_faces = None
        
        # 方法1: trimesh 格式
        if hasattr(alpha_mesh, 'vertices') and hasattr(alpha_mesh, 'faces'):
            mesh_vertices = np.array(alpha_mesh.vertices)
            mesh_faces = np.array(alpha_mesh.faces)
        # 方法2: ConvexHull 格式
        elif hasattr(alpha_mesh, 'simplices') and hasattr(alpha_mesh, 'points'):
            mesh_vertices = np.array(alpha_mesh.points)
            mesh_faces = np.array(alpha_mesh.simplices)
        # 方法3: 混合格式
        elif hasattr(alpha_mesh, 'vertices') and hasattr(alpha_mesh, 'simplices'):
            mesh_vertices = np.array(alpha_mesh.vertices)
            mesh_faces = np.array(alpha_mesh.simplices)
        else:
            print(f"  [射线投射] 无法识别网格格式，可用属性: {dir(alpha_mesh)}")
            return None
        
        if mesh_vertices is None or mesh_faces is None:
            return None
            
    except Exception as e:
        print(f"  [射线投射] 提取网格信息失败: {e}，跳过")
        return None
    
    surface_indices = np.where(boundary_mask)[0]
    surface_coords = pore_coords[surface_indices]
    n_surface = len(surface_coords)
    
    print(f"    表面孔数量: {n_surface}")
    print(f"    网格三角形数: {len(mesh_faces)}")
    
    # 预计算所有三角形的 AABB
    print("    预计算三角形 AABB...", end='', flush=True)
    mesh_triangles = mesh_vertices[mesh_faces]  # (N_faces, 3, 3)
    triangle_mins = np.min(mesh_triangles, axis=1)  # (N_faces, 3)
    triangle_maxs = np.max(mesh_triangles, axis=1)  # (N_faces, 3)
    print(" 完成")
    
    # 构建 2D 网格索引 (投影平面)
    print("    构建 2D 网格索引 (投影平面)...", end='', flush=True)
    
    # 确定投影平面范围
    coords_u = mesh_vertices[:, u_axis_idx]
    coords_v = mesh_vertices[:, v_axis_idx]
    u_min, u_max = np.min(coords_u), np.max(coords_u)
    v_min, v_max = np.min(coords_v), np.max(coords_v)
    
    # 网格参数
    cell_size = 10.0   
    
    # 网格尺寸
    grid_w = int(np.ceil((u_max - u_min) / cell_size)) + 1
    grid_h = int(np.ceil((v_max - v_min) / cell_size)) + 1
    
    # 构建网格：grid_map[(gu, gv)] = [triangle_idx1, triangle_idx2, ...]
    grid_map = {}
    
    # 将每个三角形加入到它覆盖的网格单元中
    for idx, (t_min, t_max) in enumerate(zip(triangle_mins, triangle_maxs)):
        tu_min, tu_max = t_min[u_axis_idx], t_max[u_axis_idx]
        tv_min, tv_max = t_min[v_axis_idx], t_max[v_axis_idx]
        
        gu_start = int((tu_min - u_min) / cell_size)
        gu_end = int((tu_max - u_min) / cell_size)
        gv_start = int((tv_min - v_min) / cell_size)
        gv_end = int((tv_max - v_min) / cell_size)
        
        gu_start = max(0, gu_start)
        gu_end = min(grid_w - 1, gu_end)
        gv_start = max(0, gv_start)
        gv_end = min(grid_h - 1, gv_end)
        
        for gu in range(gu_start, gu_end + 1):
            for gv in range(gv_start, gv_end + 1):
                if (gu, gv) not in grid_map:
                    grid_map[(gu, gv)] = []
                grid_map[(gu, gv)].append(idx)
    
    print(f" 完成，网格大小: {grid_w}x{grid_h}，单元尺寸: {cell_size:.1f}")
    
    # 动态生成方向向量
    dir_vec_min = np.zeros(3)
    dir_vec_min[main_axis] = -1.0
    dir_vec_max = np.zeros(3)
    dir_vec_max[main_axis] = 1.0
    
    directions = {
        'u_min': dir_vec_min,
        'u_max': dir_vec_max,
    }
    
    # 计算点集在渗透轴方向的范围
    axis_coords = pore_coords[:, main_axis]
    axis_min = np.min(axis_coords)
    axis_max = np.max(axis_coords)
    axis_range = axis_max - axis_min
    ray_offset = axis_range * 3.0
    
    face_labels = ['none'] * n_surface
    
    # 对每个方向进行射线投射
    for face_name, direction in directions.items():
        print(f"    检查 {face_name} 面...", end='', flush=True)
        visible_count = 0
        
        if face_name == 'u_min':
            ray_start_val = axis_min - ray_offset
        else:
            ray_start_val = axis_max + ray_offset
        
        for i, surface_point in enumerate(surface_coords):
            if (i + 1) % 500 == 0:
                print('.', end='', flush=True)
            
            # 生成一组射线来模拟圆柱（中心 + 多环角向均匀采样）
            rays_to_check = []
            center_origin = np.zeros(3)
            center_origin[main_axis] = ray_start_val
            center_origin[u_axis_idx] = surface_point[u_axis_idx]
            center_origin[v_axis_idx] = surface_point[v_axis_idx]
            rays_to_check.append(center_origin)
            for frac in RAY_CAST_CYLINDER_RING_FRACTIONS:
                r_ring = RAY_CAST_CYLINDER_RADIUS_NM * frac
                for k in range(RAY_CAST_CYLINDER_N_ANGULAR):
                    theta = 2.0 * np.pi * k / RAY_CAST_CYLINDER_N_ANGULAR
                    du = r_ring * np.cos(theta)
                    dv = r_ring * np.sin(theta)
                    offset_origin = center_origin.copy()
                    offset_origin[u_axis_idx] += du
                    offset_origin[v_axis_idx] += dv
                    rays_to_check.append(offset_origin)
            
            is_visible = True
            
            for ray_origin in rays_to_check:
                target_point = ray_origin.copy()
                target_point[main_axis] = surface_point[main_axis]
                
                ray_vec = target_point - ray_origin
                ray_length = np.linalg.norm(ray_vec)
                
                if ray_length < 1e-6:
                    ray_dir = direction
                else:
                    ray_dir = ray_vec / ray_length
                
                is_blocked = False
                
                ray_u = ray_origin[u_axis_idx]
                ray_v = ray_origin[v_axis_idx]
                
                gu = int((ray_u - u_min) / cell_size)
                gv = int((ray_v - v_min) / cell_size)
                
                if 0 <= gu < grid_w and 0 <= gv < grid_h:
                    candidate_indices = grid_map.get((gu, gv), [])
                else:
                    candidate_indices = []
                
                if candidate_indices:
                    for face_idx in candidate_indices:
                        triangle = mesh_triangles[face_idx]
                        t, _ = ray_triangle_intersection(ray_origin, ray_dir, triangle)
                        
                        if t is not None and 1e-6 < t < ray_length - 1.0:
                            is_blocked = True
                            break
                
                if is_blocked:
                    is_visible = False
                    break
            
            if is_visible:
                if face_labels[i] == 'none':
                    face_labels[i] = face_name
                else:
                    # 同时属于多个面
                    parts = sorted([face_labels[i], face_name])
                    face_labels[i] = '_'.join(parts)
                visible_count += 1
        
        print(f" 完成，{face_name} 面: {visible_count} 个可见孔")
    
    # 后处理步骤
    print("    执行后处理过滤...")
    
    # 将 face_labels 转换为 Numpy 数组以支持后续的高级索引过滤
    face_labels = np.array(face_labels, dtype=object)
    
    # 1. 互斥过滤
    dual_count = 0
    for i in range(len(face_labels)):
        if 'u_min' in face_labels[i] and 'u_max' in face_labels[i]:
            face_labels[i] = 'none'
            dual_count += 1
    print(f"      剔除同时属于左右面的孔: {dual_count} 个")
    
    # 2. 局部凹陷过滤 (增强版：基于局部统计)
    pore_grid = {}
    for i in range(n_surface):
        if face_labels[i] == 'none': continue
        p = surface_coords[i]
        gu = int((p[u_axis_idx] - u_min) / cell_size)
        gv = int((p[v_axis_idx] - v_min) / cell_size)
        if (gu, gv) not in pore_grid:
            pore_grid[(gu, gv)] = []
        pore_grid[(gu, gv)].append(i)
    
    outlier_threshold = 10.0 # 收紧阈值，过滤深坑
    removed_min = 0
    removed_max = 0
    
    for (gu, gv), indices in pore_grid.items():
        # 处理 u_min 面
        min_indices = [idx for idx in indices if 'u_min' in face_labels[idx]]
        if len(min_indices) > 0:
            vals = surface_coords[min_indices, main_axis]
            # 使用 10% 分位数作为基准，避免受个别极端外突点的影响
            if len(vals) >= 3:
                baseline = np.percentile(vals, 10)
            else:
                baseline = np.min(vals)
                
            for idx, val in zip(min_indices, vals):
                # 如果该点比基准点“深”太多，则剔除
                if val > baseline + outlier_threshold:
                    face_labels[idx] = 'none'
                    removed_min += 1
                    
        # 处理 u_max 面
        max_indices = [idx for idx in indices if 'u_max' in face_labels[idx]]
        if len(max_indices) > 0:
            vals = surface_coords[max_indices, main_axis]
            # 使用 90% 分位数作为基准
            if len(vals) >= 3:
                baseline = np.percentile(vals, 90)
            else:
                baseline = np.max(vals)
                
            for idx, val in zip(max_indices, vals):
                # 如果该点比基准点“深”太多，则剔除
                if val < baseline - outlier_threshold:
                    face_labels[idx] = 'none'
                    removed_max += 1

    print(f"      剔除渗透轴方向局部显著差异孔: u_min={removed_min}, u_max={removed_max}")
    
    # 2.5 全局断层过滤 (针对平行六面体侧边孔)
    def filter_by_global_gap(indices, face_type):
        if len(indices) < 10: return 0
        
        # 获取这些孔在渗透轴上的坐标
        vals = surface_coords[indices, main_axis]
        # 排序
        sorted_arg = np.argsort(vals)
        sorted_vals = vals[sorted_arg]
        
        # 计算相邻差分
        diffs = np.diff(sorted_vals)
        if len(diffs) == 0: return 0
        
        # 找到最大的断层
        max_gap_idx = np.argmax(diffs)
        max_gap = diffs[max_gap_idx]
        
        # 阈值：如果断层小于 20nm，认为可能是正常的表面起伏，不进行切割
        if max_gap < 10.0:
            return 0
            
        split_idx = max_gap_idx + 1
        n_total = len(vals)
        count_removed = 0
        
        # 只有当保留的主体部分足够大（>30%）时才执行切割，防止误删
        if face_type == 'u_min':
            # u_min (左面): 应该保留坐标较小的部分 (0 ~ split_idx)
            # 剔除坐标较大的部分 (split_idx ~ end)，这些通常是侧边或深层孔
            if split_idx > n_total * 0.20:
                remove_local_indices = sorted_arg[split_idx:]
                indices_to_remove = indices[remove_local_indices]
                face_labels[indices_to_remove] = 'none'
                count_removed = len(indices_to_remove)
        else:
            # u_max (右面): 应该保留坐标较大的部分 (split_idx ~ end)
            # 剔除坐标较小的部分 (0 ~ split_idx)
            if (n_total - split_idx) > n_total * 0.3:
                remove_local_indices = sorted_arg[:split_idx]
                indices_to_remove = indices[remove_local_indices]
                face_labels[indices_to_remove] = 'none'
                count_removed = len(indices_to_remove)
                
        return count_removed

    # 对 u_min 和 u_max 分别执行断层过滤
    # 注意：此时 face_labels 可能包含 'u_min', 'u_max' 或混合标签。这里只处理纯标签或包含标签的
    u_min_indices = np.array([i for i, label in enumerate(face_labels) if 'u_min' in label])
    gap_removed_min = filter_by_global_gap(u_min_indices, 'u_min')
    
    u_max_indices = np.array([i for i, label in enumerate(face_labels) if 'u_max' in label])
    gap_removed_max = filter_by_global_gap(u_max_indices, 'u_max')
    
    print(f"      剔除渗透轴方向全局断层孔 (侧边孔): u_min={gap_removed_min}, u_max={gap_removed_max}")

    # 2.8 基于邻域距离的密度过滤 (剔除孤立点)
    def filter_by_neighbor_distance(indices, k=5):
        if len(indices) <= k: return 0
        
        points = surface_coords[indices]
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=k+1)
        
        # 平均距离 (排除自身)
        avg_dists = np.mean(dists[:, 1:], axis=1)
        
        median_dist = np.median(avg_dists)
        mad = np.median(np.abs(avg_dists - median_dist))
        
        # 设定阈值：如果某点的邻域距离显著大于群体水平，则视为孤立点
        # 使用 max 确保阈值不会因为分布太紧凑而过低
        threshold = max(median_dist + 1.0 * mad, median_dist * 1.5)
        
        outliers = avg_dists > threshold
        if np.any(outliers):
            indices_to_remove = indices[outliers]
            face_labels[indices_to_remove] = 'none'
            return len(indices_to_remove)
        return 0

    # 更新索引并执行密度过滤
    u_min_indices = np.array([i for i, label in enumerate(face_labels) if 'u_min' in label])
    density_removed_min = filter_by_neighbor_distance(u_min_indices)
    
    u_max_indices = np.array([i for i, label in enumerate(face_labels) if 'u_max' in label])
    density_removed_max = filter_by_neighbor_distance(u_max_indices)
    
    print(f"      剔除局部稀疏/孤立孔 (邻域距离法): u_min={density_removed_min}, u_max={density_removed_max}")

    # 3. 边缘裁剪
    coords_u = surface_coords[:, u_axis_idx]
    coords_v = surface_coords[:, v_axis_idx]
    
    def get_percentiles(axis_idx):
        if axis_idx == 1: # Y 轴
            return 0.5, 99.5
        else:
            return 0.5, 99.5
            
    u_pct_low, u_pct_high = get_percentiles(u_axis_idx)
    v_pct_low, v_pct_high = get_percentiles(v_axis_idx)
    
    u_q_low, u_q_high = np.percentile(coords_u, [u_pct_low, u_pct_high])
    v_q_low, v_q_high = np.percentile(coords_v, [v_pct_low, v_pct_high])
    
    edge_removed = 0
    for i in range(n_surface):
        if face_labels[i] == 'none': continue
        u_val = surface_coords[i, u_axis_idx]
        v_val = surface_coords[i, v_axis_idx]
        
        is_edge = (
            u_val < u_q_low or u_val > u_q_high or
            v_val < v_q_low or v_val > v_q_high
        )
        if is_edge:
            face_labels[i] = 'none'
            edge_removed += 1
            
    print(f"      剔除边缘孔 (平面轴 {u_axis_idx}/{v_axis_idx}): {edge_removed} 个")
    
    # 4. 半径过滤
    radius_removed = 0
    all_surface_radii = pore_radii[surface_indices]
    for i in range(n_surface):
        if face_labels[i] == 'none': continue
        if all_surface_radii[i] <= 0.0:
            face_labels[i] = 'none'
            radius_removed += 1
            
    print(f"      剔除小孔 (半径 <= 0.0): {radius_removed} 个")
    
    face_labels = np.array(face_labels, dtype=object)
    
    # 真实半径 HTML 在「表面质心修正」之后由 write_surface_pores_real_radius_html 写出（含拟合端面与两方向箭头）
    print(f"  [射线投射] 真实半径可视化将在表面质心修正后写出（含拟合渗透端面与方向）")
    
    # 统计最终结果
    print("  [射线投射] 面分配统计 (内部详细):")
    unique_labels = np.unique(face_labels)
    for label in unique_labels:
        count = np.sum(face_labels == label)
        if label == 'none':
            print(f"    未分配 {label}: {count} 个")
        elif '_' in label:
            # 判断是否是组合标签 (edge/corner)
            parts = label.split('_')
            if len(parts) > 2:
                if len(parts) == 4:
                    print(f"    边 {label}: {count} 个")
                else:
                    print(f"    角 {label}: {count} 个")
            else:
                print(f"    面 {label}: {count} 个")
        else:
             print(f"    面 {label}: {count} 个")

    return face_labels
# ====================================================================
# 步骤2.5: 使用自动厚度校准 + 射线投射法识别表面孔
# ====================================================================
print("\n步骤2.5: 识别渗透方向并提取表面孔 (射线投射法)...")

# --- 子步骤 1: 确定渗透方向 ---
main_axis = 0 # 默认为 X（如果没有法向量/厚度信息）
target_thickness = None
normal_vec = None

# 优先使用法向量（子样本分析）
if NORMAL_VECTOR_VALS is not None:
    print("  [法向量] 使用提供的法向量作为渗透方向...")
    try:
        if len(NORMAL_VECTOR_VALS) == 2:
            # 注意：不要覆盖 networkx 的别名 nx，这里使用 nx_val/ny_val
            nx_val, ny_val = float(NORMAL_VECTOR_VALS[0]), float(NORMAL_VECTOR_VALS[1])
            normal_vec = np.array([nx_val, ny_val])
            # 归一化
            norm = np.linalg.norm(normal_vec)
            if norm > 1e-10:
                normal_vec = normal_vec / norm
            print(f"  [法向量] 归一化后的法向量(在XY平面): [{normal_vec[0]:.4f}, {normal_vec[1]:.4f}]")
        else:
            print(f"  [错误] 法向量参数个数错误，应为两个数字 NX NY")
            print("  [提示] 将使用默认X轴作为渗透方向")
    except Exception as e:
        print(f"  [错误] 解析法向量失败: {e}")
        print("  [提示] 将使用默认X轴作为渗透方向")
elif _args.thickness_file:
    print("  [厚度校准] 正在读取厚度文件...")
    try:
        thick_df = pd.read_excel(_args.thickness_file)
        if 'Sample' not in thick_df.columns:
            # 尝试标准化列名
            thick_df.columns = ['Sample', 'Thickness'] + list(thick_df.columns[2:])
        
        # 查找当前样本
        row = thick_df[thick_df['Sample'].astype(str) == SAMPLE_NAME]
        if row.empty:
            print(f"  [警告] 在厚度文件中未找到样本: {SAMPLE_NAME}")
        else:
            target_thickness = float(row.iloc[0]['Thickness'])
            print(f"  [厚度校准] 目标厚度: {target_thickness:.2f}")
            
            # 计算X和Y轴向的平均厚度（不考虑Z轴）
            thicknesses = []
            axis_list = [0, 1]  # 只考虑X和Y轴
            for axis in axis_list:
                t = calculate_average_thickness(pore_coords, pore_radii, axis)
                thicknesses.append(t)
                print(f"    轴 {axis} ({['X','Y','Z'][axis]}): {t:.2f}")
            
            # 找到最接近目标厚度的轴
            errors = [abs(t - target_thickness) for t in thicknesses]
            best_idx = np.argmin(errors)
            best_axis = axis_list[best_idx]
            print(f"  [厚度校准] 最接近的轴是 {['X','Y','Z'][best_axis]} (误差: {errors[best_idx]:.2f})")
            
            main_axis = best_axis
            
    except Exception as e:
        print(f"  [错误] 读取厚度文件失败: {e}")
        print("  [提示] 将默认使用 X 轴作为渗透方向")
else:
    print("  [提示] 未提供厚度文件或法向量，默认假设 X 轴为渗透方向")

# 物理坐标系下的渗透方向单位向量（metadata 与 θ 判定的回退方向）
if normal_vec is not None:
    _nx0, _ny0 = float(normal_vec[0]), float(normal_vec[1])
    physical_penetration_unit = np.array([_nx0, _ny0, 0.0], dtype=float)
else:
    physical_penetration_unit = np.zeros(3, dtype=float)
    physical_penetration_unit[main_axis] = 1.0
_pu_n = np.linalg.norm(physical_penetration_unit)
if _pu_n > 1e-12:
    physical_penetration_unit = physical_penetration_unit / _pu_n
else:
    physical_penetration_unit = np.array([1.0, 0.0, 0.0], dtype=float)

# 物理坐标 → 表面识别坐标（u,v,w）：p_surface = p_phys @ E_basis
E_basis = np.eye(3)

# ---------------- 在法向量方向上构造新的坐标系（如果提供了法向量） ----------------
# 默认情况下，表面识别/厚度计算都在原始坐标系下，沿 main_axis (X/Y)
pore_coords_for_surface = pore_coords
coords_for_thickness = pore_coords
thickness_axis = main_axis
axis_label = ["X", "Y", "Z"][main_axis]

if normal_vec is not None:
    # 在 XY 平面构造以法向量为渗透方向的新正交基
    try:
        nx_val, ny_val = float(normal_vec[0]), float(normal_vec[1])
        # 渗透方向 e0: 沿法向量 (nx_val, ny_val, 0)
        e0 = np.array([nx_val, ny_val, 0.0])
        # 与 e0 垂直、仍在 XY 平面的方向 e1
        e1 = np.array([-ny_val, nx_val, 0.0])
        # 保留原始 Z 轴
        e2 = np.array([0.0, 0.0, 1.0])
        E_basis = np.column_stack([e0, e1, e2])

        # 将孔坐标投影到新基底 (u,v,w)
        u = pore_coords @ e0
        v = pore_coords @ e1
        w = pore_coords @ e2
        pore_coords_rot = np.column_stack([u, v, w])

        pore_coords_for_surface = pore_coords_rot
        coords_for_thickness = pore_coords_rot
        # 在旋转坐标系下，渗透方向恒为轴 0 (u)
        main_axis = 0
        thickness_axis = 0
        axis_label = f"normal_vector({nx_val:.3f},{ny_val:.3f})"
        print(f"  [法向量] 已在XY平面构造旋转坐标系，渗透方向对齐法向量 (作为新坐标轴 u)")
    except Exception as e:
        print(f"  [警告] 基于法向量构造旋转坐标系失败，将退回到原始坐标系: {e}")
        # 如果失败，则仍然使用原始 pore_coords 和 main_axis (可能是 X/Y)
        pore_coords_for_surface = pore_coords
        coords_for_thickness = pore_coords
        thickness_axis = main_axis
        axis_label = ["X", "Y", "Z"][main_axis]

print(f"  最终用于渗透分析的方向: {axis_label} (内部轴索引 = {thickness_axis})")

# 在最终渗透方向上计算平均厚度（单位：nm）
avg_thickness_main = 0.0
try:
    avg_thickness_main = calculate_average_thickness(coords_for_thickness, pore_radii, axis=thickness_axis)
    print(f"  在渗透方向 {axis_label} 的平均厚度: {avg_thickness_main:.2f} nm（最终写入见表面质心修正后的 thickness_summary）")
except Exception as e:
    print(f"  [警告] 计算沿渗透轴平均厚度时出错: {e}")

# --- 子步骤 2: 获取 Alpha Shape Mesh ---
print(f"  [Alpha Shape] 生成表面网格...")
boundary_mask, alpha_used, alpha_mesh = identify_surface_by_alphashape(
    pore_coords_for_surface, alpha=None # 自动估算
)

# --- 子步骤 3: 射线投射识别面 ---
print(f"  [射线投射] 识别表面孔...")
face_labels = identify_faces_by_ray_casting(
    pore_coords=pore_coords_for_surface,
    pore_radii=pore_radii,
    boundary_mask=boundary_mask,
    alpha_mesh=alpha_mesh,
    sample_name=SAMPLE_NAME,
    output_dir=output_folder,
    main_axis=main_axis
)

# --- 子步骤 4: 解析结果并适配旧格式 ---
is_surface_x = np.zeros(len(pore_coords), dtype=bool)
is_surface_x_left = np.zeros(len(pore_coords), dtype=bool) # u_min
is_surface_x_right = np.zeros(len(pore_coords), dtype=bool) # u_max

if face_labels is not None:
    # 获取表面孔在原始数组中的索引
    surface_indices = np.where(boundary_mask)[0]
    
    for i, label in enumerate(face_labels):
        if i >= len(surface_indices):
            break
        
        # 获取原始孔的索引
        original_idx = surface_indices[i]
        
        if label == 'u_min':
            is_surface_x_left[original_idx] = True
            is_surface_x[original_idx] = True
        elif label == 'u_max':
            is_surface_x_right[original_idx] = True
            is_surface_x[original_idx] = True
        elif 'u_min' in label and 'u_max' in label:
            # 这种情况应该已经被后处理剔除了，但为了保险
            pass

# 射线解析完成后可选二次几何过滤（d<=r+eps；与 overall / phase4 / sub_calculate 一致时再写 Excel）
ENABLE_SURFACE_GEOM_FILTER = True
SURFACE_GEOM_TOL_FRAC = 0.05
SURFACE_GEOM_GRID_STEP_NM = 20.0
n_geom_left_in = int(np.sum(is_surface_x_left))
n_geom_right_in = int(np.sum(is_surface_x_right))
dropped_left = dropped_right = 0
if ENABLE_SURFACE_GEOM_FILTER and (np.any(is_surface_x_left) or np.any(is_surface_x_right)):
    axis_u = main_axis
    axis_v = (main_axis + 1) % 3
    axis_w = (main_axis + 2) % 3
    coords_u = pore_coords_for_surface[:, axis_u]
    coords_v = pore_coords_for_surface[:, axis_v]
    coords_w = pore_coords_for_surface[:, axis_w]
    coord_min_g = float(coords_u.min())
    coord_max_g = float(coords_u.max())
    T_global = coord_max_g - coord_min_g if coord_max_g > coord_min_g else 0.0
    v_lo, v_hi = float(coords_v.min()), float(coords_v.max())
    w_lo, w_hi = float(coords_w.min()), float(coords_w.max())
    cell = float(SURFACE_GEOM_GRID_STEP_NM)
    nv = max(1, int(np.ceil((v_hi - v_lo) / cell)))
    nw = max(1, int(np.ceil((w_hi - w_lo) / cell)))

    def _cell_of(i):
        iv = int((coords_v[i] - v_lo) / cell)
        iw = int((coords_w[i] - w_lo) / cell)
        iv = min(max(iv, 0), nv - 1)
        iw = min(max(iw, 0), nw - 1)
        return iv, iw

    _cell_idxs = defaultdict(list)
    for _i in range(len(pore_coords_for_surface)):
        _cell_idxs[_cell_of(_i)].append(_i)
    _cell_u_min = {}
    _cell_u_max = {}
    _cell_T = {}
    for ck, idxs in _cell_idxs.items():
        uu = coords_u[idxs]
        umin, umax = float(uu.min()), float(uu.max())
        _cell_u_min[ck] = umin
        _cell_u_max[ck] = umax
        _cell_T[ck] = umax - umin

    def _local_endpoints(i):
        ck = _cell_of(i)
        n_here = len(_cell_idxs.get(ck, ()))
        T_loc = _cell_T.get(ck, T_global)
        if n_here < 2 or T_loc <= 1e-9:
            return coord_min_g, coord_max_g, T_global
        return _cell_u_min[ck], _cell_u_max[ck], T_loc

    if T_global > 0:
        left_idx = np.where(is_surface_x_left)[0]
        if len(left_idx) > 0:
            keep_left = np.zeros(len(left_idx), dtype=bool)
            for _k, i in enumerate(left_idx):
                umin_i, _, T_i = _local_endpoints(i)
                eps_i = max(1.0, SURFACE_GEOM_TOL_FRAC * T_i)
                keep_left[_k] = (coords_u[i] - umin_i) <= (pore_radii[i] + eps_i)
            dropped_left = int(np.sum(~keep_left))
            if dropped_left > 0:
                is_surface_x_left[left_idx[~keep_left]] = False
        right_idx = np.where(is_surface_x_right)[0]
        if len(right_idx) > 0:
            keep_right = np.zeros(len(right_idx), dtype=bool)
            for _k, i in enumerate(right_idx):
                _, umax_i, T_i = _local_endpoints(i)
                eps_i = max(1.0, SURFACE_GEOM_TOL_FRAC * T_i)
                keep_right[_k] = (umax_i - coords_u[i]) <= (pore_radii[i] + eps_i)
            dropped_right = int(np.sum(~keep_right))
            if dropped_right > 0:
                is_surface_x_right[right_idx[~keep_right]] = False
        is_surface_x = is_surface_x_left | is_surface_x_right
        print("  [几何过滤] 射线判定后：垂直渗透轴平面网格 "
              f"{cell:.0f}×{cell:.0f} nm（{nv}×{nw} 格），局部 d<=r+eps（eps=max(1nm, "
              f"{SURFACE_GEOM_TOL_FRAC:.2f}·T_cell），单孔格回退全局 T）")
        print(f"    全局参考：T_global={T_global:.3f} nm")
        print(
            f"    u_min 侧：射线候选 {n_geom_left_in} 个 → 剔除 {dropped_left} 个 → 保留 {int(np.sum(is_surface_x_left))} 个"
        )
        print(
            f"    u_max 侧：射线候选 {n_geom_right_in} 个 → 剔除 {dropped_right} 个 → 保留 {int(np.sum(is_surface_x_right))} 个"
        )
    else:
        print("  [几何过滤] 跳过：渗透轴坐标跨度 T_global≈0")
elif not ENABLE_SURFACE_GEOM_FILTER:
    print("  [几何过滤] 已关闭（ENABLE_SURFACE_GEOM_FILTER=False），沿用射线判定结果。")

is_surface_x = is_surface_x_left | is_surface_x_right

n_surface_x_left = np.sum(is_surface_x_left)
n_surface_x_right = np.sum(is_surface_x_right)
n_surface_x_total = np.sum(is_surface_x)

geom_note = (" + 几何二次筛选" if ENABLE_SURFACE_GEOM_FILTER else "")
print(f"  识别完成（射线判定{geom_note}，将写入 Excel）:")
print(f"    渗透入口孔 (u_min): {n_surface_x_left} 个 ({100*n_surface_x_left/len(pore_coords):.1f}%)")
print(f"    渗透出口孔 (u_max): {n_surface_x_right} 个 ({100*n_surface_x_right/len(pore_coords):.1f}%)")
print(f"    总表面孔数: {n_surface_x_total} 个")

# --- 修正渗透方向（与可视化橙箭一致）：θ=arcsin(|nz|) 为与 x–y 平面夹角；θ>阈值时 平均厚度≈沿渗透轴厚度×cosθ（平行端面近似：法向间距 = 水平表观跨度×cosθ）---
TILT_REFINE_DEG = 25.0
MIN_SURFACE_EACH_SIDE = 3
thickness_excel = Path(output_folder) / f"{SAMPLE_NAME}_thickness_summary.xlsx"
metadata_path = Path(output_folder) / f"{SAMPLE_NAME}_metadata.json"
avg_thickness_reported = avg_thickness_main
n_meta = physical_penetration_unit.copy()
theta_deg_surface = None
used_tilt_refine = False
E_lab = np.asarray(E_basis, dtype=float).reshape(3, 3)
idx_left = np.where(is_surface_x_left)[0]
idx_right = np.where(is_surface_x_right)[0]
if len(idx_left) >= MIN_SURFACE_EACH_SIDE and len(idx_right) >= MIN_SURFACE_EACH_SIDE:
    sl = pore_coords_for_surface[idx_left]
    sr = pore_coords_for_surface[idx_right]
    n_surf = merged_permeation_normal_surface(sl, sr)
    if n_surf is not None:
        n_ref = (E_lab @ np.asarray(n_surf, dtype=float).reshape(3, 1)).ravel()
        n_ref = n_ref / (float(np.linalg.norm(n_ref)) + 1e-15)
        nz_abs = abs(float(n_ref[2]))
        nz_clipped = float(np.clip(nz_abs, 0.0, 1.0))
        theta_deg_surface = float(np.degrees(np.arcsin(nz_clipped)))
        n_meta = n_ref.astype(float)
        if theta_deg_surface > TILT_REFINE_DEG:
            # θ = 修正渗透方向与 x–y 平面夹角；cosθ = 水平面内分量模长 √(1−nz²)
            theta_rad = float(np.radians(theta_deg_surface))
            c_theta = float(np.cos(theta_rad))
            t_ref = float(avg_thickness_main * c_theta)
            if t_ref > 0.0 and c_theta > 1e-15:
                avg_thickness_reported = t_ref
                used_tilt_refine = True
                print(
                    f"  [倾斜厚度] θ={theta_deg_surface:.2f}° > {TILT_REFINE_DEG}°，"
                    f"修正厚度 = 沿渗透轴×cosθ = {avg_thickness_main:.2f}×{c_theta:.6f} = {t_ref:.2f} nm"
                )
            else:
                print(
                    f"  [倾斜厚度] θ={theta_deg_surface:.2f}° 但 沿渗透轴×cosθ≈0，"
                    f"保留沿渗透轴厚度 {avg_thickness_main:.2f} nm"
                )
        else:
            print(
                f"  [倾斜厚度] θ={theta_deg_surface:.2f}° ≤ {TILT_REFINE_DEG}°，"
                f"沿用沿渗透轴厚度 {avg_thickness_main:.2f} nm（仍写入修正方向 nx,ny,nz 供记录）"
            )
    else:
        print(
            "  [倾斜厚度] 无法由表面点构造修正渗透方向（质心退化），"
            "metadata 使用渗透轴方向"
        )
else:
    print(
        f"  [倾斜厚度] 两侧表面孔不足（左 {len(idx_left)} / 右 {len(idx_right)}，每侧需 ≥{MIN_SURFACE_EACH_SIDE}），"
        "跳过修正方向；metadata 使用渗透轴方向"
    )

if face_labels is not None:
    try:
        _si = np.where(boundary_mask)[0]
        _sc = pore_coords_for_surface[_si]
        _rs = pore_radii[_si]
        write_surface_pores_real_radius_html(
            _sc,
            face_labels,
            _rs,
            SAMPLE_NAME,
            output_folder,
            main_axis,
            E_basis,
            n_meta,
            theta_deg_surface,
        )
    except Exception as _e:
        print(f"  [警告] 表面半径可视化（拟合平面+方向）写入失败: {_e}")

try:
    nz_abs = float(abs(float(n_meta[2])))
    nz_clip = float(np.clip(nz_abs, 0.0, 1.0))
    phi_vs_z = float(np.degrees(np.arccos(nz_clip)))
    theta_row = (
        float(theta_deg_surface)
        if theta_deg_surface is not None and np.isfinite(theta_deg_surface)
        else np.nan
    )
    thickness_df_out = pd.DataFrame(
        [
            {
                "Sample Name": SAMPLE_NAME,
                "Main Axis Index": thickness_axis,
                "Main Axis Label": axis_label,
                # 下游 phase1 / load_thickness：主列为修正后厚度（与倾斜修正一致）
                "Average Thickness (nm)": float(avg_thickness_reported),
                "Thickness_along_axis_nm": float(avg_thickness_main),
                "Theta_deg_vs_XY": theta_row,
                "Phi_deg_vs_Z": phi_vs_z,
                "Used_tilt_refine": int(bool(used_tilt_refine)),
                "Tilt_threshold_deg": float(TILT_REFINE_DEG),
                "nx": float(n_meta[0]),
                "ny": float(n_meta[1]),
                "nz": float(n_meta[2]),
            }
        ]
    )
    thickness_df_out.to_excel(thickness_excel, index=False)
    print(
        f"  厚度汇总（最终）已保存: {thickness_excel} | "
        f"Average={avg_thickness_reported:.2f} nm, 沿轴={avg_thickness_main:.2f} nm, "
        f"θ_XY={theta_row if np.isfinite(theta_row) else '—'}°, 倾斜修正={used_tilt_refine}"
    )

    meta_out = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta_out = json.load(f)
        except Exception as e:
            print(f"  [警告] 读取既有 metadata 失败，将覆盖写入: {e}")
            meta_out = {}
    meta_out["normal_vector"] = {
        "nx": float(n_meta[0]),
        "ny": float(n_meta[1]),
        "nz": float(n_meta[2]),
    }
    meta_out["thickness_refinement"] = {
        "used_surface_centroid_normal": len(idx_left) >= MIN_SURFACE_EACH_SIDE
        and len(idx_right) >= MIN_SURFACE_EACH_SIDE,
        "theta_deg_vs_xy": theta_deg_surface,
        "theta_vs_xy_definition": "arcsin(|nz|) of merged permeation direction in lab frame (PCA merge or surface centroid chord)",
        "tilt_corrected_thickness_formula": "Thickness_along_axis_nm * cos(theta_vs_xy_rad); cos(theta)=sqrt(1-nz^2)=|n_xy|",
        "used_tilt_refine": used_tilt_refine,
        "tilt_threshold_deg": TILT_REFINE_DEG,
        "average_thickness_nm": float(avg_thickness_reported),
        "average_thickness_along_axis_nm": float(avg_thickness_main),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    print(f"  元数据已更新: {metadata_path}")
except Exception as e:
    print(f"  [警告] 更新厚度汇总或 metadata 失败: {e}")

# 创建分类数据框
classification_df = pd.DataFrame({
    'Pore ID': pore_ids,
    'Volume': pore_volumes,
    'EqRadius': pore_radii,
    'X Coord': pore_coords[:, 0],
    'Y Coord': pore_coords[:, 1],
    'Z Coord': pore_coords[:, 2],
    'Is Surface X': is_surface_x,
    'Is Surface X Left': is_surface_x_left,
    'Is Surface X Right': is_surface_x_right
})
if alpha_used is not None:
    classification_df['Alpha Used'] = alpha_used
classification_excel = Path(output_folder) / f'{SAMPLE_NAME}_pore_classification.xlsx'
classification_df.to_excel(classification_excel, index=False)
print(f"  分类结果已保存: {classification_excel}")

# ====================================================================
# 步骤3: 可视化1 - 孔隙（按半径着色，外部孔特殊标记）
# ====================================================================
print("\n步骤3: 创建孔隙的可交互三维可视化...")

# 创建子图（如果需要可以分两个视角）
fig_pores = go.Figure()

# 所有孔（按半径着色）
fig_pores.add_trace(go.Scatter3d(
    x=pore_coords[:, 0],
    y=pore_coords[:, 1],
    z=pore_coords[:, 2],
    mode='markers',
    marker=dict(
        size=pore_radii * 2,  # 缩放因子使球体可见
        color=pore_radii,
        colorscale='Viridis',
        colorbar=dict(title="Radius (nm)", x=1.1),
        line=dict(width=0.5, color='rgba(0,0,0,0.3)'),
        opacity=0.8
    ),
    name='All Pores',
    text=[f'Pore ID: {pid}<br>Radius: {r:.2f} nm<br>Volume: {v:.2f}' 
          for pid, r, v in zip(pore_ids, pore_radii, pore_volumes)],
    hovertemplate='<b>%{text}</b><extra></extra>'
))

# X轴方向的外部孔（左侧，用红色标记）
if np.any(is_surface_x_left):
    surface_x_left_mask = is_surface_x_left
    if np.any(surface_x_left_mask):
        fig_pores.add_trace(go.Scatter3d(
            x=pore_coords[surface_x_left_mask, 0],
            y=pore_coords[surface_x_left_mask, 1],
            z=pore_coords[surface_x_left_mask, 2],
            mode='markers',
            marker=dict(
                size=pore_radii[surface_x_left_mask] * 2.2,
                color='red',
                line=dict(width=1, color='darkred'),
                opacity=0.9,
                symbol='circle'
            ),
            name='Surface X-Left (Alpha Shape)',
            text=[f'Pore ID: {pid}<br>Radius: {r:.2f} nm<br>X: {x:.2f}<br><b>SURFACE X-LEFT</b>' 
                  for pid, r, x in zip(pore_ids[surface_x_left_mask], 
                                       pore_radii[surface_x_left_mask],
                                       pore_coords[surface_x_left_mask, 0])],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))

# X轴方向的外部孔（右侧，用蓝色标记）
if np.any(is_surface_x_right):
    surface_x_right_mask = is_surface_x_right
    if np.any(surface_x_right_mask):
        fig_pores.add_trace(go.Scatter3d(
            x=pore_coords[surface_x_right_mask, 0],
            y=pore_coords[surface_x_right_mask, 1],
            z=pore_coords[surface_x_right_mask, 2],
            mode='markers',
            marker=dict(
                size=pore_radii[surface_x_right_mask] * 2.2,
                color='blue',
                line=dict(width=1, color='darkblue'),
                opacity=0.9,
                symbol='circle'
            ),
            name='Surface X-Right (Alpha Shape)',
            text=[f'Pore ID: {pid}<br>Radius: {r:.2f} nm<br>X: {x:.2f}<br><b>SURFACE X-RIGHT</b>' 
                  for pid, r, x in zip(pore_ids[surface_x_right_mask], 
                                       pore_radii[surface_x_right_mask],
                                       pore_coords[surface_x_right_mask, 0])],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))

fig_pores.update_layout(
    title=dict(
        text='Pore Network - Interactive 3D Visualization (Alpha Shape)<br><sub>Green/Yellow = Internal, Red = Surface X-Left, Blue = Surface X-Right</sub>',
        x=0.5
    ),
    scene=dict(
        xaxis_title='X (nm)',
        yaxis_title='Y (nm)',
        zaxis_title='Z (nm)',
        aspectmode='data'
    ),
    width=1200,
    height=800
)

pores_html = Path(output_folder) / f'{SAMPLE_NAME}_pores_interactive.html'
fig_pores.write_html(str(pores_html))
print(f"  孔隙可视化已保存: {pores_html}")
print(f"    总孔数: {len(pore_ids)} 个（按半径着色）")
print(f"    X轴左侧表面孔: {n_surface_x_left} 个（红色球体）")
print(f"    X轴右侧外部孔: {n_surface_x_right} 个（蓝色球体）")

# ====================================================================
# 步骤4: 过滤无法通过溶质的喉
# ====================================================================
print("\n步骤4: 过滤无法通过溶质的喉...")

# 过滤：与筛分一致，喉半径须严格大于 r_s
feasible_throat_mask = throat_radii > SOLUTE_PASSAGE_GEOMETRY_MIN_NM
n_feasible = np.sum(feasible_throat_mask)
n_blocked = np.sum(~feasible_throat_mask)

print(
    f"  可通过的喉（半径 > {SOLUTE_PASSAGE_GEOMETRY_MIN_NM:.4f} nm = r_s）: {n_feasible} 个 "
    f"({100*n_feasible/len(throat_radii):.1f}%)"
)
print(
    f"  不可通过的喉（半径 ≤ {SOLUTE_PASSAGE_GEOMETRY_MIN_NM:.4f} nm）: {n_blocked} 个 "
    f"({100*n_blocked/len(throat_radii):.1f}%)"
)

feasible_throat_ids = throat_ids[feasible_throat_mask]
feasible_throat_radii = throat_radii[feasible_throat_mask]
feasible_pore1 = throat_pore1[feasible_throat_mask]
feasible_pore2 = throat_pore2[feasible_throat_mask]

# 创建图来找到连通的喉网络
print(f"  构建连通网络...")
G = nx.Graph()

# 添加可通过的喉作为边
for t_id, p1, p2, r in zip(feasible_throat_ids, feasible_pore1, feasible_pore2, feasible_throat_radii):
    if p1 in pore_ids and p2 in pore_ids:
        G.add_edge(p1, p2, throat_id=t_id, radius=r)

# 找到所有连通分量
connected_components = list(nx.connected_components(G))
n_components = len(connected_components)
component_sizes = [len(comp) for comp in connected_components]

print(f"  连通分量数: {n_components}")
print(f"  最大连通分量大小: {max(component_sizes) if component_sizes else 0} 个孔隙")
print(f"  连通分量大小统计:")
for i, size in enumerate(sorted(component_sizes, reverse=True)[:10]):
    print(f"    组件 {i+1}: {size} 个孔隙")

# 为每个孔隙分配组件ID
pore_to_component = {}
for comp_id, component in enumerate(connected_components):
    for pore_id in component:
        pore_to_component[pore_id] = comp_id

# ====================================================================
# 步骤5: 可视化2 - 可联通的喉（不同颜色标记不同连通分量）
# ====================================================================
print("\n步骤5: 创建可联通喉的可交互三维可视化...")

fig_throats = go.Figure()

# 为每个连通分量分配颜色
colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink', 'gray']
if n_components > len(colors):
    import colorsys
    colors = [colorsys.hsv_to_rgb(i/n_components, 0.8, 0.9) for i in range(n_components)]
    colors = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors]

# 绘制每个连通分量的喉
for comp_id in range(min(n_components, 8000)):  # 最多显示8000个连通分量
    component = connected_components[comp_id]
    color = colors[comp_id % len(colors)]
    
    # 找到这个分量中的所有边（喉）
    component_throats = []
    for p1 in component:
        for p2 in G.neighbors(p1):
            if p2 > p1:  # 避免重复
                edge_data = G[p1][p2]
                throat_id = edge_data['throat_id']
                throat_r = edge_data['radius']
                if p1 in pore_id_to_idx and p2 in pore_id_to_idx:
                    component_throats.append((p1, p2, throat_r))
    
    if not component_throats:
        continue
    
    # 绘制这个分量的喉（线条）
    for p1_id, p2_id, throat_r in component_throats:
        p1_idx = pore_id_to_idx[p1_id]
        p2_idx = pore_id_to_idx[p2_id]
        p1_coord = pore_coords[p1_idx]
        p2_coord = pore_coords[p2_idx]
        
        fig_throats.add_trace(go.Scatter3d(
            x=[p1_coord[0], p2_coord[0]],
            y=[p1_coord[1], p2_coord[1]],
            z=[p1_coord[2], p2_coord[2]],
            mode='lines',
            line=dict(color=color, width=throat_r * 0.5),  # 线条粗细表示半径
            showlegend=False,
            name=f'Component {comp_id+1} ({len(component)} pores)',
            hovertemplate=f'<b>Throat</b><br>Radius: {throat_r:.2f} nm<br>Component: {comp_id+1}<extra></extra>'
        ))

# 可选：添加孔隙位置（小点）
connected_pore_ids = set()
for comp in connected_components:
    connected_pore_ids.update(comp)

connected_pore_mask = np.array([pid in connected_pore_ids for pid in pore_ids])
if np.any(connected_pore_mask):
    fig_throats.add_trace(go.Scatter3d(
        x=pore_coords[connected_pore_mask, 0],
        y=pore_coords[connected_pore_mask, 1],
        z=pore_coords[connected_pore_mask, 2],
        mode='markers',
        marker=dict(size=2, color='black', opacity=0.3),
        name='Pores',
        showlegend=False,
        hovertemplate='<b>Pore</b><extra></extra>'
    ))

fig_throats.update_layout(
    title=dict(
        text=f'Feasible Throat Network - Interactive 3D Visualization<br><sub>Different colors = Different connected components (throat radius > {SOLUTE_PASSAGE_GEOMETRY_MIN_NM:.4f} nm = r_s)</sub>',
        x=0.5
    ),
    scene=dict(
        xaxis_title='X (nm)',
        yaxis_title='Y (nm)',
        zaxis_title='Z (nm)',
        aspectmode='data'
    ),
    width=1200,
    height=800
)

throats_html = Path(output_folder) / f'{SAMPLE_NAME}_feasible_throats_interactive.html'
fig_throats.write_html(str(throats_html))
print(f"  可联通喉可视化已保存: {throats_html}")

# 保存过滤和连通性结果
throats_classification_df = pd.DataFrame({
    'Throat ID': throat_ids,
    'EqRadius': throat_radii,
    'Pore ID #1': throat_pore1,
    'Pore ID #2': throat_pore2,
    'Feasible': feasible_throat_mask
})
throats_classification_excel = Path(output_folder) / f'{SAMPLE_NAME}_throat_classification.xlsx'
throats_classification_df.to_excel(throats_classification_excel, index=False)
print(f"  喉分类结果已保存: {throats_classification_excel}")

# 保存连通分量信息
component_info = []
for comp_id, component in enumerate(connected_components):
    component_pores = list(component)
    component_info.append({
        'Component ID': comp_id,
        'Component Size': len(component),
        'Pore IDs': ','.join(map(str, sorted(component_pores)))
    })
component_df = pd.DataFrame(component_info)
component_excel = Path(output_folder) / f'{SAMPLE_NAME}_connected_components.xlsx'
component_df.to_excel(component_excel, index=False)
print(f"  连通分量信息已保存: {component_excel}")

# ====================================================================
# 步骤4.5: 筛选溶剂通路（所有喉，但需满足连通性）
# ====================================================================
print("\n步骤4.5: 筛选溶剂通路（连通性检查）...")

# 构建包含所有喉的网络图（溶剂通路，无半径过滤）
G_solvent = nx.Graph()
print(f"  构建溶剂网络图（包含所有喉）...")
for t_id, p1, p2, r in zip(throat_ids, throat_pore1, throat_pore2, throat_radii):
    if p1 in pore_ids and p2 in pore_ids:
        # 计算喉长度：统一使用几何喉长 = 孔心距 d（与 GBM 模型一致）
        p1_idx = pore_id_to_idx[p1]
        p2_idx = pore_id_to_idx[p2]
        p1_coord = pore_coords[p1_idx]
        p2_coord = pore_coords[p2_idx]
        center_distance = np.linalg.norm(p2_coord - p1_coord)
        length = max(center_distance, 1e-6)
        G_solvent.add_edge(p1, p2, throat_id=t_id, radius=r, length=length)

# 找到所有连通分量
solvent_components = list(nx.connected_components(G_solvent))
n_solvent_components = len(solvent_components)
solvent_component_sizes = [len(comp) for comp in solvent_components]

print(f"  溶剂网络连通分量数: {n_solvent_components}")
print(f"  最大连通分量大小: {max(solvent_component_sizes) if solvent_component_sizes else 0} 个孔隙")
print(f"  连通分量大小统计（前10个）:")
for i, size in enumerate(sorted(solvent_component_sizes, reverse=True)[:10]):
    print(f"    组件 {i+1}: {size} 个孔隙")

# 筛选：只保留连接左右两侧外部孔的连通分量
surface_left_pore_ids = set(pore_ids[is_surface_x_left])
surface_right_pore_ids = set(pore_ids[is_surface_x_right])

solvent_penetration_components = []
for comp_id, component in enumerate(solvent_components):
    has_left = bool(component & surface_left_pore_ids)
    has_right = bool(component & surface_right_pore_ids)
    if has_left and has_right:
        solvent_penetration_components.append((comp_id, component))

n_solvent_penetration = len(solvent_penetration_components)
print(f"  溶剂渗透连通分量数（连接左右两侧）: {n_solvent_penetration}")

# 收集所有溶剂渗透组件中的喉和孔
solvent_feasible_throat_ids = set()
solvent_feasible_pores = set()
for comp_id, component in solvent_penetration_components:
    solvent_feasible_pores.update(component)
    # 收集这个分量中的所有喉
    for p1 in component:
        for p2 in G_solvent.neighbors(p1):
            if p2 in component:  # 确保都在同一个分量内
                edge_data = G_solvent[p1][p2]
                solvent_feasible_throat_ids.add(edge_data['throat_id'])

print(f"  溶剂渗透网络中:")
print(f"    孔数: {len(solvent_feasible_pores)}")
print(f"    喉数: {len(solvent_feasible_throat_ids)}")

# 保存溶剂通路筛选结果
solvent_classification_df = pd.DataFrame({
    'Throat ID': throat_ids,
    'EqRadius': throat_radii,
    'Pore ID #1': throat_pore1,
    'Pore ID #2': throat_pore2,
    'In Solvent Penetration': [t_id in solvent_feasible_throat_ids for t_id in throat_ids]
})
solvent_classification_excel = Path(output_folder) / f'{SAMPLE_NAME}_solvent_throat_classification.xlsx'
solvent_classification_df.to_excel(solvent_classification_excel, index=False)
print(f"  溶剂喉分类结果已保存: {solvent_classification_excel}")

# 保存溶剂渗透连通分量信息
solvent_component_info = []
for comp_id, component in solvent_penetration_components:
    component_pores = list(component)
    solvent_component_info.append({
        'Component ID': comp_id,
        'Component Size': len(component),
        'Pore IDs': ','.join(map(str, sorted(component_pores)))
    })
solvent_component_df = pd.DataFrame(solvent_component_info)
solvent_component_excel = Path(output_folder) / f'{SAMPLE_NAME}_solvent_penetration_components.xlsx'
solvent_component_df.to_excel(solvent_component_excel, index=False)
print(f"  溶剂渗透连通分量信息已保存: {solvent_component_excel}")

# 额外：可视化溶剂渗透网络（参考 analyze_118_pores_and_throats.py）
if n_solvent_penetration > 0:
    print("  创建溶剂渗透网络可视化...")
    fig_solvent = go.Figure()

    # 为每个溶剂渗透连通分量分配颜色
    solvent_colors = ['darkblue', 'darkgreen', 'darkred', 'purple', 'darkorange',
                      'teal', 'maroon', 'navy', 'olive', 'sienna']
    if n_solvent_penetration > len(solvent_colors):
        import colorsys
        solvent_colors = [colorsys.hsv_to_rgb(i / n_solvent_penetration, 0.8, 0.8)
                          for i in range(n_solvent_penetration)]
        solvent_colors = [
            f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
            for c in solvent_colors
        ]

    solvent_penetration_comp_ids = {comp_id for comp_id, _ in solvent_penetration_components}

    # 绘制所有连通分量（溶剂网络）
    for comp_id in range(min(n_solvent_components, 8000)):
        component = solvent_components[comp_id]

        is_penetration = comp_id in solvent_penetration_comp_ids
        if is_penetration:
            idx = list(solvent_penetration_comp_ids).index(comp_id)
            color = solvent_colors[idx % len(solvent_colors)]
            name_prefix = 'Solvent Penetration'
        else:
            color = 'lightgray'
            name_prefix = 'Non-penetration'

        component_throats = []
        for p1 in component:
            for p2 in G_solvent.neighbors(p1):
                if p2 > p1:
                    edge_data = G_solvent[p1][p2]
                    throat_id = edge_data['throat_id']
                    throat_r = edge_data['radius']
                    if p1 in pore_id_to_idx and p2 in pore_id_to_idx:
                        component_throats.append((p1, p2, throat_r))

        if not component_throats:
            continue

        for p1_id, p2_id, throat_r in component_throats:
            p1_idx = pore_id_to_idx[p1_id]
            p2_idx = pore_id_to_idx[p2_id]
            p1_coord = pore_coords[p1_idx]
            p2_coord = pore_coords[p2_idx]

            line_width = throat_r * 0.5 if is_penetration else throat_r * 0.2
            opacity = 0.9 if is_penetration else 0.2

            fig_solvent.add_trace(go.Scatter3d(
                x=[p1_coord[0], p2_coord[0]],
                y=[p1_coord[1], p2_coord[1]],
                z=[p1_coord[2], p2_coord[2]],
                mode='lines',
                line=dict(color=color, width=line_width),
                opacity=opacity,
                showlegend=False,
                name=f'{name_prefix} Component {comp_id} ({len(component)} pores)',
                hovertemplate=(
                    f'<b>Throat</b><br>Radius: {throat_r:.2f} nm<br>'
                    f'Component: {comp_id}<br>'
                    f'{"Solvent Penetration" if is_penetration else "Non-penetration"}'
                    '<extra></extra>'
                )
            ))

    # 添加所有孔隙位置（小点）
    all_pore_ids = set()
    for comp in solvent_components:
        all_pore_ids.update(comp)
    all_pore_mask = np.array([pid in all_pore_ids for pid in pore_ids])
    if np.any(all_pore_mask):
        fig_solvent.add_trace(go.Scatter3d(
            x=pore_coords[all_pore_mask, 0],
            y=pore_coords[all_pore_mask, 1],
            z=pore_coords[all_pore_mask, 2],
            mode='markers',
            marker=dict(size=2, color='black', opacity=0.3),
            name='Pores',
            showlegend=False,
            hovertemplate='<b>Pore</b><extra></extra>'
        ))

    fig_solvent.update_layout(
        title=dict(
            text=(f'{SAMPLE_NAME} Solvent Penetration Network - All Throats'
                  '<br><sub>Connected Components from Left to Right Surface (all throat radii)</sub>'),
            x=0.5
        ),
        scene=dict(
            xaxis_title='X (nm)',
            yaxis_title='Y (nm)',
            zaxis_title='Z (nm)',
            aspectmode='data'
        ),
        width=1200,
        height=800
    )

    solvent_html = Path(output_folder) / f'{SAMPLE_NAME}_solvent_penetration_network.html'
    fig_solvent.write_html(str(solvent_html))
    print(f"  溶剂渗透网络可视化已保存: {solvent_html}")
    print(f"    溶剂渗透连通分量数: {n_solvent_penetration}")
    print(f"    每个分量用不同颜色表示")

# ====================================================================
# 步骤6: 绘制目标蛋白渗透通路图（连接左右外部孔的连通分量）
# ====================================================================
print("\n步骤6: 绘制目标蛋白渗透通路图...")

print(f"  左侧外部孔数: {len(surface_left_pore_ids)}")
print(f"  右侧外部孔数: {len(surface_right_pore_ids)}")

# 先基于溶质网络连通分量找到"连接左右表面"的渗透候选组件
penetration_components_raw = []
for comp_id, component in enumerate(connected_components):
    has_left = bool(component & surface_left_pore_ids)
    has_right = bool(component & surface_right_pore_ids)
    if has_left and has_right:
        penetration_components_raw.append((comp_id, component))
        print(f"    原始连通分量 {comp_id}: {len(component)} 个孔，包含左侧和右侧外部孔")

n_penetration_components_raw = len(penetration_components_raw)
print(f"  找到 {n_penetration_components_raw} 个连接左右两侧外部孔的原始连通分量")

if n_penetration_components_raw == 0:
    n_penetration_components = 0
    print(f"  警告：没有找到连接左右两侧的连通分量")
else:
    # 在这些渗透候选组件上构建子图，并进行"结构盲端"剪枝
    penetration_pore_ids_all = set()
    for _, comp in penetration_components_raw:
        penetration_pore_ids_all.update(comp)

    # 在溶质可通行图 G 上构建只包含渗透候选孔的子图
    G_penetration = nx.Graph()
    for u, v, data in G.edges(data=True):
        if (u in penetration_pore_ids_all) and (v in penetration_pore_ids_all):
            G_penetration.add_edge(u, v, **data)

    # 反复删除度=1且不是左右表面孔的内部节点
    surface_boundary_nodes = surface_left_pore_ids | surface_right_pore_ids
    pruned_once = False
    while True:
        leaf_nodes = [
            n for n, deg in G_penetration.degree()
            if deg == 1 and (n not in surface_boundary_nodes)
        ]
        if not leaf_nodes:
            break
        pruned_once = True
        G_penetration.remove_nodes_from(leaf_nodes)

    if pruned_once:
        print(f"  结构盲端剪枝后渗透子图节点数: {G_penetration.number_of_nodes()}")
        print(f"  结构盲端剪枝后渗透子图喉数: {G_penetration.number_of_edges()}")

    # 在剪枝后的子图上重新识别渗透组件
    connected_components_pen = list(nx.connected_components(G_penetration))
    penetration_components = []
    for comp_id, component in enumerate(connected_components_pen):
        has_left = bool(component & surface_left_pore_ids)
        has_right = bool(component & surface_right_pore_ids)
        if has_left and has_right:
            penetration_components.append((comp_id, component))

    n_penetration_components = len(penetration_components)
    print(f"  剪枝后仍然连接左右两侧外部孔的渗透连通分量数: {n_penetration_components}")

    if n_penetration_components > 0:
        # 保存渗透分量信息
        penetration_info = []
        for comp_id, component in penetration_components:
            component_pores = list(component)
            penetration_info.append({
                'Component ID': comp_id,
                'Component Size': len(component),
                'Pore IDs': ','.join(map(str, sorted(component_pores)))
            })
        penetration_df = pd.DataFrame(penetration_info)
        penetration_excel = Path(output_folder) / f'{SAMPLE_NAME}_penetration_components.xlsx'
        penetration_df.to_excel(penetration_excel, index=False)
        print(f"  渗透连通分量信息已保存: {penetration_excel}")

        # 可视化渗透通路网络（基于剪枝后的 G_penetration）
        print(f"  创建渗透通路网络可视化...")
        fig_penetration = go.Figure()

        penetration_comp_ids = {comp_id for comp_id, _ in penetration_components}

        # 为每个渗透连通分量分配颜色（突出显示）
        penetration_colors = ['red', 'orange', 'yellow', 'green', 'cyan',
                              'blue', 'purple', 'magenta', 'pink']
        if n_penetration_components > len(penetration_colors):
            import colorsys
            penetration_colors = [colorsys.hsv_to_rgb(i / n_penetration_components, 0.8, 0.9)
                                  for i in range(n_penetration_components)]
            penetration_colors = [
                f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
                for c in penetration_colors
            ]

        # 绘制所有连通分量（使用剪枝后的渗透子图）
        for comp_id in range(min(len(connected_components_pen), 8000)):
            component = connected_components_pen[comp_id]

            is_penetration = comp_id in penetration_comp_ids
            if is_penetration:
                idx = list(penetration_comp_ids).index(comp_id)
                color = penetration_colors[idx % len(penetration_colors)]
                name_prefix = 'Penetration'
            else:
                color = 'lightgray'
                name_prefix = 'Non-penetration'

            component_throats = []
            for p1 in component:
                for p2 in G_penetration.neighbors(p1):
                    if p2 > p1:
                        edge_data = G_penetration[p1][p2]
                        throat_id = edge_data['throat_id']
                        throat_r = edge_data['radius']
                        if p1 in pore_id_to_idx and p2 in pore_id_to_idx:
                            component_throats.append((p1, p2, throat_r))

            if not component_throats:
                continue

            for p1_id, p2_id, throat_r in component_throats:
                p1_idx = pore_id_to_idx[p1_id]
                p2_idx = pore_id_to_idx[p2_id]
                p1_coord = pore_coords[p1_idx]
                p2_coord = pore_coords[p2_idx]

                line_width = throat_r * 0.5 if is_penetration else throat_r * 0.3
                opacity = 0.9 if is_penetration else 0.3

                fig_penetration.add_trace(go.Scatter3d(
                    x=[p1_coord[0], p2_coord[0]],
                    y=[p1_coord[1], p2_coord[1]],
                    z=[p1_coord[2], p2_coord[2]],
                    mode='lines',
                    line=dict(color=color, width=line_width),
                    opacity=opacity,
                    showlegend=False,
                    name=f'{name_prefix} Component {comp_id} ({len(component)} pores)',
                    hovertemplate=(
                        f'<b>Throat</b><br>Radius: {throat_r:.2f} nm<br>'
                        f'Component: {comp_id}<br>'
                        f'{"Penetration" if is_penetration else "Non-penetration"}'
                        '<extra></extra>'
                    )
                ))

        # 添加所有孔隙位置（小点，使用剪枝后的渗透子图节点）
        connected_pore_ids = set()
        for comp in connected_components_pen:
            connected_pore_ids.update(comp)
        connected_pore_mask = np.array([pid in connected_pore_ids for pid in pore_ids])
        if np.any(connected_pore_mask):
            fig_penetration.add_trace(go.Scatter3d(
                x=pore_coords[connected_pore_mask, 0],
                y=pore_coords[connected_pore_mask, 1],
                z=pore_coords[connected_pore_mask, 2],
                mode='markers',
                marker=dict(size=2, color='black', opacity=0.3),
                name='Pores',
                showlegend=False,
                hovertemplate='<b>Pore</b><extra></extra>'
            ))

        fig_penetration.update_layout(
            title=dict(
                text=(f'{SAMPLE_NAME} Target Solute Penetration Pathways'
                      '<br><sub>Connected Components from Left to Right Surface (throat radius &gt; '
                      f'{SOLUTE_PASSAGE_GEOMETRY_MIN_NM:.4f} nm = r_s)</sub>'),
                x=0.5
            ),
            scene=dict(
                xaxis_title='X (nm)',
                yaxis_title='Y (nm)',
                zaxis_title='Z (nm)',
                aspectmode='data'
            ),
            width=1200,
            height=800
        )

        penetration_html = Path(output_folder) / f'{SAMPLE_NAME}_penetration_pathways.html'
        fig_penetration.write_html(str(penetration_html))
        print(f"  渗透通路可视化已保存: {penetration_html}")
        print(f"    渗透连通分量数: {n_penetration_components}")
        print(f"    每个分量用不同颜色表示")

print("\n" + "=" * 70)
print("分析完成！")
print("=" * 70)
if alpha_used is not None:
    print(f"\n使用的 Alpha 参数: {alpha_used:.6f} (R = {1.0/alpha_used:.2f} nm)")
    print(f"提示：如果边界识别结果不理想，可以手动调整 alpha 参数")
print(f"\n生成的文件:")
print(f"  1. {classification_excel} - 孔隙分类结果")
print(f"  2. {pores_html} - 孔隙可交互可视化（在浏览器中打开）")
print(f"  3. {throats_classification_excel} - 喉分类结果（溶质通路）")
print(f"  4. {component_excel} - 连通分量信息（溶质通路）")
print(f"  5. {throats_html} - 可联通喉可交互可视化（在浏览器中打开）")
print(f"  6. {solvent_classification_excel} - 溶剂喉分类结果")
print(f"  7. {solvent_component_excel} - 溶剂渗透连通分量信息")

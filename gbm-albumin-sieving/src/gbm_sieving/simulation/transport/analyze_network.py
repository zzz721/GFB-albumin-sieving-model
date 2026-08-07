"""
整体样本版本：分析孔隙和喉道数据（使用 Alpha Shape 方法识别边界孔）。
与子样本版本（sub_*）区分，由 pipeline/run 直接调用本脚本。
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

try:
    import alphashape
except ImportError:
    print("警告：需要安装 alphashape 库: pip install alphashape")
    print("  正在尝试使用凸包作为 fallback...")
    alphashape = None

from scipy.spatial import cKDTree, ConvexHull

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    def _tqdm(iterable=None, total=None, desc=None):
        return iterable


def _df_to_excel_safe(df: pd.DataFrame, path: Path, *, index: bool = False, **kwargs) -> None:
    """写 xlsx；若目标被占用（常见于 Excel 未关闭）则提示后原样抛出 PermissionError。"""
    try:
        df.to_excel(path, index=index, **kwargs)
    except PermissionError:
        print(
            f"  错误：无法写入 {path.resolve()}\n"
            "  若该文件在 Excel 或其他程序中打开，请先关闭后再运行。"
        )
        raise


# 溶质的水合半径（单位：nm）
SOLUTE_HYDRATED_RADIUS = 3.55

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
    default="results_core_overall",
    help="输出结果文件夹（整体样本基础分析结果）。",
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
    "--solute-radius-nm",
    type=float,
    default=SOLUTE_HYDRATED_RADIUS,
    help="Solute hydrated radius in nm; default is albumin radius 3.55 nm.",
)
parser.add_argument(
    "--normal-vector",
    nargs=2,
    type=float,
    default=None,
    metavar=("NX", "NY"),
    help="法向量方向（可选），传入两个数 NX NY。若提供，将作为渗透方向。",
)
parser.add_argument(
    "--surface-distance-tol-frac",
    type=float,
    default=0.05,
    help="用于二次过滤表面孔的几何容差，占整体厚度 T 的比例，默认 0.05（即 0.05*T）。",
)
parser.add_argument(
    "--no-html",
    action="store_true",
    help="跳过交互式 HTML 可视化输出；批量模拟时建议开启以节省时间和磁盘空间。",
)

_args, _unknown = parser.parse_known_args()
SAMPLE_NAME = _args.sample_name
SAMPLE_DIR = Path(_args.sample_dir)
output_folder_base = Path(_args.output_dir)
RADIUS_DELTA = _args.radius_delta
SOLUTE_HYDRATED_RADIUS = float(_args.solute_radius_nm)
if SOLUTE_HYDRATED_RADIUS <= 0.0:
    raise SystemExit("--solute-radius-nm must be positive")
NORMAL_VECTOR_VALS = getattr(_args, "normal_vector", None)
SURFACE_DISTANCE_TOL_FRAC = float(getattr(_args, "surface_distance_tol_frac", 0.05))
NO_HTML = bool(getattr(_args, "no_html", False))

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
        print(f"    Alpha Shape 失败: {e}，退回凸包")
        hull = ConvexHull(pore_coords)
        boundary_mask = np.zeros(n, dtype=bool)
        boundary_mask[hull.vertices] = True
        return boundary_mask, None, None

    print("    Alpha Shape 成功，开始映射表面顶点到原始孔索引 ...")
    precision = 6
    coord_map = {}
    for i, coord in enumerate(pore_coords):
        key = tuple(np.round(coord, precision))
        if key not in coord_map:
            coord_map[key] = []
        coord_map[key].append(i)

    boundary_indices = set()
    for v in alpha_mesh.vertices:
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


def identify_faces_by_ray_casting(pore_coords: np.ndarray,
                                  pore_radii: np.ndarray,
                                  boundary_mask: np.ndarray,
                                  alpha_mesh,
                                  sample_name: str,
                                  main_axis: int = 0):
    """
    通过射线投射（模拟圆柱）识别渗透方向的两个面（u_min/u_max），并处理边和角。
    """
    if alpha_mesh is None:
        print("  [射线投射] alpha_mesh 不可用，跳过射线投射方法")
        return None
    
    print(f"  [射线投射] 准备开始，渗透方向轴索引: {main_axis} ({['X','Y','Z'][main_axis]})")
    
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
        print(f"    检查 {face_name} 面...", flush=True)
        visible_count = 0
        
        if face_name == 'u_min':
            ray_start_val = axis_min - ray_offset
        else:
            ray_start_val = axis_max + ray_offset
        
        # 使用 tqdm 显示表面孔遍历进度
        iterator = _tqdm(range(n_surface), total=n_surface, desc=f"      射线投射 {face_name}")
        for i in iterator:
            surface_point = surface_coords[i]
            
            radius = 0.55
            
            # 生成一组射线来模拟圆柱
            rays_to_check = []
            
            center_origin = np.zeros(3)
            center_origin[main_axis] = ray_start_val
            center_origin[u_axis_idx] = surface_point[u_axis_idx]
            center_origin[v_axis_idx] = surface_point[v_axis_idx]
            rays_to_check.append(center_origin)
            
            offsets = [
                (0, radius), (0, -radius),  # v 方向偏移
                (radius, 0), (-radius, 0)   # u 方向偏移
            ]
            
            for du, dv in offsets:
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
        
        print(f"    {face_name} 面可见孔数: {visible_count}")
    
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
    
    outlier_threshold = 15.0 # 收紧阈值，过滤深坑
    removed_min = 0
    removed_max = 0
    
    for (gu, gv), indices in pore_grid.items():
        # 处理 u_min 面
        min_indices = [idx for idx in indices if 'u_min' in face_labels[idx]]
        if len(min_indices) > 0:
            vals = surface_coords[min_indices, main_axis]
            # 使用 0.10 分位数作为基准，避免受个别极端外突点的影响
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
            # 使用 0.90 分位数作为基准
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
        if max_gap < 20.0:
            return 0
            
        split_idx = max_gap_idx + 1
        n_total = len(vals)
        count_removed = 0
        
        # 只有当保留的主体部分足够大（>0.30）时才执行切割，防止误删
        if face_type == 'u_min':
            # u_min (左面): 应该保留坐标较小的部分 (0 ~ split_idx)
            # 剔除坐标较大的部分 (split_idx ~ end)，这些通常是侧边或深层孔
            if split_idx > n_total * 0.3:
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
    def filter_by_neighbor_distance(indices, k=10):
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
        threshold = max(median_dist + 2 * mad, median_dist * 2)
        
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
    # 真实半径 HTML 在几何过滤之后由主流程写入，与 pore_classification 一致
    
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


def write_surface_pores_real_radius_html_filtered(
    pore_coords: np.ndarray,
    pore_radii: np.ndarray,
    boundary_mask: np.ndarray,
    is_surface_x_left: np.ndarray,
    is_surface_x_right: np.ndarray,
    sample_name: str,
    output_dir: Path,
) -> None:
    """Alpha 边界孔：按几何过滤后的 u_min/u_max 着色，与 pore_classification.xlsx 一致。"""
    if NO_HTML:
        print("  [可视化] 跳过真实半径 HTML（--no-html）")
        return
    surface_indices = np.where(boundary_mask)[0]
    if len(surface_indices) == 0:
        print("  [可视化] 跳过真实半径 HTML：无边界孔")
        return
    surface_coords = pore_coords[surface_indices]
    radii_subset = pore_radii[surface_indices]
    left_mask = is_surface_x_left[surface_indices]
    right_mask = is_surface_x_right[surface_indices]
    gray_mask = ~(left_mask | right_mask)
    fig_radius = go.Figure()
    # 灰点在下层，红/蓝在上层便于辨认
    layers = [
        (gray_mask, 'lightgray', '未分配 (内部/边缘)', 0.1),
        (left_mask, 'red', '左表面 (u_min，几何过滤后)', 0.9),
        (right_mask, 'blue', '右表面 (u_max，几何过滤后)', 0.9),
    ]
    for mask, color, name, opacity in layers:
        if not np.any(mask):
            continue
        fig_radius.add_trace(go.Scatter3d(
            x=surface_coords[mask, 0],
            y=surface_coords[mask, 1],
            z=surface_coords[mask, 2],
            mode='markers',
            marker=dict(
                size=radii_subset[mask] * 1.0,
                color=color,
                opacity=opacity,
                line=dict(width=0),
                sizemode='diameter',
                sizeref=0.5,
            ),
            name=name,
            text=[f"R={r:.2f}" for r in radii_subset[mask]],
            hoverinfo='text+name+x+y+z',
        ))
    fig_radius.update_layout(
        title=(
            f"{sample_name} 表面孔 - 半径可视化（几何过滤后，与 classification 一致）"
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
        ),
        width=1000,
        height=800,
    )
    radius_html = output_dir / f"{sample_name}_surface_pores_real_radius.html"
    fig_radius.write_html(str(radius_html))
    print(f"  [可视化] 真实半径 HTML 已保存（几何过滤后）: {radius_html}")


# ====================================================================
# 步骤2.5: 使用自动厚度校准 + 射线投射法识别表面孔
# ====================================================================
print("\n步骤2.5: 识别渗透方向并提取表面孔 (射线投射法)...")

# 在合成 GBM 模型中，固定渗透方向为 Z 轴（index=2），
# 不再根据厚度文件或法向量自动选择，避免影响 Phase1–3 的几何约定。
main_axis = 2  # 0:X, 1:Y, 2:Z
pore_coords_for_surface = pore_coords
axis_label = "Z"
print("  [渗透方向] 已固定为 Z 轴 (main_axis=2)，忽略厚度文件与法向量设置")
print(f"  最终确定的渗透方向: {axis_label} (轴索引={main_axis})")

# --- 子步骤 2: 获取 Alpha Shape Mesh ---
print(f"  [Alpha Shape] 生成表面网格...")
boundary_mask, alpha_used, alpha_mesh = identify_surface_by_alphashape(
    pore_coords_for_surface, alpha=None
)

# --- 子步骤 3: 射线投射识别面 ---
print(f"  [射线投射] 识别表面孔...")
face_labels = identify_faces_by_ray_casting(
    pore_coords=pore_coords_for_surface,
    pore_radii=pore_radii,
    boundary_mask=boundary_mask,
    alpha_mesh=alpha_mesh,
    sample_name=SAMPLE_NAME,
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

# 射线解析完成后立即二次几何过滤（再写入 Excel）：心到该侧端面距离 <= r + eps
n_geom_left_in = int(np.sum(is_surface_x_left))
n_geom_right_in = int(np.sum(is_surface_x_right))
dropped_left = 0
dropped_right = 0
if (np.any(is_surface_x_left) or np.any(is_surface_x_right)) and SURFACE_DISTANCE_TOL_FRAC > 0:
    # 使用渗透方向对应的坐标轴（main_axis）来定义厚度与距离
    axis = main_axis  # 0:X, 1:Y, 2:Z
    coords_axis = pore_coords[:, axis]
    coord_min, coord_max = float(coords_axis.min()), float(coords_axis.max())
    T = coord_max - coord_min if coord_max > coord_min else 0.0
    if T > 0:
        # 与 sub_calculate 一致：max(1 nm, frac·膜厚)，避免厚膜时仅 0.05*T 仍过大
        eps = max(1.0, SURFACE_DISTANCE_TOL_FRAC * T)
        # 入口侧（u_min）：距离 = coord - coord_min
        left_idx = np.where(is_surface_x_left)[0]
        if len(left_idx) > 0:
            d_left = coords_axis[left_idx] - coord_min
            r_left = pore_radii[left_idx]
            keep_left = d_left <= (r_left + eps)
            dropped_left = int(np.sum(~keep_left))
            if dropped_left > 0:
                is_surface_x_left[left_idx[~keep_left]] = False
        # 出口侧（u_max）：距离 = coord_max - coord
        right_idx = np.where(is_surface_x_right)[0]
        if len(right_idx) > 0:
            d_right = coord_max - coords_axis[right_idx]
            r_right = pore_radii[right_idx]
            keep_right = d_right <= (r_right + eps)
            dropped_right = int(np.sum(~keep_right))
            if dropped_right > 0:
                is_surface_x_right[right_idx[~keep_right]] = False
        # 更新汇总标记
        is_surface_x = is_surface_x_left | is_surface_x_right
        print("  [几何过滤] 射线判定后立即检验（孔心到该侧端面距离 d 须满足 d <= r+eps）：")
        print(f"    膜厚 T={T:.3f} nm，eps=max(1 nm, {SURFACE_DISTANCE_TOL_FRAC:.3f}·T) = {eps:.3f} nm")
        print(
            f"    u_min 侧：射线候选 {n_geom_left_in} 个 → 剔除 {dropped_left} 个 → 保留 {int(np.sum(is_surface_x_left))} 个"
        )
        print(
            f"    u_max 侧：射线候选 {n_geom_right_in} 个 → 剔除 {dropped_right} 个 → 保留 {int(np.sum(is_surface_x_right))} 个"
        )
    else:
        print("  [几何过滤] 跳过：渗透轴坐标跨度 T≈0")
elif SURFACE_DISTANCE_TOL_FRAC <= 0:
    print("  [几何过滤] 跳过：SURFACE_DISTANCE_TOL_FRAC<=0（未做心到端面检验）")

n_surface_x_left = np.sum(is_surface_x_left)
n_surface_x_right = np.sum(is_surface_x_right)
n_surface_x_total = np.sum(is_surface_x)

print(f"  识别完成（射线 + 几何二次筛选，将写入 Excel）:")
print(f"    渗透入口孔 (u_min): {n_surface_x_left} 个 (fraction={n_surface_x_left/len(pore_coords):.3f})")
print(f"    渗透出口孔 (u_max): {n_surface_x_right} 个 (fraction={n_surface_x_right/len(pore_coords):.3f})")
print(f"    总表面孔数: {n_surface_x_total} 个")

# 创建分类数据框
classification_df = pd.DataFrame({
    'Pore ID': pore_ids,
    'Volume': pore_volumes,
    'EqRadius': pore_radii,
    'X Coord': pore_coords[:, 0],
    'Y Coord': pore_coords[:, 1],
    'Z Coord': pore_coords[:, 2],
    'Penetration Axis Index': main_axis,
    'Penetration Axis Label': axis_label,
    'Is Surface X': is_surface_x,
    'Is Surface X Left': is_surface_x_left,
    'Is Surface X Right': is_surface_x_right
})
if alpha_used is not None:
    classification_df['Alpha Used'] = alpha_used
classification_excel = Path(output_folder) / f'{SAMPLE_NAME}_pore_classification.xlsx'
_df_to_excel_safe(classification_df, classification_excel, index=False)
print(f"  分类结果已保存: {classification_excel}")

write_surface_pores_real_radius_html_filtered(
    pore_coords,
    pore_radii,
    boundary_mask,
    is_surface_x_left,
    is_surface_x_right,
    SAMPLE_NAME,
    output_folder,
)

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
if NO_HTML:
    print("  孔隙可视化 HTML 已跳过（--no-html）")
else:
    fig_pores.write_html(str(pores_html))
    print(f"  孔隙可视化已保存: {pores_html}")
print(f"    总孔数: {len(pore_ids)} 个（按半径着色）")
print(f"    X轴左侧表面孔: {n_surface_x_left} 个（红色球体）")
print(f"    X轴右侧外部孔: {n_surface_x_right} 个（蓝色球体）")

# ====================================================================
# 步骤4: 过滤无法通过溶质的喉
# ====================================================================
print("\n步骤4: 过滤无法通过溶质的喉...")

# 过滤：只保留半径 >= 3.55nm 的喉
feasible_throat_mask = throat_radii >= SOLUTE_HYDRATED_RADIUS
n_feasible = np.sum(feasible_throat_mask)
n_blocked = np.sum(~feasible_throat_mask)

print(f"  可通过的喉（半径 >= {SOLUTE_HYDRATED_RADIUS} nm）: {n_feasible} 个 (fraction={n_feasible/len(throat_radii):.3f})")
print(f"  不可通过的喉（半径 < {SOLUTE_HYDRATED_RADIUS} nm）: {n_blocked} 个 (fraction={n_blocked/len(throat_radii):.3f})")

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
        text=f'Feasible Throat Network - Interactive 3D Visualization<br><sub>Different colors = Different connected components (throat radius >= {SOLUTE_HYDRATED_RADIUS} nm)</sub>',
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
if NO_HTML:
    print("  可联通喉可视化 HTML 已跳过（--no-html）")
else:
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
_df_to_excel_safe(throats_classification_df, throats_classification_excel, index=False)
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
_df_to_excel_safe(component_df, component_excel, index=False)
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
        # 计算喉长度
        p1_idx = pore_id_to_idx[p1]
        p2_idx = pore_id_to_idx[p2]
        p1_coord = pore_coords[p1_idx]
        p2_coord = pore_coords[p2_idx]
        length = np.linalg.norm(p2_coord - p1_coord)
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
_df_to_excel_safe(solvent_classification_df, solvent_classification_excel, index=False)
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
_df_to_excel_safe(solvent_component_df, solvent_component_excel, index=False)
print(f"  溶剂渗透连通分量信息已保存: {solvent_component_excel}")

# 额外：可视化溶剂渗透网络（仅展示沿 Z 轴最薄/中等/最厚三个“cell”）
# 暂时关闭 3-layers 可视化，避免额外的大体积 HTML 输出
if False and n_solvent_penetration > 0:
    print("  创建溶剂渗透网络可视化（三个代表性厚度 cell）...")
    z_coords = pore_coords[:, 2]
    z_min, z_max = float(z_coords.min()), float(z_coords.max())
    dz = z_max - z_min
    if dz <= 0:
        print("  警告：Z 轴范围为 0，跳过溶剂渗透网络可视化")
    else:
        # 选择 Z 方向上“薄 / 中 / 厚”三层：0.10, 0.50, 0.90 分位附近
        z_q10 = z_min + 0.1 * dz
        z_q50 = z_min + 0.5 * dz
        z_q90 = z_min + 0.9 * dz
        slab_half = 0.05 * dz  # 每层厚度为总厚度的 0.10
        masks = [
            np.abs(z_coords - z_q10) <= slab_half,
            np.abs(z_coords - z_q50) <= slab_half,
            np.abs(z_coords - z_q90) <= slab_half,
        ]
        keep_pore_mask = masks[0] | masks[1] | masks[2]
        keep_pore_ids = set(pore_ids[keep_pore_mask])
        if not keep_pore_ids:
            print("  警告：代表性厚度 cell 中无孔，跳过溶剂渗透网络可视化")
        else:
            print(
                f"    代表性厚度 cell: "
                f"薄层({np.sum(masks[0])} 孔), 中层({np.sum(masks[1])} 孔), 厚层({np.sum(masks[2])} 孔)"
            )

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
            
            # 仅绘制前三个代表性厚度 cell 中的喉
            max_components = min(n_solvent_components, 8000)
            iterator = _tqdm(
                range(max_components),
                total=max_components,
                desc="  溶剂渗透网络可视化",
            )
            for comp_id in iterator:
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
                    if p1 not in keep_pore_ids:
                        continue
                    for p2 in G_solvent.neighbors(p1):
                        if p2 <= p1 or p2 not in keep_pore_ids:
                            continue
                        edge_data = G_solvent[p1][p2]
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

            # 添加代表性 cell 中的孔位置（小点）
            all_pore_ids = set()
            for comp in solvent_components:
                all_pore_ids.update(comp)
            all_pore_mask = np.array(
                [(pid in all_pore_ids) and (pid in keep_pore_ids) for pid in pore_ids]
            )
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
                    text=(f'{SAMPLE_NAME} Solvent Penetration Network - 3 Z-slices'
                          '<br><sub>Thin / Middle / Thick layers along Z</sub>'),
                    x=0.5
                ),
                scene=dict(
                    xaxis_title='X (nm)',
                    yaxis_title='Y (nm)',
                    zaxis_title='Z (nm)',
                    aspectmode='data'
                ),
                width=1000,
                height=700
            )

            solvent_html = Path(output_folder) / f'{SAMPLE_NAME}_solvent_penetration_network_3layers.html'
            if NO_HTML:
                print("  溶剂渗透网络可视化 HTML 已跳过（--no-html）")
            else:
                fig_solvent.write_html(str(solvent_html))
                print(f"  溶剂渗透网络可视化已保存: {solvent_html}")
            print("    仅显示 Z 方向最薄/中等/最厚三个代表性 cell 的喉与孔")

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
        _df_to_excel_safe(penetration_df, penetration_excel, index=False)
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
                      '<br><sub>Connected Components from Left to Right Surface (throat radius >= '
                      f'{SOLUTE_HYDRATED_RADIUS} nm)</sub>'),
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
        if NO_HTML:
            print("  渗透通路可视化 HTML 已跳过（--no-html）")
        else:
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
if not NO_HTML:
    print(f"  2. {pores_html} - 孔隙可交互可视化（在浏览器中打开）")
print(f"  3. {throats_classification_excel} - 喉分类结果（溶质通路）")
print(f"  4. {component_excel} - 连通分量信息（溶质通路）")
if not NO_HTML:
    print(f"  5. {throats_html} - 可联通喉可交互可视化（在浏览器中打开）")
print(f"  6. {solvent_classification_excel} - 溶剂喉分类结果")
print(f"  7. {solvent_component_excel} - 溶剂渗透连通分量信息")

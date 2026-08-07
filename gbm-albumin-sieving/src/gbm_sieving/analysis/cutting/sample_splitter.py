"""
交互式样本切割工具（单次切割版本）
1. 加载样本数据（孔和喉）
2. 投影到X-Y平面并可视化
3. 用户点击三个点定义切割平面
4. 计算法向量和第二个起点
5. 显示切割结果（两条直线和两个起点）
6. 保存子样本

用法（建议在 肾脏 目录下运行，或传入绝对路径）:
    python analyze_and_calculate_2/interactive/interactive_sample_splitter.py
    python analyze_and_calculate_2/interactive/interactive_sample_splitter.py --sample-name AS317
    默认输入: 肾脏/throats_and_pores_xlsx  默认输出: 肾脏/data_subsamples
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import json

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def project_to_xy(pore_coords):
    """投影到X-Y平面（只取X和Y坐标）"""
    return pore_coords[:, [0, 1]]


def calculate_normal_vector(point1, point2):
    """
    计算边界线的法向量（在X-Y平面内）
    
    参数:
        point1, point2: 两个点的坐标 [x, y]
    
    返回:
        normal_vec: 归一化的法向量 [nx, ny]（顺时针旋转90度）
    """
    dir_vec = np.array(point2) - np.array(point1)
    # 法向量：顺时针旋转90度
    normal_vec = np.array([dir_vec[1], -dir_vec[0]])
    # 归一化
    norm = np.linalg.norm(normal_vec)
    if norm > 1e-10:
        normal_vec = normal_vec / norm
    else:
        # 如果两点重合，使用默认法向量
        normal_vec = np.array([1.0, 0.0])
    return normal_vec


def calculate_z_thickness(pore_coords):
    """计算Z轴方向的厚度"""
    z_min = pore_coords[:, 2].min()
    z_max = pore_coords[:, 2].max()
    thickness = z_max - z_min
    return thickness, z_min, z_max


def point_to_plane_distance(point_2d, ref_point, direction_vec):
    """
    计算点在X-Y平面内沿指定方向向量的投影距离
    
    参数:
        point_2d: 点的X-Y坐标 [x, y]
        ref_point: 参考点（边界线上的一点）[x, y]
        direction_vec: 方向向量 [dx, dy]（用于计算投影的方向）
    
    返回:
        distance: 沿方向向量的投影距离（有符号）
    """
    return np.dot(np.array(point_2d) - np.array(ref_point), direction_vec)


def split_pores_by_two_lines(pore_coords, pore_data, start_point1, start_point2, boundary_unit, line_spacing):
    """
    根据两条平行直线分割孔（保留两条直线之间的区域）
    
    注意：所有判断都在X-Y平面投影上进行，不考虑Z轴坐标，这是一个纯二维问题。
    
    切割逻辑：
    - 第一条直线：通过start_point1，沿着法向量方向（垂直于边界线）
    - 第二条直线：通过start_point2，沿着法向量方向（垂直于边界线）
    - 保留：在两条直线之间的孔（到两条直线的垂直距离都小于等于直线间距）
    - 切掉：在两条直线之外的孔
    
    判断方法（在X-Y平面投影上）：
    - 将所有孔的3D坐标投影到X-Y平面（只取X、Y坐标，忽略Z坐标）
    - 计算每个孔在X-Y平面上的投影点到两条直线的垂直距离（沿边界线方向，即垂直于切割直线的方向）
    - 如果点到两条直线的垂直距离都小于等于直线间距，说明在两条直线之间
    - 否则在两条直线之外
    
    参数:
        pore_coords: 所有孔的3D坐标 [N, 3]
        pore_data: 孔的DataFrame
        start_point1: 第一个起点（第一条直线的位置）[x, y]（X-Y平面坐标）
        start_point2: 第二个起点（第二条直线的位置）[x, y]（X-Y平面坐标）
        boundary_unit: 边界线方向的单位向量 [bx, by]（用于计算点到直线的垂直距离）
        line_spacing: 两条直线之间的垂直距离（Z轴厚度）
    
    返回:
        removed_pores_mask: 被切掉的孔的布尔掩码
        kept_pores_mask: 保留的孔的布尔掩码（在两条直线之间）
    """
    # 投影到X-Y平面（只取X、Y坐标，忽略Z坐标）
    pore_coords_2d = project_to_xy(pore_coords)
    
    # 计算每个孔在X-Y平面上的投影点到第一条直线的垂直距离（沿边界线方向）
    distances1 = np.array([
        point_to_plane_distance(pore_coords_2d[i], start_point1, boundary_unit)
        for i in range(len(pore_coords))
    ])
    
    # 计算每个孔在X-Y平面上的投影点到第二条直线的垂直距离（沿边界线方向）
    distances2 = np.array([
        point_to_plane_distance(pore_coords_2d[i], start_point2, boundary_unit)
        for i in range(len(pore_coords))
    ])
    
    # 计算每个点到两条直线的垂直距离（绝对值）
    abs_dist1 = np.abs(distances1)
    abs_dist2 = np.abs(distances2)
    
    # 两条直线之间的垂直距离就是Z轴厚度
    line_distance = line_spacing  # 直接使用Z轴厚度作为两条直线之间的垂直距离
    
    # 判断逻辑：如果一个点到两条直线的垂直距离都小于等于直线间的垂直距离，说明在两条直线之间
    kept_pores_mask = (abs_dist1 <= line_distance) & (abs_dist2 <= line_distance)
    
    removed_pores_mask = ~kept_pores_mask
    
    return removed_pores_mask, kept_pores_mask


def filter_throats_by_pores(throat_data, kept_pore_ids):
    """
    根据保留的孔过滤喉
    只保留连接的两个孔都在保留集合中的喉
    
    参数:
        throat_data: 喉的DataFrame
        kept_pore_ids: 保留的孔的ID集合
    
    返回:
        kept_throats_mask: 保留的喉的布尔掩码
    """
    kept_pore_set = set(kept_pore_ids)
    kept_throats_mask = []
    
    for idx, row in throat_data.iterrows():
        pore1_id = row['Pore ID #1']
        pore2_id = row['Pore ID #2']
        
        # 检查两个孔是否都在保留集合中
        in_kept = (pore1_id in kept_pore_set) and (pore2_id in kept_pore_set)
        kept_throats_mask.append(in_kept)
    
    return np.array(kept_throats_mask)


def save_sub_sample(pore_data, throat_data, pore_mask, throat_mask, 
                    sample_name, output_dir, normal_vec=None):
    """
    保存子样本到Excel文件，并记录法向量方向（渗透方向）
    
    参数:
        pore_data: 孔的DataFrame
        throat_data: 喉的DataFrame
        pore_mask: 孔的布尔掩码
        throat_mask: 喉的布尔掩码
        sample_name: 样本名（可能包含子样本后缀）
        output_dir: 输出文件夹
        normal_vec: 法向量 [nx, ny]（用于后续的渗透方向），如果为None则不保存
    """
    output_dir = Path(output_dir)
    # 提取基础样本名（例如从 "118_sub1" 提取 "118"）
    base_sample_name = sample_name.split('_sub')[0]
    # 创建以样本名字命名的子文件夹
    sample_output_dir = output_dir / base_sample_name
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 筛选数据
    sub_pores = pore_data[pore_mask].copy()
    sub_throats = throat_data[throat_mask].copy()
    
    # 生成文件名（保存到子文件夹中）
    pore_filename = sample_output_dir / f"{sample_name}_pores.xlsx"
    throat_filename = sample_output_dir / f"{sample_name}_throats.xlsx"
    
    # 保存
    sub_pores.to_excel(pore_filename, index=False)
    sub_throats.to_excel(throat_filename, index=False)
    
    print(f"  样本已保存:")
    print(f"    孔: {pore_filename} ({len(sub_pores)} 个孔)")
    print(f"    喉: {throat_filename} ({len(sub_throats)} 个喉)")
    
    # 始终写入 metadata：有法向量则写 nx/ny；否则写 null（便于下游识别「剩余区域」子样本）
    metadata_filename = sample_output_dir / f"{sample_name}_metadata.json"
    if normal_vec is not None:
        metadata = {
            "sample_name": sample_name,
            "normal_vector": {
                "nx": float(normal_vec[0]),
                "ny": float(normal_vec[1]),
            },
            "normal_vector_description": "法向量方向，用于后续的渗透方向分析",
            "coordinate_system": "X-Y平面投影",
        }
        print(f"    元数据: {metadata_filename} (法向量: [{normal_vec[0]:.4f}, {normal_vec[1]:.4f}])")
    else:
        metadata = {
            "sample_name": sample_name,
            "normal_vector": None,
            "normal_vector_description": "无切割法向量（通常为迭代剩余区域）",
            "coordinate_system": "X-Y平面投影",
        }
        print(f"    元数据: {metadata_filename} (无法向量，剩余区域)")
    with open(metadata_filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return pore_filename, throat_filename


def interactive_select_cut_plane():
    """
    交互式选择切割平面（用户点击三个点）
    
    返回:
        (point1, point2, start_point1, should_continue): 
            - point1, point2: 定义边界线的两个点
            - start_point1: 第一个起点（第一条直线的位置）
            - should_continue: 是否继续切割（False表示结束）
            如果用户取消则返回 None, None, None, False
    
    注意：第二个起点会根据第一个起点沿着边界线方向移动Z轴厚度距离自动计算
    """
    while True:
        print("\n" + "="*70)
        print("交互式切割平面选择")
        print("="*70)
        print("请在图中点击：")
        print("  - 第一个点：边界线的起点")
        print("  - 第二个点：边界线的终点（定义边界线方向）")
        print("  - 第三个点：第一个起点（第一条直线的位置）")
        print("  - 只点击三个点后关闭窗口：确认并继续切割")
        print("  - 点击第四个点：结束切割")
        print("  注意：第二个起点将自动计算（第一个起点沿边界线方向移动Z轴厚度距离）")
        print("="*70)
        
        # 使用ginput获取最多4个点（3个定义切割，第4个用于结束）
        points = plt.ginput(4, timeout=0, show_clicks=True)
        
        if len(points) < 3:
            print("错误：需要至少点击三个点")
            continue
        
        point1 = np.array(points[0])
        point2 = np.array(points[1])
        start_point1 = np.array(points[2])
        
        print(f"\n选择的点:")
        print(f"  边界线起点: ({point1[0]:.2f}, {point1[1]:.2f})")
        print(f"  边界线终点: ({point2[0]:.2f}, {point2[1]:.2f})")
        print(f"  第一个起点: ({start_point1[0]:.2f}, {start_point1[1]:.2f})")
        
        # 如果点击了第四个点，表示结束切割
        if len(points) >= 4:
            print("  检测到第四个点击，结束切割")
            return point1, point2, start_point1, False
        
        # 只有三个点，确认选择并继续
        print("  确认选择，继续切割！")
        return point1, point2, start_point1, True


def list_available_samples(sample_dir):
    """
    列出可用样本（查找所有 *_pores.xlsx 文件）
    
    返回:
        available_samples: 样本名列表
    """
    sample_dir = Path(sample_dir)
    if not sample_dir.exists():
        return []
    
    # 查找所有 *_pores.xlsx 文件
    pore_files = sorted(sample_dir.glob("*_pores.xlsx"))
    available_samples = []
    
    for pore_file in pore_files:
        # 提取样本名（去掉 _pores.xlsx 后缀）
        sample_name = pore_file.stem.replace('_pores', '')
        # 检查对应的 throats 文件是否存在
        throat_file = sample_dir / f"{sample_name}_throats.xlsx"
        if throat_file.exists():
            available_samples.append(sample_name)
    
    return available_samples


def interactive_select_sample(sample_dir):
    """
    交互式选择样本
    
    返回:
        sample_name: 选择的样本名，如果用户取消则返回 None
    """
    available_samples = list_available_samples(sample_dir)
    
    if len(available_samples) == 0:
        print(f"错误：在文件夹 {sample_dir} 中未找到任何样本文件（需要 *_pores.xlsx 和对应的 *_throats.xlsx）")
        return None
    
    print("\n" + "="*70)
    print("可用样本列表")
    print("="*70)
    for i, sample_name in enumerate(available_samples, 1):
        print(f"  {i}. {sample_name}")
    print("="*70)
    
    while True:
        try:
            choice = input(f"\n请选择样本 (1-{len(available_samples)})，或直接输入样本名: ").strip()
            
            # 如果输入的是数字，按索引选择
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_samples):
                    selected_sample = available_samples[idx]
                    print(f"已选择样本: {selected_sample}")
                    return selected_sample
                else:
                    print(f"错误：请输入 1-{len(available_samples)} 之间的数字")
                    continue
            
            # 如果直接输入了样本名，检查是否存在
            if choice in available_samples:
                print(f"已选择样本: {choice}")
                return choice
            else:
                print(f"错误：样本 '{choice}' 不存在。请从列表中选择。")
                continue
                
        except KeyboardInterrupt:
            print("\n\n用户取消选择")
            return None
        except Exception as e:
            print(f"错误：{e}")
            continue


def main():
    script_dir = Path(__file__).resolve().parent
    data_root = script_dir.parent.parent  # 肾脏
    parser = argparse.ArgumentParser(
        description="交互式样本切割工具：将样本沿用户定义的两条平行直线切割成子样本（单次切割版本）"
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        default=None,
        help="样本名字，例如 118。如果不提供，将列出可用样本供选择。",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default=str(data_root / "throats_and_pores_xlsx"),
        help="样本所在文件夹（默认: 肾脏/throats_and_pores_xlsx）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(data_root / "data_subsamples"),
        help="输出文件夹（子样本数据），默认: 肾脏/data_subsamples",
    )
    args = parser.parse_args()
    sample_dir = Path(args.sample_dir)
    if not sample_dir.is_absolute():
        sample_dir = data_root / args.sample_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = data_root / args.output_dir
    
    # 如果没有提供样本名，交互式选择
    if args.sample_name is None:
        selected_sample = interactive_select_sample(sample_dir)
        if selected_sample is None:
            print("未选择样本，程序退出")
            sys.exit(1)
        sample_name = selected_sample
    else:
        sample_name = args.sample_name
    
    # 加载数据
    print("\n" + "="*70)
    print("加载样本数据")
    print("="*70)
    
    pores_file = sample_dir / f"{sample_name}_pores.xlsx"
    throats_file = sample_dir / f"{sample_name}_throats.xlsx"
    
    if not pores_file.exists():
        print(f"错误：未找到文件 {pores_file}")
        sys.exit(1)
    if not throats_file.exists():
        print(f"错误：未找到文件 {throats_file}")
        sys.exit(1)
    
    df_pores = pd.read_excel(pores_file)
    df_throats = pd.read_excel(throats_file)
    
    print(f"  已加载 {len(df_pores)} 个孔")
    print(f"  已加载 {len(df_throats)} 个喉")
    
    # 提取坐标
    pore_coords = df_pores[['X Coord', 'Y Coord', 'Z Coord']].values
    pore_ids = df_pores['Pore ID'].values
    
    # 计算Z轴厚度
    z_thickness, z_min, z_max = calculate_z_thickness(pore_coords)
    print(f"\n  Z轴坐标范围: [{z_min:.2f}, {z_max:.2f}]")
    print(f"  Z轴厚度: {z_thickness:.2f} nm")
    
    # 投影到X-Y平面
    pore_coords_2d = project_to_xy(pore_coords)
    
    # 初始化：当前数据就是全部数据
    current_pore_data = df_pores.copy()
    current_throat_data = df_throats.copy()
    current_pore_coords = pore_coords.copy()
    current_pore_ids = pore_ids.copy()
    current_pore_coords_2d = pore_coords_2d.copy()
    cut_count = 0
    
    # 保存所有子样本和已移除的部分（用于可视化）
    all_sub_samples = []
    all_removed_coords_2d = []  # 存储所有已移除部分的2D坐标（用于可视化）
    all_removed_coords_z = []    # 存储所有已移除部分的Z坐标
    
    # 循环切割，直到用户选择结束
    while True:
        cut_count += 1
        print("\n" + "="*70)
        print(f"切割轮次 {cut_count}")
        print("="*70)
        print(f"当前剩余孔数: {len(current_pore_data)}")
        print(f"当前剩余喉数: {len(current_throat_data)}")
        
        if len(current_pore_data) == 0:
            print("  没有剩余部分，结束切割")
            break
        
        # 显示当前剩余部分
        fig_current, ax_current = plt.subplots(figsize=(10, 8))
        
        # 绘制之前已移除的部分（浅灰色，半透明）
        if len(all_removed_coords_2d) > 0:
            all_removed_2d = np.array(all_removed_coords_2d)
            ax_current.scatter(
                all_removed_2d[:, 0],
                all_removed_2d[:, 1],
                c='lightgray',
                s=10,
                alpha=0.3,
                edgecolors='none',
                linewidths=0.5,
                zorder=1,
                label=f'Previously Removed'
            )
        
        # 绘制当前剩余部分（用Z坐标着色）
        scatter_current = ax_current.scatter(
            current_pore_coords_2d[:, 0],
            current_pore_coords_2d[:, 1],
            c=current_pore_coords[:, 2],  # 用Z坐标着色
            cmap='viridis',
            s=20,
            alpha=0.6,
            edgecolors='k',
            linewidths=0.5,
            zorder=2,
            label=f'Remaining ({len(current_pore_data)} pores)'
        )
        ax_current.set_xlabel('X (nm)', fontsize=12)
        ax_current.set_ylabel('Y (nm)', fontsize=12)
        ax_current.set_title(f'Remaining Sample - Round {cut_count} ({len(current_pore_data)} pores)', 
                            fontsize=14, fontweight='bold')
        ax_current.grid(True, alpha=0.3)
        ax_current.set_aspect('equal', adjustable='box')
        plt.colorbar(scatter_current, ax=ax_current, label='Z (nm)')
        ax_current.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # 短暂暂停，确保窗口显示
        
        # 交互式选择切割平面（在当前窗口上进行）
        result = interactive_select_cut_plane()
        
        # 关闭当前窗口
        plt.close(fig_current)
        
        if result is None or result[0] is None:
            print("\n用户选择结束切割")
            break
        
        point1, point2, start_point1, should_continue = result
        
        if not should_continue:
            print("\n用户选择结束切割")
            break
        
        # 计算边界线方向向量和单位向量
        boundary_dir = np.array(point2) - np.array(point1)
        boundary_length = np.linalg.norm(boundary_dir)
        if boundary_length > 1e-10:
            boundary_unit = boundary_dir / boundary_length
        else:
            boundary_unit = np.array([1.0, 0.0])
        
        # 计算第二个起点：第一个起点沿着边界线方向移动Z轴厚度距离
        start_point2 = start_point1 + boundary_unit * z_thickness
        
        # 计算法向量
        normal_vec = calculate_normal_vector(point1, point2)
        print(f"\n  边界线方向向量: ({boundary_dir[0]:.4f}, {boundary_dir[1]:.4f})")
        print(f"  边界线长度: {boundary_length:.2f} nm")
        print(f"  法向量（归一化）: ({normal_vec[0]:.4f}, {normal_vec[1]:.4f})")
        print(f"  第一个起点: ({start_point1[0]:.2f}, {start_point1[1]:.2f})")
        print(f"  第二个起点（自动计算）: ({start_point2[0]:.2f}, {start_point2[1]:.2f})")
        print(f"  沿边界线方向移动距离（Z轴厚度）: {z_thickness:.2f} nm")
        
        # 执行切割（根据两条平行直线）
        print("\n执行切割...")
        removed_pores_mask, kept_pores_mask = split_pores_by_two_lines(
            current_pore_coords, current_pore_data, start_point1, start_point2, boundary_unit, z_thickness
        )
        
        removed_count = np.sum(removed_pores_mask)
        kept_count = np.sum(kept_pores_mask)
        
        print(f"  被切掉的孔数: {removed_count}")
        print(f"  保留的孔数: {kept_count}")
        
        if kept_count == 0:
            print("警告：所有孔都被切掉了，无法继续切割")
            break
        
        # 可视化切割结果
        fig_result, ax_result = plt.subplots(figsize=(12, 10))
        
        # 先绘制之前已移除的部分（浅灰色，半透明，在底层）
        if len(all_removed_coords_2d) > 0:
            all_removed_2d = np.array(all_removed_coords_2d)
            ax_result.scatter(
                all_removed_2d[:, 0],
                all_removed_2d[:, 1],
                c='lightgray',
                s=10,
                alpha=0.2,
                edgecolors='none',
                linewidths=0.5,
                zorder=1
            )
        
        # 绘制被切掉的部分（红色，半透明，在底层）
        if np.any(removed_pores_mask):
            removed_coords_2d = current_pore_coords_2d[removed_pores_mask]
            ax_result.scatter(
                removed_coords_2d[:, 0],
                removed_coords_2d[:, 1],
                c='red',
                s=15,
                alpha=0.2,
                edgecolors='none',
                linewidths=0.5,
                zorder=1,
                label=f'Removed ({removed_count} pores)'
            )
        
        # 绘制保留的部分（蓝色，在两条直线之间，在上层）
        kept_coords_2d = current_pore_coords_2d[kept_pores_mask]
        ax_result.scatter(
            kept_coords_2d[:, 0],
            kept_coords_2d[:, 1],
            c='blue',
            s=25,
            alpha=0.9,
            edgecolors='darkblue',
            linewidths=1.0,
            zorder=3,
            label=f'Kept ({kept_count} pores)'
        )
        
        # 绘制边界线（定义法向量方向的线，黑色虚线）
        line_x = np.array([point1[0], point2[0]])
        line_y = np.array([point1[1], point2[1]])
        ax_result.plot(line_x, line_y, 'k--', linewidth=2, alpha=0.6, zorder=2, label='Boundary Line')
        
        # 计算两条切割直线的端点（沿着法向量方向延伸）
        # 只延伸到数据点的边界附近，不要超出太多
        # 计算所有数据点到起点的距离在法向量上的投影
        all_points = current_pore_coords_2d
        dists_to_start1 = np.array([point_to_plane_distance(p, start_point1, normal_vec) for p in all_points])
        dists_to_start2 = np.array([point_to_plane_distance(p, start_point2, normal_vec) for p in all_points])
        
        # 找到数据点在法向量方向上的范围
        min_dist1, max_dist1 = dists_to_start1.min(), dists_to_start1.max()
        min_dist2, max_dist2 = dists_to_start2.min(), dists_to_start2.max()
        
        # 计算需要延伸的长度（加上一点边距）
        margin = 0.05  # 5%的边距
        range1 = max_dist1 - min_dist1
        range2 = max_dist2 - min_dist2
        line_length = max(range1, range2) * (1.0 + margin)  # 只延伸到数据点范围，加上小边距
        
        # 第一条直线：通过start_point1，沿着法向量方向（绿色粗线）
        line1_start = start_point1 - normal_vec * line_length
        line1_end = start_point1 + normal_vec * line_length
        ax_result.plot([line1_start[0], line1_end[0]], [line1_start[1], line1_end[1]], 
                      'g-', linewidth=4, alpha=0.9, zorder=5)
        
        # 第一个起点（绿色大星号）
        ax_result.scatter(
            start_point1[0], start_point1[1],
            c='green', s=300, marker='*', 
            edgecolors='darkgreen', linewidths=3,
            zorder=7
        )
        
        # 第二条直线：通过start_point2，沿着法向量方向（橙色粗线）
        line2_start = start_point2 - normal_vec * line_length
        line2_end = start_point2 + normal_vec * line_length
        ax_result.plot([line2_start[0], line2_end[0]], [line2_start[1], line2_end[1]], 
                      'orange', linewidth=4, alpha=0.9, zorder=5)
        
        # 第二个起点（橙色大星号）
        ax_result.scatter(
            start_point2[0], start_point2[1],
            c='orange', s=300, marker='*', 
            edgecolors='darkorange', linewidths=3,
            zorder=7
        )
        
        ax_result.set_xlabel('X (nm)', fontsize=12)
        ax_result.set_ylabel('Y (nm)', fontsize=12)
        ax_result.set_title(f'Cut Result - Round {cut_count}\nRed: Removed ({removed_count} pores), Blue: Kept ({kept_count} pores)', 
                           fontsize=14, fontweight='bold')
        ax_result.grid(True, alpha=0.3)
        # 不显示图注，避免遮挡图片
        ax_result.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # 保存可视化结果（保存到子文件夹中）
        sample_output_dir = output_dir / sample_name
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        vis_file = sample_output_dir / f"{sample_name}_cut_round{cut_count}_visualization.png"
        plt.savefig(vis_file, dpi=300, bbox_inches='tight')
        print(f"\n  切割结果可视化已保存: {vis_file}")
        
        # 显示结果，等待用户手动关闭
        print("  请查看切割结果，关闭窗口继续下一轮...")
        plt.show(block=True)  # 阻塞等待用户关闭窗口
        
        # 保存当前切割带对应的子样本
        # 注意：为了允许子样本之间重叠，这里使用“全体孔/喉”重新按当前切割带筛选一遍；
        # 即：即使某些孔已经在之前的子样本中，本轮只要几何上落在当前切割带内，也会被包含进新的子样本。
        print(f"\n  保存子样本 {cut_count}（基于全体孔/喉的几何筛选）...")

        # 基于全体孔坐标 / 数据，按当前切割带重新计算“保留区域”掩码
        _, kept_pores_mask_global = split_pores_by_two_lines(
            pore_coords, df_pores, start_point1, start_point2, boundary_unit, z_thickness
        )
        kept_pores_global = df_pores[kept_pores_mask_global].copy()
        kept_pore_ids_global = set(pore_ids[kept_pores_mask_global])
        kept_throats_mask_global = filter_throats_by_pores(df_throats, kept_pore_ids_global)
        kept_throats_global = df_throats[kept_throats_mask_global].copy()
        
        # 保存子样本（同时记录法向量方向作为渗透方向）
        sub_sample_name = f"{sample_name}_sub{cut_count}"
        save_sub_sample(
            kept_pores_global, kept_throats_global,
            np.ones(len(kept_pores_global), dtype=bool),
            np.ones(len(kept_throats_global), dtype=bool),
            sub_sample_name, output_dir,
            normal_vec=normal_vec  # 保存法向量方向作为渗透方向
        )
        
        # 记录子样本信息（包括法向量）
        all_sub_samples.append({
            'name': sub_sample_name,
            'pore_count': len(kept_pores_global),
            'throat_count': len(kept_throats_global),
            'normal_vector': normal_vec.copy()  # 保存法向量
        })
        
        # 更新已移除部分的坐标（用于后续可视化）
        removed_coords_2d_list = current_pore_coords_2d[removed_pores_mask].tolist()
        removed_coords_z_list = current_pore_coords[:, 2][removed_pores_mask].tolist()
        all_removed_coords_2d.extend(removed_coords_2d_list)
        all_removed_coords_z.extend(removed_coords_z_list)
        
        # 更新当前数据为被切掉的部分（剩余部分，继续下一轮）
        current_pore_data = current_pore_data[removed_pores_mask].copy()
        current_pore_coords = current_pore_coords[removed_pores_mask]
        current_pore_ids = current_pore_ids[removed_pores_mask]
        current_pore_coords_2d = current_pore_coords_2d[removed_pores_mask]
        
        # 过滤喉（保留剩余部分的喉）
        removed_pore_ids_set = set(current_pore_ids)
        removed_throats_mask = filter_throats_by_pores(current_throat_data, removed_pore_ids_set)
        current_throat_data = current_throat_data[removed_throats_mask].copy()
        
        print(f"  剩余孔数: {len(current_pore_data)}")
        print(f"  剩余喉数: {len(current_throat_data)}")
        
        if len(current_pore_data) == 0:
            print("  没有剩余部分，结束切割")
            break
    
    # 如果还有剩余部分，保存为最后一个子样本
    if len(current_pore_data) > 0:
        print("\n" + "="*70)
        print("保存剩余部分为最后一个子样本")
        print("="*70)
        
        final_sample_name = f"{sample_name}_sub{cut_count + 1}"
        # 最后一个子样本没有法向量信息（因为没有进行切割），设置为None
        save_sub_sample(
            current_pore_data, current_throat_data,
            np.ones(len(current_pore_data), dtype=bool),
            np.ones(len(current_throat_data), dtype=bool),
            final_sample_name, output_dir,
            normal_vec=None  # 最后一个子样本没有法向量信息
        )
        
        all_sub_samples.append({
            'name': final_sample_name,
            'pore_count': len(current_pore_data),
            'throat_count': len(current_throat_data),
            'normal_vector': None  # 最后一个子样本没有法向量信息
        })
    
    # 打印汇总信息
    print("\n" + "="*70)
    print("切割完成！")
    print("="*70)
    print(f"共进行了 {cut_count} 轮切割")
    print(f"共生成 {len(all_sub_samples)} 个子样本：")
    for i, sub in enumerate(all_sub_samples, 1):
        if sub.get('normal_vector') is not None:
            nv = sub['normal_vector']
            print(f"  {i}. {sub['name']}: {sub['pore_count']} 个孔, {sub['throat_count']} 个喉, "
                  f"法向量(渗透方向): [{nv[0]:.4f}, {nv[1]:.4f}]")
        else:
            print(f"  {i}. {sub['name']}: {sub['pore_count']} 个孔, {sub['throat_count']} 个喉, "
                  f"法向量(渗透方向): 未定义（剩余部分）")
    print(f"\n所有结果保存在: {output_dir / sample_name}")
    print(f"每个子样本的法向量信息保存在对应的 *_metadata.json 文件中")


if __name__ == "__main__":
    main()

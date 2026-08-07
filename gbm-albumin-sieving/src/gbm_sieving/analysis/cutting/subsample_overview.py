"""
可视化所有子样本的3D图
在一张图上显示所有切割后的子样本，用不同颜色区分

用法（默认路径相对于 肾脏 目录）:
    python analyze_and_calculate_2/interactive/visualize_sub_samples_3d.py --sample-name AS317
    默认在 肾脏/data_subsamples/<sample-name>/ 下查找子样本，输出同目录
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
import json
import re
import sys
from collections import defaultdict

import plotly.graph_objects as go


def _parse_sub_sample_index(sub_name: str) -> int | None:
    """从 '{sample}_sub{N}' 解析 N；不匹配则 None。"""
    m = re.search(r"_sub(\d+)$", sub_name)
    return int(m.group(1)) if m else None

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_sub_samples(output_dir, sample_name):
    """
    加载所有子样本
    
    参数:
        output_dir: 输出文件夹路径
        sample_name: 样本名称
    
    返回:
        sub_samples: 列表，每个元素是 {'name': 子样本名, 'pores': DataFrame, 'throats': DataFrame}
    """
    output_dir = Path(output_dir)
    sub_samples = []
    
    # 查找所有子样本文件
    pattern = f"{sample_name}_sub*_pores.xlsx"
    pore_files = sorted(output_dir.glob(pattern))
    
    if len(pore_files) == 0:
        print(f"错误：未找到子样本文件（模式: {pattern}）")
        print(f"  搜索目录: {output_dir}")
        return []
    
    print(f"找到 {len(pore_files)} 个子样本文件")

    sub_indices: list[int] = []
    for pf in pore_files:
        sn = pf.stem.replace("_pores", "")
        si = _parse_sub_sample_index(sn)
        if si is not None:
            sub_indices.append(si)
    max_sub_idx = max(sub_indices) if sub_indices else None

    for pore_file in pore_files:
        # 提取子样本名称
        sub_name = pore_file.stem.replace("_pores", "")

        # 查找对应的喉文件
        throat_file = output_dir / f"{sub_name}_throats.xlsx"

        if not throat_file.exists():
            print(f"警告：未找到对应的喉文件 {throat_file}，跳过")
            continue

        # 加载数据
        try:
            pores_df = pd.read_excel(pore_file)
            throats_df = pd.read_excel(throat_file)

            # 有法向量：metadata 里 normal_vector 为非 null 对象
            # 兼容旧版切割：剩余子样本曾不写 metadata → 将「最大编号且无 json」视为无法向量（灰色）
            has_normal = True
            meta_file = output_dir / f"{sub_name}_metadata.json"
            sub_idx = _parse_sub_sample_index(sub_name)
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                nv = meta.get("normal_vector")
                has_normal = nv is not None
            elif max_sub_idx is not None and sub_idx is not None and sub_idx == max_sub_idx:
                has_normal = False

            sub_samples.append({
                "name": sub_name,
                "pores": pores_df,
                "throats": throats_df,
                "has_normal": has_normal,
            })

            tag = "" if has_normal else " [无法向量→灰色]"
            print(f"  已加载: {sub_name} ({len(pores_df)} 个孔, {len(throats_df)} 个喉){tag}")
        except Exception as e:
            print(f"警告：加载 {sub_name} 时出错: {e}")
            continue
    
    return sub_samples


def visualize_sub_samples_3d(sub_samples, output_dir, sample_name, *, show_plot: bool = True):
    """
    在一张3D图上可视化所有子样本
    
    参数:
        sub_samples: 子样本列表
        output_dir: 输出文件夹
        sample_name: 样本名称
        show_plot: 是否在保存文件后弹出 matplotlib 交互窗口（批处理脚本可设为 False）
    """
    if len(sub_samples) == 0:
        print("错误：没有可用的子样本")
        return
    
    # 创建3D图
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    tab10 = plt.cm.tab10(np.linspace(0, 1, 10))
    plotly_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8",
    ]
    gray_mpl = (0.55, 0.55, 0.55, 1.0)
    gray_plotly = "#888888"
    # 无法向量（灰色）子样本：孔/喉略小于有色子样本
    _s_pore_mpl = 14
    _s_pore_mpl_gray = 9
    _s_pore_plotly = 2.3
    _s_pore_plotly_gray = 1.5
    _lw_throat_mpl = 0.95
    _lw_throat_mpl_gray = 0.55
    _lw_throat_plotly = 3.5
    _lw_throat_plotly_gray = 2.0

    sub_colors_mpl: list = []
    sub_colors_plotly: list = []
    color_idx = 0
    for sub in sub_samples:
        has_normal = bool(sub.get("has_normal", True))
        if has_normal:
            ci = color_idx % len(tab10)
            sub_colors_mpl.append(tab10[ci])
            sub_colors_plotly.append(plotly_palette[color_idx % len(plotly_palette)])
            color_idx += 1
        else:
            sub_colors_mpl.append(gray_mpl)
            sub_colors_plotly.append(gray_plotly)

    # 全局孔 ID → 坐标、EqRadius、该孔所在子样本的显示色
    pore_lookup: dict[int, dict] = {}
    for si, sub in enumerate(sub_samples):
        df = sub["pores"]
        cm = sub_colors_mpl[si]
        cp = sub_colors_plotly[si]
        has_er = "EqRadius" in df.columns
        for _, row in df.iterrows():
            try:
                pid = int(row["Pore ID"])
            except Exception:
                continue
            if has_er:
                er = float(pd.to_numeric(row["EqRadius"], errors="coerce"))
                if not np.isfinite(er):
                    er = 0.0
            else:
                er = 0.0
            pore_lookup[pid] = {
                "x": float(row["X Coord"]),
                "y": float(row["Y Coord"]),
                "z": float(row["Z Coord"]),
                "r": er,
                "c_mpl": cm,
                "c_plotly": cp,
            }

    # 喉：线段颜色与两端 EqRadius 较大的那个孔一致（同色 = 该孔所属子样本色）
    throat_segments: list[tuple[float, float, float, float, float, float, object, str]] = []
    for sub in sub_samples:
        dft = sub["throats"]
        if dft is None or len(dft) == 0:
            continue
        if not {"Pore ID #1", "Pore ID #2"}.issubset(dft.columns):
            continue
        for _, trow in dft.iterrows():
            try:
                p1 = int(trow["Pore ID #1"])
                p2 = int(trow["Pore ID #2"])
            except Exception:
                continue
            if p1 not in pore_lookup or p2 not in pore_lookup:
                continue
            a, b = pore_lookup[p1], pore_lookup[p2]
            if a["r"] >= b["r"]:
                cm_seg, cp_seg = a["c_mpl"], a["c_plotly"]
            else:
                cm_seg, cp_seg = b["c_mpl"], b["c_plotly"]
            throat_segments.append(
                (a["x"], a["y"], a["z"], b["x"], b["y"], b["z"], cm_seg, cp_seg)
            )

    print(f"  喉线段: {len(throat_segments)}（颜色与两端 EqRadius 较大孔一致）")

    # 先画喉（细线），再画孔
    for xa, ya, za, xb, yb, zb, cm_seg, cp in throat_segments:
        _lw = _lw_throat_mpl_gray if cp == gray_plotly else _lw_throat_mpl
        ax.plot(
            [xa, xb],
            [ya, yb],
            [za, zb],
            color=cm_seg,
            linewidth=_lw,
            alpha=0.45,
            solid_capstyle="round",
        )

    for si, sub in enumerate(sub_samples):
        pores_df = sub["pores"]
        x = pores_df["X Coord"].values
        y = pores_df["Y Coord"].values
        z = pores_df["Z Coord"].values
        c_mpl = sub_colors_mpl[si]
        _sp = _s_pore_mpl_gray if not bool(sub.get("has_normal", True)) else _s_pore_mpl
        ax.scatter(
            x, y, z,
            c=[c_mpl],
            s=_sp,
            alpha=0.6,
            edgecolors="none",
            label=f"{sub['name']} ({len(pores_df)} pores)",
        )

    ax.set_title(
        f"3D Visualization of All Sub-samples - {sample_name}\nTotal: {len(sub_samples)} sub-samples",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=9)

    # 去掉三维坐标系背景（刻度、网格、 pane），保留标题与图例
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    
    # 设置相等的坐标轴比例
    # 获取所有坐标的范围
    all_x = []
    all_y = []
    all_z = []
    for sub in sub_samples:
        pores_df = sub['pores']
        all_x.extend(pores_df['X Coord'].values)
        all_y.extend(pores_df['Y Coord'].values)
        all_z.extend(pores_df['Z Coord'].values)
    
    x_min = x_max = y_min = y_max = z_min = z_max = None
    if len(all_x) > 0:
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        z_min, z_max = min(all_z), max(all_z)
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2
        
        ax.set_xlim([x_center - x_range/2, x_center + x_range/2])
        ax.set_ylim([y_center - y_range/2, y_center + y_range/2])
        ax.set_zlim([z_center - z_range/2, z_center + z_range/2])
    
    # 保存静态PNG图片
    output_file = Path(output_dir) / f"{sample_name}_all_sub_samples_3d.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n3D静态可视化已保存: {output_file}")
    
    # 使用 Plotly 生成交互式3D图（喉线 + 孔点；与 PNG 一致）
    fig_html = go.Figure()

    lines_by_hex: dict[str, dict[str, list]] = defaultdict(lambda: {"x": [], "y": [], "z": []})
    for xa, ya, za, xb, yb, zb, _cm, cp in throat_segments:
        bucket = lines_by_hex[cp]
        bucket["x"].extend([xa, xb, None])
        bucket["y"].extend([ya, yb, None])
        bucket["z"].extend([za, zb, None])
    for cp, d in lines_by_hex.items():
        _lwp = _lw_throat_plotly_gray if cp == gray_plotly else _lw_throat_plotly
        fig_html.add_trace(
            go.Scatter3d(
                x=d["x"],
                y=d["y"],
                z=d["z"],
                mode="lines",
                line=dict(color=cp, width=_lwp),
                opacity=0.45,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    for si, sub in enumerate(sub_samples):
        pores_df = sub["pores"]
        x = pores_df["X Coord"].values
        y = pores_df["Y Coord"].values
        z = pores_df["Z Coord"].values
        mc = sub_colors_plotly[si]
        _ms = _s_pore_plotly_gray if not bool(sub.get("has_normal", True)) else _s_pore_plotly
        fig_html.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=_ms,
                    color=mc,
                    opacity=0.7,
                    line=dict(width=0, color=mc),
                ),
                name=f"{sub['name']} ({len(pores_df)} pores)",
            )
        )

    scene_kwargs: dict = {}
    if x_min is not None:
        scene_kwargs = dict(
            xaxis=dict(
                visible=False,
                range=[x_center - x_range / 2, x_center + x_range / 2],
            ),
            yaxis=dict(
                visible=False,
                range=[y_center - y_range / 2, y_center + y_range / 2],
            ),
            zaxis=dict(
                visible=False,
                range=[z_center - z_range / 2, z_center + z_range / 2],
            ),
            bgcolor="rgba(0,0,0,0)",
            aspectmode="data",
        )

    fig_html.update_layout(
        scene=scene_kwargs,
        title=f"3D Interactive Visualization of All Sub-samples - {sample_name}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            x=1.02,
            y=1,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    
    html_file = Path(output_dir) / f"{sample_name}_all_sub_samples_3d.html"
    fig_html.write_html(str(html_file), include_plotlyjs='cdn')
    print(f"3D交互式可视化已保存(HTML): {html_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def visualize_radius_3d(sub_samples, output_dir, sample_name):
    """
    生成按真实孔半径(EqRadius)定大小、按子样本分色的 3D 图（去轴去网格去背景）。

    点的大小 = EqRadius，同一个 sub-sample 同色，不同 sub-sample 不同色。

    输出：
      - PNG: matplotlib 静态图，无坐标轴/网格/背景
      - HTML: Plotly 交互图，无坐标轴/网格/背景
    """
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(sub_samples), 1)))

    # 先计算全局范围，用于统一标度
    global_x_min = global_x_max = global_y_min = global_y_max = global_z_min = global_z_max = None
    for sub in sub_samples:
        df = sub["pores"]
        x = df["X Coord"].values.astype(float)
        y = df["Y Coord"].values.astype(float)
        z = df["Z Coord"].values.astype(float)
        if global_x_min is None:
            global_x_min, global_x_max = x.min(), x.max()
            global_y_min, global_y_max = y.min(), y.max()
            global_z_min, global_z_max = z.min(), z.max()
        else:
            global_x_min = min(global_x_min, x.min())
            global_x_max = max(global_x_max, x.max())
            global_y_min = min(global_y_min, y.min())
            global_y_max = max(global_y_max, y.max())
            global_z_min = min(global_z_min, z.min())
            global_z_max = max(global_z_max, z.max())

    x_range = global_x_max - global_x_min
    y_range = global_y_max - global_y_min
    z_range = global_z_max - global_z_min
    x_center = 0.5 * (global_x_max + global_x_min)
    y_center = 0.5 * (global_y_max + global_y_min)
    z_center = 0.5 * (global_z_max + global_z_min)

    # 用于归一化 s 的全局最小半径
    all_r_flat = []
    for sub in sub_samples:
        df = sub["pores"]
        if "EqRadius" not in df.columns:
            continue
        r = pd.to_numeric(df["EqRadius"], errors="coerce").values
        all_r_flat.extend(r[np.isfinite(r)].tolist())
    r_min = float(np.min(all_r_flat)) if all_r_flat else 1.0
    s_min = 2.0
    s_max = 50.0

    R_CAP_NORMAL = 20.0
    R_CAP_NO_NORMAL = 10.0

    def _cap_for_sub(sub: dict) -> float:
        return R_CAP_NORMAL if sub.get("has_normal", True) else R_CAP_NO_NORMAL

    def _size(r: np.ndarray, cap: float) -> np.ndarray:
        rc = np.clip(r, r_min, cap)
        return s_min + (rc - r_min) / max(cap - r_min, 1e-12) * (s_max - s_min)

    # --- 静态 PNG (matplotlib, 去轴) ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for i, sub in enumerate(sub_samples):
        df = sub["pores"]
        if "EqRadius" not in df.columns:
            continue
        x = df["X Coord"].values.astype(float)
        y = df["Y Coord"].values.astype(float)
        z = df["Z Coord"].values.astype(float)
        r = pd.to_numeric(df["EqRadius"], errors="coerce").values
        valid = np.isfinite(r)
        if not valid.any():
            continue
        x, y, z, r = x[valid], y[valid], z[valid], r[valid]
        ax.scatter(
            x, y, z,
            s=_size(r, _cap_for_sub(sub)),
            c=[colors[i]],
            alpha=0.7,
            edgecolors="none",
            label=sub["name"],
        )
    ax.set_xlim(x_center - x_range * 0.525, x_center + x_range * 0.525)
    ax.set_ylim(y_center - y_range * 0.525, y_center + y_range * 0.525)
    ax.set_zlim(z_center - z_range * 0.525, z_center + z_range * 0.525)
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")

    png_file = Path(output_dir) / f"{sample_name}_radius_3d.png"
    fig.tight_layout(pad=0)
    fig.savefig(png_file, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  半径 3D 静态图已保存: {png_file}")

    # --- 交互式 HTML (Plotly, 去轴) ---
    plotly_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    fig_html = go.Figure()
    for i, sub in enumerate(sub_samples):
        df = sub["pores"]
        if "EqRadius" not in df.columns:
            continue
        x = df["X Coord"].values.astype(float)
        y = df["Y Coord"].values.astype(float)
        z = df["Z Coord"].values.astype(float)
        r = pd.to_numeric(df["EqRadius"], errors="coerce").values
        valid = np.isfinite(r)
        if not valid.any():
            continue
        x, y, z, r = x[valid], y[valid], z[valid], r[valid]
        sz = _size(r, _cap_for_sub(sub))
        mc = plotly_colors[i % len(plotly_colors)]
        fig_html.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(
                size=sz,
                color=mc,
                opacity=0.7,
                # Plotly WebGL 默认会给圆点加浅色描边（看起来像白圈）
                line=dict(width=0, color=mc),
            ),
            name=sub["name"],
        ))
    fig_html.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="rgba(0,0,0,0)",
            aspectmode="data",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        title=None,
    )
    html_file = Path(output_dir) / f"{sample_name}_radius_3d.html"
    fig_html.write_html(str(html_file), include_plotlyjs="cdn")
    print(f"  半径 3D 交互图已保存: {html_file}")


def main():
    script_dir = Path(__file__).resolve().parent
    data_root = script_dir.parent.parent  # 肾脏
    parser = argparse.ArgumentParser(
        description="可视化所有子样本的3D图"
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        required=True,
        help="样本名字，例如 AS317",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(data_root / "data_subsamples"),
        help="子样本输入顶层文件夹（默认: 肾脏/data_subsamples）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出文件夹；默认与 --input-dir 一致。",
    )
    parser.add_argument(
        "--radius-3d",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="额外生成按真实孔半径(EqRadius)着色的 3D 图（去轴去背景）。",
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = data_root / args.input_dir
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    if not output_dir.is_absolute():
        output_dir = data_root / args.output_dir
    sample_input_dir = input_dir / args.sample_name
    sample_output_dir = output_dir / args.sample_name
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not sample_input_dir.exists():
        print(f"错误：子样本输入文件夹不存在: {sample_input_dir}")
        sys.exit(1)
    
    print("="*70)
    print("加载子样本数据")
    print("="*70)
    
    # 加载所有子样本（从 input_dir/<sample-name>/ 中查找）
    sub_samples = load_sub_samples(sample_input_dir, args.sample_name)
    
    if len(sub_samples) == 0:
        print("错误：没有找到任何子样本")
        sys.exit(1)
    
    print(f"\n共加载 {len(sub_samples)} 个子样本")
    
    print("\n" + "="*70)
    print("生成3D可视化")
    print("="*70)
    
    # 可视化（图片保存在 output_dir/<sample-name>/ 中）
    visualize_sub_samples_3d(sub_samples, sample_output_dir, args.sample_name)
    
    if bool(args.radius_3d):
        print("\n" + "="*70)
        print("生成按孔半径着色的 3D 图")
        print("="*70)
        visualize_radius_3d(sub_samples, sample_output_dir, args.sample_name)
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)


if __name__ == "__main__":
    main()


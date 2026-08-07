"""
在 X–Y 投影上用三点定义长方形区域，将区域内孔/喉导出为「大子样本」。

交互（与 interactive_correlation_length 一致的两点 + 第三点）：
- 第 1、2 点：只确定第一条边的方向 **u** = normalize(P2−P1)，过第 1 点的无限长直线 L1 沿 **u**。
- 第 3 点 P3：过 P3 作与 **u** 垂直的直线（第二条边的方向 **v**⊥**u**）；与 L1 的交点 **I** 即垂足
  I = P1 + ((P3−P1)·**u**)**u**（P3 到 L1 的正交投影）。
- 第四顶点 D = P1 + P3 − I，矩形顶点顺序 P1 → I → P3 → D（对边平行）。

- 可选叠加 alpha shape 边界（--alpha 同 correlation）
- 元数据记录前两点 P1→P2 的单位方向作为 X–Y 平面渗透方向 permeation_direction_xy
- 窗口底部：「✓ 确认」保存当前长方形；「↺ 重选区域」清空后重新点击三点
- 输出目录默认：肾脏/large_subsamples/<样本名>/

用法（建议在 肾脏 目录下）:
    python analyze_and_calculate_2/interactive/interactive_large_subsample_box.py
    python analyze_and_calculate_2/interactive/interactive_large_subsample_box.py --sample-name AS317
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.widgets import Button

# 与同目录 interactive_correlation_length 共用工具与 alpha shape 逻辑
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from interactive_correlation_length import (
    get_alpha_shape_polygon_2d,
    list_available_samples,
    project_to_xy,
)

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def interactive_select_sample(sample_dir: Path):
    """交互式选择样本（与 interactive_sample_splitter 一致）"""
    available = list_available_samples(sample_dir)
    if not available:
        print(f"错误：在 {sample_dir} 中未找到样本（需 *_pores.xlsx 与对应 *_throats.xlsx）")
        return None
    print("\n" + "=" * 70)
    print("可用样本列表")
    print("=" * 70)
    for i, name in enumerate(available, 1):
        print(f"  {i}. {name}")
    print("=" * 70)
    while True:
        try:
            choice = input(f"\n请选择样本 (1-{len(available)})，或直接输入样本名: ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    print(f"已选择样本: {available[idx]}")
                    return available[idx]
                print(f"错误：请输入 1-{len(available)} 之间的数字")
                continue
            if choice in available:
                print(f"已选择样本: {choice}")
                return choice
            print(f"错误：样本 '{choice}' 不存在。")
        except KeyboardInterrupt:
            print("\n\n用户取消选择")
            return None


def filter_throats_by_pores(throat_data: pd.DataFrame, kept_pore_ids: set) -> np.ndarray:
    """只保留两端孔 ID 均在保留集合中的喉。"""
    mask = []
    for _, row in throat_data.iterrows():
        p1 = row["Pore ID #1"]
        p2 = row["Pore ID #2"]
        mask.append((p1 in kept_pore_ids) and (p2 in kept_pore_ids))
    return np.array(mask, dtype=bool)


def rectangle_vertices_from_three_points(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    前两点定方向 u，第三点 P3；I 为 P3 到过 P1 沿 u 的直线的垂足；D = P1 + P3 - I。
    返回 (4,2) 顶点 [P1, I, P3, D] 与单位向量 u；退化时返回 None。
    """
    p1 = np.asarray(p1, dtype=float).ravel()[:2]
    p2 = np.asarray(p2, dtype=float).ravel()[:2]
    p3 = np.asarray(p3, dtype=float).ravel()[:2]
    d12 = p2 - p1
    nu = np.linalg.norm(d12)
    if nu < 1e-9:
        return None
    u = d12 / nu
    # 垂足：P3 到直线 P1 + t u 的投影
    t = float(np.dot(p3 - p1, u))
    i_pt = p1 + t * u
    d_pt = p1 + p3 - i_pt
    # 面积（平行四边形 | (I-P1) x (P3-I) |）
    e1 = i_pt - p1
    e2 = p3 - i_pt
    area2 = abs(e1[0] * e2[1] - e1[1] * e2[0])
    if area2 < 1e-12:
        return None
    verts = np.vstack([p1, i_pt, p3, d_pt])
    return verts, u


def pore_mask_in_xy_polygon(pore_coords: np.ndarray, quad_vertices: np.ndarray) -> np.ndarray:
    """孔心 X–Y 投影落在凸四边形内部（含边界）。"""
    xy = project_to_xy(pore_coords)
    q = np.asarray(quad_vertices, dtype=float)
    path = MplPath(np.vstack([q, q[0:1]]))
    return path.contains_points(xy)


def interactive_select_rectangle_three_points(
    pore_coords_2d: np.ndarray, sample_name: str, polygon=None
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    与 interactive_correlation_length 类似：点 1→2 有预览线；点 3 后画出长方形；
    用底部按钮「✓ 确认」保存结果，「重选区域」清空后重新点击三点。
    成功时返回 (quad_vertices, u_xy)：
      - quad_vertices: (4,2) 顶点 [P1, I, P3, D]
      - u_xy: (2,) 单位向量，前两点 P1→P2 方向（X–Y 平面渗透方向）
    失败返回 None。
    """
    points: list[tuple[float, float]] = []
    state: dict = {"quad": None, "u_edge": None, "confirmed": False}

    fig, ax = plt.subplots(figsize=(10, 8))
    if polygon is not None and hasattr(polygon, "exterior") and polygon.exterior is not None:
        xb, yb = polygon.exterior.xy
        ax.plot(xb, yb, "r-", linewidth=2, label="Alpha shape 边界", zorder=2)
    ax.scatter(
        pore_coords_2d[:, 0],
        pore_coords_2d[:, 1],
        c="steelblue",
        s=15,
        alpha=0.6,
        edgecolors="none",
        zorder=1,
    )
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_title(
        f"样本 {sample_name}：请依次点击三点 — 1、2 定边方向，3 定长方形区域"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    if polygon is not None:
        ax.legend(loc="upper right")
    fig.tight_layout(rect=[0.03, 0.12, 0.97, 0.96])

    preview_line, = ax.plot([], [], "g-", linewidth=2, alpha=0.85, zorder=4)
    edge_line, = ax.plot([], [], "g-", linewidth=2.5, alpha=0.95, zorder=4)
    rect_patch_holder: list = [None]
    corner_scatter_holder: list = [None]
    # 第三点移动鼠标时的半透明预览长方形（与最终确认区分开）
    preview_rect_patch_holder: list = [None]

    def clear_preview_rect():
        pr = preview_rect_patch_holder[0]
        if pr is not None:
            pr.remove()
            preview_rect_patch_holder[0] = None

    def redraw_rect(verts: np.ndarray):
        clear_preview_rect()
        if rect_patch_holder[0] is not None:
            rect_patch_holder[0].remove()
            rect_patch_holder[0] = None
        if corner_scatter_holder[0] is not None:
            corner_scatter_holder[0].remove()
            corner_scatter_holder[0] = None
        poly = MplPolygon(
            verts,
            closed=True,
            facecolor="orange",
            edgecolor="darkorange",
            linewidth=2,
            alpha=0.28,
            zorder=3,
        )
        ax.add_patch(poly)
        rect_patch_holder[0] = poly
        sc = ax.scatter(
            verts[:, 0],
            verts[:, 1],
            c=["lime", "cyan", "magenta", "yellow"],
            s=80,
            zorder=5,
            edgecolors="k",
            linewidths=0.5,
        )
        corner_scatter_holder[0] = sc

    def reset_selection():
        """清空点击与长方形，从第一点重新选起。"""
        points.clear()
        state["quad"] = None
        state["u_edge"] = None
        clear_preview_rect()
        if rect_patch_holder[0] is not None:
            rect_patch_holder[0].remove()
            rect_patch_holder[0] = None
        if corner_scatter_holder[0] is not None:
            corner_scatter_holder[0].remove()
            corner_scatter_holder[0] = None
        preview_line.set_data([], [])
        edge_line.set_data([], [])
        ax.set_title(
            f"样本 {sample_name}：请依次点击三点 — 1、2 定边方向，3 定长方形区域"
        )
        fig.canvas.draw_idle()
        print("  已重选：请重新点击三点划定区域。")

    def on_confirm_clicked(_event):
        if state["quad"] is None or state["u_edge"] is None:
            print("  请先完成三点框选（出现橙色长方形），再点「✓ 确认」。")
            return
        state["confirmed"] = True
        plt.close(fig)

    def on_reset_clicked(_event):
        reset_selection()

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        points.append((float(event.xdata), float(event.ydata)))
        if len(points) == 2:
            preview_line.set_data([], [])
            p1, p2 = np.array(points[0]), np.array(points[1])
            edge_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            ax.set_title(
                f"样本 {sample_name}：移动鼠标可预览半透明长方形，点击确定第三点"
            )
        if len(points) == 3:
            p1, p2, p3 = np.array(points[0]), np.array(points[1]), np.array(points[2])
            out = rectangle_vertices_from_three_points(p1, p2, p3)
            if out is None:
                print("  提示：前两点过近或三点共线/面积过小，请重选第三点。")
                points.pop()
                ax.set_title(
                    f"样本 {sample_name}：请重新点击第三点（与第一条边垂直方向上的角点）"
                )
                fig.canvas.draw_idle()
                return
            verts, u_unit = out
            state["quad"] = verts
            state["u_edge"] = np.asarray(u_unit, dtype=float).ravel()[:2].copy()
            redraw_rect(verts)
            ax.set_title(
                f"样本 {sample_name}：点击下方「✓ 确认」保存，或「重选区域」重新划定"
            )
        fig.canvas.draw_idle()

    def on_motion(event):
        # 已选两点、鼠标移出坐标轴时去掉预览，避免残留
        if len(points) == 2:
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                clear_preview_rect()
                fig.canvas.draw_idle()
                return
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if len(points) == 1:
            clear_preview_rect()
            p0 = points[0]
            preview_line.set_data([p0[0], event.xdata], [p0[1], event.ydata])
            fig.canvas.draw_idle()
            return
        if len(points) == 2:
            # 用当前鼠标位置作为临时第三点，实时画半透明预览长方形
            p1 = np.array(points[0])
            p2 = np.array(points[1])
            p3 = np.array([event.xdata, event.ydata])
            out = rectangle_vertices_from_three_points(p1, p2, p3)
            clear_preview_rect()
            if out is not None:
                verts, _u = out
                pr = MplPolygon(
                    verts,
                    closed=True,
                    facecolor="orange",
                    edgecolor="darkorange",
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.2,
                    zorder=2.5,
                )
                ax.add_patch(pr)
                preview_rect_patch_holder[0] = pr
            fig.canvas.draw_idle()
            return

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    cid_motion = fig.canvas.mpl_connect("motion_notify_event", on_motion)

    # 底部按钮（figure 坐标）
    ax_ok = fig.add_axes([0.28, 0.02, 0.18, 0.07])
    ax_rst = fig.add_axes([0.52, 0.02, 0.22, 0.07])
    btn_ok = Button(ax_ok, "✓ 确认")
    btn_rst = Button(ax_rst, "↺ 重选区域")
    btn_ok.on_clicked(on_confirm_clicked)
    btn_rst.on_clicked(on_reset_clicked)

    print(
        "  请点击：1) 起点  2) 第二点（定第一条边的方向）  "
        "3) 第三点：可预览半透明长方形后点击；然后点「✓ 确认」保存，或「重选区域」重来。"
    )
    plt.show(block=False)
    # 等待用户点「确认」（或提前关闭窗口 = 取消）
    while plt.fignum_exists(fig.number) and not state["confirmed"]:
        plt.pause(0.05)

    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_motion)

    if not state["confirmed"]:
        if plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

    return state["quad"], state["u_edge"]


def save_large_subsample(
    pore_data: pd.DataFrame,
    throat_data: pd.DataFrame,
    pore_mask: np.ndarray,
    throat_mask: np.ndarray,
    sample_name: str,
    output_subdir: Path,
    tag: str,
    quad_vertices_xy_nm: np.ndarray,
    permeation_direction_xy: np.ndarray,
) -> None:
    """保存到 output_subdir / f'{sample_name}_{tag}_*.xlsx' 与 metadata.json"""
    output_subdir = Path(output_subdir)
    output_subdir.mkdir(parents=True, exist_ok=True)

    base = f"{sample_name}_{tag}"
    sub_pores = pore_data[pore_mask].copy()
    sub_throats = throat_data[throat_mask].copy()

    pore_path = output_subdir / f"{base}_pores.xlsx"
    throat_path = output_subdir / f"{base}_throats.xlsx"
    meta_path = output_subdir / f"{base}_metadata.json"

    sub_pores.to_excel(pore_path, index=False)
    sub_throats.to_excel(throat_path, index=False)

    q = np.asarray(quad_vertices_xy_nm, dtype=float)
    u = np.asarray(permeation_direction_xy, dtype=float).ravel()[:2]
    nu = float(np.linalg.norm(u))
    if nu > 1e-12:
        u = u / nu
    else:
        u = np.array([1.0, 0.0], dtype=float)
    verts_list = [
        {"name": "P1_start", "x_nm": float(q[0, 0]), "y_nm": float(q[0, 1])},
        {"name": "I_foot", "x_nm": float(q[1, 0]), "y_nm": float(q[1, 1])},
        {"name": "P3", "x_nm": float(q[2, 0]), "y_nm": float(q[2, 1])},
        {"name": "D", "x_nm": float(q[3, 0]), "y_nm": float(q[3, 1])},
    ]
    meta = {
        "tool": "interactive_large_subsample_box",
        "source_sample": sample_name,
        "tag": tag,
        "rectangle_vertices_xy_nm": verts_list,
        "vertex_order": "P1 → I(垂足) → P3 → D，对边平行",
        "permeation_direction_xy": {
            "ux": float(u[0]),
            "uy": float(u[1]),
            "description": "X–Y 平面单位向量，由前两点 P1→P2 定义（与第一条边同向），用于渗透/流动方向参考",
        },
        "coordinate_system": "X–Y 为孔心投影平面 (nm)",
        "n_pores": int(len(sub_pores)),
        "n_throats": int(len(sub_throats)),
        "note": "X–Y 平面长方形内保留孔（投影）；喉要求两端孔均在保留集合内",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"  已保存大子样本:")
    print(f"    孔: {pore_path} ({len(sub_pores)} 个)")
    print(f"    喉: {throat_path} ({len(sub_throats)} 个)")
    print(f"    元数据: {meta_path}")
    print(f"    渗透方向 (X–Y 单位向量, P1→P2): ux={u[0]:.6f}, uy={u[1]:.6f}")


def main():
    script_dir = Path(__file__).resolve().parent
    data_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description="X–Y 平面矩形框选，导出框内孔/喉至 large_subsamples"
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        default=None,
        help="样本名；不提供则交互式选择",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default=str(data_root / "throats_and_pores_xlsx"),
        help="样本目录（默认 肾脏/throats_and_pores_xlsx）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(data_root / "large_subsamples"),
        help="大子样本根目录（默认 肾脏/large_subsamples），其下再建样本子文件夹",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="largebox",
        help="输出文件名后缀：{样本名}_{tag}_pores.xlsx（默认 largebox）",
    )
    parser.add_argument(
        "--alpha",
        type=str,
        default="auto",
        help="Alpha shape：auto=自动估计；0=凸包；或正数手动指定（与 correlation_length 一致）",
    )
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    if not sample_dir.is_absolute():
        sample_dir = data_root / args.sample_dir

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = data_root / args.output_dir

    if args.sample_name is None:
        sample_name = interactive_select_sample(sample_dir)
        if sample_name is None:
            sys.exit(1)
    else:
        sample_name = args.sample_name

    pores_file = sample_dir / f"{sample_name}_pores.xlsx"
    throats_file = sample_dir / f"{sample_name}_throats.xlsx"
    if not pores_file.exists():
        print(f"错误：未找到 {pores_file}")
        sys.exit(1)
    if not throats_file.exists():
        print(f"错误：未找到 {throats_file}")
        sys.exit(1)

    df_pores = pd.read_excel(pores_file)
    df_throats = pd.read_excel(throats_file)
    pore_coords = df_pores[["X Coord", "Y Coord", "Z Coord"]].values
    pore_coords_2d = project_to_xy(pore_coords)

    alpha_auto = str(args.alpha).strip().lower() == "auto"
    alpha_val = 0.0
    if not alpha_auto:
        try:
            alpha_val = float(args.alpha)
        except (ValueError, TypeError):
            alpha_auto = True
    polygon, alpha_used = get_alpha_shape_polygon_2d(pore_coords_2d, alpha=alpha_val, alpha_auto=alpha_auto)
    if polygon is None:
        print("  提示：未得到 2D 边界多边形（点数过少或缺少依赖），仅显示散点")
    else:
        print(f"  Alpha shape alpha={alpha_used:.6f}（{'自动' if alpha_auto else '手动'}）")

    sel = interactive_select_rectangle_three_points(pore_coords_2d, sample_name, polygon=polygon)
    if sel is None:
        print("未获得有效长方形区域，取消保存。")
        sys.exit(1)
    quad, u_perm = sel

    pore_mask = pore_mask_in_xy_polygon(pore_coords, quad)
    kept_ids = set(df_pores.loc[pore_mask, "Pore ID"].values)
    throat_mask = filter_throats_by_pores(df_throats, kept_ids)

    if not np.any(pore_mask):
        print("长方形内无孔，不保存。")
        sys.exit(1)

    sample_out = out_root / sample_name
    save_large_subsample(
        df_pores,
        df_throats,
        pore_mask,
        throat_mask,
        sample_name,
        sample_out,
        tag=args.tag.strip() or "largebox",
        quad_vertices_xy_nm=quad,
        permeation_direction_xy=u_perm,
    )
    print("\n完成。")


if __name__ == "__main__":
    main()

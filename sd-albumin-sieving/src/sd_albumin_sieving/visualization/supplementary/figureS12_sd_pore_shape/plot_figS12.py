from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(__file__).resolve().parents[5] / "outputs"
REFERENCE_LENGTH_NM = 50.2439981515067
THRESHOLD_LENGTH_NM = 52.379922487355266


def apply_arial_font() -> None:
    matplotlib.rcParams.update(
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


def _geometry() -> tuple[float, float, float]:
    vertices = pd.read_csv(HERE / "frozen_data" / "SDsubunitCorrect_closed_angle_117_expanded_symmetric_hexagon.csv")
    xy = vertices[["x_nm", "y_nm"]].to_numpy(float)
    current_major = float(np.ptp(xy[:, 1]))
    sides = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    slanted = float(np.min(sides))
    vertical = float(np.max(sides))
    return current_major, slanted, vertical


def _hexagon(major_nm: float, slanted_nm: float, vertical_nm: float) -> tuple[np.ndarray, float]:
    half_major = major_nm / 2.0
    half_vertical = vertical_nm / 2.0
    delta = half_major - half_vertical
    half_minor = math.sqrt(max(0.0, slanted_nm**2 - delta**2))
    vertices = np.array([[0, -half_major], [half_minor, -half_vertical], [half_minor, half_vertical], [0, half_major], [-half_minor, half_vertical], [-half_minor, -half_vertical]], float)
    return vertices, min(half_minor, half_major * half_minor / slanted_nm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Supplementary Fig. S12 SD-pore shape schematic.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "supplementary" / "figureS12_sd_pore_shape")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    apply_arial_font()
    rows = pd.read_csv(HERE / "frozen_data" / "SDsubunit_flexible_hexagon_by_slit_width.csv")
    lengths = pd.to_numeric(rows["estimated_protein_length_nm"], errors="coerce").dropna().to_numpy(float)
    current_major, slanted, vertical = _geometry()
    specs = [("Maximum", float(np.max(lengths)), "#d92943"), ("Mean", float(np.mean(lengths)), "#f0a22e"), ("Minimum", float(np.min(lengths)), "#189a5b")]
    threshold, _ = _hexagon(current_major * THRESHOLD_LENGTH_NM / REFERENCE_LENGTH_NM, slanted, vertical)
    threshold = np.vstack([threshold, threshold[0]])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.55))
    for ax, (label, length, color) in zip(axes, specs):
        vertices, inradius = _hexagon(current_major * length / REFERENCE_LENGTH_NM, slanted, vertical)
        closed = np.vstack([vertices, vertices[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=3.0)
        ax.fill(closed[:, 0], closed[:, 1], color=color, alpha=0.08)
        ax.plot(threshold[:, 0], threshold[:, 1], color="#202020", linestyle=(0, (4, 3)), linewidth=2.0)
        ax.axhline(0, color="#d9dee2", linewidth=0.8); ax.axvline(0, color="#d9dee2", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box"); ax.set_xlim(-6, 6); ax.set_ylim(-10, 10); ax.set_xticks(np.arange(-6, 7, 3))
        ax.set_title(f"{label} pore shape\nSD width={length:.2f} nm, r={inradius:.2f} nm", fontsize=14, pad=14); ax.tick_params(labelsize=12)
    for ax in axes[:2]:
        for xy, xytext in [((0, 7.6), (0, 9.0)), ((0, -7.6), (0, -9.0)), ((-2.8, 0), (-1.2, 0)), ((2.8, 0), (1.2, 0))]:
            ax.annotate("", xy=xy, xytext=xytext, arrowprops=dict(arrowstyle="-|>", color="#5c6f82", linewidth=1.5))
    fig.suptitle("Large-pore shape varies with SD width", fontsize=20, weight="bold")
    fig.supxlabel("Short-axis length (nm)", fontsize=16, y=0.145); fig.supylabel("Long-axis length (nm)", fontsize=16, x=0.045)
    fig.legend(handles=[Line2D([0], [0], color="#202020", linestyle=(0, (4, 3)), linewidth=2.2), Line2D([0], [0], color="#d92943", linewidth=3.0), Line2D([0], [0], color="#5c6f82", marker=">", linewidth=1.5)], labels=["albumin ellipsoid pass threshold", "pore shape", "shape change: max -> min"], loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=3, frameon=False, fontsize=12)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.72, bottom=0.26, wspace=0.48)
    output = args.out_dir.resolve() / "figS12.png"; output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.08); plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

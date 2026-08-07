"""
一键运行：对子样本进行分析和计算（pipelines 层，仅依赖 core_analysis 新结构）。
1. 读取子样本文件夹中的所有子样本
2. 对每个子样本读取法向量（渗透方向）
3. 运行孔隙与喉道分析 + 表面识别（core_analysis/subsamples/sub_*）
4. 运行溶剂/溶质输运计算与整体筛分系数（固定使用 sub_calculate_sieving_coefficient_new.py）
5. 若存在整体样本文件，对切割子样本模式也对整体样本执行分析（overall_*），结果保存为 <sample-name>_tot；大子样本模式（--large-subsamples）不跑全体，避免与框选大子样本混淆
6. 生成汇总表格（主口径为 ``筛过系数 (C_out/C_in, *)``；不再写入修正 Theta / J 基筛分列）

执行顺序（每个子样本，同一 output 子目录）：
  先 sub_analyze_pores_and_throats → 写出 pore_classification、溶剂网络等；
  再 sub_calculate_sieving_coefficient_new → 读取**同目录刚生成**的上述文件并求解压强。
  一次 pipeline 连续成功跑通时，分类与压强、筛分内 [出入口压强自检] 在逻辑上应对齐，不存在「正常运算却压强配错边界」的设计缺陷。
  若离线把「仅几何表面」与压强逐孔比对而困惑，或手工只重跑 analyze/sieving 其中一步，可能看到表间不一致——见上一级 `README_batch_processing.md`「表面分类与压强边界」。

用途：
    python pipelines/run_sub_sample_pipeline.py --sub-sample-dir data_subsamples --sample-name AS317
    # 大子样本（默认输入 large_subsamples/、默认输出 results_core_large_subsamples/，不与切割子样本混淆）
    python pipelines/run_sub_sample_pipeline.py --large-subsamples --sample-name AS317
    # 默认就是 new_J；可直接配合盒约束 LSQ / 稳定化（转发给子脚本）
    python pipelines/run_sub_sample_pipeline.py --large-subsamples --sample-name AS338 --new-j-box-lsq
    python pipelines/run_sub_sample_pipeline.py --large-subsamples --sample-name AS338 --new-j-stabilization 1e-10
    # 大子样本：省略 --sample-name 时自动处理 large_subsamples 下全部含大子样本的样本文件夹
    python pipelines/run_sub_sample_pipeline.py --large-subsamples

依赖：
    - core_analysis/subsamples/sub_analyze_pores_and_throats.py
    - core_analysis/subsamples/sub_calculate_sieving_coefficient_new.py
    - core_analysis/overall/overall_*.py（整体样本）
"""

import re
import os
import subprocess
import sys
import json
import shutil
import shlex
from pathlib import Path
import pandas as pd
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from gbm_sieving.data_io.aggregation import resolve_sieving_results_xlsx
from gbm_sieving.data_io.filenames import artifact_filename

# 筛分结果 xlsx 中「全溶剂网络出口总通量」行名
_PARAM_Q_FULL_SOLVENT = (
    "Total Solvent Flow (Q, full solvent network, nL/s)",
    "Total Solvent Flow (Q_total, nL/s)",  # 旧列名；新版子/整体脚本中该值为全网络 Q
)
_PARAM_Q_SUBNET_DIAG = "Diagnostic: Q_solute_subnet at exit (nL/s)"

# 筛分结果 xlsx 中「主筛过系数」行名（new 口径）
_PARAM_PRIMARY_SIEVING = (
    "sieving_coefficient (C_out/C_in)",
)
_PARAM_SIEVING_NEW_J_DIAG = (
    "sieving_coefficient (C_out/C_in, new_J)",
    "sieving_coefficient (C_out_mean/C_in, new_J, diagnostic)",
)
_PARAM_SIEVING_ITERATIVE = (
    "sieving_coefficient (C_out/C_in, iterative)",
)


def _param_is_full_network_solvent_flow(parameter) -> bool:
    p = parameter if isinstance(parameter, str) else str(parameter)
    return p in _PARAM_Q_FULL_SOLVENT


# 汇总 xlsx：与 sub_calculate_sieving_coefficient_new 一致，主口径为 C_out/C_in（不再写 Theta_corrected / J 基筛分）
_SUMMARY_SIEVING_COLUMNS_ORDER: list[str] = [
    "子样本名称",
    "Type",
    "筛过系数 (C_out/C_in, primary)",
    "筛过系数 (C_out/C_in, new_J)",
    "筛过系数 (C_out/C_in, iterative)",
    "溶剂通量 (Q_全网络, nL/s)",
    "溶剂通量 (Q_total, nL/s)",
    "溶剂通量 (Q_子网络, nL/s)",
    "Thickness (nm)",
]


def _reorder_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    first = [c for c in _SUMMARY_SIEVING_COLUMNS_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    return df[first + rest]


def _ensure_cout_cin_summary_columns(row: dict) -> None:
    """汇总表始终含 C_out/C_in 与 Q 列，缺省填 NaN。"""
    for k in (
        "筛过系数 (C_out/C_in, primary)",
        "筛过系数 (C_out/C_in, new_J)",
        "筛过系数 (C_out/C_in, iterative)",
        "溶剂通量 (Q_全网络, nL/s)",
        "溶剂通量 (Q_total, nL/s)",
        "溶剂通量 (Q_子网络, nL/s)",
    ):
        if k not in row:
            row[k] = np.nan


def _summary_row_from_sieving_result_df(
    df: pd.DataFrame,
    *,
    sub_label: str,
    sample_type: str,
    thickness_file: Path | None,
) -> dict:
    """从子/整体筛分 Parameter/Value 表构建汇总行（仅 C_out/C_in 与 Q、厚度等，不含修正 Theta）。"""
    row: dict = {"子样本名称": sub_label, "Type": sample_type}
    q_total_nls: float | None = None
    q_subnet_nls: float | None = None
    for _, r in df.iterrows():
        p = r.get("Parameter")
        if p is None or (isinstance(p, float) and pd.isna(p)):
            continue
        ps = str(p).strip()
        val = r.get("Value")
        if ps in _PARAM_PRIMARY_SIEVING:
            row["筛过系数 (C_out/C_in, primary)"] = val
        elif ps in _PARAM_SIEVING_NEW_J_DIAG:
            row["筛过系数 (C_out/C_in, new_J)"] = val
        elif ps in _PARAM_SIEVING_ITERATIVE:
            row["筛过系数 (C_out/C_in, iterative)"] = val
        elif _param_is_full_network_solvent_flow(ps):
            q_total_nls = val
        elif ps == _PARAM_Q_SUBNET_DIAG:
            q_subnet_nls = val
        elif ps == "Penetration axis index (0=X, 1=Y, 2=Z)":
            row["渗透轴索引"] = val
        elif ps == "Permeation plane cross-sectional area (nm^2)":
            row["渗透面面积 (nm²)"] = val
        elif ps == "Permeation plane cross-sectional area (µm^2)":
            row["渗透面面积 (µm²)"] = val
    if row.get("筛过系数 (C_out/C_in, primary)") is None:
        if row.get("筛过系数 (C_out/C_in, new_J)") is not None:
            row["筛过系数 (C_out/C_in, primary)"] = row["筛过系数 (C_out/C_in, new_J)"]
    row["溶剂通量 (Q_全网络, nL/s)"] = q_total_nls
    row["溶剂通量 (Q_total, nL/s)"] = q_total_nls
    row["溶剂通量 (Q_子网络, nL/s)"] = q_subnet_nls
    _ensure_cout_cin_summary_columns(row)
    if thickness_file is not None and thickness_file.exists():
        try:
            row["Thickness (nm)"] = load_thickness_nm(thickness_file)
            merge_thickness_summary_columns(row, thickness_file)
        except Exception:
            row["Thickness (nm)"] = np.nan
    else:
        row["Thickness (nm)"] = np.nan
    return row


def _resolve_sub_sieving_results_xlsx(sample_output_dir: Path, sub_name: str) -> Path | None:
    """返回当前项目规范的筛分汇总文件。"""
    return resolve_sieving_results_xlsx(sample_output_dir, sub_name)


def _resolve_iterative_sieving_results_xlsx(
    sample_output_dir: Path,
    sub_name: str,
    *,
    result_tag: str,
) -> Path | None:
    d = sample_output_dir / sub_name
    if not d.is_dir():
        return None
    tag = (result_tag or "").strip()
    tag_suffix = f"_{tag}" if tag else ""
    p = d / f"{sub_name}_sieving_results_iterative{tag_suffix}.xlsx"
    return p if p.is_file() else None


def _append_or_update_iterative_rows_into_classic_sieving(
    sample_output_dir: Path,
    sub_name: str,
    *,
    iterative_result_tag: str,
) -> None:
    """
    将 iterative 结果中的 J/Q 与 J/Q 口径筛过系数写入当前筛分汇总文件。
    """
    it_path = _resolve_iterative_sieving_results_xlsx(
        sample_output_dir,
        sub_name,
        result_tag=iterative_result_tag,
    )
    if it_path is None:
        return
    try:
        it_df = pd.read_excel(it_path)
    except Exception:
        return
    if "Parameter" not in it_df.columns or "Value" not in it_df.columns:
        return
    p2v: dict[str, float] = {}
    for _, r in it_df.iterrows():
        p = r.get("Parameter")
        if p is None:
            continue
        try:
            p2v[str(p).strip()] = float(r.get("Value"))
        except Exception:
            continue

    j_tot = p2v.get("Total Solute Flow (J_out, m3/s)", np.nan)
    q_tot = p2v.get("Total Solvent Flow (Q_out, m3/s)", np.nan)
    if np.isfinite(j_tot) and np.isfinite(q_tot) and abs(q_tot) > 0.0:
        sieve_iter = float(j_tot / q_tot)
    else:
        sieve_iter = np.nan

    classic_path = _resolve_sub_sieving_results_xlsx(sample_output_dir, sub_name)
    if classic_path is not None and classic_path.is_file():
        try:
            base_df = pd.read_excel(classic_path)
        except Exception:
            base_df = pd.DataFrame(columns=["Parameter", "Value"])
    else:
        classic_dir = sample_output_dir / sub_name
        classic_dir.mkdir(parents=True, exist_ok=True)
        classic_path = classic_dir / artifact_filename(sub_name, "sieving_summary")
        base_df = pd.DataFrame(columns=["Parameter", "Value"])

    if "Parameter" not in base_df.columns:
        base_df["Parameter"] = []
    if "Value" not in base_df.columns:
        base_df["Value"] = []

    rows_to_upsert = {
        "Total Solute Flux (J_s_total, iterative)": j_tot,
        "Total Solvent Flow (Q_out, iterative, m3/s)": q_tot,
        "sieving_coefficient (C_out/C_in, iterative)": sieve_iter,
    }
    for param, val in rows_to_upsert.items():
        mask = base_df["Parameter"].astype(str).str.strip() == param
        if mask.any():
            base_df.loc[mask, "Value"] = val
        else:
            base_df = pd.concat(
                [
                    base_df,
                    pd.DataFrame([{"Parameter": param, "Value": val}]),
                ],
                ignore_index=True,
            )
    base_df.to_excel(classic_path, index=False)
    print(f"已将 iterative J/Q 结果写入: {classic_path}")


def read_normal_from_metadata(metadata: dict):
    """
    从 metadata.json 读取 X–Y 渗透方向，供 sub_analyze --normal-vector。
    支持：
      - interactive_sample_splitter: normal_vector.nx / .ny
      - interactive_large_subsample_box: permeation_direction_xy.ux / .uy
    """
    if not isinstance(metadata, dict):
        return None
    if "normal_vector" in metadata:
        nv = metadata["normal_vector"]
        if isinstance(nv, dict) and "nx" in nv and "ny" in nv:
            try:
                return [float(nv["nx"]), float(nv["ny"])]
            except (TypeError, ValueError):
                pass
    p = metadata.get("permeation_direction_xy")
    if isinstance(p, dict) and "ux" in p and "uy" in p:
        try:
            return [float(p["ux"]), float(p["uy"])]
        except (TypeError, ValueError):
            pass
    return None


def _resolve_sub_sample_root(sub_sample_dir_base: Path, sample_name: str, pattern: str) -> tuple[Path, list]:
    """
    在顶层或 <sample_name>/ 子文件夹中查找匹配 pattern 的 *_pores.xlsx。
    返回 (实际使用的目录, pore_files 列表)。
    """
    pore_files = sorted(sub_sample_dir_base.glob(pattern))
    use_dir = sub_sample_dir_base
    if len(pore_files) == 0:
        sample_sub_dir = sub_sample_dir_base / sample_name
        if sample_sub_dir.exists():
            pore_files = sorted(sample_sub_dir.glob(pattern))
            if len(pore_files) > 0:
                print(f"在子文件夹中找到文件: {sample_sub_dir}")
                use_dir = sample_sub_dir
    return use_dir, pore_files


def load_sub_sample_metadata(sub_sample_dir, sample_name):
    """
    加载切割子样本（模式 {sample_name}_sub*_pores.xlsx）及元数据。
    返回: list of {'name', 'pores_file', 'throats_file', 'normal_vec': [nx, ny] or None}
    """
    sub_sample_dir = Path(sub_sample_dir)
    pattern = f"{sample_name}_sub*_pores.xlsx"
    use_dir, pore_files = _resolve_sub_sample_root(sub_sample_dir, sample_name, pattern)
    if len(pore_files) == 0:
        print(f"错误：未找到子样本文件（模式: {pattern}）")
        return []
    print(f"找到 {len(pore_files)} 个子样本文件（切割子样本模式）")
    sub_samples = []
    for pore_file in pore_files:
        sub_name = pore_file.stem.replace("_pores", "")
        throat_file = use_dir / f"{sub_name}_throats.xlsx"
        metadata_file = use_dir / f"{sub_name}_metadata.json"
        if not throat_file.exists():
            continue
        normal_vec = None
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    normal_vec = read_normal_from_metadata(json.load(f))
            except Exception:
                pass
        sub_samples.append(
            {
                "name": sub_name,
                "pores_file": pore_file,
                "throats_file": throat_file,
                "normal_vec": normal_vec,
            }
        )
    return sub_samples


def load_large_subsample_metadata(sub_sample_dir, sample_name, *, quiet: bool = False):
    """
    加载 large_subsamples 下的大子样本：匹配 {sample_name}_*_pores.xlsx，排除切割命名 {sample_name}_subN_。
    元数据：permeation_direction_xy（interactive_large_subsample_box）或 normal_vector（切割工具）。
    """
    sub_sample_dir = Path(sub_sample_dir)
    pattern = f"{sample_name}_*_pores.xlsx"
    use_dir, pore_files = _resolve_sub_sample_root(sub_sample_dir, sample_name, pattern)
    if len(pore_files) == 0:
        if not quiet:
            print(f"错误：未找到大子样本文件（模式: {pattern}，目录: {sub_sample_dir}）")
        return []

    def is_classic_sub(p: Path) -> bool:
        stem = p.stem.replace("_pores", "")
        return re.match(rf"^{re.escape(sample_name)}_sub\d+$", stem) is not None

    # 排除切割子样本命名 AS317_sub1，保留 AS317_largebox、AS317_region1 等
    filtered = [p for p in pore_files if not is_classic_sub(p)]
    if not filtered:
        print(
            "提示：仅发现切割子样本命名 (*_subN_pores)；大子样本模式已排除此类文件。"
            "若需跑切割子样本，请勿使用 --large-subsamples。"
        )
        return []
    pore_files = sorted(filtered)
    print(f"找到 {len(pore_files)} 个大子样本文件（large_subsamples 模式，已排除 *_subN_*）")
    sub_samples = []
    for pore_file in pore_files:
        sub_name = pore_file.stem.replace("_pores", "")
        throat_file = use_dir / f"{sub_name}_throats.xlsx"
        metadata_file = use_dir / f"{sub_name}_metadata.json"
        if not throat_file.exists():
            print(f"  跳过（无喉表）: {sub_name}")
            continue
        normal_vec = None
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    normal_vec = read_normal_from_metadata(json.load(f))
            except Exception:
                pass
        sub_samples.append(
            {
                "name": sub_name,
                "pores_file": pore_file,
                "throats_file": throat_file,
                "normal_vec": normal_vec,
            }
        )
    return sub_samples


def discover_large_sample_names(sub_sample_dir_base: Path) -> list[str]:
    """
    在 large_subsamples 顶层下扫描子文件夹，若其中存在至少一个大子样本（非 *_subN_*）则加入列表。
    """
    sub_sample_dir_base = Path(sub_sample_dir_base)
    names: list[str] = []
    for p in sorted(sub_sample_dir_base.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        sub = load_large_subsample_metadata(sub_sample_dir_base, p.name, quiet=True)
        if len(sub) > 0:
            names.append(p.name)
    return names


def run_script(cmd: list, desc: str):
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(SRC_ROOT), env.get("PYTHONPATH", "")) if part
        )
        subprocess.run(cmd, check=True, text=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n执行 {desc} 时出错，返回码: {e.returncode}")
        return False
    return True


def load_thickness_nm(thickness_file: Path) -> float:
    """从厚度汇总表读取 Average Thickness (nm)（倾斜修正后的主厚度）。"""
    df = pd.read_excel(thickness_file)
    if "Average Thickness (nm)" not in df.columns:
        raise ValueError(f"厚度表需包含列 'Average Thickness (nm)': {thickness_file}")
    return float(df["Average Thickness (nm)"].iloc[0])


def merge_thickness_summary_columns(row: dict, thickness_file: Path) -> None:
    """将 sub/overall 写出的扩展列并入汇总行（角度、沿轴厚度、法向等）。"""
    try:
        df = pd.read_excel(thickness_file)
    except Exception:
        return
    if df is None or len(df) == 0:
        return
    r0 = df.iloc[0]
    extra_map = {
        "Thickness_along_axis_nm": "Thickness_along_axis_nm",
        "Theta_deg_vs_XY": "Theta_deg_vs_XY",
        "Phi_deg_vs_Z": "Phi_deg_vs_Z",
        "Used_tilt_refine": "Used_tilt_refine",
        "Tilt_threshold_deg": "Tilt_threshold_deg",
        "nx": "nx",
        "ny": "ny",
        "nz": "nz",
    }
    for excel_col, row_key in extra_map.items():
        if excel_col not in df.columns:
            continue
        v = r0[excel_col]
        if pd.isna(v):
            row[row_key] = np.nan
        elif excel_col == "Used_tilt_refine":
            row[row_key] = int(float(v))
        else:
            row[row_key] = float(v)


def plot_thickness_vs_log_sieving(
    summary_df: pd.DataFrame,
    sample_output_dir: Path,
    sample_name: str,
    png_filename: str | None = None,
):
    """画厚度 vs log10(筛过系数) 散点图，仅含有效筛过系数的样本，WT/AS 不同颜色。主口径：C_out/C_in。"""
    col_sieving = "筛过系数 (C_out/C_in, primary)"
    if col_sieving not in summary_df.columns:
        col_sieving = "筛过系数 (Theta_default)"
    if col_sieving not in summary_df.columns:
        col_sieving = (
            "筛过系数 (Theta_corrected)"
            if "筛过系数 (Theta_corrected)" in summary_df.columns
            else "筛过系数 (Theta_global)"
        )
    col_thickness = "Thickness (nm)"
    col_type = "Type"
    if col_sieving not in summary_df.columns or col_thickness not in summary_df.columns:
        return
    sieving = pd.to_numeric(summary_df[col_sieving], errors="coerce")
    thickness = pd.to_numeric(summary_df[col_thickness], errors="coerce")
    valid = (sieving > 0) & thickness.notna()
    if not valid.any():
        return
    x = thickness[valid].values
    y = np.log10(sieving[valid].values)
    types = summary_df.loc[valid, col_type].values if col_type in summary_df.columns else None
    fig, ax = plt.subplots(figsize=(6, 4))
    if types is not None and len(np.unique(types)) >= 2:
        for t in ["WT", "AS"]:
            mask = types == t
            if mask.any():
                ax.scatter(x[mask], y[mask], label=t, alpha=0.7)
    else:
        ax.scatter(x, y, alpha=0.7)
    ax.set_xlabel("Thickness (nm)")
    ax.set_ylabel("log10(Sieving coefficient)")
    ax.set_title(f"Thickness vs log10(C_out/C_in) — {sample_name}")
    if types is not None and len(np.unique(types)) >= 2:
        ax.legend()
    plt.tight_layout()
    out_path = sample_output_dir / (png_filename or f"{sample_name}_thickness_vs_log_sieving.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"散点图已保存: {out_path}")


def run_one_sample_pipeline(
    sample_name: str,
    args: argparse.Namespace,
    project_root: Path,
    data_root: Path,
    sub_sample_dir_base: Path,
    output_dir_base: Path,
    *,
    fail_fast: bool,
) -> bool:
    """处理单个样本（切割子样本或大子样本下的一个 sample 子文件夹）。"""
    sub_sample_dir = sub_sample_dir_base / sample_name
    if not sub_sample_dir.exists():
        print(f"错误：未找到该样本的子样本文件夹: {sub_sample_dir}")
        if fail_fast:
            sys.exit(1)
        return False

    sample_output_dir = output_dir_base / sample_name
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    _sieving_py = "gbm_sieving.simulation.transport.sieving_solver"
    print(f"筛分步骤：默认仅 new_J（{_sieving_py}）")
    run_iterative = bool(getattr(args, "iterative", False) or getattr(args, "iterative_only", False))
    if run_iterative:
        print(
            f"迭代步骤：已启用（{'仅迭代' if getattr(args, 'iterative_only', False) else '与现有筛分并行'}），"
            "输出文件名使用 *_iterative* 后缀，不覆盖现有筛分结果。"
        )

    if args.large_subsamples:
        print("模式: 大子样本（large_subsamples，interactive_large_subsample_box）")
        sub_samples = load_large_subsample_metadata(sub_sample_dir, sample_name)
    else:
        print("模式: 切割子样本（data_subsamples，*_subN_*）")
        sub_samples = load_sub_sample_metadata(sub_sample_dir, sample_name)
    if len(sub_samples) == 0:
        print(f"错误：样本 {sample_name} 未找到可用的子样本文件。")
        if fail_fast:
            sys.exit(1)
        return False

    normals = [s["normal_vec"] for s in sub_samples if s.get("normal_vec") is not None]
    avg_normal_vec = None
    if len(normals) > 0:
        nx_mean = sum(n[0] for n in normals) / len(normals)
        ny_mean = sum(n[1] for n in normals) / len(normals)
        norm_len = (nx_mean ** 2 + ny_mean ** 2) ** 0.5
        if norm_len > 1e-10:
            avg_normal_vec = (nx_mean / norm_len, ny_mean / norm_len)

    successful_samples = []
    failed_samples = []
    for i, sub in enumerate(sub_samples, 1):
        sub_name = sub["name"]
        normal_vec = sub["normal_vec"]
        if normal_vec is None:
            failed_samples.append(sub_name)
            continue
        cmd1 = [
            sys.executable,
            "-m",
            "gbm_sieving.analysis.network_extraction.extract_network",
            "--sample-name",
            sub_name,
            "--sample-dir",
            str(sub_sample_dir),
            "--output-dir",
            str(sample_output_dir),
            "--radius-delta",
            str(args.radius_delta),
            "--normal-vector",
            str(normal_vec[0]),
            str(normal_vec[1]),
        ]
        if not run_script(cmd1, f"步骤1：{sub_name} - 孔/喉分析"):
            failed_samples.append(sub_name)
            continue
        classic_ok = True
        if not getattr(args, "iterative_only", False):
            cmd2 = [
                sys.executable,
                "-m",
                _sieving_py,
                "--sample-name",
                sub_name,
                "--sample-dir",
                str(sub_sample_dir),
                "--output-dir",
                str(sample_output_dir),
                "--radius-delta",
                str(args.radius_delta),
            ]
            if args.supernode_merge:
                cmd2.append("--supernode-merge")
            cmd2.extend(["--pressure-physics-tol", str(args.pressure_physics_tol)])
            cmd2.extend(["--pressure-flow-tol-pa", str(args.pressure_flow_tol_pa)])
            if args.assembly_row_audit:
                cmd2.append("--assembly-row-audit")
            cmd2.extend(["--linear-solver", str(args.linear_solver)])
            if args.compare_linear_solvers:
                cmd2.append("--compare-linear-solvers")
            cmd2.extend(["--gmres-rtol", str(args.gmres_rtol)])
            cmd2.extend(["--gmres-atol", str(args.gmres_atol)])
            cmd2.extend(["--gmres-maxiter", str(args.gmres_maxiter)])
            cmd2.extend(["--ilut-drop-tol", str(args.ilut_drop_tol)])
            cmd2.extend(["--ilut-fill-factor", str(args.ilut_fill_factor)])
            cmd2.extend(["--conc-equilibrate", str(getattr(args, "conc_equilibrate", "none"))])
            if float(args.new_j_stabilization) != 0.0:
                cmd2.extend(["--new-j-stabilization", str(args.new_j_stabilization)])
            if args.new_j_box_lsq:
                cmd2.append("--new-j-box-lsq")
                cmd2.extend(["--new-j-c-min", str(args.new_j_c_min)])
                cmd2.extend(["--new-j-c-max", str(args.new_j_c_max)])
                cmd2.extend(["--new-j-box-ridge", str(args.new_j_box_ridge)])

            if getattr(args, "concentration_clip_refine", False):
                cmd2.append("--concentration-clip-refine")
                cmd2.extend(["--concentration-clip-lo", str(args.concentration_clip_lo)])
                cmd2.extend(["--concentration-clip-hi", str(args.concentration_clip_hi)])
                cmd2.extend(["--clip-refine-max-outer", str(args.clip_refine_max_outer)])
                cmd2.extend(["--clip-refine-inner-maxiter", str(args.clip_refine_inner_maxiter)])
            cmd2.extend(["--weak-link-percentile", str(args.weak_link_percentile)])
            cmd2.extend(["--new-j-pe-adv-only-above", str(args.new_j_pe_adv_only_above)])
            cmd2.extend(["--new-j-pe-diff-only-below", str(args.new_j_pe_diff_only_below)])
            classic_ok = run_script(cmd2, f"步骤2：{sub_name} - 筛分系数")
        else:
            print(f"步骤2：{sub_name} - 已跳过现有筛分（--iterative-only）")

        iterative_ok = True
        if run_iterative:
            cmd_it = [
                sys.executable,
                str(project_root / "core_analysis" / "subsamples" / "sub_calculate_sieving_coefficient_iterative.py"),
                "--sample-name",
                sub_name,
                "--sample-dir",
                str(sub_sample_dir),
                "--output-dir",
                str(sample_output_dir),
                "--result-tag",
                str(getattr(args, "iterative_result_tag", "equiv_C_v0")),
            ]
            extra = str(getattr(args, "iterative_extra_args", "") or "").strip()
            # 默认把 new_J 矩阵浓度分布作为 explicit hold-check 起点（可被用户手动参数覆盖）
            if (
                classic_ok
                and (not getattr(args, "iterative_only", False))
                and ("--explicit-hold-start-concentration-xlsx" not in extra)
            ):
                _new_conc_candidates = [
                    sample_output_dir / sub_name / f"{sub_name}_concentration_distribution_equiv_C_v0.xlsx",
                    sample_output_dir / sub_name / f"{sub_name}_concentration_distribution.xlsx",
                ]
                _new_conc_file = next((p for p in _new_conc_candidates if p.exists()), None)
                if _new_conc_file is not None:
                    cmd_it.extend(
                        [
                            "--explicit-hold-start-concentration-xlsx",
                            str(_new_conc_file),
                        ]
                    )
                    print(
                        f"步骤3：{sub_name} - hold-check 起点使用 new_J 矩阵浓度文件: {_new_conc_file.name}"
                    )
                else:
                    print(
                        f"步骤3：{sub_name} - 未找到 new_J 矩阵浓度文件，hold-check 将回退到 explicit_best_residual"
                    )
            if extra:
                cmd_it.extend(shlex.split(extra))
            iterative_ok = run_script(cmd_it, f"步骤3：{sub_name} - 迭代筛分")
            if iterative_ok:
                _append_or_update_iterative_rows_into_classic_sieving(
                    sample_output_dir,
                    sub_name,
                    iterative_result_tag=str(getattr(args, "iterative_result_tag", "equiv_C_v0")),
                )

        if classic_ok and iterative_ok:
            successful_samples.append(sub_name)
        else:
            failed_samples.append(sub_name)

    total_sample_success = False
    total_sample_dir = data_root / "throats_and_pores_xlsx"
    total_pores = total_sample_dir / f"{sample_name}_pores.xlsx"
    total_throats = total_sample_dir / f"{sample_name}_throats.xlsx"
    if args.large_subsamples:
        print("大子样本模式：跳过全体样本（throats_and_pores_xlsx 的 overall_* 与 *_tot）。")
    elif total_pores.exists() and total_throats.exists():
        print("整理版第一阶段仅运行子样本核心流程；旧 overall 分支暂不复制或调用。")

    sample_type = "AS" if sample_name.upper().startswith("AS") else "WT"
    if len(successful_samples) > 0 or total_sample_success:
        summary_data = []
        for sub_name in successful_samples:
            result_file = _resolve_sub_sieving_results_xlsx(sample_output_dir, sub_name)
            if result_file is None or not result_file.exists():
                continue
            try:
                df = pd.read_excel(result_file)
                thickness_file = sample_output_dir / sub_name / f"{sub_name}_thickness_summary.xlsx"
                row = _summary_row_from_sieving_result_df(
                    df,
                    sub_label=sub_name,
                    sample_type=sample_type,
                    thickness_file=thickness_file,
                )
                if len(row) >= 4:
                    summary_data.append(row)
            except Exception:
                pass
        if total_sample_success:
            tot_name = f"{sample_name}_tot"
            tot_dir = sample_output_dir / tot_name
            tot_file = tot_dir / artifact_filename(sample_name, "sieving_summary")
            if tot_file.is_file():
                try:
                    df = pd.read_excel(tot_file)
                    tot_thickness_file = sample_output_dir / tot_name / f"{sample_name}_thickness_summary.xlsx"
                    row = _summary_row_from_sieving_result_df(
                        df,
                        sub_label=tot_name,
                        sample_type=sample_type,
                        thickness_file=tot_thickness_file,
                    )
                    if len(row) >= 4:
                        summary_data.append(row)
                except Exception:
                    pass
        if summary_data:
            summary_df = _reorder_summary_columns(pd.DataFrame(summary_data))
            if args.large_subsamples:
                summary_file = sample_output_dir / f"{sample_name}_large_subsamples_summary.xlsx"
                png_name = f"{sample_name}_large_subsamples_thickness_vs_log_sieving.png"
            else:
                summary_file = sample_output_dir / f"{sample_name}_sub_samples_summary.xlsx"
                png_name = f"{sample_name}_thickness_vs_log_sieving.png"
            summary_df.to_excel(summary_file, index=False)
            print(f"汇总表格已保存: {summary_file}")
            plot_thickness_vs_log_sieving(summary_df, sample_output_dir, sample_name, png_filename=png_name)
    print(f"成功: {len(successful_samples)}/{len(sub_samples)}")
    if failed_samples:
        print(f"未跑通（缺渗透方向或脚本失败）: {', '.join(failed_samples)}")
        if args.large_subsamples:
            print("  大子样本请在 *_metadata.json 中提供 permeation_direction_xy.ux/uy（或 normal_vector.nx/ny）。")
    if total_sample_success:
        print(f"整体样本: {sample_name}_tot")
    print(f"结果目录: {sample_output_dir}")
    return not failed_samples


def main():
    parser = argparse.ArgumentParser(description="对子样本进行分析和计算")
    parser.add_argument("--sub-sample-dir", type=str, default="data_subsamples")
    parser.add_argument(
        "--sample-name",
        type=str,
        default=None,
        help="样本名（子文件夹名）。切割子样本模式必填；--large-subsamples 时可省略，将自动扫描并处理全部样本。",
    )
    parser.add_argument("--output-dir", type=str, default="results_core_subsamples")
    parser.add_argument("--radius-delta", type=float, default=0.0)
    parser.add_argument(
        "--iterative",
        action="store_true",
        help="额外运行 iterative 迭代筛分脚本（输出 *_iterative* 文件，不覆盖现有筛分结果）。",
    )
    parser.add_argument(
        "--iterative-only",
        action="store_true",
        help="仅运行 iterative 迭代筛分（仍会执行步骤1 analyze 以生成前置分类文件）。",
    )
    parser.add_argument(
        "--iterative-result-tag",
        type=str,
        default="equiv_C_v0",
        help="传给 iterative 脚本的 --result-tag（用于输出文件名后缀）。",
    )
    parser.add_argument(
        "--iterative-extra-args",
        type=str,
        default="",
        help=(
            "透传给 iterative 脚本的附加参数字符串。示例："
            "'--min-steps 20 --tol 1e-10 --save-history-every 10'"
        ),
    )
    parser.add_argument(
        "--large-subsamples",
        action="store_true",
        help=(
            "大子样本模式：从 large_subsamples 读取 {sample}_*_pores.xlsx（排除切割命名 *_subN_*），"
            "metadata 支持 permeation_direction_xy（interactive_large_subsample_box）。"
            "未显式改 --sub-sample-dir 时，顶层目录默认为 large_subsamples（仍使用子文件夹 <sample-name>/）。"
            "未显式改 --output-dir 时，默认输出到 results_core_large_subsamples，避免与切割子样本的 results_core_subsamples 混淆。"
            "省略 --sample-name 时自动扫描顶层下所有子文件夹并逐个处理（仅包含有大子样本 xlsx 的目录）。"
        ),
    )
    parser.add_argument(
        "--supernode-merge",
        action="store_true",
        help="传给筛分计算：启用 B0 超节点合并；默认不传则每孔一节点。",
    )
    parser.add_argument(
        "--pressure-physics-tol",
        type=float,
        default=10.0,
        help="传给筛分：Kirchhoff 解须在出入口压强带内，否则子进程 exit(1)；负值关闭检验。默认 10 Pa。",
    )
    parser.add_argument(
        "--assembly-row-audit",
        dest="assembly_row_audit",
        action="store_true",
        help="传给筛分子脚本：组装 A 后打印非零大行矩阵元的分项（按喉 Q、Kc、diff、Pe 等）。",
    )
    parser.add_argument(
        "--new-j-stabilization",
        type=float,
        default=0.0,
        help="传给筛分 new_J：对角稳定化 A += ε·I；默认 0 不传（与子脚本一致）。",
    )
    parser.add_argument(
        "--pressure-flow-tol-pa",
        type=float,
        default=1e-9,
        help="传给筛分：喉两端 |ΔP| 低于此值(Pa)时视为无压力驱动对流，仅保留扩散（若有）。",
    )
    parser.add_argument(
        "--new-j-box-lsq",
        dest="new_j_box_lsq",
        action="store_true",
        help="传给筛分：new_J 用盒约束 lsq_linear（[new-j-c-min,new-j-c-max]）。",
    )
    parser.add_argument(
        "--no-new-j-box-lsq",
        dest="new_j_box_lsq",
        action="store_false",
        help="显式关闭 new_J 盒约束 LSQ（默认）。",
    )
    parser.set_defaults(new_j_box_lsq=False)
    parser.add_argument(
        "--new-j-c-min",
        type=float,
        default=0.0,
        help="仅与 --new-j-box-lsq 一起传给子脚本：浓度下界。",
    )
    parser.add_argument(
        "--new-j-c-max",
        type=float,
        default=1.0,
        help="仅与 --new-j-box-lsq 一起传给子脚本：浓度上界。",
    )
    parser.add_argument(
        "--new-j-box-ridge",
        type=float,
        default=1e-10,
        help="仅与 --new-j-box-lsq 一起传给子脚本：岭项 λ。",
    )
    parser.add_argument(
        "--new-j",
        dest="solve_new_j",
        action="store_true",
        help=(
            "已弃用：new_J 现为默认口径。该参数保留兼容旧命令行。"
        ),
    )
    parser.set_defaults(solve_new_j=False)
    parser.add_argument(
        "--linear-solver",
        type=str,
        default="direct",
        choices=("direct", "gmres-ilut"),
        help="传给筛分子脚本：线性求解器（direct 或 gmres-ilut）。",
    )
    parser.add_argument(
        "--compare-linear-solvers",
        dest="compare_linear_solvers",
        action="store_true",
        help="传给筛分子脚本：同次运行对比 direct 与 gmres-ilut（打印残差与解差）。",
    )
    parser.set_defaults(compare_linear_solvers=False)
    parser.add_argument(
        "--gmres-rtol",
        type=float,
        default=1e-50,
        help="传给筛分子脚本：GMRES 相对容差 rtol。",
    )
    parser.add_argument(
        "--gmres-atol",
        type=float,
        default=0.0,
        help="传给筛分子脚本：GMRES 绝对容差 atol。",
    )
    parser.add_argument(
        "--gmres-maxiter",
        type=int,
        default=3000,
        help="传给筛分子脚本：GMRES 最大迭代次数（默认 3000）。",
    )
    parser.add_argument(
        "--ilut-drop-tol",
        type=float,
        default=1e-4,
        help="传给筛分子脚本：ILUT drop_tol。",
    )
    parser.add_argument(
        "--ilut-fill-factor",
        type=float,
        default=20.0,
        help="传给筛分子脚本：ILUT fill_factor。",
    )
    parser.add_argument(
        "--conc-equilibrate",
        type=str,
        default="none",
        choices=("none", "row", "row-col"),
        help=(
            "传给筛分子脚本：浓度方程 Ax=b 求解前等式缩放；"
            "none=关闭，row=行平衡，row-col=行/列平衡（与子脚本 --conc-equilibrate 一致）。"
        ),
    )
    parser.add_argument(
        "--concentration-clip-refine",
        dest="concentration_clip_refine",
        action="store_true",
        help="传给筛分：direct 越界则 clip 初值 + GMRES+ILUT 外循环（见子脚本帮助）。",
    )
    parser.set_defaults(concentration_clip_refine=False)
    parser.add_argument(
        "--concentration-clip-lo",
        type=float,
        default=0.0,
        help="与 --concentration-clip-refine 一起传给筛分。",
    )
    parser.add_argument(
        "--concentration-clip-hi",
        type=float,
        default=1.0,
        help="与 --concentration-clip-refine 一起传给筛分。",
    )
    parser.add_argument(
        "--clip-refine-max-outer",
        type=int,
        default=5,
        help="与 --concentration-clip-refine 一起传给筛分：外循环上限。",
    )
    parser.add_argument(
        "--clip-refine-inner-maxiter",
        type=int,
        default=10000,
        help="与 --concentration-clip-refine 一起传给筛分：每轮 GMRES 最大迭代（与 sub_calculate_sieving_coefficient_new 默认一致）。",
    )
    parser.add_argument(
        "--weak-link-percentile",
        type=float,
        default=1.0,
        help="传给筛分 D2.8：在当前可通行喉上取 |Q| 与 D_eff 的 p 分位（默认 1 即最弱 1%%），二者同时处于弱尾才剪枝。",
    )
    parser.add_argument(
        "--new-j-pe-adv-only-above",
        type=float,
        default=100.0,
        help="传给筛分：|Pe| 超过此值时该喉仅组装纯对流；0 关闭分段。",
    )
    parser.add_argument(
        "--new-j-pe-diff-only-below",
        type=float,
        default=0.01,
        help="传给筛分：|Pe| 低于此值时该喉仅 Fick 扩散；0 恢复旧口径（仅 |Pe|<1e-14）。",
    )
    args = parser.parse_args()

    if args.iterative or args.iterative_only:
        parser.error("迭代验证求解器尚未纳入整理版核心；请先使用默认 sieving_solver。")

    # 脚本在 pipelines/，项目根为 analyze_and_calculate_2，数据根为 肾脏
    script_dir = Path(__file__).resolve().parent
    project_root = PROJECT_ROOT
    data_root = script_dir.parent.parent

    output_dir_arg = args.output_dir
    if args.large_subsamples and output_dir_arg == "results_core_subsamples":
        output_dir_arg = "results_core_large_subsamples"

    sub_sample_dir_arg = args.sub_sample_dir
    if args.large_subsamples and sub_sample_dir_arg == "data_subsamples":
        sub_sample_dir_arg = "large_subsamples"

    sub_sample_dir_base = Path(sub_sample_dir_arg)
    if not sub_sample_dir_base.is_absolute():
        sub_sample_dir_base = data_root / sub_sample_dir_arg
    if not sub_sample_dir_base.exists():
        print(f"错误：子样本顶层文件夹不存在: {sub_sample_dir_base}")
        sys.exit(1)

    output_dir_base = Path(output_dir_arg)
    if not output_dir_base.is_absolute():
        output_dir_base = data_root / output_dir_arg
    if args.large_subsamples and args.output_dir == "results_core_subsamples":
        print(
            f"大子样本默认输出根目录: {output_dir_base}（与切割子样本的 results_core_subsamples 分开）"
        )
    output_dir_base.mkdir(parents=True, exist_ok=True)

    if args.large_subsamples:
        if args.sample_name:
            sample_names = [args.sample_name]
        else:
            sample_names = discover_large_sample_names(sub_sample_dir_base)
            if not sample_names:
                print("错误：在 large_subsamples 顶层下未发现任何含大子样本 xlsx 的样本子文件夹。")
                sys.exit(1)
            print(
                f"未指定 --sample-name，将依次处理 {len(sample_names)} 个样本: {', '.join(sample_names)}"
            )
    else:
        if not args.sample_name:
            print("错误：切割子样本模式必须指定 --sample-name")
            sys.exit(1)
        sample_names = [args.sample_name]

    fail_fast = len(sample_names) == 1
    all_successful = True
    for sample_name in sample_names:
        print("\n" + "=" * 70)
        print(f"样本: {sample_name}")
        print("=" * 70)
        sample_successful = run_one_sample_pipeline(
            sample_name,
            args,
            project_root,
            data_root,
            sub_sample_dir_base,
            output_dir_base,
            fail_fast=fail_fast,
        )
        all_successful = all_successful and sample_successful

    if not all_successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""
阶段四：全模型筛过系数与对比
- 在合成网络上计算筛过系数
- WT vs AS 合成模型筛过系数对比
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gbm_sieving.data_io.filenames import artifact_filename
from gbm_sieving.data_io.units import nl_per_s_to_m3_per_s
from gbm_sieving.paths import OUTPUT_ROOT, PROJECT_ROOT, SRC_ROOT
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = True

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

    def _tqdm(iterable=None, total=None, desc=None, **kwargs):
        return iterable

SIEVING_VALID_EPS = 1e-12


def _finite_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _sanitize_parsed_sieving(parsed: dict) -> dict:
    """Keep raw diagnostics, but expose only physical C_out/C_in values in [0, 1]."""
    raw = None
    for key in ("sieving_coefficient", "Theta_corrected_concentration", "Theta_global"):
        raw = _finite_float(parsed.get(key))
        if raw is not None:
            break
    if raw is None:
        return parsed

    if -SIEVING_VALID_EPS <= raw < 0.0:
        fixed = 0.0
    elif 1.0 < raw <= 1.0 + SIEVING_VALID_EPS:
        fixed = 1.0
    elif raw < -SIEVING_VALID_EPS or raw > 1.0 + SIEVING_VALID_EPS:
        parsed["sieving_coefficient_raw"] = raw
        parsed["sieving_invalid_reason"] = "C_out/C_in outside [0, 1]"
        for key in ("sieving_coefficient", "Theta_corrected_concentration", "Theta_global"):
            if key in parsed:
                parsed[f"{key}_raw"] = parsed.get(key)
                parsed[key] = np.nan
        return parsed
    else:
        fixed = raw

    for key in ("sieving_coefficient", "Theta_corrected_concentration", "Theta_global"):
        if key in parsed:
            parsed[key] = fixed
    return parsed


def _valid_sieving_or_none(result: dict) -> float | None:
    sc = result.get("sieving_coefficient")
    if sc is None:
        sc = result.get("Theta_corrected_concentration")
    if sc is None:
        sc = result.get("Theta_global")
    val = _finite_float(sc)
    if val is None:
        return None
    if -SIEVING_VALID_EPS <= val < 0.0:
        return 0.0
    if 1.0 < val <= 1.0 + SIEVING_VALID_EPS:
        return 1.0
    if val < -SIEVING_VALID_EPS or val > 1.0 + SIEVING_VALID_EPS:
        return None
    return val


def resolve_sieving_results_xlsx(output_dir: Path, sample_name: str) -> Path | None:
    """Resolve the canonical sieving-summary result."""
    base = Path(output_dir) / sample_name
    result = base / artifact_filename(sample_name, "sieving_summary")
    return result if result.is_file() else None


def _get_param_substr(df: pd.DataFrame, name_substr: str):
    if "Parameter" not in df.columns or "Value" not in df.columns:
        return None
    row = df[df["Parameter"].astype(str).str.contains(name_substr, na=False, regex=False)]
    if len(row) == 0:
        return None
    v = row["Value"].iloc[0]
    return float(v) if pd.notna(v) else None


def _get_param_value_substr(df: pd.DataFrame, name_substr: str):
    if "Parameter" not in df.columns or "Value" not in df.columns:
        return None
    row = df[df["Parameter"].astype(str).str.contains(name_substr, na=False, regex=False)]
    if len(row) == 0:
        return None
    v = row["Value"].iloc[0]
    return v if pd.notna(v) else None


def _first_finite_sieving_cout_cin_from_df(df: pd.DataFrame) -> float | None:
    """
    从 Parameter/Value 表读取筛过系数 C_out/C_in。

    新表可能有多行：``..., new_J`` / 旧版单行。这里按 new_J -> 任意首个有限值取值。
    """
    if "Parameter" not in df.columns or "Value" not in df.columns:
        return None
    m = df["Parameter"].astype(str)
    mask = (
        m.str.contains("sieving_coefficient", na=False, regex=False)
        & m.str.contains("C_out", na=False, regex=False)
        & m.str.contains("C_in", na=False, regex=False)
    )
    sub = df.loc[mask, ["Parameter", "Value"]]
    if len(sub) == 0:
        return None

    def _first_finite_where(pred) -> float | None:
        for _, r in sub.iterrows():
            if not pred(str(r["Parameter"])):
                continue
            v = r["Value"]
            if pd.notna(v):
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(fv):
                    return fv
        return None

    v = _first_finite_where(lambda s: "new_J" in s)
    if v is not None:
        return v
    for _, r in sub.iterrows():
        vv = r["Value"]
        if pd.notna(vv):
            try:
                fv = float(vv)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                return fv
    return None


def parse_sieving_results_dataframe(df: pd.DataFrame) -> dict | None:
    """
    解析 Parameter/Value 筛分表（sub_calculate_sieving_coefficient_new equiv_C_v0）。

    主口径：``sieving_coefficient = C_out / C_in``（行名含 ``sieving_coefficient (C_out/C_in)``）。
    兼容旧表：PRIMARY Theta_global、F2.5 修正等。旧键 ``Theta_global`` / ``Theta_corrected_concentration``
    在新表上与 ``sieving_coefficient`` 同步为同一数值，便于旧代码读取。
    """
    if df is None or len(df) == 0:
        return None
    parsed: dict = {}

    if "Parameter" in df.columns and "Value" in df.columns:
        sc = _first_finite_sieving_cout_cin_from_df(df)
        if sc is not None:
            parsed["sieving_coefficient"] = sc
            parsed["Theta_corrected_concentration"] = sc
            parsed["Theta_global"] = sc

    if "Theta_global" not in parsed and "Theta_global" in df.columns:
        v = df["Theta_global"].iloc[0]
        parsed["Theta_global"] = float(v) if pd.notna(v) else 0.0
    elif "Parameter" in df.columns and "Value" in df.columns and "Theta_global" not in parsed:
        m = df["Parameter"].astype(str)
        prim_new = df[m.str.contains("PRIMARY", na=False) & m.str.contains("Theta_global", na=False) & m.str.contains("new_J", na=False)]
        if len(prim_new) > 0:
            v = prim_new["Value"].iloc[0]
            if pd.notna(v):
                parsed["Theta_global"] = float(v)
        if "Theta_global" not in parsed:
            row = df[m.str.contains("Theta_global", na=False, regex=False)]
            if len(row) > 0:
                v = row["Value"].iloc[0]
                parsed["Theta_global"] = float(v) if pd.notna(v) else 0.0

    if "Parameter" not in df.columns or "Value" not in df.columns:
        return parsed if parsed else None

    q_alb = _get_param_substr(df, "Total Albumin Flow Rate (Q_alb_total, equiv_C)")
    if q_alb is None:
        q_alb = _get_param_substr(df, "Total Solute Flux (J_s_total, new_J)")
    if q_alb is None:
        q_alb = _get_param_substr(df, "Total Solute Flux")
    parsed["Q_alb_total_times_C0"] = q_alb
    # Backward-compatible alias for older plotting scripts/results.
    parsed["J_s_total_times_C0"] = q_alb

    parsed["C0"] = _get_param_substr(df, "Plasma Concentration (C0)")
    if parsed.get("C0") is None:
        parsed["C0"] = _get_param_substr(df, "Plasma Concentration")

    if parsed.get("Theta_corrected_concentration") is None:
        theta_corr = None
        for sub in (
            "Secondary (F2.5 conc.): Overall Sieving Theta_corrected (summary;",
            "Estimated Urine Sieving by Concentration",
        ):
            theta_corr = _get_param_substr(df, sub)
            if theta_corr is not None:
                break
        parsed["Theta_corrected_concentration"] = theta_corr

    parsed["C_urine_bulk_flow_weighted"] = _get_param_substr(df, "Estimated Urine Bulk Concentration (Flow-weighted)")
    if parsed.get("Theta_corrected_concentration") is None:
        c_urine_bulk = parsed.get("C_urine_bulk_flow_weighted")
        c0_for_corr = parsed.get("C0")
        if c_urine_bulk is not None and c0_for_corr is not None and c0_for_corr > 0:
            parsed["Theta_corrected_concentration"] = c_urine_bulk / c0_for_corr

    if parsed.get("sieving_coefficient") is None and parsed.get("Theta_corrected_concentration") is not None:
        parsed["sieving_coefficient"] = float(parsed["Theta_corrected_concentration"])
    if parsed.get("sieving_coefficient") is None and parsed.get("Theta_global") is not None:
        parsed["sieving_coefficient"] = float(parsed["Theta_global"])

    q_m3s = _get_param_substr(df, "Q_total_full_solvent")
    if q_m3s is None:
        q_nls = _get_param_substr(df, "Total Solvent Flow (Q, full solvent network")
        if q_nls is not None:
            q_m3s = float(nl_per_s_to_m3_per_s(q_nls))
    if q_m3s is None:
        q_m3s = _get_param_substr(df, "Q_total (m")
    if q_m3s is None:
        q_nls = _get_param_substr(df, "Q_total, nL/s")
        if q_nls is None:
            q_nls = _get_param_substr(df, "full solvent network, nL/s")
        if q_nls is None:
            q_nls = _get_param_substr(df, "nL/s)")
        q_m3s = float(nl_per_s_to_m3_per_s(q_nls)) if q_nls is not None else None
    parsed["Q_total_m3s"] = q_m3s

    q_sol_nls = _get_param_substr(df, "Diagnostic: Q_solute_subnet at exit (nL/s)")
    parsed["Q_solute_subnet_m3s"] = (
        float(nl_per_s_to_m3_per_s(q_sol_nls)) if q_sol_nls is not None else None
    )

    n_exit_solute = _get_param_substr(df, "N exit nodes in solute graph")
    if n_exit_solute is not None:
        parsed["N_exit_nodes_solute_graph"] = int(n_exit_solute)
    n_entrance_solute = _get_param_substr(df, "N entrance nodes in solute graph")
    if n_entrance_solute is not None:
        parsed["N_entrance_nodes_solute_graph"] = int(n_entrance_solute)
    solution_status = _get_param_value_substr(df, "new_J solution status")
    if solution_status is not None:
        parsed["new_J_solution_status"] = str(solution_status)

    theta_global_val = parsed.get("Theta_global")
    if (
        parsed.get("Theta_corrected_concentration") is None
        and theta_global_val is not None
        and q_m3s is not None
        and q_m3s > 0
        and abs(theta_global_val) <= 1e-20
    ):
        parsed["Theta_corrected_concentration"] = 0.0

    if (
        parsed.get("sieving_coefficient") is None
        and parsed.get("N_exit_nodes_solute_graph") == 0
        and q_m3s is not None
        and q_m3s > 0
    ):
        parsed["sieving_coefficient"] = 0.0
        parsed["Theta_corrected_concentration"] = 0.0
        parsed["Theta_global"] = 0.0
        parsed["Q_alb_total_times_C0"] = 0.0
        parsed["J_s_total_times_C0"] = 0.0
        parsed["Q_solute_subnet_m3s"] = 0.0
        parsed["zero_sieving_reason"] = "no_solute_exit_nodes"

    if parsed:
        parsed = _sanitize_parsed_sieving(parsed)
    return parsed if parsed else None


def run_sieving_calculation(
    pores_file: Path,
    throats_file: Path,
    sample_name: str,
    output_dir: Path,
    run_calculation: bool = True,
    phase4_html: bool = False,
    quiet: bool = False,
) -> dict:
    """
    在合成网络上运行整体分析 + 筛过系数计算。
    Run the packaged network analyzer and the unified transport solver.
    Result reads use the canonical project filename.
    """
    sample_dir = pores_file.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_ROOT), child_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    def _run_child(cmd: list[str], **kwargs):
        if quiet:
            return subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **kwargs,
            )
        return subprocess.run(cmd, check=True, **kwargs)
    
    if run_calculation:
        # 第一步：整体孔/喉分析 + Alpha Shape 表面孔识别 + 通路筛选
        cmd_analyze = [
            sys.executable,
            "-m",
            "gbm_sieving.simulation.transport.analyze_network",
            "--sample-name", sample_name,
            "--sample-dir", str(sample_dir),
            "--output-dir", str(output_dir),
        ]
        if not phase4_html:
            cmd_analyze.append("--no-html")
        print(f"运行整体分析 (Alpha Shape + 表面孔识别): {' '.join(cmd_analyze)}")
        try:
            # 直接在当前终端中流式输出整体分析脚本的日志与进度条
            _run_child(
                cmd_analyze,
                env=child_env,
            )
        except subprocess.CalledProcessError as e:
            print(f"整体分析脚本执行错误: {e}")
            return None

        # 前置拦截：若整体分析后“溶剂渗透连通分量”为空，则直接判定不连通并跳过筛分浓度求解
        solvent_comp_file = Path(output_dir) / sample_name / f"{sample_name}_solvent_penetration_components.xlsx"
        try:
            if solvent_comp_file.exists():
                _df_comp = pd.read_excel(solvent_comp_file)
                if len(_df_comp) == 0:
                    print(
                        "检测到无溶剂渗透连通分量（no_exit_nodes 预判）："
                        f"跳过筛过系数浓度求解并返回不连通结果 -> {sample_name}"
                    )
                    return {
                        "sieving_coefficient": 0.0,
                        "Theta_global": 0.0,
                        "Theta_corrected_concentration": 0.0,
                        "Q_total_m3s": 0.0,
                        "no_exit_nodes": True,
                        "skipped_sieving_due_to_no_exit": True,
                    }
        except Exception:
            # 读取预判文件失败时，回退原流程继续筛分计算
            pass
        
        # 第二步：统一筛分求解
        cmd_sieve = [
            sys.executable,
            "-m",
            "gbm_sieving.simulation.transport.sieving_solver",
            "--sample-name", sample_name,
            "--sample-dir", str(sample_dir),
            "--output-dir", str(output_dir),
        ]
        if not phase4_html:
            cmd_sieve.append("--no-html")
        print(f"运行筛过系数计算: {' '.join(cmd_sieve)}")
        
        try:
            _run_child(
                cmd_sieve,
                cwd=str(PROJECT_ROOT),
                env=child_env,
            )
        except subprocess.CalledProcessError as e:
            print(f"筛过系数脚本执行错误: {e}")
            return None
    
    results_file = resolve_sieving_results_xlsx(output_dir, sample_name)
    if results_file is not None:
        df = pd.read_excel(results_file)
        parsed = parse_sieving_results_dataframe(df)
        if parsed:
            return parsed
        if len(df) > 0 and "Parameter" in df.columns and "Value" in df.columns:
            m = df["Parameter"].astype(str)
            row = df[
                m.str.contains("sieving_coefficient", na=False, regex=False)
                & m.str.contains("C_out", na=False, regex=False)
            ]
            if len(row) > 0:
                v = row["Value"].iloc[0]
                if pd.notna(v):
                    sc = float(v)
                    return _sanitize_parsed_sieving({
                        "sieving_coefficient": sc,
                        "Theta_global": sc,
                        "Theta_corrected_concentration": sc,
                    })
            row = df[df["Parameter"].astype(str).str.contains("Theta_global", na=False, regex=False)]
            if len(row) > 0:
                v = row["Value"].iloc[0]
                return _sanitize_parsed_sieving({"Theta_global": float(v) if pd.notna(v) else 0.0})
        if len(df) > 0:
            return _sanitize_parsed_sieving(df.to_dict("records")[0])
    return None


def calculate_sieving_for_synthetic_networks(
    synthetic_networks_dir: Path,
    output_dir: Path,
    sample_types: list = None,
    reuse_existing_results: bool = False,
    quiet: bool = False,
) -> dict:
    """
    对合成网络计算筛过系数
    """
    if sample_types is None:
        sample_types = ["WT", "AS"]
    
    results = {}
    
    for sample_type in sample_types:
        print(f"\n=== 计算 {sample_type} 合成网络筛过系数 ===")
        
        # 查找该类型的所有合成网络
        pattern = f"synthetic_{sample_type}_*_pores.xlsx"
        pore_files = list(synthetic_networks_dir.glob(pattern))
        
        if not pore_files:
            print(f"  未找到 {sample_type} 合成网络")
            continue
        
        n_samples = len(pore_files)
        print(f"  共找到 {n_samples} 个 {sample_type} 合成网络")
        
        results[sample_type] = []
        
        # 使用 tqdm 显示处理进度条（若环境已安装 tqdm）
        iterator = _tqdm(
            list(enumerate(pore_files, start=1)),
            total=n_samples,
            disable=quiet,
            desc=f"{sample_type} 合成网络",
        )
        
        for idx, pores_file in iterator:
            sample_name = pores_file.stem.replace("_pores", "")
            throats_file = pores_file.parent / f"{sample_name}_throats.xlsx"
            
            if not throats_file.exists():
                print(f"  [{idx}/{n_samples}] 跳过 {sample_name}（无喉文件）")
                continue
            
            print(f"  [{idx}/{n_samples}] 处理 {sample_name}...")
            result = run_sieving_calculation(
                pores_file,
                throats_file,
                sample_name,
                output_dir,
                run_calculation=not reuse_existing_results,
                quiet=quiet,
            )
            
            if result:
                result["sample_name"] = sample_name
                results[sample_type].append(result)
                sc = result.get("sieving_coefficient")
                if sc is None:
                    sc = result.get("Theta_corrected_concentration")
                if sc is None:
                    sc = result.get("Theta_global")
                if sc is not None and not (isinstance(sc, float) and np.isnan(sc)):
                    print(f"    单次整体筛过系数 (C_out/C_in): {float(sc):.6g}")
                else:
                    print("    单次整体筛过系数: （解析失败）")
    
    return results


def compare_wt_vs_as(results: dict, output_dir: Path):
    """
    WT vs AS 合成模型筛过系数对比
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取筛过系数（主口径：sieving_coefficient = C_out/C_in）
    wt_sieving = []
    as_sieving = []
    
    if "WT" in results:
        for r in results["WT"]:
            sc = _valid_sieving_or_none(r)
            if sc is not None:
                wt_sieving.append(float(sc))
    
    if "AS" in results:
        for r in results["AS"]:
            sc = _valid_sieving_or_none(r)
            if sc is not None:
                as_sieving.append(float(sc))
    
    if not wt_sieving and not as_sieving:
        print("警告：无有效的筛过系数数据")
        return
    
    # 统计
    comparison = {}
    if wt_sieving:
        comparison["WT"] = {
            "mean": float(np.mean(wt_sieving)),
            "std": float(np.std(wt_sieving)),
            "median": float(np.median(wt_sieving)),
            "min": float(np.min(wt_sieving)),
            "max": float(np.max(wt_sieving)),
            "n": len(wt_sieving),
        }
        print(f"\nWT 筛过系数:")
        print(f"  均值: {comparison['WT']['mean']:.6f}")
        print(f"  标准差: {comparison['WT']['std']:.6f}")
        print(f"  中位数: {comparison['WT']['median']:.6f}")
        print(f"  范围: [{comparison['WT']['min']:.6f}, {comparison['WT']['max']:.6f}]")
        print(f"  样本数: {comparison['WT']['n']}")
    
    if as_sieving:
        comparison["AS"] = {
            "mean": float(np.mean(as_sieving)),
            "std": float(np.std(as_sieving)),
            "median": float(np.median(as_sieving)),
            "min": float(np.min(as_sieving)),
            "max": float(np.max(as_sieving)),
            "n": len(as_sieving),
        }
        print(f"\nAS 筛过系数:")
        print(f"  均值: {comparison['AS']['mean']:.6f}")
        print(f"  标准差: {comparison['AS']['std']:.6f}")
        print(f"  中位数: {comparison['AS']['median']:.6f}")
        print(f"  范围: [{comparison['AS']['min']:.6f}, {comparison['AS']['max']:.6f}]")
        print(f"  样本数: {comparison['AS']['n']}")
    
    # 保存 WT / AS 统计对比结果
    comparison_path = output_dir / "sieving_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n对比结果已保存: {comparison_path}")

    # 汇总：
    # - sieving_Q_weighted：各网络筛过系数（C_out/C_in）按全网溶剂通量 Q 加权平均，sum(theta_k * Q_k) / sum(Q_k)
    # - Qalb_over_Qwater_aggregate：sum(Q_alb)_k / (C0 * sum(Q_water,k))，
    #   仅当各 run 结果表中仍有白蛋白总流率列时有效；
    #   与「浓度场定义的 C_out/C_in」一般不等价（含扩散/阻碍等），与 Q 加权 C 也不一定相同，仅作对照。
    aggregate_rows = []
    for sample_type in ("WT", "AS"):
        type_results = results.get(sample_type, [])
        sum_js_times_c0 = 0.0
        sum_q = 0.0
        c0_used = 1.0
        sum_q_for_siev = 0.0
        sum_siev_q = 0.0
        n_q_positive = 0
        n_used = 0
        n_excluded_theta_abnormal = 0
        for r in type_results:
            td = _valid_sieving_or_none(r)
            if td is None:
                n_excluded_theta_abnormal += 1
                continue
            n_used += 1
            js = r.get("Q_alb_total_times_C0")
            if js is None:
                js = r.get("J_s_total_times_C0")
            q = r.get("Q_total_m3s")
            c0 = r.get("C0")
            if c0 is not None and c0 > 0:
                c0_used = float(c0)
            if js is not None and np.isfinite(float(js)):
                sum_js_times_c0 += float(js)
            if q is not None and q > 0:
                sum_q += float(q)
            q_default = float(q) if (q is not None and not (isinstance(q, float) and np.isnan(q))) else 0.0
            if q_default > 0:
                n_q_positive += 1
                sum_q_for_siev += q_default
                sum_siev_q += float(td) * q_default
        j_over_q_aggregate = np.nan
        if sum_q > 0 and c0_used > 0 and sum_js_times_c0 > 0:
            j_over_q_aggregate = (sum_js_times_c0 / c0_used) / sum_q
        sieving_q_weighted = np.nan
        if sum_q_for_siev > 0:
            sieving_q_weighted = sum_siev_q / sum_q_for_siev
        aggregate_rows.append({
            "sample_type": sample_type,
            "n_samples": n_used,
            "n_excluded_sieving_not_in_0_1": n_excluded_theta_abnormal,
            "n_q_positive_used_in_denominator": n_q_positive,
            "sum_albumin_flow_rate_times_C0": sum_js_times_c0,
            "sum_solvent_flux_m3s": sum_q,
            "C0_used": c0_used,
            "Qalb_over_Qwater_aggregate_sumQalb_over_sumQwater_C0": j_over_q_aggregate,
            "sieving_coefficient_Q_weighted_Cout_over_Cin": sieving_q_weighted,
        })

    df_aggregate = pd.DataFrame(aggregate_rows)
    aggregate_xlsx = output_dir / "sieving_aggregate_by_flux.xlsx"
    df_aggregate.to_excel(aggregate_xlsx, index=False)
    print(f"按总通量汇总结果已保存: {aggregate_xlsx}")

    # 按样本输出筛过系数汇总表（主列：C_out/C_in）
    rows = []
    for st, lst in results.items():
        for r in lst:
            sc = _valid_sieving_or_none(r)
            if sc is None:
                continue
            rows.append(
                {
                    "sample_type": st,
                    "sample_name": r.get("sample_name", ""),
                    "sieving_coefficient_Cout_over_Cin": float(sc),
                    "Theta_global_legacy": r.get("Theta_global"),
                    "Theta_corrected_concentration_legacy": r.get("Theta_corrected_concentration"),
                }
            )
    if rows:
        df_summary = pd.DataFrame(rows)
        summary_path = output_dir / "sieving_summary_by_sample.xlsx"
        df_summary.to_excel(summary_path, index=False)
        print(f"样本级筛过系数汇总已保存: {summary_path}")
    
    # 箱型图 → sieving_boxplot.png
    if wt_sieving or as_sieving:
        data_to_plot = []
        labels = []
        if wt_sieving:
            data_to_plot.append(wt_sieving)
            labels.append("WT")
        if as_sieving:
            data_to_plot.append(as_sieving)
            labels.append("AS")
        if data_to_plot:
            fig, ax = plt.subplots(figsize=(6, 4))
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            bp["boxes"][0].set_facecolor("lightblue")
            if len(bp["boxes"]) > 1:
                bp["boxes"][1].set_facecolor("lightcoral")
            ax.set_ylabel("Sieving coefficient (C_out / C_in)", fontsize=12)
            ax.set_title("WT vs AS sieving coefficient (C_out / C_in)", fontsize=14)
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plot_path = output_dir / "sieving_boxplot.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"箱型图已保存: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="阶段四：全模型筛过系数与对比"
    )
    parser.add_argument(
        "--synthetic-networks-dir",
        type=str,
        default=str(OUTPUT_ROOT / "phase3_synthetic"),
        help="合成网络目录（默认：肾脏/gbm_full_model/results/phase3_synthetic）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_ROOT / "phase4_sieving"),
        help="输出目录（默认：肾脏/gbm_full_model/results/phase4_sieving）",
    )
    parser.add_argument(
        "--sample-types",
        type=str,
        nargs="+",
        choices=("WT", "AS"),
        default=["WT", "AS"],
        help="要计算的样本类型（默认：WT AS）",
    )
    parser.add_argument(
        "--reuse-existing-results",
        action="store_true",
        help="仅基于已存在的 phase4 结果汇总，不重新运行单样本筛过系数计算。",
    )
    args = parser.parse_args()

    synthetic_networks_dir = Path(args.synthetic_networks_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("阶段四：全模型筛过系数与对比")
    print("=" * 60)

    # 4.1 在合成网络上计算筛过系数
    print("\n[4.1] 在合成网络上计算筛过系数...")
    results = calculate_sieving_for_synthetic_networks(
        synthetic_networks_dir,
        output_dir,
        sample_types=args.sample_types,
        reuse_existing_results=args.reuse_existing_results,
    )

    # 4.2 WT vs AS 合成模型筛过系数对比
    print("\n[4.2] WT vs AS 合成模型筛过系数对比...")
    compare_wt_vs_as(results, output_dir)

    print("\n" + "=" * 60)
    print("阶段四完成！")
    print("=" * 60)
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    main()

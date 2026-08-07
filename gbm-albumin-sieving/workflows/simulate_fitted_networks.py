"""
单 cell、均一厚度模拟流程（n>=1 兼容）。断点续跑：使用 ``--skip-existing``（会自动保留 ``run_result``，等效 ``--no-clear``），跳过已有 Phase3+Phase4 的 run，仅补未完成项。

``--batch-repeat K``（K>1）：整套流程重复 K 次，每次输出隔离在 ``results/run_result/repeat_000/`` … ``repeat_{K-1}/``（含 phase3、phase4、thickness_sampling、physical_params、汇总图等）。

``thickness_sampling/sampled_thickness_n*_{WT|AS}.json``：管道在 AS/WT **厚度下限抬升**后会**回写**与 Phase3 一致的列表（``persist_effective_sampled_thickness_json``），避免下游仍读原始抽样值。

输出精简为 6 类：
1) phase3_synthetic/networks/：每次模拟的网络 xlsx
2) physical_params/：每 run 物理量记录 + 按厚度 bin 汇总图（柱状图 + Phase2 拟合；含 overlap 喉 R_throat/R_cap 与 Phase2 拟合对比，需 overlap_throat_R_ratio_by_thickness_bin.json）
3) phase4_sieving/：每个网络筛过系数 + sieving_boxplot.png
4) physical_params/thickness_distribution.png：WT/AS 厚度柱状图 + 拟合曲线
5) phase4_sieving/sieving_boxplot.png：WT/AS 筛过系数箱型图
6) pore_quintile_throat_mixing_synthetic/：默认不生成；可用 ``--run-quintile-mixing`` 开启（置换诊断图默认 n_perm=99）
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# 使用英文标签，避免中文白框
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = True

_root = PROJECT_ROOT


def _default_fitted_parameter_dir(name: str) -> Path:
    """Prefer project-local results, then the adjacent legacy data location."""
    local = PROJECT_ROOT / "results" / name
    legacy = PROJECT_ROOT.parent / "gbm_full_model" / "results" / name
    if local.is_dir():
        return local
    if legacy.is_dir():
        return legacy
    return local
BIN_WIDTH_NM = 10.0
_PROGRESS_STREAM = sys.stdout
_PROGRESS_TOTAL = 0
_PROGRESS_DONE = 0
_PROGRESS_ENABLED = False


def _utf8_child_env() -> dict:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC_ROOT), env.get("PYTHONPATH", "")) if part
    )
    return env


@contextlib.contextmanager
def _suppress_stdout_if(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
        yield


def _progress_start(total: int, *, enabled: bool) -> None:
    global _PROGRESS_TOTAL, _PROGRESS_DONE, _PROGRESS_ENABLED
    _PROGRESS_TOTAL = max(0, int(total))
    _PROGRESS_DONE = 0
    _PROGRESS_ENABLED = bool(enabled)
    if _PROGRESS_ENABLED:
        _progress_render()


def _progress_render() -> None:
    if not _PROGRESS_ENABLED:
        return
    total = max(1, _PROGRESS_TOTAL)
    frac = min(1.0, max(0.0, _PROGRESS_DONE / total))
    width = 40
    filled = int(round(width * frac))
    bar = "#" * filled + "-" * (width - filled)
    _PROGRESS_STREAM.write(f"\rSimulating [{bar}] {_PROGRESS_DONE}/{_PROGRESS_TOTAL} runs")
    _PROGRESS_STREAM.flush()


def _progress_advance(step: int = 1) -> None:
    global _PROGRESS_DONE
    if not _PROGRESS_ENABLED:
        return
    _PROGRESS_DONE = min(_PROGRESS_TOTAL, _PROGRESS_DONE + int(step))
    _progress_render()


def _progress_finish() -> None:
    if not _PROGRESS_ENABLED:
        return
    if _PROGRESS_DONE < _PROGRESS_TOTAL:
        _progress_render()
    _PROGRESS_STREAM.write("\n")
    _PROGRESS_STREAM.flush()

# 与 phase3_synthetic_network.TOL_NM_OVERLAP / phase2 overlap 统计一致
TOL_OVERLAP_NM = 0.2


def _intersection_circle_radius_nm_run(d: float, R1: float, R2: float, tol: float = TOL_OVERLAP_NM) -> float:
    """两球交线圆半径 R_cap；无交线圆时 nan（与 Phase3/Phase2 定义一致）。"""
    if not (np.isfinite(d) and np.isfinite(R1) and np.isfinite(R2)):
        return float("nan")
    if R1 <= 0 or R2 <= 0 or d <= 0:
        return float("nan")
    if d >= R1 + R2 - tol:
        return float("nan")
    if d <= abs(R1 - R2) + tol:
        return float("nan")
    a1 = (d * d + R1 * R1 - R2 * R2) / (2.0 * d)
    sq = R1 * R1 - a1 * a1
    if sq <= 0:
        return float("nan")
    return float(np.sqrt(sq))


def _sample_thickness_gmm(n: int, gmm_dict: dict, rng: np.random.Generator) -> np.ndarray:
    if not gmm_dict or gmm_dict.get("type") != "gmm":
        return np.full(n, np.nan)
    weights = np.asarray(gmm_dict["weights"], dtype=float)
    means = np.asarray(gmm_dict["means"], dtype=float)
    sigmas = np.sqrt(np.asarray(gmm_dict["covariances"], dtype=float))
    components = rng.choice(len(weights), size=int(n), p=weights)
    return rng.normal(means[components], sigmas[components]).astype(float)


def _sample_thickness_parametric(n: int, entry: dict, rng: np.random.Generator) -> np.ndarray:
    name = entry.get("distribution")
    params = entry.get("params")
    if not name or params is None:
        raise ValueError("Thickness distribution entry is missing distribution or params.")
    dist = getattr(stats, str(name))
    return np.asarray(dist.rvs(*params, size=int(n), random_state=rng), dtype=float)


def _sample_thickness_bootstrap_from_raw(
    n: int,
    phase1_dir: Path,
    sample_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    raw = Path(phase1_dir) / "thickness_data_raw.xlsx"
    if not raw.exists():
        raise FileNotFoundError(
            f"KDE thickness sampling needs raw thickness data, but this file was not found: {raw}"
        )
    df = pd.read_excel(raw)
    if "thickness" not in df.columns or "is_wt" not in df.columns:
        raise ValueError(f"{raw} must contain thickness and is_wt columns.")
    want_wt = str(sample_type).upper() == "WT"
    values = df[df["is_wt"] == want_wt]["thickness"].dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"No raw {sample_type} thickness values found in {raw}.")
    return values[rng.integers(0, values.size, size=int(n))].astype(float)


def _sample_thickness_kde(
    n: int,
    entry: dict,
    phase1_dir: Path,
    sample_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    kde_data = entry.get("kde_data")
    if kde_data is not None:
        arr = np.asarray(kde_data, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size >= 2:
            kde = gaussian_kde(arr)
            return np.asarray(kde.resample(int(n), seed=rng)).reshape(-1).astype(float)
    return _sample_thickness_bootstrap_from_raw(n, phase1_dir, sample_type, rng)


def _resolve_thickness_distribution_entry(
    data: dict,
    sample_type: str,
    wt_fit: str,
    as_fit: str,
) -> tuple[str, dict]:
    sample_type = str(sample_type).upper()
    if sample_type == "AS":
        fit = str(as_fit).lower()
        key = {"kde": "AS_kde", "gmm": "AS_gmm", "auto": "AS"}.get(fit)
        if key is None:
            raise ValueError(f"Unsupported AS thickness fit: {as_fit}")
        entry = data.get(key)
        if entry:
            return key, entry
        raise ValueError(f"Missing AS thickness distribution entry: {key}")

    if sample_type == "WT":
        fit = str(wt_fit).lower()
        if fit == "kde":
            for key in ("WT_kde", "WT"):
                entry = data.get(key)
                if entry and (key == "WT_kde" or entry.get("type") == "kde"):
                    return key, entry
        elif fit == "norm":
            entry = data.get("WT_norm")
            if entry:
                return "WT_norm", entry
            entry = data.get("WT")
            if entry and entry.get("type") == "parametric" and str(entry.get("distribution")) == "norm":
                return "WT", entry
        else:
            raise ValueError(f"Unsupported WT thickness fit: {wt_fit}")
        raise ValueError(f"Missing WT thickness distribution for fit={wt_fit}")

    entry = data.get(sample_type)
    if not entry:
        raise ValueError(f"Missing thickness distribution entry for {sample_type}")
    return sample_type, entry


def _draw_sampled_thickness(
    *,
    phase1_dir: Path,
    output_dir: Path,
    n: int,
    sample_type: str,
    seed: int | None,
    wt_fit: str = "kde",
    as_fit: str = "kde",
) -> np.ndarray:
    params_file = Path(phase1_dir) / "thickness_distribution_parameters.json"
    if not params_file.exists():
        raise FileNotFoundError(f"Missing thickness distribution parameters: {params_file}")
    with open(params_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    entry_key, entry = _resolve_thickness_distribution_entry(data, sample_type, wt_fit, as_fit)
    rng = np.random.default_rng(seed)
    kind = entry.get("type")
    if kind == "gmm":
        values = _sample_thickness_gmm(n, entry, rng)
    elif kind == "parametric":
        values = _sample_thickness_parametric(n, entry, rng)
    elif kind == "kde":
        values = _sample_thickness_kde(n, entry, phase1_dir, sample_type, rng)
    else:
        raise ValueError(f"Unsupported thickness distribution type for {sample_type}: {kind}")

    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{sample_type} thickness sampling produced non-finite values.")
    values = np.sort(np.maximum(values, 1.0))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"sampled_thickness_n{int(n)}_{sample_type}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": int(n),
                "sample_type": str(sample_type),
                "thickness_nm": values.tolist(),
                "entry_key": entry_key,
                "entry_type": kind,
                "wt_fit": str(wt_fit).lower() if str(sample_type).upper() == "WT" else None,
                "as_fit": str(as_fit).lower() if str(sample_type).upper() == "AS" else None,
            },
            f,
            indent=2,
        )

    if int(n) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(values, bins=min(30, max(10, int(n) // 3)), color="steelblue", alpha=0.8, edgecolor="white")
        axes[0].set_xlabel("Thickness (nm)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"{sample_type} thickness distribution (n={int(n)})")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(np.arange(int(n)), values, "b.-", markersize=4)
        axes[1].set_xlabel("Run index i")
        axes[1].set_ylabel("Thickness (nm)")
        axes[1].set_title(f"{sample_type} thickness vs run index")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(output_dir / f"sampled_thickness_n{int(n)}_{sample_type}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return values


try:
    from gbm_sieving.data_io.aggregation import (
        run_sieving_calculation,
        calculate_sieving_for_synthetic_networks,
        compare_wt_vs_as,
        resolve_sieving_results_xlsx,
        parse_sieving_results_dataframe,
        _first_finite_sieving_cout_cin_from_df,
    )
except ImportError:  # 延迟到 main 报错
    run_sieving_calculation = None
    calculate_sieving_for_synthetic_networks = None
    compare_wt_vs_as = None
    resolve_sieving_results_xlsx = None
    parse_sieving_results_dataframe = None
    _first_finite_sieving_cout_cin_from_df = None

SIEVING_VALID_EPS = 1e-12


def _phase4_sieving_results_path(phase4_dir: Path, sample_name: str) -> Path | None:
    """返回当前项目规范的筛分汇总文件。"""
    if resolve_sieving_results_xlsx is not None:
        return resolve_sieving_results_xlsx(phase4_dir, sample_name)
    return None


def _prune_phase4_sample_dir(sample_dir: Path, *, keep_radius_cache: bool = False) -> None:
    """默认只保留最终筛过系数结果，减少批量模拟的 I/O 和磁盘占用。"""
    if not sample_dir.is_dir():
        return
    keep = {p.name for p in sample_dir.glob("*__sieving_summary*.xlsx")}
    if not keep:
        return
    if keep_radius_cache:
        cache_patterns = [
            "*_pore_classification.xlsx",
            "*_solvent_throat_classification.xlsx",
            "*_solvent_penetration_components.xlsx",
            "*_pressure_distribution_equiv_C_v0.xlsx",
        ]
        for pattern in cache_patterns:
            keep.update(p.name for p in sample_dir.glob(pattern))
    for path in sample_dir.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink(missing_ok=True)


try:
    from gbm_sieving.analysis.parameter_fitting.fit_thickness_trends import (
        analyze_and_plot_pore_quintile_throat_mixing,
        collect_synthetic_run_quintile_mixing_dataframe,
    )
except ImportError:
    collect_synthetic_run_quintile_mixing_dataframe = None
    analyze_and_plot_pore_quintile_throat_mixing = None


def _sieving_cout_cin_from_parsed(parsed: dict | None) -> float | None:
    """主口径：C_out/C_in（与 phase4 parse 的 sieving_coefficient 一致；兼容旧键名）。"""
    if not parsed:
        return None
    t = parsed.get("sieving_coefficient")
    if t is None:
        t = parsed.get("Theta_corrected_concentration")
    if t is None:
        t = parsed.get("Theta_global")
    if t is None:
        return None
    try:
        f = float(t)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _sieving_cout_cin_retry_ok(s: float | None) -> bool:
    """同厚度重试判定：C_out/C_in 必须是物理范围 [0, 1] 内的有限值。"""
    if s is None:
        return False
    if not np.isfinite(s):
        return False
    return -SIEVING_VALID_EPS <= float(s) <= 1.0 + SIEVING_VALID_EPS


def _sanitize_sieving_value(value) -> tuple[float, str]:
    """Return a physical C_out/C_in value; invalid values become NaN with a reason."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan"), "missing_or_nonfinite"
    if not np.isfinite(v):
        return float("nan"), "missing_or_nonfinite"
    if -SIEVING_VALID_EPS <= v < 0.0:
        return 0.0, ""
    if 1.0 < v <= 1.0 + SIEVING_VALID_EPS:
        return 1.0, ""
    if v < -SIEVING_VALID_EPS:
        return float("nan"), "theta_lt_0"
    if v > 1.0 + SIEVING_VALID_EPS:
        return float("nan"), "theta_gt_1"
    return v, ""


def _phase4_has_no_exit_nodes(phase4_dir: Path, sample_name: str) -> bool | None:
    """
    从单样本筛分结果表判断是否“无出口节点”。

    返回：
    - True：明确读到出口节点数且 <=0
    - False：明确读到出口节点数且 >0
    - None：结果缺失/无法解析
    """
    res_file = _phase4_sieving_results_path(phase4_dir, sample_name)
    if res_file is None or (not res_file.exists()):
        return None
    try:
        df = pd.read_excel(res_file)
    except Exception:
        return None
    if len(df) == 0 or "Parameter" not in df.columns or "Value" not in df.columns:
        return None
    m = df["Parameter"].astype(str)
    row = df[m.str.contains("N exit nodes in solute graph", na=False, regex=False)]
    if len(row) == 0:
        row = df[m.str.contains("exit nodes", na=False) & m.str.contains("solute graph", na=False)]
    if len(row) == 0:
        return None
    v = row["Value"].iloc[0]
    if pd.isna(v):
        return None
    try:
        n_exit = int(float(v))
    except (TypeError, ValueError):
        return None
    return n_exit <= 0

# Phase2 fit PDF 计算（与 phase3 一致，用于 phase2_synthetic 绘图）
try:
    from gbm_sieving.analysis.parameter_fitting.fit_pore_throat_parameters import _radius_fitted_pdf as _phase2_radius_fitted_pdf
    from gbm_sieving.analysis.parameter_fitting.fit_pore_throat_parameters import _radius_gmm_pdf as _phase2_gmm_pdf
    from gbm_sieving.analysis.parameter_fitting.fit_pore_throat_parameters import _metric_fitted_pdf as _phase2_metric_fitted_pdf
except ImportError:
    _phase2_radius_fitted_pdf = None
    _phase2_gmm_pdf = None
    _phase2_metric_fitted_pdf = None


def _radius_fit_pdf(x: np.ndarray, fit: dict) -> np.ndarray:
    """从 Phase2 半径/喉长度 fit 计算 PDF，与 phase3 的 _radius_fit_pdf 一致。"""
    x = np.asarray(x, dtype=float)
    if _phase2_radius_fitted_pdf is None or _phase2_gmm_pdf is None:
        return np.zeros_like(x)
    f = fit.get("fit") or fit
    try:
        if f.get("fit_type") == "gmm" and f.get("n_components"):
            w = f.get("weights", [1.0])
            mu = f.get("means", [0.0])
            cov = f.get("covariances", [1e-10])
            return _phase2_gmm_pdf(x, w, mu, cov)
        dist = f.get("distribution") or {}
        name = dist.get("name", "")
        params = dist.get("params") or []
        if name and params:
            return _phase2_radius_fitted_pdf(x, name, params)
    except Exception:
        pass
    return np.zeros_like(x)


def ensure_sampled_thickness(
    phase1_dir: Path,
    output_dir: Path,
    n: int,
    sample_type: str,
    seed: int = None,
    wt_fit: str = "kde",
    as_fit: str = "kde",
):
    """若尚无 n 个厚度抽样则在当前流程内抽样并返回厚度列表。

    注意：``run_uniform_sieving_pipeline`` 在 AS/WT 下限约束后会用 ``persist_effective_sampled_thickness_json``
    覆盖同一路径 JSON，使列表与 Phase3 ``--geometry-thickness-mean`` 一致。
    """
    out_json = output_dir / f"sampled_thickness_n{n}_{sample_type}.json"
    if out_json.exists():
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        _wt_fit_ok = True
        _as_fit_ok = True
        if str(sample_type).upper() == "WT":
            _wt_fit_ok = (data.get("wt_fit") == str(wt_fit).lower())
        if str(sample_type).upper() == "AS":
            _as_fit_ok = (data.get("as_fit") == str(as_fit).lower())
        if data.get("n") == n and data.get("sample_type") == sample_type and _wt_fit_ok and _as_fit_ok:
            return np.array(data["thickness_nm"])
    return _draw_sampled_thickness(
        phase1_dir=phase1_dir,
        output_dir=output_dir,
        n=n,
        sample_type=sample_type,
        seed=seed,
        wt_fit=wt_fit,
        as_fit=as_fit,
    )


def _draw_thickness_batch_uncached(
    *,
    phase1_dir: Path,
    output_dir: Path,
    sample_type: str,
    n_draw: int,
    seed: int | None,
    wt_fit: str = "kde",
    as_fit: str = "kde",
) -> np.ndarray:
    """强制重新抽样一批厚度（不复用既有 json）。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return _draw_sampled_thickness(
        phase1_dir=phase1_dir,
        output_dir=output_dir,
        n=int(n_draw),
        sample_type=str(sample_type),
        seed=seed,
        wt_fit=wt_fit,
        as_fit=as_fit,
    ).reshape(-1)


def rejection_resample_thickness_below_min(
    *,
    thickness_values: np.ndarray,
    min_thickness_nm: float,
    phase1_dir: Path,
    thickness_sampling_dir: Path,
    sample_type: str,
    seed: int | None,
    wt_fit: str = "kde",
    as_fit: str = "kde",
    max_rounds: int = 2000,
) -> tuple[np.ndarray, int]:
    """对低于下限的样本做拒绝采样，直到全部 >= min_thickness_nm。"""
    arr = np.asarray(thickness_values, dtype=float).copy()
    min_t = float(min_thickness_nm)
    bad_idx = np.where(arr < min_t)[0]
    if bad_idx.size == 0:
        return arr, 0

    tmp_dir = Path(thickness_sampling_dir) / "_rejection_resample_tmp" / str(sample_type)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    replaced_total = int(bad_idx.size)
    try:
        round_id = 0
        while bad_idx.size > 0:
            round_id += 1
            if round_id > int(max_rounds):
                raise RuntimeError(
                    f"[{sample_type}] 拒绝采样超过最大轮次 {max_rounds}，仍有 {bad_idx.size} 个厚度 < {min_t:.3f} nm"
                )
            need = int(bad_idx.size)
            n_draw = max(16, need * 4)
            draw_seed = None if seed is None else int(seed) + 100_000_019 * round_id + 97 * n_draw
            draws = _draw_thickness_batch_uncached(
                phase1_dir=phase1_dir,
                output_dir=tmp_dir,
                sample_type=sample_type,
                n_draw=n_draw,
                seed=draw_seed,
                wt_fit=wt_fit,
                as_fit=as_fit,
            )
            ok = draws[np.isfinite(draws) & (draws >= min_t)]
            if ok.size == 0:
                continue
            fill_n = min(need, int(ok.size))
            arr[bad_idx[:fill_n]] = ok[:fill_n]
            bad_idx = np.where(arr < min_t)[0]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return arr, replaced_total


def rejection_resample_thickness_outside_range(
    *,
    thickness_values: np.ndarray,
    min_thickness_nm: float | None,
    max_thickness_nm: float | None,
    phase1_dir: Path,
    thickness_sampling_dir: Path,
    sample_type: str,
    seed: int | None,
    wt_fit: str = "kde",
    as_fit: str = "kde",
    max_rounds: int = 2000,
) -> tuple[np.ndarray, int]:
    """Optional targeted-thickness rejection sampling for small diagnostic runs."""
    arr = np.asarray(thickness_values, dtype=float).copy()
    min_t = None if min_thickness_nm is None else float(min_thickness_nm)
    max_t = None if max_thickness_nm is None else float(max_thickness_nm)
    if min_t is not None and max_t is not None and min_t > max_t:
        raise ValueError(
            f"target thickness min ({min_t:.3f} nm) must be <= max ({max_t:.3f} nm)"
        )

    def _ok(values: np.ndarray) -> np.ndarray:
        mask = np.isfinite(values)
        if min_t is not None:
            mask &= values >= min_t
        if max_t is not None:
            mask &= values <= max_t
        return mask

    bad_idx = np.where(~_ok(arr))[0]
    if bad_idx.size == 0:
        return arr, 0

    tmp_dir = Path(thickness_sampling_dir) / "_target_range_resample_tmp" / str(sample_type)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    replaced_total = int(bad_idx.size)
    try:
        round_id = 0
        while bad_idx.size > 0:
            round_id += 1
            if round_id > int(max_rounds):
                lo_txt = "-inf" if min_t is None else f"{min_t:.3f}"
                hi_txt = "+inf" if max_t is None else f"{max_t:.3f}"
                raise RuntimeError(
                    f"[{sample_type}] targeted thickness rejection sampling exceeded "
                    f"{max_rounds} rounds; {bad_idx.size} values remain outside "
                    f"[{lo_txt}, {hi_txt}] nm"
                )
            need = int(bad_idx.size)
            n_draw = max(64, need * 8)
            draw_seed = None if seed is None else int(seed) + 200_000_033 * round_id + 193 * n_draw
            draws = _draw_thickness_batch_uncached(
                phase1_dir=phase1_dir,
                output_dir=tmp_dir,
                sample_type=sample_type,
                n_draw=n_draw,
                seed=draw_seed,
                wt_fit=wt_fit,
                as_fit=as_fit,
            )
            ok = draws[_ok(draws)]
            if ok.size == 0:
                continue
            fill_n = min(need, int(ok.size))
            arr[bad_idx[:fill_n]] = ok[:fill_n]
            bad_idx = np.where(~_ok(arr))[0]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return arr, replaced_total


def persist_effective_sampled_thickness_json(
    output_dir: Path, n: int, sample_type: str, thickness_nm: np.ndarray
) -> Path:
    """把**实际用于 Phase3** 的厚度列表写回 ``sampled_thickness_n{n}_{sample_type}.json``。

    初始厚度 JSON 只保存 Phase1 原始抽样；管道里对 AS/WT 做厚度下限抬升后若不同步，
    ``plot_flux_vs_thickness`` 等仍读旧 JSON，会与真实几何不一致。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"sampled_thickness_n{n}_{sample_type}.json"
    arr = np.asarray(thickness_nm, dtype=float).reshape(-1)
    if int(arr.size) != int(n):
        raise ValueError(f"persist_effective_sampled_thickness_json: len(thickness_nm)={arr.size} != n={n}")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"n": int(n), "sample_type": str(sample_type), "thickness_nm": arr.tolist()},
            f,
            indent=2,
        )
    print(f"[{sample_type}] 已同步 Phase3 有效厚度列表 -> {out_json.name}（n={n}）")
    return out_json


def _phase3_effective_seed(
    *,
    use_explicit_seed: bool,
    base_seed: int,
    run_index: int,
    sieving_retry_index: int,
    phase3_attempt_index: int,
) -> int | None:
    """Return the exact seed currently passed to phase3 for this attempt."""
    if not use_explicit_seed:
        return None
    return int(base_seed) + int(run_index) + int(sieving_retry_index) * 1_000_000 + (int(phase3_attempt_index) - 1) * 100_000


def _write_seed_ledger(run_result_dir: Path, records: list[dict]) -> None:
    """Persist per-run seed/provenance records for reproducible synthetic pools."""
    run_result_dir = Path(run_result_dir)
    run_result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_result_dir / "seed_ledger.csv"
    json_path = run_result_dir / "seed_ledger.json"
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"随机种子记录已保存: {csv_path}")


def _metric_fit_pdf_core(x: np.ndarray, name: str, params: list) -> np.ndarray:
    """与 phase2_pore_throat_parameters._metric_fitted_pdf 一致；不依赖可选导入。"""
    x = np.asarray(x, dtype=float)
    if name == "norm":
        return stats.norm.pdf(x, *params)
    if name == "lognorm":
        return stats.lognorm.pdf(x, *params)
    if name == "gamma":
        return stats.gamma.pdf(x, *params)
    if name == "beta":
        return stats.beta.pdf(x, *params)
    return np.zeros_like(x)


def _metric_fit_pdf(x: np.ndarray, fit: dict) -> np.ndarray:
    """
    Phase2 `density_frac_by_thickness_bins_analysis.json` 中各指标的 PDF。
    - 对 rho_pore / rho_throat：子样本级「孔/喉数量密度」(1/nm³)，不是体积分数。
    - 对 frac_pore / frac_throat：子样本级「孔/喉体积分数」。
    （与 `pore_count_fraction_by_thickness_bin.json` 的全局孔数量分数不同源。）
    """
    x = np.asarray(x, dtype=float)
    f = fit.get("fit") or fit
    dist = f.get("distribution") or {}
    name = dist.get("name", "")
    params = dist.get("params") or []
    if not name or not params:
        return np.zeros_like(x)
    if _phase2_metric_fitted_pdf is not None:
        try:
            return _phase2_metric_fitted_pdf(x, name, params)
        except Exception:
            pass
    return _metric_fit_pdf_core(x, name, params)


def _phase2_density_frac_hist_nbins(n: int) -> int:
    """
    与 phase2_pore_throat_parameters.plot_density_frac_by_thickness_bins 中
    ``n_bins = min(15, max(5, len(values) // 2))`` 一致，柱宽（等分 [min,max]）与 Phase2 图同一规则。
    """
    return min(15, max(5, int(n) // 2))


def _sample_from_metric_fit_bin(fit_dict: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """从 Phase2 某厚度 bin 的 density fit 抽样（与 _metric_fit_pdf 分布一致）。"""
    f = fit_dict.get("fit") or fit_dict
    dist = f.get("distribution") or {}
    name = str(dist.get("name", ""))
    params = list(dist.get("params") or [])
    if not name or not params:
        mu = float(f.get("mean", 0.0))
        sig = max(float(f.get("std", 1e-10)), 1e-15)
        return rng.normal(mu, sig, n)
    try:
        if name == "norm" and len(params) >= 2:
            return stats.norm.rvs(
                loc=float(params[0]), scale=float(params[1]), size=n, random_state=rng
            )
        if name == "lognorm":
            return stats.lognorm.rvs(*params, size=n, random_state=rng)
        if name == "gamma":
            return stats.gamma.rvs(*params, size=n, random_state=rng)
        if name == "beta":
            return stats.beta.rvs(*params, size=n, random_state=rng)
    except Exception:
        pass
    mu = float(f.get("mean", 0.0))
    sig = max(float(f.get("std", 1e-10)), 1e-15)
    return rng.normal(mu, sig, n)


def _phase2_ratio_kde_pdf(
    fit_p_bin: dict,
    fit_t_bin: dict,
    x_plot: np.ndarray,
    *,
    n_mc: int = 200_000,
) -> np.ndarray | None:
    """
    Phase2 同 bin 内 **ρ_pore、ρ_throat（数量密度）** 的独立边际拟合（非 frac 体积分数）→
    R = ρ_throat/ρ_pore = N_throat/N_pore（同一体积内）的 Monte Carlo + KDE；
    在 x_plot 上返回 PDF（与 gaussian_kde 一致，积分≈1）。
    """
    rng = np.random.default_rng(0)
    try:
        P = _sample_from_metric_fit_bin(fit_p_bin, n_mc, rng)
        T = _sample_from_metric_fit_bin(fit_t_bin, n_mc, rng)
    except Exception:
        return None
    P = np.maximum(np.asarray(P, dtype=float), 1e-15)
    T = np.asarray(T, dtype=float)
    R = T / P
    R = R[np.isfinite(R) & (R > 0)]
    if len(R) < 80:
        return None
    lo, hi = np.quantile(R, [0.002, 0.998])
    R = R[(R >= lo) & (R <= hi)]
    if len(R) < 50:
        return None
    try:
        kde = gaussian_kde(R)
        return kde(x_plot)
    except Exception:
        return None


def _validate_permeation_direction_for_sample(
    pores_file: Path,
    thickness_nm: float,
    eps_nm: float = 1e-6,
) -> tuple[bool, str]:
    """
    合成平面 cell：渗透沿 **全局 Z（膜厚方向）**，与 phase3 bbox [0,w]×[0,h]×[0,T] 一致。
    - 入口候选：贴 z=0 面，z - r <= 0
    - 出口候选：贴 z=T 面，z + r >= T
    要求：max(z_entrance) < min(z_exit) - eps

    说明：此前误用 X 与 geometry_width 作检验，与「沿厚度渗透」不符，已改为 Z 与膜厚 T。
    """
    if not pores_file.exists():
        return False, f"未找到孔文件: {pores_file}"
    try:
        df = pd.read_excel(pores_file)
    except Exception as e:
        return False, f"读取孔文件失败: {e}"

    if "Z Coord" not in df.columns or "EqRadius" not in df.columns:
        return False, "孔文件缺少 Z Coord 或 EqRadius 列"

    z = pd.to_numeric(df["Z Coord"], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(df["EqRadius"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(z) & np.isfinite(r) & (r >= 0) & (float(thickness_nm) > 0)
    z = z[valid]
    r = r[valid]
    if len(z) == 0:
        return False, "有效孔坐标为空"

    T = float(thickness_nm)
    entrance_mask = (z - r) <= 0.0
    exit_mask = (z + r) >= T

    # 边界判据为空时兜底：按分位数取两端孔（沿 Z）
    if not np.any(entrance_mask) or not np.any(exit_mask):
        q_lo = np.nanpercentile(z, 5.0)
        q_hi = np.nanpercentile(z, 95.0)
        entrance_mask = z <= q_lo
        exit_mask = z >= q_hi

    if not np.any(entrance_mask) or not np.any(exit_mask):
        return False, "入口/出口候选为空"

    ent_max = float(np.max(z[entrance_mask]))
    ex_min = float(np.min(z[exit_mask]))
    ok = ent_max < (ex_min - eps_nm)
    detail = (
        f"(Z) ent_max={ent_max:.6f}, ex_min={ex_min:.6f}, delta={ex_min-ent_max:.6f}, T={T:.6f}, "
        f"n_ent={int(np.sum(entrance_mask))}, n_exit={int(np.sum(exit_mask))}"
    )
    return ok, detail


def _collect_run_stats(phase3_networks_dir, thickness_sampling_dir, phase4_dir, sample_types, n_runs, geometry_size, phase1_dir=None):
    """收集各 run 的厚度与完整物理量（与 Phase2 定义一致）。"""
    rows = []
    thickness_sampling_dir = Path(thickness_sampling_dir)
    phase3_networks_dir = Path(phase3_networks_dir)
    # xlsx 优先在 networks 子目录，若无则尝试 phase3 根目录（兼容旧输出）
    xlsx_dirs = [phase3_networks_dir]
    if phase3_networks_dir.parent.exists():
        xlsx_dirs.append(phase3_networks_dir.parent)
    # 厚度 json 优先从 thickness_sampling，其次从 phase1_thickness
    search_dirs = [thickness_sampling_dir]
    if phase1_dir:
        search_dirs.append(Path(phase1_dir))
    for sample_type in sample_types:
        json_path = None
        for d in search_dirs:
            if not d.exists():
                continue
            p = d / f"sampled_thickness_n{n_runs}_{sample_type}.json"
            if p.exists():
                json_path = p
                break
            alt = list(d.glob(f"sampled_thickness_n*_{sample_type}.json"))
            if alt:
                json_path = alt[0]
                break
        if json_path is None:
            print(f"  [physical_params] 未找到厚度 json (sample_type={sample_type}, n={n_runs})")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        thickness_list = data.get("thickness_nm", [])
        for i in range(min(n_runs, len(thickness_list))):
            T = float(thickness_list[i])
            volume = geometry_size * geometry_size * T
            sample_name = f"synthetic_{sample_type}_run{i}_plane"
            pores_file = throats_file = None
            for d in xlsx_dirs:
                pf = d / f"{sample_name}_pores.xlsx"
                tf = d / f"{sample_name}_throats.xlsx"
                if pf.exists() and tf.exists():
                    pores_file, throats_file = pf, tf
                    break
            rec = {
                "sample_type": sample_type, "run": i, "thickness_nm": T,
                "n_pore": np.nan, "n_throat": np.nan,
                "rho_pore": np.nan, "frac_pore": np.nan,
                "rho_throat": np.nan, "frac_throat": np.nan,
                "n_cross_cell_throat": np.nan,
                "n_throat_no_cross": np.nan,
                "rho_throat_no_cross": np.nan,
                "frac_throat_no_cross": np.nan,
                "pore_radii": np.array([]), "throat_radii": np.array([]),
                "throat_lengths": np.array([]), "effective_throat_lengths": np.array([]),
                "overlap_throat_R_ratios": np.array([]),
                "pore_degrees": np.array([]),                 "mean_pore_per_throat": np.array([]),
                "throat_r_for_scatter": np.array([]),
                "overlap_mean_pore_scatter": np.array([], dtype=bool),
                "throat_overlap_mask": np.array([], dtype=bool),
                "Theta_global": np.nan,
            }
            if pores_file is not None and throats_file is not None:
                try:
                    df_p = pd.read_excel(pores_file)
                    df_t = pd.read_excel(throats_file)
                    r_p = df_p["EqRadius"].values if "EqRadius" in df_p.columns else np.array([])
                    r_t = df_t["EqRadius"].values if "EqRadius" in df_t.columns else np.array([])
                    L_t = df_t["Length"].values if "Length" in df_t.columns else np.array([])
                    if "Is Cross Cell" in df_t.columns:
                        cross_mask = df_t["Is Cross Cell"].fillna(False).astype(bool).to_numpy()
                    elif "Throat Kind" in df_t.columns:
                        cross_mask = (
                            df_t["Throat Kind"].fillna("").astype(str).str.lower().eq("cross_cell_nonoverlap").to_numpy()
                        )
                    else:
                        cross_mask = np.zeros(len(df_t), dtype=bool)
                    p1_col = "Pore ID #1" if "Pore ID #1" in df_t.columns else "Pore1"
                    p2_col = "Pore ID #2" if "Pore ID #2" in df_t.columns else "Pore2"
                    pore_ids = df_p["Pore ID"].values if "Pore ID" in df_p.columns else np.arange(len(df_p))
                    pid_to_idx = {int(pid): j for j, pid in enumerate(pore_ids)}
                    vol_pore = np.sum((4.0 / 3.0) * np.pi * np.maximum(r_p, 0) ** 3) if len(r_p) > 0 else 0
                    vol_throat = 0
                    vol_throat_no_cross = 0
                    mean_pore_list, throat_r_for_scatter = [], []
                    overlap_mean_scatter: list[bool] = []
                    throat_overlap_list: list[bool] = []
                    Lraw_list = []
                    overlap_ratio_list: list[float] = []
                    for t_idx in range(len(df_t)):
                        row = df_t.iloc[t_idx]
                        rt = r_t[t_idx] if t_idx < len(r_t) else 0
                        lt = L_t[t_idx] if t_idx < len(L_t) else 1e-6
                        vol_throat += np.pi * rt ** 2 * lt
                        if t_idx >= len(cross_mask) or not bool(cross_mask[t_idx]):
                            vol_throat_no_cross += np.pi * rt ** 2 * lt
                        is_ov_full = False
                        if p1_col in df_t.columns and p2_col in df_t.columns:
                            a, b = int(row[p1_col]), int(row[p2_col])
                            if a in pid_to_idx and b in pid_to_idx:
                                d = float(L_t[t_idx]) if t_idx < len(L_t) else 0.0
                                r1 = float(r_p[pid_to_idx[a]])
                                r2 = float(r_p[pid_to_idx[b]])
                                is_ov_full = np.isfinite(d) and d < r1 + r2 - TOL_OVERLAP_NM
                                r_avg = 0.5 * (r_p[pid_to_idx[a]] + r_p[pid_to_idx[b]])
                                mean_pore_list.append(r_avg)
                                throat_r_for_scatter.append(rt)
                                overlap_mean_scatter.append(is_ov_full)
                                Lraw_list.append(d - r1 - r2)
                                if d < r1 + r2 - TOL_OVERLAP_NM:
                                    rcap = _intersection_circle_radius_nm_run(d, r1, r2)
                                    if np.isfinite(rcap) and rcap > 0:
                                        overlap_ratio_list.append(float(rt) / rcap)
                        throat_overlap_list.append(is_ov_full)
                    deg = np.zeros(len(pore_ids), dtype=int)
                    if p1_col in df_t.columns and p2_col in df_t.columns:
                        for _, row in df_t.iterrows():
                            a, b = int(row[p1_col]), int(row[p2_col])
                            if a in pid_to_idx:
                                deg[pid_to_idx[a]] += 1
                            if b in pid_to_idx and b != a:
                                deg[pid_to_idx[b]] += 1
                    rec["n_pore"] = len(r_p)
                    rec["n_throat"] = len(r_t)
                    rec["n_cross_cell_throat"] = int(np.sum(cross_mask)) if len(cross_mask) > 0 else 0
                    rec["n_throat_no_cross"] = int(len(r_t) - rec["n_cross_cell_throat"])
                    rec["rho_pore"] = len(r_p) / volume if volume > 0 else np.nan
                    rec["frac_pore"] = vol_pore / volume if volume > 0 else np.nan
                    rec["rho_throat"] = len(r_t) / volume if volume > 0 else np.nan
                    rec["frac_throat"] = vol_throat / volume if volume > 0 else np.nan
                    rec["rho_throat_no_cross"] = rec["n_throat_no_cross"] / volume if volume > 0 else np.nan
                    rec["frac_throat_no_cross"] = vol_throat_no_cross / volume if volume > 0 else np.nan
                    rec["pore_radii"] = r_p
                    rec["throat_radii"] = r_t
                    rec["throat_lengths"] = L_t
                    rec["effective_throat_lengths"] = np.array(Lraw_list, dtype=float) if Lraw_list else np.array([])
                    rec["pore_degrees"] = deg
                    rec["mean_pore_per_throat"] = np.array(mean_pore_list) if mean_pore_list else np.array([])
                    rec["throat_r_for_scatter"] = np.array(throat_r_for_scatter) if throat_r_for_scatter else np.array([])
                    rec["overlap_mean_pore_scatter"] = (
                        np.array(overlap_mean_scatter, dtype=bool) if overlap_mean_scatter else np.array([], dtype=bool)
                    )
                    rec["throat_overlap_mask"] = (
                        np.array(throat_overlap_list, dtype=bool) if throat_overlap_list else np.array([], dtype=bool)
                    )
                    rec["overlap_throat_R_ratios"] = (
                        np.array(overlap_ratio_list, dtype=float) if overlap_ratio_list else np.array([])
                    )
                except Exception:
                    pass
            if phase4_dir:
                res_file = _phase4_sieving_results_path(phase4_dir, sample_name)
                if res_file is not None:
                    try:
                        df_s = pd.read_excel(res_file)
                        if parse_sieving_results_dataframe is not None:
                            parsed_s = parse_sieving_results_dataframe(df_s)
                            if parsed_s:
                                sc = parsed_s.get("sieving_coefficient")
                                if sc is None:
                                    sc = parsed_s.get("Theta_corrected_concentration")
                                if sc is None:
                                    sc = parsed_s.get("Theta_global")
                                if sc is not None:
                                    rec["Theta_global"] = float(sc)
                        elif len(df_s) > 0 and "Theta_global" in df_s.columns:
                            rec["Theta_global"] = float(df_s["Theta_global"].iloc[0])
                    except Exception:
                        pass
            rows.append(rec)
    return rows


def run_physical_params(
    physical_params_dir: Path,
    phase3_networks_dir: Path,
    thickness_sampling_dir: Path,
    phase2_dir: Path,
    phase1_dir: Path,
    phase4_dir: Path,
    sample_types: list,
    n_runs: int,
    geometry_size: float,
):
    """
    收集每 run 物理量，保存 run_stats_per_sample.xlsx，按厚度 bin 汇总绘图 + Phase2 拟合，
    厚度分布图，散点图（无厚度 bin）。
    """
    physical_params_dir = Path(physical_params_dir)
    phase2_dir = Path(phase2_dir)
    phase1_dir = Path(phase1_dir)
    physical_params_dir.mkdir(parents=True, exist_ok=True)

    run_info = _collect_run_stats(
        phase3_networks_dir, thickness_sampling_dir, phase4_dir, sample_types, n_runs, geometry_size, phase1_dir
    )
    if not run_info:
        print("无有效 run 数据，跳过 physical_params")
        return

    # 保存标量记录（不含数组列）
    scalar_cols = [
        "sample_type", "run", "thickness_nm", "n_pore", "n_throat",
        "n_cross_cell_throat", "n_throat_no_cross",
        "rho_pore", "frac_pore", "rho_throat", "rho_throat_no_cross",
        "frac_throat", "frac_throat_no_cross", "Theta_global",
    ]
    df_rec = pd.DataFrame([{k: r[k] for k in scalar_cols if k in r} for r in run_info])
    df_rec.to_excel(physical_params_dir / "run_stats_per_sample.xlsx", index=False)
    print(f"已保存: {physical_params_dir / 'run_stats_per_sample.xlsx'}")

    # 加载 Phase2 参数
    radius_file = phase2_dir / "radius_by_thickness_bins_analysis.json"
    throat_len_file = phase2_dir / "throat_length_analysis.json"
    degree_file = phase2_dir / "degree_by_thickness_bins_analysis.json"
    density_file = phase2_dir / "density_frac_by_thickness_bins_analysis.json"
    throat_len_data = {}
    degree_data = {}
    density_data = {}
    pore_radius_dist = throat_radius_dist = {}
    if radius_file.exists():
        with open(radius_file, "r", encoding="utf-8") as f:
            rd = json.load(f)
        pore_radius_dist = rd.get("pore", {})
        throat_radius_dist = rd.get("throat", {})
    if throat_len_file.exists():
        with open(throat_len_file, "r", encoding="utf-8") as f:
            throat_len_data = json.load(f)
    overlap_ratio_file = phase2_dir / "overlap_throat_R_ratio_by_thickness_bin.json"
    overlap_ratio_data: dict = {}
    if overlap_ratio_file.exists():
        with open(overlap_ratio_file, "r", encoding="utf-8") as f:
            overlap_ratio_data = json.load(f)
    if degree_file.exists():
        with open(degree_file, "r", encoding="utf-8") as f:
            degree_data = json.load(f)
    if density_file.exists():
        with open(density_file, "r", encoding="utf-8") as f:
            density_data = json.load(f)

    def _plot_continuous(metric_name, xlabel, get_vals, phase2_dist_by_st, use_radius_pdf=True, x_range_fn=None):
        for st in sample_types:
            dist = phase2_dist_by_st.get(st, {})
            bin_edges = np.array(dist.get("bin_edges", []), dtype=float)
            bins_fit = dist.get("bins", [])
            if len(bin_edges) < 2:
                continue
            n_bins = len(bin_edges) - 1
            vals_per_bin = [[] for _ in range(n_bins)]
            for r in run_info:
                if r["sample_type"] != st:
                    continue
                k = np.clip(np.searchsorted(bin_edges, r["thickness_nm"], side="right") - 1, 0, n_bins - 1)
                v = get_vals(r)
                if len(v) > 0:
                    vals_per_bin[k].extend(v.tolist())
            bins_with_data = [i for i in range(n_bins) if len(vals_per_bin[i]) > 0]
            if not bins_with_data:
                continue
            n_panels = len(bins_with_data)
            n_cols = min(4, n_panels)
            n_rows = (n_panels + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
            if n_rows == 1:
                axes = np.atleast_2d(axes)
            axes_flat = axes.flatten()
            for idx, k in enumerate(bins_with_data):
                ax = axes_flat[idx]
                vals = np.array(vals_per_bin[k], dtype=float)
                n_bins_hist = min(30, max(5, len(vals) // 3))
                ax.hist(vals, bins=n_bins_hist, weights=np.ones_like(vals) / len(vals),
                        color="steelblue", alpha=0.8, edgecolor="white", label="Simulated")
                if k < len(bins_fit) and bins_fit[k].get("fit"):
                    if x_range_fn is not None and len(vals) > 0:
                        x_min, x_max = x_range_fn(vals)
                    else:
                        x_min = max(0.1, float(vals.min()) * 0.8) if len(vals) > 0 else 0.5
                        x_max = float(vals.max()) * 1.2 if len(vals) > 0 else 50
                    x_plot = np.linspace(x_min, x_max, 200)
                    pdf_vals = _radius_fit_pdf(x_plot, bins_fit[k]) if use_radius_pdf else _metric_fit_pdf(x_plot, bins_fit[k])
                    if np.any(pdf_vals > 0):
                        bw = (vals.max() - vals.min()) / n_bins_hist if n_bins_hist and vals.max() > vals.min() else 1e-6
                        ax.plot(x_plot, pdf_vals * bw, "r-", linewidth=2, label="Phase2 fit")
                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_ylabel("Probability", fontsize=10)
                ax.set_title(f"T in [{bin_edges[k]:.0f},{bin_edges[k+1]:.0f}) nm (n={len(vals)})", fontsize=10)
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
            for idx in range(len(bins_with_data), len(axes_flat)):
                axes_flat[idx].set_visible(False)
            plt.suptitle(f"{metric_name}: Simulated vs Phase2 fit ({st})", fontsize=12, fontweight="bold")
            plt.tight_layout()
            fname = metric_name.lower().replace(" ", "_").replace("/", "_") + "_" + st + ".png"
            fig.savefig(physical_params_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # 喉半径：按 overlap/非overlap 拆成双色堆叠柱状图
    def _plot_throat_radius_overlap(metric_name, xlabel, phase2_dist_by_st, use_radius_pdf=True, x_range_fn=None):
        for st in sample_types:
            dist = phase2_dist_by_st.get(st, {})
            bin_edges = np.array(dist.get("bin_edges", []), dtype=float)
            bins_fit = dist.get("bins", [])
            if len(bin_edges) < 2:
                continue
            n_bins = len(bin_edges) - 1

            non_vals_per_bin = [[] for _ in range(n_bins)]
            ov_vals_per_bin = [[] for _ in range(n_bins)]
            for r in run_info:
                if r["sample_type"] != st:
                    continue
                k = np.clip(np.searchsorted(bin_edges, r["thickness_nm"], side="right") - 1, 0, n_bins - 1)
                v = np.asarray(r.get("throat_radii", np.array([])), dtype=float)
                if v.size == 0:
                    continue
                ov_mask = np.asarray(r.get("throat_overlap_mask", np.array([], dtype=bool)), dtype=bool)
                if ov_mask.size != v.size:
                    # 与旧 run 兼容：如果缺 overlap mask，则默认当作 non-overlap
                    non_vals_per_bin[k].extend(v.tolist())
                    continue
                non_vals_per_bin[k].extend(v[~ov_mask].tolist())
                ov_vals_per_bin[k].extend(v[ov_mask].tolist())

            bins_with_data = [i for i in range(n_bins) if (len(non_vals_per_bin[i]) + len(ov_vals_per_bin[i])) > 0]
            if not bins_with_data:
                continue

            n_panels = len(bins_with_data)
            n_cols = min(4, n_panels)
            n_rows = (n_panels + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
            if n_rows == 1:
                axes = np.atleast_2d(axes)
            axes_flat = axes.flatten()

            for idx, k in enumerate(bins_with_data):
                ax = axes_flat[idx]
                vals_non = np.array(non_vals_per_bin[k], dtype=float)
                vals_ov = np.array(ov_vals_per_bin[k], dtype=float)
                n_total = int(vals_non.size + vals_ov.size)
                vals_total = np.concatenate([vals_non, vals_ov]) if n_total > 0 else np.array([], dtype=float)

                if n_total == 0:
                    ax.set_visible(False)
                    continue

                n_bins_hist = min(30, max(5, n_total // 3))
                vmin, vmax = float(vals_total.min()), float(vals_total.max())
                if vmax <= vmin:
                    vmax = vmin + 1e-9

                w_non = np.ones_like(vals_non, dtype=float) / n_total if vals_non.size > 0 else np.array([])
                w_ov = np.ones_like(vals_ov, dtype=float) / n_total if vals_ov.size > 0 else np.array([])
                ax.hist(
                    [vals_non, vals_ov],
                    bins=n_bins_hist,
                    range=(vmin, vmax),
                    weights=[w_non, w_ov],
                    stacked=True,
                    color=["steelblue", "coral"],
                    alpha=0.8,
                    edgecolor="white",
                    label=["Non-overlap throat", "Overlap throat"],
                )

                if k < len(bins_fit) and bins_fit[k].get("fit"):
                    vals_for_fit = vals_total
                    if x_range_fn is not None and len(vals_for_fit) > 0:
                        x_min, x_max = x_range_fn(vals_for_fit)
                    else:
                        x_min = max(0.1, float(vals_for_fit.min()) * 0.8) if len(vals_for_fit) > 0 else 0.5
                        x_max = float(vals_for_fit.max()) * 1.2 if len(vals_for_fit) > 0 else 50
                    x_plot = np.linspace(x_min, x_max, 200)
                    pdf_vals = _radius_fit_pdf(x_plot, bins_fit[k]) if use_radius_pdf else _metric_fit_pdf(x_plot, bins_fit[k])
                    if np.any(pdf_vals > 0):
                        bw = (float(vals_for_fit.max()) - float(vals_for_fit.min())) / n_bins_hist if n_bins_hist and vals_for_fit.max() > vals_for_fit.min() else 1e-6
                        ax.plot(x_plot, pdf_vals * bw, "r-", linewidth=2, label="Phase2 fit")

                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_ylabel("Probability", fontsize=10)
                ax.set_title(f"T in [{bin_edges[k]:.0f},{bin_edges[k+1]:.0f}) nm (n={n_total})", fontsize=10)
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)

            for idx in range(len(bins_with_data), len(axes_flat)):
                axes_flat[idx].set_visible(False)

            plt.suptitle(f"{metric_name}: Simulated vs Phase2 fit ({st})", fontsize=12, fontweight="bold")
            plt.tight_layout()
            fname = metric_name.lower().replace(" ", "_").replace("/", "_") + "_" + st + ".png"
            fig.savefig(physical_params_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # 孔半径、喉半径、喉长度
    pore_dist = {st: pore_radius_dist.get(st, {}) for st in sample_types}
    throat_dist = {st: throat_radius_dist.get(st, {}) for st in sample_types}
    _plot_continuous("Pore radius", "Pore radius (nm)", lambda r: r["pore_radii"], pore_dist)
    _plot_continuous("Throat radius", "Throat radius (nm)", lambda r: r["throat_radii"], throat_dist)
    throat_len_dist = {st: throat_len_data.get(st, {}) for st in sample_types}
    _plot_continuous("Throat length", "Throat length (nm)", lambda r: r["throat_lengths"], throat_len_dist)

    if overlap_ratio_data:
        def _get_overlap_R_ratios_for_plot(r):
            """与 Phase2 拟合同口径：ratio 按 ratio_clip_max_for_fit 裁剪后再与拟合曲线对比。"""
            v = r.get("overlap_throat_R_ratios", np.array([]))
            if len(v) == 0:
                return v
            st = r["sample_type"]
            cmax = float(overlap_ratio_data.get(st, {}).get("ratio_clip_max_for_fit", 10.0))
            return np.clip(np.asarray(v, dtype=float), 1e-9, cmax)

        overlap_ratio_dist = {st: overlap_ratio_data.get(st, {}) for st in sample_types}
        _plot_continuous(
            "Overlap throat R ratio",
            r"$R_{\mathrm{throat}}/R_{\mathrm{cap}}$ (overlap)",
            _get_overlap_R_ratios_for_plot,
            overlap_ratio_dist,
        )

    # 孔度数
    for st in sample_types:
        dist = degree_data.get(st, {})
        bin_edges = np.array(dist.get("bin_edges", []), dtype=float)
        bins_deg = dist.get("bins", [])
        if len(bin_edges) < 2:
            continue
        n_bins = len(bin_edges) - 1
        deg_per_bin = [[] for _ in range(n_bins)]
        for r in run_info:
            if r["sample_type"] != st:
                continue
            k = np.clip(np.searchsorted(bin_edges, r["thickness_nm"], side="right") - 1, 0, n_bins - 1)
            if len(r["pore_degrees"]) > 0:
                deg_per_bin[k].extend(r["pore_degrees"].tolist())
        bins_with_data = [i for i in range(n_bins) if len(deg_per_bin[i]) > 0]
        if not bins_with_data:
            continue
        n_panels = len(bins_with_data)
        n_cols = min(4, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = np.atleast_2d(axes)
        axes_flat = axes.flatten()
        for idx, k in enumerate(bins_with_data):
            ax = axes_flat[idx]
            vals = np.array(deg_per_bin[k], dtype=int)
            d_min, d_max = int(vals.min()), int(vals.max())
            if d_max == d_min:
                d_max = d_min + 1
            bins_int = np.arange(d_min - 0.5, d_max + 1.5, 1.0)
            ax.hist(vals, bins=bins_int, weights=np.ones_like(vals, dtype=float) / len(vals),
                    color="steelblue", alpha=0.8, edgecolor="white", label="Simulated")
            if k < len(bins_deg) and bins_deg[k].get("fit"):
                f = bins_deg[k].get("fit") or bins_deg[k]
                if f.get("fit_type") == "empirical":
                    dv = np.array(f.get("degree_values", []), dtype=int)
                    cnt = np.array(f.get("counts", []), dtype=float)
                    total = float(np.sum(cnt))
                    if total > 0 and len(dv) > 0:
                        ax.plot(dv, cnt / total, "ro-", linewidth=2, markersize=4, label="Phase2 empirical PMF")
                else:
                    d_range = np.arange(d_min, d_max + 1)
                    lam = max(float(f.get("mean", np.mean(vals))), 1e-3)
                    ax.plot(d_range, stats.poisson.pmf(d_range, mu=lam), "ro-", linewidth=2, markersize=4, label="Phase2 fit PMF")
            ax.set_xlabel("Pore degree", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            ax.set_title(f"T in [{bin_edges[k]:.0f},{bin_edges[k+1]:.0f}) nm (n={len(vals)})", fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        for idx in range(len(bins_with_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.suptitle(f"Pore degree: Simulated vs Phase2 fit ({st})", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(physical_params_dir / f"pore_degree_{st}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # rho_pore, rho_throat：数量密度；frac_pore, frac_throat：体积分数（均来自 density_frac_by_thickness_bins_analysis）
    def _get_rho_pore(r):
        v = r.get("rho_pore")
        return np.array([v]) if v is not None and np.isfinite(v) else np.array([])
    def _get_frac_pore(r):
        v = r.get("frac_pore")
        return np.array([v]) if v is not None and np.isfinite(v) else np.array([])
    def _get_rho_throat(r):
        v = r.get("rho_throat_no_cross")
        if v is None or not np.isfinite(v):
            v = r.get("rho_throat")
        return np.array([v]) if v is not None and np.isfinite(v) else np.array([])
    def _get_frac_throat(r):
        v = r.get("frac_throat_no_cross")
        if v is None or not np.isfinite(v):
            v = r.get("frac_throat")
        return np.array([v]) if v is not None and np.isfinite(v) else np.array([])
    for metric, xlabel, get_vals in [
        ("rho_pore", "Pore count density (1/nm^3)", _get_rho_pore),
        ("frac_pore", "Pore volume fraction", _get_frac_pore),
        ("rho_throat", "Throat count density excluding cross-cell throats (1/nm^3)", _get_rho_throat),
        ("frac_throat", "Throat volume fraction excluding cross-cell throats", _get_frac_throat),
    ]:
        phase2_dist = {st: density_data.get(metric, {}).get(st, {}) for st in sample_types}
        for st in sample_types:
            dist = phase2_dist.get(st, {})
            bin_edges = np.array(dist.get("bin_edges", []), dtype=float)
            bins_fit = dist.get("bins", [])
            if len(bin_edges) < 2:
                continue
            n_bins = len(bin_edges) - 1
            vals_per_bin = []
            for _ in range(n_bins):
                vals_per_bin.append([])
            for r in run_info:
                if r["sample_type"] != st:
                    continue
                k = np.clip(np.searchsorted(bin_edges, r["thickness_nm"], side="right") - 1, 0, n_bins - 1)
                v = get_vals(r)
                if len(v) > 0:
                    vals_per_bin[k].extend(v.tolist())
            bins_with_data = [i for i in range(n_bins) if len(vals_per_bin[i]) > 0]
            if not bins_with_data:
                continue
            n_panels = len(bins_with_data)
            n_cols = min(4, n_panels)
            n_rows = (n_panels + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
            if n_rows == 1:
                axes = np.atleast_2d(axes)
            axes_flat = axes.flatten()
            for idx, k in enumerate(bins_with_data):
                ax = axes_flat[idx]
                vals = np.array(vals_per_bin[k], dtype=float)
                n_bins_hist = _phase2_density_frac_hist_nbins(len(vals))
                # 预取 fit 参数，用于 vals 全相同时的 bins 范围
                fit_mean = fit_std = None
                if k < len(bins_fit) and bins_fit[k].get("fit"):
                    f = bins_fit[k].get("fit") or bins_fit[k]
                    fit_mean = float(f.get("mean", 0))
                    fit_std = max(float(f.get("std", 1e-6)), 1e-10)
                # 当 vals 全相同时：用「单一宽 bin」画柱状图；若用多窄 bin 则只有一格有值，视觉上像一条竖线
                bin_width = 1.0
                if len(vals) > 0 and vals.min() == vals.max():
                    v0 = float(vals[0])
                    eps = max(abs(v0) * 0.1, 1e-10) if v0 != 0 else 0.01
                    if "frac" in metric:
                        eps = min(eps, 0.05)
                    else:
                        # rho（数量密度）量级 ~1e-4~1e-3，需足够 eps 使柱状图与曲线可见
                        eps = max(eps, 1e-4, (fit_std or 1e-5) * 2)
                    bins_hist = [v0 - eps, v0 + eps]
                    bin_width = 2.0 * eps
                else:
                    bins_hist = n_bins_hist
                    vmin, vmax = float(vals.min()), float(vals.max())
                    if vmax > vmin:
                        bin_width = (vmax - vmin) / float(n_bins_hist)
                    else:
                        bin_width = max((fit_std or 1e-10), 1e-10)
                # 与 Phase2 plot_density_frac_by_thickness_bins 一致：柱为概率质量（柱高之和=1），红线为 pdf×bin_width，y 轴同量级；避免 density=True 时柱极高、曲线被压成贴底直线
                # 勿用变量名 bin_edges 接收 hist 返回值，否则会覆盖 Phase2 厚度 bin_edges，导致 set_title 时 IndexError
                counts, hist_edges, _patches = ax.hist(
                    vals,
                    bins=bins_hist,
                    weights=np.ones_like(vals) / len(vals),
                    density=False,
                    color="steelblue",
                    alpha=0.8,
                    edgecolor="white",
                    label="Simulated",
                )
                if len(hist_edges) >= 2:
                    bw = float(np.median(np.diff(hist_edges)))
                    if bw > 0:
                        bin_width = bw
                if k < len(bins_fit) and bins_fit[k].get("fit"):
                    fbin = bins_fit[k].get("fit") or bins_fit[k]
                    fit_mean = float(fbin.get("mean", 0))
                    fit_std = max(float(fbin.get("std", 1e-6)), 1e-10)
                    # x 范围需同时覆盖模拟数据与 Phase2 拟合，否则 PDF 在图上为 0
                    if len(vals) > 0:
                        x_min = min(float(vals.min()) * 0.9, fit_mean - 6 * fit_std)
                        x_max = max(float(vals.max()) * 1.1, fit_mean + 6 * fit_std)
                    else:
                        x_min = fit_mean - 6 * fit_std
                        x_max = fit_mean + 6 * fit_std
                    if "frac" in metric:
                        x_min = max(1e-6, x_min)
                        x_max = min(1.0, x_max)
                    else:
                        x_min = max(1e-10, x_min)
                    if x_max <= x_min:
                        x_max = x_min + max(fit_std, 1e-10)
                    x_plot = np.linspace(x_min, x_max, 200)
                    pdf_vals = _metric_fit_pdf(x_plot, bins_fit[k])
                    curve_y = pdf_vals * bin_width
                    if np.any(pdf_vals > 0):
                        ax.plot(x_plot, curve_y, "r-", linewidth=2, label="Phase2 fit")
                    y_hi = max(
                        float(np.max(counts)) if len(counts) else 0.0,
                        float(np.nanmax(curve_y)) if np.any(np.isfinite(curve_y)) else 0.0,
                    )
                    ax.set_ylim(0, max(y_hi * 1.12, 1e-12))
                    xpad = (x_max - x_min) * 0.03 if x_max > x_min else 1e-9
                    ax.set_xlim(x_min - xpad, x_max + xpad)
                else:
                    y_hi = float(np.max(counts)) if len(counts) else 0.0
                    ax.set_ylim(0, max(y_hi * 1.12, 1e-12))
                    if len(vals) > 0:
                        v_lo, v_hi = float(vals.min()), float(vals.max())
                        vpad = (v_hi - v_lo) * 0.08 if v_hi > v_lo else max(abs(v_lo) * 0.08, 1e-9)
                        ax.set_xlim(v_lo - vpad, v_hi + vpad)
                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_ylabel("Probability", fontsize=10)
                ax.set_title(f"T in [{bin_edges[k]:.0f},{bin_edges[k+1]:.0f}) nm (n={len(vals)})", fontsize=10)
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
            for idx in range(len(bins_with_data), len(axes_flat)):
                axes_flat[idx].set_visible(False)
            _kind = " (volume fraction)" if "frac" in metric else " (count density, not volume fraction)"
            plt.suptitle(f"{xlabel}: Simulated vs Phase2 fit ({st}){_kind}", fontsize=12, fontweight="bold")
            plt.tight_layout()
            fig.savefig(physical_params_dir / f"{metric}_{st}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ρ_throat/ρ_pore = N_throat/N_pore（同体积，数量比）；Phase2 边际来自 ρ（数量密度），非 frac（体积分数）
    def _get_rho_ratio(r):
        rp = r.get("rho_pore")
        rt = r.get("rho_throat_no_cross")
        if rt is None or not np.isfinite(rt):
            rt = r.get("rho_throat")
        if rp is None or rt is None or not np.isfinite(rp) or not np.isfinite(rt) or float(rp) <= 0:
            return np.array([])
        return np.array([float(rt) / float(rp)])

    for st in sample_types:
        dist = density_data.get("rho_pore", {}).get(st, {})
        bin_edges = np.array(dist.get("bin_edges", []), dtype=float)
        bins_fit_p = density_data.get("rho_pore", {}).get(st, {}).get("bins", [])
        bins_fit_t = density_data.get("rho_throat", {}).get(st, {}).get("bins", [])
        if len(bin_edges) < 2:
            continue
        n_bins = len(bin_edges) - 1
        vals_per_bin = [[] for _ in range(n_bins)]
        for r in run_info:
            if r["sample_type"] != st:
                continue
            kk = np.clip(np.searchsorted(bin_edges, r["thickness_nm"], side="right") - 1, 0, n_bins - 1)
            v = _get_rho_ratio(r)
            if len(v) > 0:
                vals_per_bin[kk].extend(v.tolist())
        bins_with_data = [i for i in range(n_bins) if len(vals_per_bin[i]) > 0]
        if not bins_with_data:
            continue
        n_panels = len(bins_with_data)
        n_cols = min(4, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = np.atleast_2d(axes)
        axes_flat = axes.flatten()
        for idx, k in enumerate(bins_with_data):
            ax = axes_flat[idx]
            vals = np.array(vals_per_bin[k], dtype=float)
            n_bins_hist = _phase2_density_frac_hist_nbins(len(vals))
            if len(vals) > 0 and vals.min() == vals.max():
                v0 = float(vals[0])
                eps = max(abs(v0) * 0.05, 1e-6, float(np.std(vals)) * 2 if len(vals) > 1 else 0.01)
                bins_hist = [v0 - eps, v0 + eps]
            else:
                bins_hist = n_bins_hist
            # density=True：纵轴为 PDF，与 Phase2 KDE 曲线同量纲可比；显式 ylim 避免单 bin 时柱极高、曲线贴底
            dens_hist, _edges, _p = ax.hist(
                vals,
                bins=bins_hist,
                weights=np.ones_like(vals) / len(vals),
                density=True,
                color="steelblue",
                alpha=0.85,
                edgecolor="white",
                label="Simulated",
            )
            ax.axvline(
                float(np.mean(vals)),
                color="darkorange",
                linestyle="--",
                linewidth=1.5,
                label=f"Sim. mean={np.mean(vals):.4g}",
            )
            pdf_vals = None
            if (
                k < len(bins_fit_p)
                and k < len(bins_fit_t)
                and bins_fit_p[k].get("fit")
                and bins_fit_t[k].get("fit")
            ):
                fp = bins_fit_p[k].get("fit") or bins_fit_p[k]
                ft = bins_fit_t[k].get("fit") or bins_fit_t[k]
                fit_mean_p = float(fp.get("mean", 0))
                fit_std_p = max(float(fp.get("std", 1e-6)), 1e-10)
                fit_mean_t = float(ft.get("mean", 0))
                fit_std_t = max(float(ft.get("std", 1e-6)), 1e-10)
                rng_q = np.random.default_rng(42)
                Pq = _sample_from_metric_fit_bin(bins_fit_p[k], 40000, rng_q)
                Tq = _sample_from_metric_fit_bin(bins_fit_t[k], 40000, rng_q)
                Rq = Tq / np.maximum(Pq, 1e-15)
                Rq = Rq[np.isfinite(Rq) & (Rq > 0)]
                if len(vals) > 0:
                    x_min = min(float(np.min(vals)) * 0.88, float(np.quantile(Rq, 0.01)) if len(Rq) > 20 else float(np.min(vals)))
                    x_max = max(float(np.max(vals)) * 1.12, float(np.quantile(Rq, 0.99)) if len(Rq) > 20 else float(np.max(vals)))
                else:
                    x_min = float(np.quantile(Rq, 0.01)) if len(Rq) > 20 else 0.01
                    x_max = float(np.quantile(Rq, 0.99)) if len(Rq) > 20 else 10.0
                x_min = max(1e-9, float(x_min))
                if x_max <= x_min:
                    x_max = x_min + max(fit_std_t / max(fit_mean_p, 1e-15), 0.01)
                x_plot = np.linspace(x_min, x_max, 200)
                pdf_vals = _phase2_ratio_kde_pdf(bins_fit_p[k], bins_fit_t[k], x_plot)
                if pdf_vals is not None and np.any(np.isfinite(pdf_vals)) and np.nanmax(pdf_vals) > 0:
                    ax.plot(
                        x_plot,
                        pdf_vals,
                        "r-",
                        linewidth=2,
                        label="Phase2 fit (rho marginals, count density -> N_t/N_p)",
                    )
                    xpad = (x_max - x_min) * 0.02 if x_max > x_min else 1e-9
                    ax.set_xlim(x_min - xpad, x_max + xpad)
            y_top = float(np.max(dens_hist)) if len(dens_hist) else 0.0
            if pdf_vals is not None and np.any(np.isfinite(pdf_vals)):
                y_top = max(y_top, float(np.nanmax(pdf_vals)))
            ax.set_ylim(0, max(y_top * 1.15, 1e-15))
            ax.set_xlabel(
                "N_throat / N_pore = rho_throat / rho_pore (count ratio, not volume frac.)",
                fontsize=9,
            )
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(f"T in [{bin_edges[k]:.0f},{bin_edges[k+1]:.0f}) nm (n={len(vals)})", fontsize=10)
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True, alpha=0.3)
        for idx in range(len(bins_with_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.suptitle(
            f"Count ratio N_throat/N_pore (=rho_throat/rho_pore) vs Phase2 rho marginals ({st}); "
            f"not frac_throat/frac_pore",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(physical_params_dir / f"rho_throat_over_rho_pore_{st}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 散点图：喉-孔半径、喉长度-半径（不区分厚度），叠加 Phase1_5 线性拟合
    phase1_5_dir = _root / "results" / "phase1_5_param_vs_thickness"
    fits_all = {}
    fits_path = phase1_5_dir / "phase1_5_linear_fits.json"
    if fits_path.exists():
        try:
            with open(fits_path, "r", encoding="utf-8") as f:
                fits_all = json.load(f)
        except Exception:
            fits_all = {}

    # 散点图按 sample_type 分开，WT/AS 各一个子图
    data_by_st = {}
    for r in run_info:
        st = r["sample_type"]
        if st not in data_by_st:
            data_by_st[st] = {
                "mean_pore": [],
                "throat_r_scatter": [],
                "overlap_mean_pore_scatter": [],
                "throat_r": [],
                "throat_L": [],
                "overlap_throat": [],
                "pore_radii": [],
                "pore_degrees": [],
            }
        if len(r.get("mean_pore_per_throat", [])) > 0 and len(r.get("throat_r_for_scatter", [])) == len(r["mean_pore_per_throat"]):
            data_by_st[st]["mean_pore"].extend(r["mean_pore_per_throat"].tolist())
            data_by_st[st]["throat_r_scatter"].extend(r["throat_r_for_scatter"].tolist())
            om = r.get("overlap_mean_pore_scatter", np.array([], dtype=bool))
            om = np.asarray(om, dtype=bool)
            if om.size == len(r["mean_pore_per_throat"]):
                data_by_st[st]["overlap_mean_pore_scatter"].extend(om.tolist())
            else:
                data_by_st[st]["overlap_mean_pore_scatter"].extend([False] * len(r["mean_pore_per_throat"]))
        if len(r.get("throat_radii", [])) > 0 and len(r.get("throat_lengths", [])) == len(r["throat_radii"]):
            data_by_st[st]["throat_r"].extend(r["throat_radii"].tolist())
            data_by_st[st]["throat_L"].extend(r["throat_lengths"].tolist())
            ot = r.get("throat_overlap_mask", np.array([], dtype=bool))
            ot = np.asarray(ot, dtype=bool)
            if ot.size == len(r["throat_radii"]):
                data_by_st[st]["overlap_throat"].extend(ot.tolist())
            else:
                data_by_st[st]["overlap_throat"].extend([False] * len(r["throat_radii"]))
        if len(r.get("pore_radii", [])) > 0 and len(r.get("pore_degrees", [])) == len(r["pore_radii"]):
            data_by_st[st]["pore_radii"].extend(r["pore_radii"].tolist())
            data_by_st[st]["pore_degrees"].extend(r["pore_degrees"].tolist())
    for st in list(data_by_st.keys()):
        for k in data_by_st[st]:
            data_by_st[st][k] = np.array(data_by_st[st][k])

    # 喉-孔半径散点图：WT/AS 各一子图
    sts_with_pore = [st for st in sample_types if st in data_by_st and len(data_by_st[st]["mean_pore"]) > 10]
    if sts_with_pore:
        fig, axes = plt.subplots(1, len(sts_with_pore), figsize=(5 * len(sts_with_pore), 5))
        if len(sts_with_pore) == 1:
            axes = [axes]
        fits_rel = fits_all.get("throat_radius_vs_mean_pore_radius", {})
        for i, st in enumerate(sts_with_pore):
            ax = axes[i]
            mp = data_by_st[st]["mean_pore"]
            tr = data_by_st[st]["throat_r_scatter"]
            ax.scatter(mp, tr, alpha=0.1, s=2, c="steelblue", label="Simulated")
            x_min, x_max = mp.min(), mp.max()
            if x_max <= x_min:
                x_max = x_min + 1.0
            x_line = np.linspace(x_min, x_max, 200)
            if np.std(mp) > 1e-10 and np.std(tr) > 1e-10:
                slope, intercept, r_val, _, _ = stats.linregress(mp, tr)
                ax.plot(x_line, slope * x_line + intercept, "darkorange", linewidth=2, label=f"Simulated fit r={r_val:.3f}")
            fit_one = fits_rel.get(st, {})
            if fit_one and "slope" in fit_one and "intercept" in fit_one:
                s, b = float(fit_one["slope"]), float(fit_one["intercept"])
                r_p = fit_one.get("pearson_r")
                lbl = f"Phase1_5" + (f" r={float(r_p):.3f}" if r_p is not None else "")
                ax.plot(x_line, s * x_line + b, "r--", linewidth=2, label=lbl)
            ax.set_xlabel("Mean pore radius (nm)", fontsize=10)
            ax.set_ylabel("Throat radius (nm)", fontsize=10)
            ax.set_title(f"Throat vs mean pore radius ({st})")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(physical_params_dir / "throat_pore_radius_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 喉长度-半径散点图：WT/AS 各一子图
    sts_with_throat = [st for st in sample_types if st in data_by_st and len(data_by_st[st]["throat_r"]) > 10]
    if sts_with_throat:
        fig, axes = plt.subplots(1, len(sts_with_throat), figsize=(5 * len(sts_with_throat), 5))
        if len(sts_with_throat) == 1:
            axes = [axes]
        fits_rel = fits_all.get("throat_radius_vs_length", {})
        for i, st in enumerate(sts_with_throat):
            ax = axes[i]
            tr = data_by_st[st]["throat_r"]
            tl = data_by_st[st]["throat_L"]
            ax.scatter(tr, tl, alpha=0.1, s=2, c="steelblue", label="Simulated")
            x_min, x_max = tr.min(), tr.max()
            if x_max <= x_min:
                x_max = x_min + 1.0
            x_line = np.linspace(x_min, x_max, 200)
            if np.std(tr) > 1e-10 and np.std(tl) > 1e-10:
                slope, intercept, r_val, _, _ = stats.linregress(tr, tl)
                ax.plot(x_line, slope * x_line + intercept, "darkorange", linewidth=2, label=f"Simulated fit r={r_val:.3f}")
            fit_one = fits_rel.get(st, {})
            if fit_one and "slope" in fit_one and "intercept" in fit_one:
                s, b = float(fit_one["slope"]), float(fit_one["intercept"])
                r_p = fit_one.get("pearson_r")
                lbl = f"Phase1_5" + (f" r={float(r_p):.3f}" if r_p is not None else "")
                ax.plot(x_line, s * x_line + b, "r--", linewidth=2, label=lbl)
            ax.set_xlabel("Throat radius (nm)", fontsize=10)
            ax.set_ylabel("Throat length (nm)", fontsize=10)
            ax.set_title(f"Throat length vs radius ({st})")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(physical_params_dir / "throat_length_radius_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 孔度数-半径散点图：WT/AS 各一子图，模拟线性拟合 + Phase1_5 拟合
    sts_with_pore_deg = [st for st in sample_types if st in data_by_st and len(data_by_st[st]["pore_radii"]) > 10]
    if sts_with_pore_deg:
        fig, axes = plt.subplots(1, len(sts_with_pore_deg), figsize=(5 * len(sts_with_pore_deg), 5))
        if len(sts_with_pore_deg) == 1:
            axes = [axes]
        fits_rel = fits_all.get("pore_degree_vs_radius", {})
        for i, st in enumerate(sts_with_pore_deg):
            ax = axes[i]
            deg = data_by_st[st]["pore_degrees"]
            rad = data_by_st[st]["pore_radii"]
            ax.scatter(deg, rad, alpha=0.1, s=2, c="steelblue", label="Simulated")
            x_min, x_max = deg.min(), deg.max()
            if x_max <= x_min:
                x_max = x_min + 1.0
            x_line = np.linspace(x_min, x_max, 200)
            if np.std(deg) > 1e-10 and np.std(rad) > 1e-10:
                slope, intercept, r_val, _, _ = stats.linregress(deg, rad)
                ax.plot(x_line, slope * x_line + intercept, "darkorange", linewidth=2, label=f"Simulated fit r={r_val:.3f}")
            fit_one = fits_rel.get(st, {})
            if fit_one and "slope" in fit_one and "intercept" in fit_one:
                s, b = float(fit_one["slope"]), float(fit_one["intercept"])
                r_p = fit_one.get("pearson_r")
                lbl = f"Phase1_5" + (f" r={float(r_p):.3f}" if r_p is not None else "")
                ax.plot(x_line, s * x_line + b, "r--", linewidth=2, label=lbl)
            ax.set_xlabel("Pore degree", fontsize=10)
            ax.set_ylabel("Pore radius (nm)", fontsize=10)
            ax.set_title(f"Pore degree vs radius ({st})")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(physical_params_dir / "pore_degree_vs_radius_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 厚度分布：柱状图 + Phase1 GMM 拟合曲线
    params_file = phase1_dir / "thickness_distribution_parameters.json"
    gmm_by_st = {}
    if params_file.exists():
        with open(params_file, "r", encoding="utf-8") as f:
            gmm_by_st = json.load(f)
    fig, axes = plt.subplots(1, len(sample_types), figsize=(5 * len(sample_types), 4))
    if len(sample_types) == 1:
        axes = [axes]
    for i, st in enumerate(sample_types):
        sub = [r for r in run_info if r["sample_type"] == st]
        if not sub:
            continue
        T = np.array([r["thickness_nm"] for r in sub])
        ax = axes[i]
        n_bins = min(25, max(8, len(T) // 2))
        ax.hist(T, bins=n_bins, density=True, alpha=0.7, label=st, edgecolor="white", color="steelblue")
        gmm = gmm_by_st.get(st)
        if gmm and gmm.get("type") == "gmm":
            w = np.array(gmm.get("weights", [1]))
            mu = np.array(gmm.get("means", [150]))
            cov = np.array(gmm.get("covariances", [1]))
            sigma = np.sqrt(cov)
            x_plot = np.linspace(max(1, T.min() - 20), T.max() + 20, 300)
            pdf = np.zeros_like(x_plot)
            for wi, mi, si in zip(w, mu, sigma):
                pdf += wi * stats.norm.pdf(x_plot, loc=mi, scale=si)
            ax.plot(x_plot, pdf, "r-", linewidth=2, label="Phase1 GMM fit")
        ax.set_xlabel("Thickness (nm)")
        ax.set_ylabel("Density")
        ax.set_title(f"{st} thickness (n={len(T)})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(physical_params_dir / "thickness_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"physical_params 已保存: {physical_params_dir}")


def _plot_thickness_sampling_combined(
    thickness_sampling_dir: Path,
    phase1_dir: Path,
    sample_types: list,
    n_runs: int,
):
    """n 次模拟厚度柱状图 + Phase1 AS/WT 拟合曲线，画在一张图上，保存到 thickness_sampling。"""
    thickness_sampling_dir = Path(thickness_sampling_dir)
    phase1_dir = Path(phase1_dir)
    params_file = phase1_dir / "thickness_distribution_parameters.json"
    gmm_by_st = {}
    if params_file.exists():
        with open(params_file, "r", encoding="utf-8") as f:
            gmm_by_st = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"AS": "steelblue", "WT": "coral"}
    x_min_all, x_max_all = float("inf"), float("-inf")

    for st in sample_types:
        json_path = thickness_sampling_dir / f"sampled_thickness_n{n_runs}_{st}.json"
        if not json_path.exists():
            alt = list(thickness_sampling_dir.glob(f"sampled_thickness_n*_{st}.json"))
            json_path = alt[0] if alt else None
        if json_path is None:
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        T = np.array(data.get("thickness_nm", []))
        if len(T) == 0:
            continue
        n_bins = min(25, max(8, len(T) // 2))
        ax.hist(T, bins=n_bins, density=True, alpha=0.6, label=f"{st} (n={len(T)})",
                edgecolor="white", color=colors.get(st, "gray"))
        x_min_all = min(x_min_all, float(T.min()))
        x_max_all = max(x_max_all, float(T.max()))

        gmm = gmm_by_st.get(st)
        if gmm and gmm.get("type") == "gmm":
            w = np.array(gmm.get("weights", [1]))
            mu = np.array(gmm.get("means", [150]))
            cov = np.array(gmm.get("covariances", [1]))
            sigma = np.sqrt(cov)
            x_plot = np.linspace(max(1, T.min() - 30), T.max() + 30, 300)
            pdf = np.zeros_like(x_plot)
            for wi, mi, si in zip(w, mu, sigma):
                pdf += wi * stats.norm.pdf(x_plot, loc=mi, scale=si)
            ax.plot(x_plot, pdf, color=colors.get(st, "gray"), linewidth=2, linestyle="--",
                    label=f"Phase1 {st} GMM fit")
            x_min_all = min(x_min_all, float(np.min(mu - 3 * sigma)))
            x_max_all = max(x_max_all, float(np.max(mu + 3 * sigma)))

    if x_min_all < float("inf"):
        ax.set_xlim(left=max(1, x_min_all - 10), right=x_max_all + 10)
    ax.set_xlabel("Thickness (nm)")
    ax.set_ylabel("Density")
    ax.set_title("Thickness sampling: simulated histogram + Phase1 AS/WT fit")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(thickness_sampling_dir / "thickness_sampling_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {thickness_sampling_dir / 'thickness_sampling_distribution.png'}")


def main():
    parser = argparse.ArgumentParser(description="单 cell 均一厚度：抽样 n 个厚度，轮着用，跑 n 次 Phase3+Phase4，箱型图")
    parser.add_argument("--n-runs", type=int, default=1, help="总模拟次数，默认 1（测试）")
    parser.add_argument("--sample-types", nargs="+", default=["AS", "WT"], help="样本类型，默认 AS WT")
    parser.add_argument("--geometry-size", type=float, default=400, help="单 cell 平面边长 (nm)，默认 400")
    parser.add_argument(
        "--target-thickness-min-nm",
        type=float,
        default=None,
        help="Optional lower bound for targeted thickness rejection sampling.",
    )
    parser.add_argument(
        "--target-thickness-max-nm",
        type=float,
        default=None,
        help="Optional upper bound for targeted thickness rejection sampling.",
    )
    parser.add_argument("--phase1-dir", type=str, default=None)
    parser.add_argument(
        "--wt-thickness-fit",
        type=str,
        choices=("kde", "norm"),
        default="kde",
        help="WT 厚度采样拟合类型（默认 kde；可选 norm）。",
    )
    parser.add_argument(
        "--as-thickness-fit",
        type=str,
        choices=("kde", "gmm", "auto"),
        default="kde",
        help="AS 厚度采样拟合类型（默认 kde；可选 gmm/auto）。",
    )
    parser.add_argument("--phase2-dir", type=str, default=None)
    parser.add_argument("--phase3-dir", type=str, default=None)
    parser.add_argument("--phase4-dir", type=str, default=None)
    parser.add_argument(
        "--run-result-dir",
        type=str,
        default=None,
        help="根输出目录；默认 gbm_full_model/results/run_result。用于小规模测试时可指定独立目录，避免覆盖正式结果。",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-clear", action="store_true", help="不清除 run_result，保留上次结果；默认会清除")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "断点续跑：若某 run 已有 Phase3 双 xlsx 且已有 Phase4 sieving 结果 xlsx，则跳过该 run，不覆盖。"
            "若仅有 Phase3 无 Phase4，则只补算 Phase4。须与 --no-clear 同用，否则会先清空目录。"
        ),
    )
    parser.add_argument(
        "--phase3-retries",
        type=int,
        default=5,
        help="Phase3 子进程失败后的重试次数（默认5，表示最多尝试5次）。",
    )
    parser.add_argument(
        "--phase3-html",
        action="store_true",
        help="Phase3 生成 HTML 可视化（默认关闭；批量跑易触发底层库崩溃，建议需要时再开）",
    )
    parser.add_argument(
        "--phase4-html",
        action="store_true",
        help="Phase4 生成交互式 HTML 可视化（默认关闭；批量跑建议关闭以节省时间和磁盘空间）",
    )
    parser.add_argument(
        "--keep-phase4-diagnostics",
        action="store_true",
        help=(
            "保留每个样本 phase4_sieving 目录下的分类/连通分量/压力等诊断文件。"
            "默认关闭：成功写出最终筛过系数结果后，只保留 *__sieving_summary*.xlsx。"
        ),
    )
    parser.add_argument(
        "--keep-phase4-radius-cache",
        action="store_true",
        help=(
            "保留用于后续不同溶质半径比较的最小 Phase4 缓存："
            "pore_classification、solvent_throat_classification、"
            "solvent_penetration_components、pressure_distribution。"
            "默认关闭；若 --keep-phase4-diagnostics 或 --phase4-html 开启，则会保留完整诊断。"
        ),
    )
    parser.add_argument(
        "--direction-check-eps-nm",
        type=float,
        default=1e-6,
        help="入口/出口在渗透方向（合成：全局 Z）上顺序检验容差（nm），默认 1e-6。",
    )
    parser.add_argument(
        "--sieving-abnormal-max-retries",
        type=int,
        default=2,
        help=(
            "筛过系数 C_out/C_in < -1e-12 或非有限/缺失时，同厚度重新模拟："
            "Phase3（新随机种子）+ Phase4。本参数为「首轮之外的额外轮数」，默认 2（即最多共 3 轮）。"
            "断点续跑仅补 Phase4 时：首轮沿用已有 xlsx；若仍异常，从第 2 轮起会重跑 Phase3。"
        ),
    )
    parser.add_argument(
        "--retry-disconnected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "是否对“无出口节点/不连通”样本做同厚度重生重算。"
            "默认开启：按 --retry-disconnected-prob 概率重生。"
            "关闭（--no-retry-disconnected）时：保留当前结果继续后续 run，Q 仍参与整体筛过系数汇总。"
        ),
    )
    parser.add_argument(
        "--retry-disconnected-prob",
        type=float,
        default=1.0,
        help=(
            "触发条件为 no_exit_nodes 时，执行同厚度重生的概率（默认 1.0）。"
            "取值范围 [0,1]；0 表示从不因 no_exit_nodes 重生，1 表示总是重生（受最大重试轮次限制）。"
        ),
    )
    parser.add_argument(
        "--run-quintile-mixing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "对合成网络的每 run xlsx 做孔五分位组内喉 frac_intra（与阶段 1.5 同口径），"
            "输出到 pore_quintile_throat_mixing_synthetic/（默认关闭；需要时可加 --run-quintile-mixing 开启；"
            "置换见下方 n-perm）。"
        ),
    )
    parser.add_argument(
        "--run-quintile-mixing-n-perm",
        type=int,
        default=99,
        help="上述检验的置换次数；0 不做置换、不生成 perm 诊断图（默认 99，与阶段1.5 的 Δdiag/p 图一致；大批量可改 0）。",
    )
    parser.add_argument(
        "--run-quintile-mixing-min-throats",
        type=int,
        default=30,
        help="置换检验要求的最小喉数（默认 30）。",
    )
    parser.add_argument(
        "--phase3-pore-q4-spatial-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "传给 Phase3 的 --pore-radius-q4-spatial-bias（默认关）；"
            "开启：--phase3-pore-q4-spatial-bias。"
        ),
    )
    parser.add_argument(
        "--phase3-pore-placement-cell-nm",
        type=float,
        default=50.0,
        help="传给 Phase3 的 --pore-placement-cell-nm（默认 50 nm）。",
    )
    parser.add_argument(
        "--phase3-throat-neighbor-nm",
        type=float,
        default=30.0,
        help="传给 Phase3 的 --throat-neighbor-nm（默认 30 nm）。",
    )
    parser.add_argument(
        "--phase3-rho-sampling-scope",
        choices=["run", "cell"],
        default="run",
        help=(
            "传给 Phase3 的 --rho-sampling-scope。run=每个 synthetic run/厚度 bin 抽一次 rho，"
            "用于保留 Phase2 子样本级 count-density 方差；cell=每个 cell 独立抽样，汇总后更窄。默认 run。"
        ),
    )
    parser.add_argument(
        "--phase3-pore-q4-bias-strength",
        type=float,
        default=0.55,
        help="与 Phase3 --pore-radius-q4-bias-strength 一致，默认 0.55。",
    )
    parser.add_argument(
        "--phase3-rho-mean-correction-rel-tol",
        type=float,
        default=None,
        help="兼容旧参数：传给 Phase3 的 --rho-mean-correction-rel-tol；默认不用共享参数，孔/喉分别设置。",
    )
    parser.add_argument(
        "--phase3-rho-mean-correction-scale-min",
        type=float,
        default=None,
        help="兼容旧参数：传给 Phase3 的 --rho-mean-correction-scale-min；默认不用共享参数，孔/喉分别设置。",
    )
    parser.add_argument(
        "--phase3-rho-mean-correction-scale-max",
        type=float,
        default=None,
        help="兼容旧参数：传给 Phase3 的 --rho-mean-correction-scale-max；默认不用共享参数，孔/喉分别设置。",
    )
    parser.add_argument(
        "--phase3-pore-rho-mean-correction-rel-tol",
        type=float,
        default=None,
        help="传给 Phase3 的 --pore-rho-mean-correction-rel-tol；孔数量密度弱回正触发阈值，默认 inf（关闭）。",
    )
    parser.add_argument(
        "--phase3-pore-rho-mean-correction-scale-min",
        type=float,
        default=None,
        help="传给 Phase3 的 --pore-rho-mean-correction-scale-min；孔数量密度弱回正最小 scale，默认 0.5。",
    )
    parser.add_argument(
        "--phase3-pore-rho-mean-correction-scale-max",
        type=float,
        default=None,
        help="传给 Phase3 的 --pore-rho-mean-correction-scale-max；孔数量密度弱回正最大 scale，默认 2.0。",
    )
    parser.add_argument(
        "--phase3-throat-rho-mean-correction-rel-tol",
        type=float,
        default=None,
        help="传给 Phase3 的 --throat-rho-mean-correction-rel-tol；喉数量密度弱回正触发阈值，默认 inf（关闭）。",
    )
    parser.add_argument(
        "--phase3-throat-rho-mean-correction-scale-min",
        type=float,
        default=None,
        help="传给 Phase3 的 --throat-rho-mean-correction-scale-min；喉数量密度弱回正最小 scale，默认 0.5。",
    )
    parser.add_argument(
        "--phase3-throat-rho-mean-correction-scale-max",
        type=float,
        default=None,
        help="传给 Phase3 的 --throat-rho-mean-correction-scale-max；喉数量密度弱回正最大 scale，默认 2.0。",
    )
    parser.add_argument(
        "--batch-repeat",
        type=int,
        default=1,
        help=(
            "完整管线重复次数：每次独立厚度抽样、Phase3/4、汇总与 physical_params 等。"
            "大于 1 时输出写入 results/run_result/repeat_000/、repeat_001/ … 彼此隔离。默认 1 与原先一致。"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出完整运行细节（恢复旧的 terminal 输出）。默认关闭，仅显示模拟总进度条。",
    )
    args = parser.parse_args()

    if run_sieving_calculation is None or calculate_sieving_for_synthetic_networks is None or compare_wt_vs_as is None:
        print("错误：无法导入 phase4_sieving_comparison，请确认脚本位于 gbm_full_model 目录下。")
        raise SystemExit(1)

    phase1_dir = Path(args.phase1_dir) if args.phase1_dir else _default_fitted_parameter_dir("phase1_thickness")
    phase2_dir = Path(args.phase2_dir) if args.phase2_dir else _default_fitted_parameter_dir("phase2_parameters")
    batch_repeat = max(1, int(args.batch_repeat))
    if args.verbose and batch_repeat > 1 and (args.phase3_dir or args.phase4_dir):
        print(
            "警告：--batch-repeat>1 且指定了 --phase3-dir 或 --phase4-dir，"
            "各次重复将写入同一自定义目录，可能互相覆盖。"
        )
    if getattr(args, "skip_existing", False) and not args.no_clear:
        if args.verbose:
            print("提示：已指定 --skip-existing，将自动启用 --no-clear（避免删除已有 run）。")
        args.no_clear = True
    if args.verbose and args.seed is None:
        print(
            "提示：未指定 --seed；seed_ledger 会记录 Phase3/重试 seed，"
            "但厚度抽样 seed 为 None，完整复现实验池建议正式运行时指定 --seed。"
        )
    base_run_result = Path(args.run_result_dir) if args.run_result_dir else _root / "results" / "run_result"
    total_progress_runs = max(0, int(args.n_runs)) * len(args.sample_types) * batch_repeat
    _progress_start(total_progress_runs, enabled=not bool(args.verbose))
    try:
        with _suppress_stdout_if(not bool(args.verbose)):
            for batch_rep_index in range(batch_repeat):
                if batch_repeat > 1:
                    run_result_dir = base_run_result / f"repeat_{batch_rep_index:03d}"
                    print(
                        f"\n========== 完整管线批次 {batch_rep_index + 1}/{batch_repeat} -> {run_result_dir} ==========\n"
                    )
                else:
                    run_result_dir = base_run_result
                if not args.no_clear and run_result_dir.exists():
                    shutil.rmtree(run_result_dir, ignore_errors=True)
                    print(f"已清除 {run_result_dir}")
                _run_uniform_pipeline_single_batch(
                    args, run_result_dir, phase1_dir, phase2_dir, batch_rep_index, batch_repeat
                )
            print("全部批次完成。")
    finally:
        _progress_finish()
    return


def _run_uniform_pipeline_single_batch(
    args,
    run_result_dir: Path,
    phase1_dir: Path,
    phase2_dir: Path,
    batch_rep_index: int,
    batch_repeat_total: int,
) -> None:
    phase3_dir = Path(args.phase3_dir) if args.phase3_dir else run_result_dir / "phase3_synthetic"
    phase3_networks_dir = phase3_dir / "networks"  # 1. 每次模拟的 xlsx
    phase4_dir = Path(args.phase4_dir) if args.phase4_dir else run_result_dir / "phase4_sieving"
    physical_params_dir = run_result_dir / "physical_params"  # 2. 物理量记录 + 按厚度 bin 汇总图（含厚度分布）
    thickness_sampling_dir = run_result_dir / "thickness_sampling"

    phase3_networks_dir.mkdir(parents=True, exist_ok=True)
    phase4_dir.mkdir(parents=True, exist_ok=True)
    physical_params_dir.mkdir(parents=True, exist_ok=True)
    thickness_sampling_dir.mkdir(parents=True, exist_ok=True)

    n = args.n_runs
    size = args.geometry_size
    failed_phase3_runs = []
    seed_ledger_records: list[dict] = []

    max_sieving_tries = 1 + max(0, args.sieving_abnormal_max_retries)
    if batch_repeat_total > 1:
        thickness_seed = (args.seed if args.seed is not None else 0) + batch_rep_index * 1_000_003
    else:
        thickness_seed = args.seed
    base_seed = (args.seed if args.seed is not None else 0) + (
        batch_rep_index * 1_000_003 if batch_repeat_total > 1 else 0
    )
    use_explicit_seed = (args.seed is not None) or (max_sieving_tries > 1) or (batch_repeat_total > 1)
    retry_disconnected_prob = float(np.clip(getattr(args, "retry_disconnected_prob", 0.8), 0.0, 1.0))
    for sample_type in args.sample_types:
        if str(sample_type).upper() == "WT":
            print(f"[WT] 厚度采样拟合类型：{args.wt_thickness_fit}")
        if str(sample_type).upper() == "AS":
            print(f"[AS] 厚度采样拟合类型：{args.as_thickness_fit}")
        thickness_values = ensure_sampled_thickness(
            phase1_dir,
            thickness_sampling_dir,
            n,
            sample_type,
            thickness_seed,
            wt_fit=args.wt_thickness_fit,
            as_fit=args.as_thickness_fit,
        )
        if str(sample_type).upper() == "AS":
            as_min_thickness_nm = 25.0
            rejection_seed = thickness_seed
            thickness_values, n_resampled = rejection_resample_thickness_below_min(
                thickness_values=thickness_values,
                min_thickness_nm=as_min_thickness_nm,
                phase1_dir=phase1_dir,
                thickness_sampling_dir=thickness_sampling_dir,
                sample_type=sample_type,
                seed=rejection_seed,
                wt_fit=args.wt_thickness_fit,
                as_fit=args.as_thickness_fit,
            )
            if n_resampled > 0:
                print(
                    f"[AS] 厚度下限拒绝采样：{n_resampled}/{len(thickness_values)} 个样本 "
                    f"由 < {as_min_thickness_nm:.1f} nm 重新采样至 >= {as_min_thickness_nm:.1f} nm"
                )
        elif str(sample_type).upper() == "WT":
            wt_min_thickness_nm = 50.0
            rejection_seed = None if thickness_seed is None else int(thickness_seed) + 1_000_000
            thickness_values, n_resampled = rejection_resample_thickness_below_min(
                thickness_values=thickness_values,
                min_thickness_nm=wt_min_thickness_nm,
                phase1_dir=phase1_dir,
                thickness_sampling_dir=thickness_sampling_dir,
                sample_type=sample_type,
                seed=rejection_seed,
                wt_fit=args.wt_thickness_fit,
                as_fit=args.as_thickness_fit,
            )
            if n_resampled > 0:
                print(
                    f"[WT] 厚度下限拒绝采样：{n_resampled}/{len(thickness_values)} 个样本 "
                    f"由 < {wt_min_thickness_nm:.1f} nm 重新采样至 >= {wt_min_thickness_nm:.1f} nm"
                )
        else:
            rejection_seed = None
            n_resampled = 0
        target_min_nm = getattr(args, "target_thickness_min_nm", None)
        target_max_nm = getattr(args, "target_thickness_max_nm", None)
        target_range_seed = None
        n_target_range_resampled = 0
        if target_min_nm is not None or target_max_nm is not None:
            target_range_seed = None if thickness_seed is None else int(thickness_seed) + 2_000_000
            thickness_values, n_target_range_resampled = rejection_resample_thickness_outside_range(
                thickness_values=thickness_values,
                min_thickness_nm=target_min_nm,
                max_thickness_nm=target_max_nm,
                phase1_dir=phase1_dir,
                thickness_sampling_dir=thickness_sampling_dir,
                sample_type=sample_type,
                seed=target_range_seed,
                wt_fit=args.wt_thickness_fit,
                as_fit=args.as_thickness_fit,
            )
            lo_txt = "-inf" if target_min_nm is None else f"{float(target_min_nm):.1f}"
            hi_txt = "+inf" if target_max_nm is None else f"{float(target_max_nm):.1f}"
            print(
                f"[{sample_type}] targeted thickness range [{lo_txt}, {hi_txt}] nm: "
                f"resampled {n_target_range_resampled}/{len(thickness_values)} values"
            )
        persist_effective_sampled_thickness_json(
            thickness_sampling_dir, n, sample_type, thickness_values
        )
        shared_rho_rel_tol = args.phase3_rho_mean_correction_rel_tol
        shared_rho_scale_min = args.phase3_rho_mean_correction_scale_min
        shared_rho_scale_max = args.phase3_rho_mean_correction_scale_max
        phase3_pore_rho_rel_tol = (
            args.phase3_pore_rho_mean_correction_rel_tol
            if args.phase3_pore_rho_mean_correction_rel_tol is not None
            else (shared_rho_rel_tol if shared_rho_rel_tol is not None else float("inf"))
        )
        phase3_pore_rho_scale_min = (
            args.phase3_pore_rho_mean_correction_scale_min
            if args.phase3_pore_rho_mean_correction_scale_min is not None
            else (shared_rho_scale_min if shared_rho_scale_min is not None else 0.5)
        )
        phase3_pore_rho_scale_max = (
            args.phase3_pore_rho_mean_correction_scale_max
            if args.phase3_pore_rho_mean_correction_scale_max is not None
            else (shared_rho_scale_max if shared_rho_scale_max is not None else 2.0)
        )
        phase3_throat_rho_rel_tol = (
            args.phase3_throat_rho_mean_correction_rel_tol
            if args.phase3_throat_rho_mean_correction_rel_tol is not None
            else (shared_rho_rel_tol if shared_rho_rel_tol is not None else float("inf"))
        )
        phase3_throat_rho_scale_min = (
            args.phase3_throat_rho_mean_correction_scale_min
            if args.phase3_throat_rho_mean_correction_scale_min is not None
            else (shared_rho_scale_min if shared_rho_scale_min is not None else 0.5)
        )
        phase3_throat_rho_scale_max = (
            args.phase3_throat_rho_mean_correction_scale_max
            if args.phase3_throat_rho_mean_correction_scale_max is not None
            else (shared_rho_scale_max if shared_rho_scale_max is not None else 2.0)
        )
        for i in range(n):
            T = float(thickness_values[i])
            sample_name = f"synthetic_{sample_type}_run{i}_plane"
            seed_record = {
                "batch_repeat_index": int(batch_rep_index),
                "batch_repeat_total": int(batch_repeat_total),
                "sample_type": str(sample_type),
                "run": int(i),
                "sample_name": sample_name,
                "thickness_nm": T,
                "geometry_size_nm": float(size),
                "phase3_pore_placement_cell_nm": float(args.phase3_pore_placement_cell_nm),
                "phase3_throat_neighbor_nm": float(args.phase3_throat_neighbor_nm),
                "phase3_rho_sampling_scope": str(args.phase3_rho_sampling_scope),
                "phase3_rho_mean_correction_rel_tol": (
                    None if args.phase3_rho_mean_correction_rel_tol is None else float(args.phase3_rho_mean_correction_rel_tol)
                ),
                "phase3_rho_mean_correction_scale_min": (
                    None if args.phase3_rho_mean_correction_scale_min is None else float(args.phase3_rho_mean_correction_scale_min)
                ),
                "phase3_rho_mean_correction_scale_max": (
                    None if args.phase3_rho_mean_correction_scale_max is None else float(args.phase3_rho_mean_correction_scale_max)
                ),
                "phase3_pore_rho_mean_correction_rel_tol": float(phase3_pore_rho_rel_tol),
                "phase3_pore_rho_mean_correction_scale_min": float(phase3_pore_rho_scale_min),
                "phase3_pore_rho_mean_correction_scale_max": float(phase3_pore_rho_scale_max),
                "phase3_throat_rho_mean_correction_rel_tol": float(phase3_throat_rho_rel_tol),
                "phase3_throat_rho_mean_correction_scale_min": float(phase3_throat_rho_scale_min),
                "phase3_throat_rho_mean_correction_scale_max": float(phase3_throat_rho_scale_max),
                "global_seed_arg": None if args.seed is None else int(args.seed),
                "base_seed": int(base_seed),
                "use_explicit_phase3_seed": bool(use_explicit_seed),
                "thickness_seed": None if thickness_seed is None else int(thickness_seed),
                "thickness_rejection_seed": None if rejection_seed is None else int(rejection_seed),
                "thickness_resampled_below_min_count_for_type": int(n_resampled),
                "target_thickness_min_nm": None if target_min_nm is None else float(target_min_nm),
                "target_thickness_max_nm": None if target_max_nm is None else float(target_max_nm),
                "thickness_target_range_seed": None if target_range_seed is None else int(target_range_seed),
                "thickness_target_range_resampled_count_for_type": int(n_target_range_resampled),
                "wt_thickness_fit": str(args.wt_thickness_fit),
                "as_thickness_fit": str(args.as_thickness_fit),
                "keep_phase4_diagnostics": bool(args.keep_phase4_diagnostics),
                "keep_phase4_radius_cache": bool(args.keep_phase4_radius_cache),
                "phase3_seed_attempts": "",
                "phase3_seed_final": None,
                "phase3_sieving_retry_final": None,
                "phase3_attempt_final": None,
                "phase3_status": "pending",
                "phase4_status": "pending",
                "phase4_result_path": "",
                "theta_cout_cin": None,
                "no_exit_nodes": None,
                "no_exit_rng_seed": None,
                "no_exit_retry_prob": float(retry_disconnected_prob),
                "skip_existing": bool(args.skip_existing),
                "phase4_only_existing_phase3": False,
            }
            cmd_phase3 = [
                sys.executable,
                "-m",
                "gbm_sieving.simulation.network_generation.generator",
                "--sample-type", sample_type,
                "--geometry-type", "plane",
                "--geometry-width", str(size),
                "--geometry-height", str(size),
                "--geometry-thickness-mean", str(T),
                "--no-thickness-field",
                "--no-plots",
                "--phase1-dir", str(phase1_dir),
                "--phase2-dir", str(phase2_dir),
                "--output-dir", str(phase3_networks_dir),
                "--sample-name", sample_name,
                "--pore-placement-cell-nm", str(float(args.phase3_pore_placement_cell_nm)),
                "--throat-neighbor-nm", str(float(args.phase3_throat_neighbor_nm)),
                "--rho-sampling-scope", str(args.phase3_rho_sampling_scope),
                "--pore-rho-mean-correction-rel-tol", str(float(phase3_pore_rho_rel_tol)),
                "--pore-rho-mean-correction-scale-min", str(float(phase3_pore_rho_scale_min)),
                "--pore-rho-mean-correction-scale-max", str(float(phase3_pore_rho_scale_max)),
                "--throat-rho-mean-correction-rel-tol", str(float(phase3_throat_rho_rel_tol)),
                "--throat-rho-mean-correction-scale-min", str(float(phase3_throat_rho_scale_min)),
                "--throat-rho-mean-correction-scale-max", str(float(phase3_throat_rho_scale_max)),
            ]
            if args.phase3_rho_mean_correction_rel_tol is not None:
                cmd_phase3.extend(["--rho-mean-correction-rel-tol", str(float(args.phase3_rho_mean_correction_rel_tol))])
            if args.phase3_rho_mean_correction_scale_min is not None:
                cmd_phase3.extend(["--rho-mean-correction-scale-min", str(float(args.phase3_rho_mean_correction_scale_min))])
            if args.phase3_rho_mean_correction_scale_max is not None:
                cmd_phase3.extend(["--rho-mean-correction-scale-max", str(float(args.phase3_rho_mean_correction_scale_max))])
            if args.phase3_html:
                cmd_phase3.append("--html")
            if bool(getattr(args, "phase3_pore_q4_spatial_bias", True)):
                cmd_phase3.append("--pore-radius-q4-spatial-bias")
            else:
                cmd_phase3.append("--no-pore-radius-q4-spatial-bias")
            cmd_phase3.extend(
                ["--pore-radius-q4-bias-strength", str(float(args.phase3_pore_q4_bias_strength))]
            )

            pores_p = phase3_networks_dir / f"{sample_name}_pores.xlsx"
            throats_p = phase3_networks_dir / f"{sample_name}_throats.xlsx"
            phase4_res = _phase4_sieving_results_path(phase4_dir, sample_name)

            if args.skip_existing and pores_p.exists() and throats_p.exists() and phase4_res is not None:
                print(
                    f"[{sample_type}] 第 {i+1}/{n} 次: 厚度 = {T:.2f} nm，"
                    f"已存在 Phase3+Phase4，跳过 {sample_name}"
                )
                seed_record.update(
                    {
                        "phase3_status": "skipped_existing",
                        "phase4_status": "skipped_existing",
                        "phase4_result_path": str(phase4_res),
                    }
                )
                seed_ledger_records.append(seed_record)
                _progress_advance()
                continue

            phase4_only = (
                args.skip_existing
                and pores_p.exists()
                and throats_p.exists()
                and phase4_res is None
            )
            if phase4_only:
                seed_record["phase4_only_existing_phase3"] = True
                print(
                    f"[{sample_type}] 第 {i+1}/{n} 次: 厚度 = {T:.2f} nm，"
                    f"断点续跑：先仅补 Phase4（沿用已有 Phase3 xlsx）；若筛过异常则同厚度重跑 Phase3+Phase4 — {sample_name}"
                )
            else:
                print(f"[{sample_type}] 第 {i+1}/{n} 次: 厚度 = {T:.2f} nm, 生成 {sample_name} ...")
            max_attempts = max(1, args.phase3_retries)
            # no_exit_nodes 概率重生随机源（固定种子，保证可复现）
            _sample_type_code = sum(ord(ch) for ch in str(sample_type))
            _no_exit_rng_seed = base_seed + i * 1000003 + _sample_type_code * 1009
            seed_record["no_exit_rng_seed"] = int(_no_exit_rng_seed)
            _rng_no_exit = np.random.default_rng(_no_exit_rng_seed)
            phase3_seed_attempts: list[int | None] = []
            run_recorded = False

            for sv in range(max_sieving_tries):
                last_error = None
                phase3_ok = False
                # 首轮且断点「仅有 Phase3、无 Phase4」：先不跑 Phase3，直接对已有网络算筛过；异常则后续轮次重跑 Phase3
                skip_phase3 = bool(
                    phase4_only and sv == 0 and pores_p.exists() and throats_p.exists()
                )
                if skip_phase3:
                    print(f"  沿用已有 Phase3 xlsx → Phase4（sv={sv+1}/{max_sieving_tries}）")
                    phase3_ok = True
                    seed_record.update(
                        {
                            "phase3_status": "existing_phase3",
                            "phase3_sieving_retry_final": int(sv),
                            "phase3_attempt_final": None,
                        }
                    )
                else:
                    for attempt in range(1, max_attempts + 1):
                        cmd_phase3_attempt = list(cmd_phase3)
                        if use_explicit_seed:
                            effective_seed = _phase3_effective_seed(
                                use_explicit_seed=use_explicit_seed,
                                base_seed=base_seed,
                                run_index=i,
                                sieving_retry_index=sv,
                                phase3_attempt_index=attempt,
                            )
                            phase3_seed_attempts.append(effective_seed)
                            cmd_phase3_attempt += ["--random-seed", str(effective_seed)]
                        else:
                            effective_seed = None
                            phase3_seed_attempts.append(None)
                        show_child_output = attempt == 1 and sv == 0
                        try:
                            result = subprocess.run(
                                cmd_phase3_attempt,
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                                env=_utf8_child_env(),
                            )
                            if show_child_output and result.stdout:
                                print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
                            if show_child_output and result.stderr:
                                print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
                            result.check_returncode()
                            pores_file = phase3_networks_dir / f"{sample_name}_pores.xlsx"
                            ok_dir, detail_dir = _validate_permeation_direction_for_sample(
                                pores_file=pores_file,
                                thickness_nm=T,
                                eps_nm=args.direction_check_eps_nm,
                            )
                            if ok_dir:
                                phase3_ok = True
                                seed_record.update(
                                    {
                                        "phase3_status": "ok",
                                        "phase3_seed_final": effective_seed,
                                        "phase3_sieving_retry_final": int(sv),
                                        "phase3_attempt_final": int(attempt),
                                    }
                                )
                                break
                            last_error = subprocess.CalledProcessError(
                                returncode=2,
                                cmd=cmd_phase3_attempt,
                                output=result.stdout,
                                stderr=f"[direction-check-failed] {detail_dir}",
                            )
                        except subprocess.CalledProcessError as e:
                            last_error = e

                if not phase3_ok:
                    if sv < max_sieving_tries - 1:
                        continue
                    print(f"Phase3 最终失败（已尝试 {max_attempts} 次），跳过该 run: {sample_name}")
                    if last_error is not None:
                        stderr_tail = (last_error.stderr or "")[-4000:]
                        returncode = last_error.returncode
                    else:
                        stderr_tail = ""
                        returncode = None
                    failed_phase3_runs.append(
                        {
                            "sample_type": sample_type,
                            "run": i,
                            "sample_name": sample_name,
                            "thickness_nm": T,
                            "attempts": max_attempts,
                            "returncode": returncode,
                            "stderr_tail": stderr_tail,
                        }
                    )
                    seed_record.update(
                        {
                            "phase3_status": "failed",
                            "phase4_status": "not_run",
                            "phase3_seed_attempts": json.dumps(phase3_seed_attempts, ensure_ascii=False),
                        }
                    )
                    seed_ledger_records.append(seed_record)
                    run_recorded = True
                    break

                pores_p = phase3_networks_dir / f"{sample_name}_pores.xlsx"
                throats_p = phase3_networks_dir / f"{sample_name}_throats.xlsx"
                if not throats_p.exists():
                    if sv >= max_sieving_tries - 1:
                        print(f"  已达最大筛分重试，跳过该 run: {sample_name}")
                        failed_phase3_runs.append(
                            {
                                "sample_type": sample_type,
                                "run": i,
                                "sample_name": sample_name,
                                "thickness_nm": T,
                                "attempts": max_attempts,
                                "returncode": None,
                                "stderr_tail": "[missing-throats-after-phase3]",
                            }
                        )
                        seed_record.update(
                            {
                                "phase3_status": "missing_throats_after_phase3",
                                "phase4_status": "not_run",
                                "phase3_seed_attempts": json.dumps(phase3_seed_attempts, ensure_ascii=False),
                            }
                        )
                        seed_ledger_records.append(seed_record)
                        run_recorded = True
                        break
                    continue

                res = run_sieving_calculation(
                    pores_p,
                    throats_p,
                    sample_name,
                    phase4_dir,
                    run_calculation=True,
                    phase4_html=bool(args.phase4_html),
                    quiet=not bool(args.verbose),
                )
                no_exit_flag = None
                if isinstance(res, dict) and bool(res.get("no_exit_nodes", False)):
                    no_exit_flag = True
                else:
                    no_exit_flag = _phase4_has_no_exit_nodes(phase4_dir, sample_name)
                seed_record["no_exit_nodes"] = None if no_exit_flag is None else bool(no_exit_flag)
                if no_exit_flag is True and bool(getattr(args, "retry_disconnected", False)):
                    if sv < (max_sieving_tries - 1):
                        _u = float(_rng_no_exit.random())
                        _do_retry = (_u < retry_disconnected_prob)
                        if _do_retry:
                            continue
                th = _sieving_cout_cin_from_parsed(res)
                phase4_res_now = _phase4_sieving_results_path(phase4_dir, sample_name)
                if (
                    phase4_res_now is not None
                    and not bool(getattr(args, "keep_phase4_diagnostics", False))
                    and not bool(getattr(args, "phase4_html", False))
                ):
                    _prune_phase4_sample_dir(
                        phase4_dir / sample_name,
                        keep_radius_cache=bool(getattr(args, "keep_phase4_radius_cache", False)),
                    )
                seed_record.update(
                    {
                        "phase4_status": "calculated",
                        "phase4_result_path": "" if phase4_res_now is None else str(phase4_res_now),
                        "theta_cout_cin": None if th is None else float(th),
                        "phase3_seed_attempts": json.dumps(phase3_seed_attempts, ensure_ascii=False),
                    }
                )
                if _sieving_cout_cin_retry_ok(th):
                    seed_record["phase4_status"] = "ok"
                    break
            else:
                print(
                    f"  警告：{sample_name} 修正筛过仍 < -1e-12 或缺失，已保留最后一次 Phase4 结果"
                )
                seed_record["phase4_status"] = "kept_abnormal"
            if not run_recorded:
                seed_record["phase3_seed_attempts"] = json.dumps(phase3_seed_attempts, ensure_ascii=False)
                seed_ledger_records.append(seed_record)
                run_recorded = True
            _progress_advance()

    if seed_ledger_records:
        _write_seed_ledger(run_result_dir, seed_ledger_records)

    if failed_phase3_runs:
        failed_df = pd.DataFrame(failed_phase3_runs)
        failed_csv = run_result_dir / "phase3_failed_runs.csv"
        failed_df.to_csv(failed_csv, index=False, encoding="utf-8-sig")
        print(f"Phase3 失败 run 清单已保存: {failed_csv} (共 {len(failed_phase3_runs)} 条)")

    # Phase4：各 run 已在循环内算过；此处复用已有结果做汇总与箱线图
    print("Phase4：基于已有单样本结果汇总（reuse_existing_results）...")
    results_phase4 = calculate_sieving_for_synthetic_networks(
        phase3_networks_dir,
        phase4_dir,
        sample_types=args.sample_types,
        reuse_existing_results=True,
        quiet=not bool(args.verbose),
    )
    compare_wt_vs_as(results_phase4, phase4_dir)

    def _get_param(df: pd.DataFrame, name_substr: str) -> float | None:
        if df is None or len(df) == 0 or "Parameter" not in df.columns or "Value" not in df.columns:
            return None
        row = df[df["Parameter"].astype(str).str.contains(name_substr, na=False, regex=False)]
        if len(row) == 0:
            return None
        v = row["Value"].iloc[0]
        return float(v) if pd.notna(v) else None

    def _get_q_total_m3s(df: pd.DataFrame) -> float | None:
        """Read Q_total in m^3/s from Phase4 result. Tries several Parameter substrings (Excel may store ^ as ³)."""
        if df is None or len(df) == 0 or "Parameter" not in df.columns or "Value" not in df.columns:
            return None
        # 首选：全溶剂网络口径（用于单样本筛过分母）
        v = _get_param(df, "Q_total_full_solvent")
        if v is not None:
            return v
        # sub_calculate_sieving_coefficient_new: Value is nL/s.
        v = _get_param(df, "Total Solvent Flow (Q, full solvent network")
        if v is not None:
            return v * 1e-12
        v = _get_param(df, "full solvent network, nL/s")
        if v is not None:
            return v * 1e-12
        # Prefer row that is clearly m^3/s: "Q_total (m" matches "Total Solvent Flow Q_total (m^3/s)" or "(m³/s)"
        for sub in ("Q_total (m", "m^3/s)", "m³/s)", "m3/s)", "m3s)"):
            v = _get_param(df, sub)
            if v is not None:
                return v
        # Fallback: nL/s row -> m^3/s = value * 1e-12
        v = _get_param(df, "nL/s)")
        if v is not None:
            return v * 1e-12
        return None

    def _get_diagnostic_bool(df: pd.DataFrame, name_substr: str) -> bool | None:
        if df is None or len(df) == 0 or "Parameter" not in df.columns or "Value" not in df.columns:
            return None
        row = df[df["Parameter"].astype(str).str.contains(name_substr, na=False)]
        if len(row) == 0:
            return None
        v = row["Value"].iloc[0]
        if pd.isna(v):
            return None
        return bool(v) if isinstance(v, (bool, np.bool_)) else bool(float(v))

    def _get_diagnostic_int(df: pd.DataFrame, name_substr: str) -> int | None:
        if df is None or len(df) == 0 or "Parameter" not in df.columns or "Value" not in df.columns:
            return None
        row = df[df["Parameter"].astype(str).str.contains(name_substr, na=False)]
        if len(row) == 0:
            return None
        v = row["Value"].iloc[0]
        if pd.isna(v):
            return None
        return int(float(v))

    # 筛过系数：每个 run 都统计，不连通=0 也计入；并读取 Q_alb_total*C0、C0、Q_total 用于整体流率比
    rows = []
    for st in args.sample_types:
        for i in range(n):
            name = f"synthetic_{st}_run{i}_plane"
            res_file = _phase4_sieving_results_path(phase4_dir, name)
            theta = 0.0
            theta_raw = None
            Q_alb_times_C0 = None
            C0_run = None
            Q_total_m3s = None
            theta_corrected = None
            diag_pressure_ok = None
            diag_conc_ok = None
            diag_theta_neg = None
            diag_theta_nan = None
            diag_conc_neg = None
            diag_n_reversed = None
            stable = True
            if res_file is not None:
                try:
                    df = pd.read_excel(res_file)
                    if len(df) > 0:
                        if parse_sieving_results_dataframe is not None:
                            parsed_run = parse_sieving_results_dataframe(df)
                            if parsed_run:
                                sc = parsed_run.get("sieving_coefficient")
                                if sc is None:
                                    sc = parsed_run.get("Theta_corrected_concentration")
                                if sc is None:
                                    sc = parsed_run.get("Theta_global")
                                if sc is None and _first_finite_sieving_cout_cin_from_df is not None:
                                    sc = _first_finite_sieving_cout_cin_from_df(df)
                                theta_raw = parsed_run.get("sieving_coefficient_raw", sc)
                                if sc is not None:
                                    theta = float(sc)
                                Q_alb_times_C0 = parsed_run.get("Q_alb_total_times_C0")
                                if Q_alb_times_C0 is None:
                                    Q_alb_times_C0 = parsed_run.get("J_s_total_times_C0")
                                C0_run = parsed_run.get("C0")
                                Q_total_m3s = parsed_run.get("Q_total_m3s")
                                theta_corrected = parsed_run.get("Theta_corrected_concentration")
                        else:
                            if "Theta_global" in df.columns:
                                v = df["Theta_global"].iloc[0]
                                theta = float(v) if pd.notna(v) else 0.0
                            elif "Parameter" in df.columns and "Value" in df.columns:
                                if _first_finite_sieving_cout_cin_from_df is not None:
                                    fv = _first_finite_sieving_cout_cin_from_df(df)
                                    if fv is not None:
                                        theta = float(fv)
                                if theta == 0.0:
                                    mp = df["Parameter"].astype(str)
                                    row_sc = df[
                                        mp.str.contains("sieving_coefficient", na=False, regex=False)
                                        & mp.str.contains("C_out", na=False, regex=False)
                                    ]
                                    if len(row_sc) > 0:
                                        v = row_sc["Value"].iloc[0]
                                        theta_raw = float(v) if pd.notna(v) else None
                                        theta = float(v) if pd.notna(v) else 0.0
                                if theta == 0.0:
                                    row_theta = df[df["Parameter"].astype(str).str.contains("Theta_global", na=False, regex=False)]
                                    if len(row_theta) > 0:
                                        v = row_theta["Value"].iloc[0]
                                        theta_raw = float(v) if pd.notna(v) else None
                                        theta = float(v) if pd.notna(v) else 0.0
                        if C0_run is None and "Parameter" in df.columns and "Value" in df.columns:
                            C0_run = _get_param(df, "Plasma Concentration (C0)")
                        if C0_run is None and "Parameter" in df.columns and "Value" in df.columns:
                            C0_run = _get_param(df, "Entrance Solute Concentration")
                        if C0_run is None and "Parameter" in df.columns and "Value" in df.columns:
                            C0_run = _get_param(df, "Plasma Concentration")
                        if "Parameter" in df.columns and "Value" in df.columns:
                            if Q_alb_times_C0 is None:
                                Q_alb_times_C0 = _get_param(df, "Total Albumin Flow Rate (Q_alb_total, equiv_C)")
                            if Q_alb_times_C0 is None:
                                Q_alb_times_C0 = _get_param(df, "Total Solute Flux (J_s_total, new_J)")
                            if Q_alb_times_C0 is None:
                                Q_alb_times_C0 = _get_param(df, "Total Solute Flux")
                            if Q_total_m3s is None:
                                Q_total_m3s = _get_q_total_m3s(df)
                            if theta_corrected is None:
                                theta_corrected = _get_param(df, "Estimated Urine Sieving by Concentration")
                            if theta_corrected is None:
                                c_urine_bulk = _get_param(df, "Estimated Urine Bulk Concentration (Flow-weighted)")
                                if c_urine_bulk is not None and C0_run is not None and C0_run > 0:
                                    theta_corrected = c_urine_bulk / C0_run
                            if theta_corrected is None and Q_total_m3s is not None and Q_total_m3s > 0 and abs(theta) <= 1e-20:
                                theta_corrected = 0.0
                            diag_pressure_ok = _get_diagnostic_bool(df, "Pressure solve OK")
                            diag_conc_ok = _get_diagnostic_bool(df, "Concentration solve OK")
                            diag_theta_neg = _get_diagnostic_bool(df, "Theta negative")
                            diag_theta_nan = _get_diagnostic_bool(df, "Theta NaN")
                            diag_conc_neg = _get_diagnostic_bool(df, "Concentration has negative")
                            diag_n_reversed = _get_diagnostic_int(df, "C_exit>C_upstream")
                            stable_row = _get_diagnostic_bool(df, "Diagnostic: Stable")
                            if stable_row is not None:
                                stable = stable_row
                            else:
                                stable = (diag_pressure_ok is not False) and (diag_conc_ok is not False) and (diag_conc_neg is not True)
                except Exception:
                    pass
            rows.append({
                "sample_type": st, "run": i, "sample_name": name, "Theta_global": theta,
                "Theta_raw": theta_raw if theta_raw is not None else theta,
                "Q_alb_total_times_C0": Q_alb_times_C0,
                "J_s_total_times_C0": Q_alb_times_C0,  # compatibility alias
                "C0": C0_run, "Q_total_m3s": Q_total_m3s,
                "Theta_corrected_concentration": theta_corrected,
                "stable": stable,
                "diag_pressure_ok": diag_pressure_ok, "diag_conc_ok": diag_conc_ok,
                "diag_theta_negative": diag_theta_neg, "diag_theta_nan": diag_theta_nan,
                "diag_conc_negative": diag_conc_neg, "diag_n_reversed": diag_n_reversed,
            })
    if rows:
        df_all = pd.DataFrame(rows)
        # 默认口径：与 sub 一致，主指标为 sieving_coefficient = C_out/C_in（解析后写入 Theta_global / Theta_corrected 列）
        raw_from_legacy = df_all["Theta_corrected_concentration"].where(
            df_all["Theta_corrected_concentration"].notna(), df_all["Theta_global"]
        )
        df_all["Theta_default_raw"] = df_all["Theta_raw"].where(df_all["Theta_raw"].notna(), raw_from_legacy)
        theta_filtered = []
        theta_invalid_reason = []
        for v in pd.to_numeric(df_all["Theta_default_raw"], errors="coerce"):
            fixed, reason = _sanitize_sieving_value(v)
            theta_filtered.append(fixed)
            theta_invalid_reason.append(reason)
        df_all["Theta_default"] = theta_filtered
        df_all["Theta_invalid_reason"] = theta_invalid_reason
        df_all["Q_default_m3s"] = pd.to_numeric(df_all["Q_total_m3s"], errors="coerce").fillna(0.0)
        # 统计摘要（含不连通=0；无解/越界值保留为 NaN，不参与均值）
        stats = {}
        for st in args.sample_types:
            sub_st = df_all[df_all["sample_type"] == st]
            vals = pd.to_numeric(sub_st["Theta_default"], errors="coerce").to_numpy(dtype=float)
            raw_vals = pd.to_numeric(sub_st["Theta_default_raw"], errors="coerce").to_numpy(dtype=float)
            vals_valid = vals[np.isfinite(vals)]
            n_zero = int(np.sum(np.isfinite(vals) & np.isclose(vals, 0.0, atol=1e-20)))
            n_unstable = int(np.sum(sub_st["stable"].apply(lambda x: x is False)))
            vals_stable = sub_st[sub_st["stable"] == True]["Theta_global"].values
            stats[st] = {
                "mean": float(np.mean(vals_valid)) if len(vals_valid) else None,
                "std": float(np.std(vals_valid)) if len(vals_valid) else None,
                "median": float(np.median(vals_valid)) if len(vals_valid) else None,
                "min": float(np.min(vals_valid)) if len(vals_valid) else None,
                "max": float(np.max(vals_valid)) if len(vals_valid) else None,
                "n": len(vals),
                "n_valid_sieving": int(len(vals_valid)),
                "n_connected": int(np.sum(np.isfinite(vals) & (vals > 0))),
                "n_disconnected": n_zero,
                "n_unstable": n_unstable,
                "n_invalid_gt1": int(np.sum(np.isfinite(raw_vals) & (raw_vals > 1.0 + SIEVING_VALID_EPS))),
                "n_invalid_lt0": int(np.sum(np.isfinite(raw_vals) & (raw_vals < -SIEVING_VALID_EPS))),
                "n_missing_or_nonfinite": int(np.sum(~np.isfinite(raw_vals))),
            }
        if seed_ledger_records:
            seed_df = pd.DataFrame(seed_ledger_records)
            merge_cols = [
                "sample_type",
                "run",
                "sample_name",
                "thickness_nm",
                "global_seed_arg",
                "base_seed",
                "thickness_seed",
                "thickness_rejection_seed",
                "phase3_seed_final",
                "phase3_sieving_retry_final",
                "phase3_attempt_final",
                "phase3_status",
                "phase4_status",
                "no_exit_rng_seed",
                "no_exit_retry_prob",
            ]
            merge_cols = [c for c in merge_cols if c in seed_df.columns]
            df_all = df_all.merge(
                seed_df[merge_cols],
                on=["sample_type", "run", "sample_name"],
                how="left",
                suffixes=("", "_seedlog"),
            )
        df_all["Stable"] = df_all["stable"].map(lambda x: "Yes" if x else "No")
        per_run_path = phase4_dir / "sieving_per_run.xlsx"
        df_all.to_excel(per_run_path, index=False)
        print(f"  每 run 结果（含稳定性）已保存: {per_run_path}")
        # 整体筛过系数（按 WT/AS 分别汇总）：
        # Theta_default 即 C_out/C_in（与 sub 主结果一致）；按 Q 加权平均 sum(θ×Q)/sum(Q)。
        # 另给「通量构造」对照：Σ(θ×Q×C0)/(C0×ΣQ)，与 ΣJ/(C0×ΣQ) 在 θ 由浓度场定义时不保证相同。
        # 仅纳入 0<=Theta_default<=1 且 Q>0 的 run。
        # 溶质通量累加用与 Theta_default 自洽的量：J ≡ theta×Q×C0（conc·m^3/s）。
        aggregate_flux = {}
        for st in args.sample_types:
            sub = df_all[df_all["sample_type"] == st]
            sum_Q_alb_times_C0 = 0.0
            sum_Q = 0.0
            c0_used = 1.0
            sum_q_for_corr = 0.0
            sum_theta_corr_q = 0.0
            n_excluded_qalb_nonfinite = 0
            n_excluded_q_nonfinite = 0
            n_excluded_theta_nonfinite = 0
            n_excluded_theta_outside_01 = 0
            n_runs_in_aggregate = 0
            for _, r in sub.iterrows():
                js_c0 = r.get("Q_alb_total_times_C0")
                if js_c0 is None:
                    js_c0 = r.get("J_s_total_times_C0")
                q = r.get("Q_total_m3s")
                c0 = r.get("C0")
                theta_default = r.get("Theta_default")
                theta_default_raw = r.get("Theta_default_raw")
                if c0 is not None and c0 > 0:
                    c0_used = float(c0)
                js_val = None
                if js_c0 is not None:
                    try:
                        js_val = float(js_c0)
                    except (TypeError, ValueError):
                        js_val = None
                q_val = None
                if q is not None:
                    try:
                        q_val = float(q)
                    except (TypeError, ValueError):
                        q_val = None
                if q is not None and q_val is not None and not np.isfinite(q_val):
                    n_excluded_q_nonfinite += 1

                try:
                    theta_val = float(theta_default)
                except (TypeError, ValueError):
                    theta_val = float("nan")
                try:
                    theta_raw_val = float(theta_default_raw)
                except (TypeError, ValueError):
                    theta_raw_val = float("nan")
                theta_in_01 = np.isfinite(theta_val) and 0.0 <= theta_val <= 1.0
                theta_raw_outside_01 = (
                    np.isfinite(theta_raw_val)
                    and (theta_raw_val < -SIEVING_VALID_EPS or theta_raw_val > 1.0 + SIEVING_VALID_EPS)
                )
                q_ok = q_val is not None and np.isfinite(q_val) and q_val > 0

                c_row_f = float(c0_used) if c0_used > 0 else 1.0
                if c0 is not None:
                    try:
                        cx = float(c0)
                        if cx > 0:
                            c_row_f = cx
                    except (TypeError, ValueError):
                        pass

                if theta_in_01 and q_ok:
                    n_runs_in_aggregate += 1
                    sum_Q_alb_times_C0 += theta_val * q_val * c_row_f
                elif js_c0 is not None and (js_val is None or not np.isfinite(js_val)):
                    n_excluded_qalb_nonfinite += 1

                if q_ok:
                    if theta_in_01:
                        sum_Q += q_val

                q_default = r.get("Q_default_m3s")
                qd_val = None
                if q_default is not None:
                    try:
                        qd_val = float(q_default)
                    except (TypeError, ValueError):
                        qd_val = None
                if qd_val is not None and np.isfinite(qd_val) and qd_val > 0:
                    if np.isfinite(theta_val):
                        if 0.0 <= theta_val <= 1.0:
                            sum_q_for_corr += qd_val
                            sum_theta_corr_q += theta_val * qd_val
                        else:
                            n_excluded_theta_outside_01 += 1
                    elif theta_raw_outside_01:
                        n_excluded_theta_outside_01 += 1
                    else:
                        n_excluded_theta_nonfinite += 1
            if sum_Q > 0 and c0_used > 0:
                theta_agg_flux = (sum_Q_alb_times_C0 / c0_used) / sum_Q
            else:
                theta_agg_flux = np.nan
            if sum_q_for_corr > 0:
                theta_agg_corrected = sum_theta_corr_q / sum_q_for_corr
            else:
                theta_agg_corrected = np.nan
            aggregate_flux[st] = {
                "sum_albumin_flow_rate": sum_Q_alb_times_C0,
                "sum_solute_flux": sum_Q_alb_times_C0,  # compatibility alias
                "sum_solvent_flux_m3s": sum_Q,
                "C0_used": c0_used,
                "Theta_aggregate": float(theta_agg_corrected) if not np.isnan(theta_agg_corrected) else None,
                "Theta_aggregate_corrected_concentration_qweighted": float(theta_agg_corrected) if not np.isnan(theta_agg_corrected) else None,
                "Theta_aggregate_by_flux": float(theta_agg_flux) if not np.isnan(theta_agg_flux) else None,
                "n_runs_in_aggregate": n_runs_in_aggregate,
                "n_excluded_qalb_nonfinite": n_excluded_qalb_nonfinite,
                "n_excluded_js_nonfinite": n_excluded_qalb_nonfinite,  # compatibility alias
                "n_excluded_q_nonfinite": n_excluded_q_nonfinite,
                "n_excluded_theta_nonfinite": n_excluded_theta_nonfinite,
                "n_excluded_theta_outside_01": n_excluded_theta_outside_01,
            }
        for st in args.sample_types:
            agg = aggregate_flux.get(st, {})
            print(
                f"  [{st}] 白蛋白总流率之和 Σ(θ×Q×C0) (0<=θ<=1,Q>0) = {agg.get('sum_albumin_flow_rate', 0):.6e}, "
                f"溶剂流量之和 = {agg.get('sum_solvent_flux_m3s', 0):.6e} m^3/s, "
                f"整体筛过系数(C_out/C_in, Q加权) = {agg.get('Theta_aggregate_corrected_concentration_qweighted')}, "
                f"整体筛过系数(θ×Q×C0 总流率构造) = {agg.get('Theta_aggregate_by_flux')}, "
                f"纳入汇总的 run 数 = {agg.get('n_runs_in_aggregate', 0)}, "
                f"排除 js非有限/Q非有限/θ非有限/θ<-1e-12 = "
                f"{agg.get('n_excluded_js_nonfinite', 0)}/"
                f"{agg.get('n_excluded_q_nonfinite', 0)}/"
                f"{agg.get('n_excluded_theta_nonfinite', 0)}/"
                f"{agg.get('n_excluded_theta_outside_01', 0)}"
            )

        # 求解/稳定性诊断汇总（便于检查失败或不稳定 run）
        diagnostics_summary = {}
        for st in args.sample_types:
            sub = df_all[df_all["sample_type"] == st]
            n_pressure_fail = int(np.sum(sub["diag_pressure_ok"].apply(lambda x: x is False)))
            n_conc_fail = int(np.sum(sub["diag_conc_ok"].apply(lambda x: x is False)))
            n_theta_neg = int(np.sum(sub["diag_theta_negative"].apply(lambda x: x is True)))
            n_theta_nan = int(np.sum(sub["diag_theta_nan"].apply(lambda x: x is True)))
            n_conc_neg = int(np.sum(sub["diag_conc_negative"].apply(lambda x: x is True)))
            n_has_reversed = int(np.sum(sub["diag_n_reversed"].apply(lambda x: x is not None and x > 0)))
            diagnostics_summary[st] = {
                "n_pressure_solve_fail": n_pressure_fail,
                "n_concentration_solve_fail": n_conc_fail,
                "n_theta_negative": n_theta_neg,
                "n_theta_nan_inf": n_theta_nan,
                "n_concentration_has_negative": n_conc_neg,
                "n_runs_with_exit_reversed": n_has_reversed,
            }
        stats["diagnostics"] = diagnostics_summary
        stats["aggregate_policy"] = (
            "顶层 AS/WT 的 mean/std/median/min/max/n 仅使用物理范围内的 Theta_default；"
            "Theta_default 原始值 <0 或 >1 的 run 记为无效 NaN，不作为不连通样本。"
            "整体汇总 aggregate_by_flux：仅纳入 0<=Theta_default<=1 且 Q_total_m3s>0 的 run；"
            "溶质通量 ΣJ 用每 run 的 θ×Q×C0（与 Theta_default 自洽）。"
            "n_excluded_*：在「未同时满足纳入条件」时的诊断计数——"
            "theta_outside_01（保留旧字段名）在 Q_default_m3s>0 且原始 theta 有限但位于 [0,1] 外时累加；"
            "theta_nonfinite 在 Q_default>0 且 theta 非有限时累加；"
            "albumin_flow_nonfinite 在未能用 θ×Q×C0 纳入、且表中 Total Albumin Flow Rate 非有限时累加；"
            "q_nonfinite 在 Q_total 非有限时累加。"
            "n_runs_in_aggregate：纳入 Σ(θ×Q×C0) 与 ΣQ 的 run 数。"
        )
        n_total = len(df_all)
        any_issue = any(
            diagnostics_summary[st]["n_pressure_solve_fail"] > 0
            or diagnostics_summary[st]["n_concentration_solve_fail"] > 0
            or diagnostics_summary[st]["n_theta_negative"] > 0
            or diagnostics_summary[st]["n_theta_nan_inf"] > 0
            for st in args.sample_types
        )
        if any_issue:
            print("  求解/稳定性诊断:")
            for st in args.sample_types:
                d = diagnostics_summary[st]
                if d["n_pressure_solve_fail"] or d["n_concentration_solve_fail"] or d["n_theta_negative"] or d["n_theta_nan_inf"]:
                    print(f"    [{st}] 压力求解失败={d['n_pressure_solve_fail']}, 浓度求解失败={d['n_concentration_solve_fail']}, Theta<0={d['n_theta_negative']}, Theta NaN/Inf={d['n_theta_nan_inf']}, 负浓度={d['n_concentration_has_negative']}, 出口逆梯度 run 数={d['n_runs_with_exit_reversed']}")

        stats["aggregate_by_flux"] = aggregate_flux
        with open(phase4_dir / "sieving_statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"筛过系数统计: {stats}")

        # 箱型图：仅稳定 run，标题标出 不稳定 与 Theta=0 不连通 数量
        df_stable = df_all[df_all["stable"] == True]
        data = [
            pd.to_numeric(df_stable[df_stable["sample_type"] == t]["Theta_default"], errors="coerce")
            .dropna()
            .values
            for t in args.sample_types
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            bp = ax.boxplot(data, tick_labels=args.sample_types)
        except TypeError:
            bp = ax.boxplot(data, labels=args.sample_types)
        ax.set_ylabel("Sieving coefficient (default=corrected)")
        title = f"WT vs AS (n={n} runs/type, stable run)"
        n_unstable_total = sum(stats[st]["n_unstable"] for st in args.sample_types)
        n_d_total = sum(stats[st]["n_disconnected"] for st in args.sample_types)
        if n_unstable_total > 0 or n_d_total > 0:
            title += f"\n not stable {n_unstable_total}, Theta=0 no connection {n_d_total}"
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        fig.savefig(phase4_dir / "sieving_boxplot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存: {phase4_dir / 'sieving_boxplot.png'}")

        # 直方图：仅稳定 run，筛过系数分布（含 0）
        fig2, axes = plt.subplots(1, len(args.sample_types), figsize=(5 * len(args.sample_types), 4))
        if len(args.sample_types) == 1:
            axes = [axes]
        for i, st in enumerate(args.sample_types):
            vals = (
                pd.to_numeric(df_stable[df_stable["sample_type"] == st]["Theta_default"], errors="coerce")
                .dropna()
                .values
            )
            axes[i].hist(vals, bins=min(20, max(5, len(vals) // 2)) if len(vals) > 0 else 10, edgecolor="white", alpha=0.8)
            axes[i].axvline(0, color="red", linestyle="--", linewidth=1, label="Theta=0")
            axes[i].set_xlabel("Theta_default (corrected if available)")
            axes[i].set_ylabel("Count")
            axes[i].set_title(f"{st} stable run n={len(vals)}\nnot stable {stats[st]['n_unstable']}, Theta=0 no connection {stats[st]['n_disconnected']}")
            axes[i].legend(frameon=False)
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        fig2.savefig(phase4_dir / "sieving_histogram.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"已保存: {phase4_dir / 'sieving_histogram.png'}")

        # 厚度 vs 筛过系数散点图：仅稳定 run，且 Theta_default > 0（厚度表读完后只画一次，避免重复保存）
        thickness_map: dict = {}
        for st in args.sample_types:
            json_path = thickness_sampling_dir / f"sampled_thickness_n{n}_{st}.json"
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data_json = json.load(f)
                    thickness_map[st] = data_json.get("thickness_nm", [])
                except Exception:
                    thickness_map[st] = []
        if thickness_map:
            df_all["thickness_nm"] = np.nan
            for st, T_list in thickness_map.items():
                for i in range(len(T_list)):
                    mask = (df_all["sample_type"] == st) & (df_all["run"] == i)
                    if mask.any():
                        df_all.loc[mask, "thickness_nm"] = float(T_list[i])
            df_pos = df_all[(df_all["stable"] == True) & (df_all["Theta_default"] > 0) & (df_all["thickness_nm"].notna())]
            if len(df_pos) > 0:
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                colors = {"WT": "tab:blue", "AS": "tab:orange"}
                for st in args.sample_types:
                    sub = df_pos[df_pos["sample_type"] == st]
                    if len(sub) == 0:
                        continue
                    x = sub["thickness_nm"].to_numpy(dtype=float)
                    y_log = np.log10(sub["Theta_default"].to_numpy(dtype=float))
                    ax3.scatter(
                        x,
                        y_log,
                        alpha=0.7,
                        s=20,
                        label=st,
                        color=colors.get(st, "gray"),
                        edgecolors="none",
                    )
                ax3.set_xlabel("Thickness (nm)")
                ax3.set_ylabel("log10(Theta_default)")
                ax3.set_title(
                    f"Thickness vs log10(Theta_default) (stable, Theta>0)\n"
                    f"not stable {n_unstable_total}, Theta=0 disconnected {n_d_total}"
                )
                ax3.legend(frameon=False)
                ax3.grid(True, alpha=0.3)
                fig3.savefig(phase4_dir / "thickness_vs_sieving_scatter.png", dpi=150, bbox_inches="tight")
                plt.close(fig3)
                print(f"已保存: {phase4_dir / 'thickness_vs_sieving_scatter.png'}")

    # thickness_sampling：n 次模拟厚度柱状图 + Phase1 AS/WT 拟合曲线，画在一起
    _plot_thickness_sampling_combined(thickness_sampling_dir, phase1_dir, args.sample_types, n)

    # physical_params：每 run 物理量记录 + 按厚度 bin 汇总图 + 厚度分布 + 散点图
    run_physical_params(
        physical_params_dir, phase3_networks_dir, thickness_sampling_dir, phase2_dir, phase1_dir,
        phase4_dir, args.sample_types, n, size
    )
    if args.run_quintile_mixing:
        if collect_synthetic_run_quintile_mixing_dataframe is None or analyze_and_plot_pore_quintile_throat_mixing is None:
            print("提示：无法导入 phase1_5_param_vs_thickness，已跳过 --run-quintile-mixing。")
        else:
            print("\n[孔五分位-喉] 合成 run 复验（与阶段 1.5 同口径）...")
            df_q = collect_synthetic_run_quintile_mixing_dataframe(
                phase3_networks_dir,
                thickness_sampling_dir,
                args.sample_types,
                n,
                phase1_dir=phase1_dir,
                n_perm=max(0, int(args.run_quintile_mixing_n_perm)),
                random_seed=(thickness_seed if batch_repeat_total > 1 else args.seed),
                min_throats_for_perm=int(args.run_quintile_mixing_min_throats),
            )
            q_out = run_result_dir / "pore_quintile_throat_mixing_synthetic"
            analyze_and_plot_pore_quintile_throat_mixing(df_q, q_out)
            print(f"  [孔五分位-喉] 已写入: {q_out}")
    print("完成。")


if __name__ == "__main__":
    main()


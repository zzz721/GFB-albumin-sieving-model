"""
阶段三：基底膜几何与合成孔喉网络

当前假设：喉无方向性（不考虑 Q/S），喉连接在候选内均匀随机选取。
所用参数：几何、密度(rho/frac)、孔/喉半径分布（均来自 Phase2）；方向性默认 0。

- 几何与厚度空间
- 孔心放置与数量密度
- 孔半径赋值与体积密度、无重叠
- 喉连接（无方向性时为等权随机；可选方向性加权）
- 喉半径赋值与体积密度
- 导出与筛过流程对接
"""

import argparse
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm as _tqdm_raw
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    _tqdm_raw = None


def _tqdm(iterable=None, total=None, desc=None, **kwargs):
    """Enable tqdm only on interactive TTY terminals."""
    if not _HAS_TQDM or _tqdm_raw is None:
        return iterable
    is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
    if not is_tty:
        return iterable
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", False)
    kwargs.setdefault("mininterval", 0.3)
    return _tqdm_raw(iterable=iterable, total=total, desc=desc, **kwargs)

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# 与 Phase2 完全一致的 PDF 计算（用于拟合曲线叠加）
from gbm_sieving.analysis.parameter_fitting.fit_pore_throat_parameters import (
    _metric_fitted_pdf as _phase2_metric_fitted_pdf,
    _radius_fitted_pdf as _phase2_radius_fitted_pdf,
    _radius_gmm_pdf as _phase2_gmm_pdf,
)
from gbm_sieving.paths import OUTPUT_ROOT

# 与真实样本 overlap/交界面分析一致的几何容差（nm）
TOL_NM_OVERLAP = 0.2

# overlap 喉：R_cap 相对「本批 overlap 两端孔平均半径的中位数」越小，对 k 放大越多（仍受平均孔径上界约束）
OVERLAP_K_SMALL_CAP_BOOST = 0.8
OVERLAP_K_BIAS_MAX_FACTOR = 2.2


def _overlap_k_bias_for_small_Rcap(cap_j_nm: float, mean_pore_median_nm: float) -> float:
    """
    cap_j_nm：该喉的几何 R_cap。
    mean_pore_median_nm：本批 overlap 喉上 mean(R_pore1,R_pore2) 的中位数。
    R_cap 相对典型孔径越小，返回因子 >1 越大（上限 OVERLAP_K_BIAS_MAX_FACTOR）。
    """
    if not np.isfinite(cap_j_nm) or not np.isfinite(mean_pore_median_nm) or mean_pore_median_nm <= 0:
        return 1.0
    excess = max(0.0, (mean_pore_median_nm - float(cap_j_nm)) / (mean_pore_median_nm + 1e-9))
    b = 1.0 + OVERLAP_K_SMALL_CAP_BOOST * excess
    return float(min(b, OVERLAP_K_BIAS_MAX_FACTOR))


def _intersection_circle_radius_nm(d: float, R1: float, R2: float) -> float:
    """
    两球心距 d，半径 R1、R2（同单位，如 nm）。返回交线圆半径；无交线圆时 nan。
    几何条件：|R1-R2| < d < R1+R2，并留出 TOL_NM_OVERLAP 的容差。
    """
    if not (np.isfinite(d) and np.isfinite(R1) and np.isfinite(R2)):
        return float("nan")
    if R1 <= 0 or R2 <= 0 or d <= 0:
        return float("nan")
    # 外离或外切：视为无交线圆
    if d >= R1 + R2 - TOL_NM_OVERLAP:
        return float("nan")
    # 一球含于另一球：视为无交线圆
    if d <= abs(R1 - R2) + TOL_NM_OVERLAP:
        return float("nan")
    a1 = (d * d + R1 * R1 - R2 * R2) / (2.0 * d)
    sq = R1 * R1 - a1 * a1
    if sq <= 0:
        return float("nan")
    return float(np.sqrt(sq))


def _gmm_pdf(x: np.ndarray, gmm_dict: dict) -> np.ndarray:
    """Phase1 GMM 的 PDF。gmm_dict 含 type='gmm', weights, means, covariances, n_components。"""
    x = np.asarray(x, dtype=float)
    if gmm_dict.get("type") != "gmm":
        return np.zeros_like(x)
    w = np.array(gmm_dict["weights"])
    mu = np.array(gmm_dict["means"])
    cov = np.array(gmm_dict["covariances"])
    sigma = np.sqrt(cov)
    pdf = np.zeros_like(x)
    for k in range(len(w)):
        pdf += w[k] * stats.norm.pdf(x, loc=mu[k], scale=sigma[k])
    return pdf


def _gmm_sample(n: int, gmm_dict: dict, rng: np.random.Generator = None) -> np.ndarray:
    """从 Phase1 GMM 采样，用于构造逆 CDF。"""
    if gmm_dict.get("type") != "gmm":
        return np.full(n, np.nan)
    w = np.array(gmm_dict["weights"])
    mu = np.array(gmm_dict["means"])
    cov = np.array(gmm_dict["covariances"])
    sigma = np.sqrt(cov)
    if rng is None:
        rng = np.random.default_rng()
    out = np.empty(n)
    for i in range(n):
        k = rng.choice(len(w), p=w)
        out[i] = rng.normal(mu[k], sigma[k])
    return out


def _sample_rho_from_fit(fit: dict, n: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    从 Phase2 拟合分布采样。支持 norm, lognorm, gamma, beta, gmm。
    fit 来自 density_frac_by_thickness_bins_analysis.json 的 bins[].fit。
    返回 n 个正数（孔数量密度，1/nm^3）。方案 A：norm 采样后 clip 到 >= 1e-10。
    """
    if rng is None:
        rng = np.random.default_rng()
    if not fit or "fit" not in fit:
        return None
    f = fit.get("fit") or fit
    dist = f.get("distribution") or {}
    name = dist.get("name", "")
    params = dist.get("params") or []
    params = [float(p) for p in params]

    try:
        if name == "norm" and len(params) >= 2:
            scale = max(params[1], 1e-10)
            samples = stats.norm.rvs(loc=params[0], scale=scale, size=n, random_state=rng)
        elif name == "lognorm" and len(params) >= 3:
            # scipy lognorm: s, loc, scale
            samples = stats.lognorm.rvs(params[0], loc=params[1], scale=params[2], size=n, random_state=rng)
        elif name == "gamma" and len(params) >= 3:
            samples = stats.gamma.rvs(params[0], loc=params[1], scale=params[2], size=n, random_state=rng)
        elif name == "beta" and len(params) >= 4:
            # beta fit uses floc=0, fscale=1 -> (a, b, 0, 1)
            samples = stats.beta.rvs(params[0], params[1], loc=params[2] if len(params) > 2 else 0,
                                     scale=params[3] if len(params) > 3 else 1, size=n, random_state=rng)
        elif f.get("fit_type") == "gmm":
            w = np.array(f.get("weights", [1.0]))
            mu = np.array(f.get("means", [f.get("mean", 1e-6)]))
            cov = np.array(f.get("covariances", [1e-10]))
            sigma = np.sqrt(cov)
            samples = np.empty(n)
            for i in range(n):
                k = rng.choice(len(w), p=w)
                samples[i] = rng.normal(mu[k], sigma[k])
        else:
            # fallback: use mean if available
            mean = f.get("mean", 1e-6)
            std = f.get("std", 1e-7)
            samples = stats.norm.rvs(loc=mean, scale=max(std, 1e-10), size=n, random_state=rng)
    except Exception:
        mean = f.get("mean", 1e-6)
        std = f.get("std", 1e-7)
        samples = stats.norm.rvs(loc=mean, scale=max(std, 1e-10), size=n, random_state=rng)

    return np.maximum(np.asarray(samples, dtype=float), 1e-10)


def _stochastic_round_nonneg(x: float, rng: np.random.Generator) -> int:
    """
    对非负实数做随机舍入：
    n = floor(x) + Bernoulli(frac(x))，保证 E[n]=x，且 n>=0。
    """
    if not np.isfinite(x) or x <= 0:
        return 0
    x = float(x)
    lo = int(np.floor(x))
    frac = x - lo
    if frac <= 0:
        return lo
    return lo + (1 if float(rng.random()) < frac else 0)


def _sample_radius_from_fit(fit: dict, n: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    从 Phase2 孔半径拟合分布采样。支持 norm, lognorm, gamma, gmm。
    fit 来自 radius_by_thickness_bins_analysis.json 的 pore.bins[].fit。
    返回 n 个正半径 (nm)，最小 0.5 nm。
    """
    if rng is None:
        rng = np.random.default_rng()
    if not fit or "fit" not in fit:
        return None
    f = fit.get("fit") or fit
    dist = f.get("distribution") or {}
    name = dist.get("name", "")
    params = dist.get("params") or []
    params = [float(p) for p in params]
    try:
        if name == "norm" and len(params) >= 2:
            samples = stats.norm.rvs(loc=params[0], scale=params[1], size=n, random_state=rng)
        elif name == "lognorm" and len(params) >= 3:
            samples = stats.lognorm.rvs(params[0], loc=params[1], scale=params[2], size=n, random_state=rng)
        elif name == "gamma" and len(params) >= 3:
            samples = stats.gamma.rvs(params[0], loc=params[1], scale=params[2], size=n, random_state=rng)
        elif f.get("fit_type") == "gmm":
            w = np.array(f.get("weights", [1.0]))
            mu = np.array(f.get("means", [f.get("mean", 5.0)]))
            cov = np.array(f.get("covariances", [1.0]))
            sigma = np.sqrt(cov)
            samples = np.empty(n)
            for i in range(n):
                k = rng.choice(len(w), p=w)
                samples[i] = rng.normal(mu[k], sigma[k])
        else:
            mean = f.get("mean", 5.0)
            std = f.get("std", 1.0)
            samples = stats.norm.rvs(loc=mean, scale=max(std, 1e-6), size=n, random_state=rng)
    except Exception:
        mean = f.get("mean", 5.0)
        std = f.get("std", 1.0)
        samples = stats.norm.rvs(loc=mean, scale=max(std, 1e-6), size=n, random_state=rng)
    return np.maximum(np.asarray(samples, dtype=float), 0.5)


def _sample_ratio_from_fit(
    fit: dict,
    n: int,
    rng: np.random.Generator = None,
    k_min: float = 0.02,
    k_max: float = 20.0,
) -> np.ndarray | None:
    """
    从 Phase2 拟合采样无量纲比值 k（如 overlap 喉 R_throat / R_cap）。
    与 _sample_radius_from_fit 相同分布族，但不使用 0.5 nm 物理下限；采样后 clip 到 [k_min, k_max]。
    """
    if rng is None:
        rng = np.random.default_rng()
    if not fit or "fit" not in fit:
        return None
    f = fit.get("fit") or fit
    dist = f.get("distribution") or {}
    name = dist.get("name", "")
    params = dist.get("params") or []
    params = [float(p) for p in params]
    try:
        if name == "norm" and len(params) >= 2:
            samples = stats.norm.rvs(loc=params[0], scale=params[1], size=n, random_state=rng)
        elif name == "lognorm" and len(params) >= 3:
            samples = stats.lognorm.rvs(params[0], loc=params[1], scale=params[2], size=n, random_state=rng)
        elif name == "gamma" and len(params) >= 3:
            samples = stats.gamma.rvs(params[0], loc=params[1], scale=params[2], size=n, random_state=rng)
        elif f.get("fit_type") == "gmm":
            w = np.array(f.get("weights", [1.0]))
            mu = np.array(f.get("means", [f.get("mean", 1.2)]))
            cov = np.array(f.get("covariances", [1.0]))
            sigma = np.sqrt(cov)
            samples = np.empty(n)
            for i in range(n):
                k0 = rng.choice(len(w), p=w)
                samples[i] = rng.normal(mu[k0], sigma[k0])
        else:
            mean = f.get("mean", 1.2)
            std = f.get("std", 0.2)
            samples = stats.norm.rvs(loc=mean, scale=max(std, 1e-6), size=n, random_state=rng)
    except Exception:
        mean = f.get("mean", 1.2)
        std = f.get("std", 0.2)
        samples = stats.norm.rvs(loc=mean, scale=max(std, 1e-6), size=n, random_state=rng)
    return np.clip(np.asarray(samples, dtype=float), k_min, k_max)


def _sample_frac_metric_from_bin(bin_obj: dict, rng: np.random.Generator) -> float:
    """
    从 Phase2 子样本级「超限比例」bin 的 norm 拟合采样 f，并 clip 到 [0,1]。
    bin_obj 为 throat_gt_min_pore_frac JSON 中 bins[] 的一项。
    """
    fit = bin_obj.get("fit") or {}
    dist = fit.get("distribution") or {}
    name = dist.get("name", "")
    params = dist.get("params") or []
    try:
        if name == "norm" and len(params) >= 2:
            x = float(rng.normal(float(params[0]), float(params[1])))
        else:
            x = float(fit.get("mean", 0.0))
    except Exception:
        x = float(fit.get("mean", 0.0))
    return float(np.clip(x, 0.0, 1.0))


def _sample_degree_from_empirical(fit: dict, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    从 Phase2 孔度数经验 CDF 逆变换采样。
    fit 含 fit_type="empirical", degree_values, cdf。
    返回非负整数数组，长度为 n。
    """
    if rng is None:
        rng = np.random.default_rng()
    if not fit:
        return np.zeros(n, dtype=int)
    f = fit.get("fit") or fit
    if f.get("fit_type") != "empirical":
        return np.zeros(n, dtype=int)
    dv = np.array(f.get("degree_values", []), dtype=int)
    cdf = np.array(f.get("cdf", []), dtype=float)
    if len(dv) == 0 or len(cdf) == 0 or len(dv) != len(cdf):
        mean = float(f.get("mean", 2.0))
        return np.maximum(rng.poisson(max(mean, 0.0), size=n).astype(int), 0)
    u = rng.random(size=n)
    # u ~ U(0,1), 返回 degree_values[min{i : cdf[i] >= u}]
    idx = np.searchsorted(cdf, u, side="left")
    idx = np.clip(idx, 0, len(dv) - 1)
    return dv[idx]


def _sample_degree_from_fit(fit: dict, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    从 Phase2 孔度数分布采样。支持经验 CDF（empirical）与参数拟合（poisson, nbinom）。
    fit 来自 degree_by_thickness_bins_analysis.json 的 bins[].fit。
    返回非负整数数组，长度为 n。
    """
    if rng is None:
        rng = np.random.default_rng()
    if not fit:
        return np.zeros(n, dtype=int)
    f = fit.get("fit") or fit
    if f.get("fit_type") == "empirical":
        return _sample_degree_from_empirical(fit, n, rng)
    # 参数拟合（向后兼容旧 Phase2 输出）
    dist = f.get("distribution") or {}
    name = dist.get("name", "")
    params = dist.get("params") or []
    try:
        if name == "poisson" and len(params) >= 1:
            lam = max(float(params[0]), 0.0)
            samples = rng.poisson(lam, size=n)
        elif name == "nbinom" and len(params) >= 2:
            n_nb = max(float(params[0]), 1e-3)
            p_nb = float(params[1])
            p_nb = min(max(p_nb, 1e-6), 1.0 - 1e-6)
            samples = rng.negative_binomial(n_nb, p_nb, size=n)
        else:
            mean = float(f.get("mean", 2.0))
            mean = max(mean, 0.0)
            samples = rng.poisson(mean, size=n)
    except Exception:
        mean = float(f.get("mean", 2.0))
        mean = max(mean, 0.0)
        samples = rng.poisson(mean, size=n)
    samples = np.maximum(np.round(samples).astype(int), 0)
    return samples


def _fit_pdf(x: np.ndarray, fit: dict) -> np.ndarray:
    """从 Phase2 fit 计算 PDF，与 Phase2 的 _metric_fitted_pdf / GMM 完全一致（用于 rho/frac）。"""
    x = np.asarray(x, dtype=float)
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
            return _phase2_metric_fitted_pdf(x, name, params)
    except Exception:
        pass
    return np.zeros_like(x)


def _radius_fit_pdf(x: np.ndarray, fit: dict) -> np.ndarray:
    """从 Phase2 半径 fit 计算 PDF，与 Phase2 的 _radius_fitted_pdf / _radius_gmm_pdf 完全一致。"""
    x = np.asarray(x, dtype=float)
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


def _throat_length_pdf(length: float, fit: dict) -> float:
    """
    从 Phase2 喉长度按厚度区间的 fit 计算 PDF（结构与半径拟合相同，可复用 _radius_fit_pdf）。
    返回标量，若 fit 无效则返回 1e-10。
    """
    if not fit or not fit.get("fit"):
        return 1e-10
    x = np.array([float(length)], dtype=float)
    pdf = _radius_fit_pdf(x, fit)
    return float(np.maximum(pdf[0], 1e-10))


def _sample_joint_from_gmm_2d(
    gmm_dict: dict,
    n: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    从 Phase2 拟合的 2D GMM 中采样 (x, y) 点。gmm_dict 结构与 joint_gmm_* 一致。
    返回形状 (n, 2) 的数组。
    """
    if rng is None:
        rng = np.random.default_rng()
    if not gmm_dict:
        return np.full((0, 2), np.nan)
    weights = np.asarray(gmm_dict.get("weights", []), dtype=float)
    means = np.asarray(gmm_dict.get("means", []), dtype=float)
    covs = np.asarray(gmm_dict.get("covariances", []), dtype=float)
    K = weights.shape[0]
    if (
        K == 0
        or means.shape != (K, 2)
        or covs.shape != (K, 2, 2)
        or np.any(~np.isfinite(weights))
    ):
        return np.full((0, 2), np.nan)
    p = weights / np.maximum(weights.sum(), 1e-16)
    comp_idx = rng.choice(K, size=n, p=p)
    samples = np.empty((n, 2), dtype=float)
    for k in range(K):
        mask = comp_idx == k
        if not np.any(mask):
            continue
        try:
            samples[mask] = rng.multivariate_normal(
                mean=means[k],
                cov=covs[k],
                size=int(mask.sum()),
            )
        except Exception:
            samples[mask] = means[k]
    return samples


def _sample_conditional_from_joint_gmm_1d(
    x: np.ndarray,
    gmm_dict: dict,
    rng: np.random.Generator | None = None,
    min_std: float = 1e-3,
) -> np.ndarray:
    """
    从 Phase2 拟合的 2D GMM p(x, y) 中抽样条件分布 y|x（此处 y 为喉半径）。
    gmm_dict 结构与 Phase2 joint_gmm_* 一致，含 n_components, weights, means(K,2), covariances(K,2,2)。
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    if not gmm_dict:
        return np.full(n, np.nan)
    weights = np.asarray(gmm_dict.get("weights", []), dtype=float)
    means = np.asarray(gmm_dict.get("means", []), dtype=float)
    covs = np.asarray(gmm_dict.get("covariances", []), dtype=float)
    K = weights.shape[0]
    if (
        K == 0
        or means.shape != (K, 2)
        or covs.shape != (K, 2, 2)
        or np.any(~np.isfinite(weights))
    ):
        return np.full(n, np.nan)
    # 组件参数
    mu_x = means[:, 0]
    mu_y = means[:, 1]
    var_x = covs[:, 0, 0]
    cov_xy = covs[:, 0, 1]
    cov_yx = covs[:, 1, 0]
    var_y = covs[:, 1, 1]

    # 1) 计算给定 x 时各组件的后验权重 α_k(x) ∝ w_k * N(x | μ_xk, σ_xk)
    sigma_x = np.sqrt(np.maximum(var_x, 1e-12))
    # 形状 (n, K)
    px = stats.norm.pdf(x[:, None], loc=mu_x[None, :], scale=sigma_x[None, :])
    resp = weights[None, :] * px
    sum_resp = resp.sum(axis=1, keepdims=True)
    invalid = sum_resp[:, 0] <= 0
    sum_resp[sum_resp <= 0] = 1.0
    resp /= sum_resp

    # 2) 为每个样本抽取一个组件索引
    u = rng.random(n)
    cum_resp = np.cumsum(resp, axis=1)
    comp_idx = (cum_resp < u[:, None]).sum(axis=1)
    comp_idx = np.clip(comp_idx, 0, K - 1)

    # 3) 根据选中的组件计算条件分布参数 y|x：一维高斯
    sel_var_x = var_x[comp_idx]
    sel_cov_yx = cov_yx[comp_idx]
    sel_cov_xy = cov_xy[comp_idx]
    sel_var_y = var_y[comp_idx]
    sel_mu_x = mu_x[comp_idx]
    sel_mu_y = mu_y[comp_idx]

    eps = 1e-12
    gain = sel_cov_yx / (sel_var_x + eps)
    cond_mean = sel_mu_y + gain * (x - sel_mu_x)
    cond_var = sel_var_y - sel_cov_yx * sel_cov_xy / (sel_var_x + eps)
    cond_std = np.sqrt(np.maximum(cond_var, min_std**2))

    y = rng.normal(cond_mean, cond_std)
    return np.maximum(y, 0.5)


def _rho_pore_at_thickness(T: np.ndarray, density_params: dict, sample_type: str) -> np.ndarray:
    """Phase2 线性回归: rho_pore(T) = slope*T + intercept，截断为非负。"""
    rho_info = (density_params or {}).get("rho_pore", {}).get(sample_type, {})
    reg = rho_info.get("linear_regression", {})
    slope = reg.get("slope", 0.0)
    intercept = reg.get("intercept", rho_info.get("mean", 1e-6))
    rho = np.asarray(T, dtype=float) * slope + intercept
    return np.maximum(rho, 0.0)


def _gmm_probability_in_interval(a: float, b: float, gmm_dict: dict) -> float:
    """Phase1 GMM 在区间 [a, b] 上的概率质量。"""
    if not gmm_dict or gmm_dict.get("type") != "gmm":
        return 0.0
    w = np.array(gmm_dict["weights"])
    mu = np.array(gmm_dict["means"])
    sigma = np.sqrt(np.array(gmm_dict["covariances"]))
    p = 0.0
    for k in range(len(w)):
        p += w[k] * (stats.norm.cdf(b, loc=mu[k], scale=sigma[k]) - stats.norm.cdf(a, loc=mu[k], scale=sigma[k]))
    return float(np.clip(p, 0.0, 1.0))


def _generate_thickness_field_plane(
    width_nm: float,
    height_nm: float,
    gmm_dict: dict,
    grid_step_nm: float = 10000.0,
    correlation_length_nm: float = 5000.0,
    rng: np.random.Generator = None,
    sample_type: str | None = None,
) -> tuple:
    """
    在平面上生成连续厚度场 T(x,y)：高斯随机场 + GMM 逆 CDF 变换。
    返回 (T_grid, x_1d, y_1d, interp, volume_nm3)。
    T_grid 形状 (ny, nx)，对应 x_1d, y_1d；interp 为 RectBivariateSpline，可求 T(x,y)。
    """
    if rng is None:
        rng = np.random.default_rng()
    nx = max(2, int(np.ceil(width_nm / grid_step_nm)) + 1)
    ny = max(2, int(np.ceil(height_nm / grid_step_nm)) + 1)
    x_1d = np.linspace(0, width_nm, nx)
    y_1d = np.linspace(0, height_nm, ny)
    dx = x_1d[1] - x_1d[0] if nx > 1 else width_nm
    dy = y_1d[1] - y_1d[0] if ny > 1 else height_nm

    # 2D 高斯随机场 Z（均值0方差1，空间相关）
    sigma_pixels = correlation_length_nm / min(dx, dy)
    W = rng.standard_normal((ny, nx))
    Z = gaussian_filter(W, sigma=float(sigma_pixels), mode="reflect")
    Z = (Z - np.mean(Z)) / (np.std(Z) + 1e-12)

    # GMM 逆 CDF：用大量采样构造分位数
    n_quantile = 100000
    gmm_samples = _gmm_sample(n_quantile, gmm_dict, rng)
    # AS：对极小厚度做拒绝采样重抽样，再施加软下限 40 nm
    if sample_type == "AS":
        # 先把 < 30 nm 的样本重抽（最多重试若干次，避免死循环）
        mask_too_small = gmm_samples < 35.0
        n_retry = 0
        while np.any(mask_too_small) and n_retry < 5:
            gmm_samples[mask_too_small] = _gmm_sample(int(mask_too_small.sum()), gmm_dict, rng)
            mask_too_small = gmm_samples < 35.0
            n_retry += 1
        # 若仍有极小值，直接截断到 30 nm
        gmm_samples = np.maximum(gmm_samples, 35.0)
        # 软下限：整体再抬到至少 40 nm
        gmm_samples = np.maximum(gmm_samples, 35.0)
    else:
        gmm_samples = np.clip(gmm_samples, 1.0, None)  # 厚度至少 1 nm
    u_quantiles = np.linspace(1e-6, 1 - 1e-6, n_quantile)
    sorted_samples = np.sort(gmm_samples)

    # Z -> u = Phi(Z) -> T
    u = stats.norm.cdf(Z)
    u = np.clip(u, 1e-6, 1 - 1e-6)
    T_grid = np.interp(u.ravel(), u_quantiles, sorted_samples).reshape(Z.shape)
    T_grid = np.maximum(T_grid, 1.0)

    # 插值器：interp(x, y) = T(x,y)。RectBivariateSpline 要求 z[i,j]=T(x_1d[i],y_1d[j])，故 z = T_grid.T
    interp = RectBivariateSpline(x_1d, y_1d, T_grid.T, kx=1, ky=1)

    # 体积：梯形法则
    volume_nm3 = float(np.sum(T_grid)) * dx * dy
    return T_grid, x_1d, y_1d, interp, volume_nm3


class GBMSyntheticNetwork:
    """
    合成 GBM 孔喉网络生成器
    """
    
    def __init__(
        self,
        sample_type: str,  # "WT" or "AS"
        geometry_type: str = "plane",  # "plane" or "sphere"
        geometry_params: dict = None,
        thickness_distribution: dict = None,
        density_params: dict = None,
        radius_params: dict = None,
        directionality_params: dict = None,
        pore_count_fraction_by_thickness_bin: dict = None,
        rho_pore_dist_by_thickness_bin: dict = None,
        frac_pore_dist_by_thickness_bin: dict = None,
        pore_radius_dist_by_thickness_bin: dict = None,
        throat_radius_dist_by_thickness_bin: dict = None,
        throat_radius_dist_by_length_bin: dict = None,
        pore_throat_radius_correlation: dict = None,
        rho_throat_dist_by_thickness_bin: dict = None,
        frac_throat_dist_by_thickness_bin: dict = None,
        throat_length_dist_by_thickness_bin: dict = None,
        effective_throat_length_dist_by_thickness_bin: dict = None,
        overlap_throat_R_ratio_dist_by_thickness_bin: dict = None,
        throat_gt_min_pore_frac_dist_by_thickness_bin: dict = None,
        nonoverlap_cross_density_dist_by_thickness_bin: dict = None,
        degree_dist_by_thickness_bin: dict = None,
        random_seed: int = None,
        pore_radius_q4_spatial_bias: bool = True,
        pore_radius_q4_bias_strength: float = 0.55,
    ):
        """
        初始化合成网络生成器
        
        Parameters:
        -----------
        sample_type : str
            "WT" or "AS"
        geometry_type : str
            "plane" (平面展开) or "sphere" (球面)
        geometry_params : dict
            几何参数，如 {"width": 10000, "height": 10000, "thickness": 150} (平面)
            或 {"radius": 5000, "thickness_mean": 150} (球面)
        thickness_distribution : dict
            厚度分布参数（从阶段一获得）
        density_params : dict
            密度参数（从阶段二获得）
        radius_params : dict
            半径分布参数（从阶段二获得）
        directionality_params : dict
            方向性参数（从阶段二获得）
        random_seed : int
            随机种子
        """
        self.sample_type = sample_type
        self.geometry_type = geometry_type
        self.geometry_params = geometry_params or {}
        self.thickness_distribution = thickness_distribution or {}
        self.density_params = density_params or {}
        self.radius_params = radius_params or {}
        self.directionality_params = directionality_params or {}
        self.pore_count_fraction_by_thickness_bin = pore_count_fraction_by_thickness_bin or {}
        self.rho_pore_dist_by_thickness_bin = rho_pore_dist_by_thickness_bin or {}
        self.frac_pore_dist_by_thickness_bin = frac_pore_dist_by_thickness_bin or {}
        self.pore_radius_dist_by_thickness_bin = pore_radius_dist_by_thickness_bin or {}
        self.throat_radius_dist_by_thickness_bin = throat_radius_dist_by_thickness_bin or {}
        self.throat_radius_dist_by_length_bin = throat_radius_dist_by_length_bin or {}
        self.pore_throat_radius_correlation = pore_throat_radius_correlation or {}
        self.rho_throat_dist_by_thickness_bin = rho_throat_dist_by_thickness_bin or {}
        self.frac_throat_dist_by_thickness_bin = frac_throat_dist_by_thickness_bin or {}
        self.throat_length_dist_by_thickness_bin = throat_length_dist_by_thickness_bin or {}
        self.effective_throat_length_dist_by_thickness_bin = effective_throat_length_dist_by_thickness_bin or {}
        self.overlap_throat_R_ratio_dist_by_thickness_bin = overlap_throat_R_ratio_dist_by_thickness_bin or {}
        self.throat_gt_min_pore_frac_dist_by_thickness_bin = throat_gt_min_pore_frac_dist_by_thickness_bin or {}
        self.nonoverlap_cross_density_dist_by_thickness_bin = (
            nonoverlap_cross_density_dist_by_thickness_bin or {}
        )
        self.degree_dist_by_thickness_bin = degree_dist_by_thickness_bin or {}
        self._gt_min_pore_frac_report: dict | None = None
        self._throat_target_debug: dict | None = None
        self._pore_rho_sampling_debug: dict | None = None
        self._throat_rho_sampling_debug: dict | None = None
        self.pore_radius_q4_spatial_bias = bool(pore_radius_q4_spatial_bias)
        self.pore_radius_q4_bias_strength = float(np.clip(pore_radius_q4_bias_strength, 0.0, 1.0))
        self._random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        self.pore_coords = None
        self.pore_radii = None
        self.pore_ids = None
        self.throat_pore1 = None
        self.throat_pore2 = None
        self.throat_radii = None
    
    def generate_geometry(self):
        """
        3.1 生成几何与厚度空间
        平面时：若提供 Phase1 的 GMM 厚度分布，则生成连续厚度场 T(x,y)，体积 = ∫∫ T dx dy；
        否则使用均匀厚度。
        """
        if self.geometry_type == "plane":
            width = self.geometry_params.get("width", 5000)  # nm
            height = self.geometry_params.get("height", 5000)  # nm
            thickness_mean = self.geometry_params.get("thickness_mean", 150)  # nm
            grid_step = self.geometry_params.get("grid_step_nm", 50.0)
            correlation_length = self.geometry_params.get("correlation_length_nm", 50000.0)
            rng = np.random.default_rng(self._random_seed)

            gmm = self.thickness_distribution.get(self.sample_type) if self.thickness_distribution else None
            if gmm and gmm.get("type") == "gmm":
                T_grid, x_1d, y_1d, interp, volume_nm3 = _generate_thickness_field_plane(
                    width,
                    height,
                    gmm,
                    grid_step_nm=grid_step,
                    correlation_length_nm=correlation_length,
                    rng=rng,
                    sample_type=self.sample_type,
                )
                self.thickness_field_grid = T_grid
                self._thickness_x_1d = x_1d
                self._thickness_y_1d = y_1d
                self._thickness_interp = interp
                self.volume = volume_nm3
                self.thickness = float(np.max(T_grid))  # 用于 bbox 上界
                self.bbox = np.array([[0, 0, 0], [width, height, self.thickness]])
                self._plane_width = width
                self._plane_height = height
            else:
                self.thickness_field_grid = None
                self._thickness_interp = None
                self._plane_width = width
                self._plane_height = height
                self.thickness = thickness_mean
                self.volume = width * height * self.thickness
                self.bbox = np.array([[0, 0, 0], [width, height, self.thickness]])
            
        elif self.geometry_type == "sphere":
            # 球面：球壳
            radius = self.geometry_params.get("radius", 5000)  # nm
            thickness_mean = self.geometry_params.get("thickness_mean", 150)  # nm
            
            self.thickness = thickness_mean
            self.volume = 4 * np.pi * radius**2 * self.thickness  # 球壳体积近似
            
            self.bbox = None  # 球面不需要 bbox
            self.sphere_radius = radius
        
        print(f"几何生成完成: {self.geometry_type}, 体积 = {self.volume:.2e} nm^3")
    
    def place_pores(self):
        """
        3.2 孔心放置与数量密度
        有厚度场时（按您指定的流程）：
        1) 统计各个厚度区间的网格数量与体积；
        2) 根据 Phase2 孔在不同厚度区间的数量分数分布给各区间分配分数；
        3) 总孔数 N_total 由体积与 Phase2 密度反推，再按分数分到各区间；
        4) 区间内按每格体积占比反推每格孔数，再在格体积内随机撒点（非均匀网格）。
        """
        target_rho_pore = self.density_params.get("rho_pore", {}).get(self.sample_type, {}).get("mean", 1e-6)  # 1/nm^3

        if self.geometry_type == "plane":
            width = getattr(self, "_plane_width", self.geometry_params.get("width", 5000))
            height = getattr(self, "_plane_height", self.geometry_params.get("height", 5000))
            rho_dist_for_cells = self.rho_pore_dist_by_thickness_bin.get(self.sample_type, {})
            has_rho_cell_fit = bool(rho_dist_for_cells.get("bin_edges")) and bool(rho_dist_for_cells.get("bins"))

            if getattr(self, "_thickness_interp", None) is not None or has_rho_cell_fit:
                # 按厚度区间统计网格 → 按 Phase2 孔数量分数分布分配 → 体积反推数量 → 格内随机撒点
                # In --no-thickness-field mode we still use cells, but each
                # cell has the same sample-level thickness.
                thickness_interp = getattr(self, "_thickness_interp", None)
                uniform_cell_T = max(
                    1.0,
                    float(getattr(self, "thickness", self.geometry_params.get("thickness_mean", 150.0))),
                )
                cell_nm = max(float(self.geometry_params.get("pore_placement_cell_nm", 50.0)), 1e-6)
                pore_frac = self.pore_count_fraction_by_thickness_bin.get(self.sample_type)

                nx_cell = max(1, int(np.ceil(width / cell_nm)))
                ny_cell = max(1, int(np.ceil(height / cell_nm)))
                cells = []
                for j in range(ny_cell):
                    y_lo = j * cell_nm
                    y_hi = min((j + 1) * cell_nm, height)
                    dy = y_hi - y_lo
                    for i in range(nx_cell):
                        x_lo = i * cell_nm
                        x_hi = min((i + 1) * cell_nm, width)
                        dx = x_hi - x_lo
                        x_center = (x_lo + x_hi) / 2.0
                        y_center = (y_lo + y_hi) / 2.0
                        if thickness_interp is not None:
                            T_cell = float(thickness_interp(x_center, y_center, grid=False))
                        else:
                            T_cell = uniform_cell_T
                        T_cell = max(1.0, T_cell)
                        volume_cell = dx * dy * T_cell
                        cells.append({
                            "T": T_cell, "V": volume_cell,
                            "x_lo": x_lo, "x_hi": x_hi, "y_lo": y_lo, "y_hi": y_hi,
                        })
                if not cells:
                    self.pore_coords = np.zeros((0, 3))
                    print("孔放置: 无网格单元")
                    return

                T_arr = np.array([c["T"] for c in cells])
                T_min, T_max = float(np.min(T_arr)), float(np.max(T_arr))

                # 优先使用 Phase2 的孔在不同厚度区间的数量分数分布（bin_edges + fraction_per_bin）
                rho_bin_edges = np.array(rho_dist_for_cells.get("bin_edges", []), dtype=float)
                rho_bin_edges = np.array(rho_dist_for_cells.get("bin_edges", []), dtype=float)
                rho_bins_for_cells = rho_dist_for_cells.get("bins", [])
                if len(rho_bin_edges) >= 2 and len(rho_bins_for_cells) > 0:
                    bin_edges = rho_bin_edges
                    fraction_per_bin = None
                    n_bins = len(bin_edges) - 1
                elif pore_frac and "bin_edges" in pore_frac and "fraction_per_bin" in pore_frac:
                    bin_edges = np.array(pore_frac["bin_edges"], dtype=float)
                    fraction_per_bin = np.array(pore_frac["fraction_per_bin"], dtype=float)
                    n_bins = len(bin_edges) - 1
                    if n_bins <= 0 or len(fraction_per_bin) != n_bins:
                        bin_edges = np.linspace(max(1.0, T_min), T_max, 21)
                        fraction_per_bin = None
                else:
                    n_thickness_bins = int(self.geometry_params.get("n_thickness_bins", 20))
                    bin_edges = np.linspace(max(1.0, T_min), T_max, n_thickness_bins + 1)
                    n_bins = n_thickness_bins
                    fraction_per_bin = None

                bin_idx = np.searchsorted(bin_edges, T_arr, side="right") - 1
                bin_idx = np.clip(bin_idx, 0, n_bins - 1)

                N_total = 0.0
                for c in cells:
                    rho = _rho_pore_at_thickness(np.array([c["T"]]), self.density_params, self.sample_type)[0]
                    N_total += rho * c["V"]
                N_total = int(np.round(N_total))

                n_grid_per_bin = np.zeros(n_bins, dtype=int)
                V_per_bin = np.zeros(n_bins)
                for i, c in enumerate(_tqdm(cells, total=len(cells), desc="统计厚度区间体积")):
                    k = bin_idx[i]
                    n_grid_per_bin[k] += 1
                    V_per_bin[k] += c["V"]

                has_grid = n_grid_per_bin > 0
                if fraction_per_bin is not None and len(fraction_per_bin) == n_bins:
                    fraction_per_bin = np.asarray(fraction_per_bin, dtype=float)
                    if np.any(has_grid):
                        f_sum = fraction_per_bin[has_grid].sum()
                        if f_sum > 0:
                            fraction_per_bin = fraction_per_bin.copy()
                            fraction_per_bin[~has_grid] = 0.0
                            fraction_per_bin[has_grid] /= f_sum
                else:
                    V_total = V_per_bin.sum()
                    fraction_per_bin = np.where(V_total > 0, V_per_bin / V_total, 1.0 / n_bins)

                # 直接分配 rho（数量分数），孔数 n = rho × V 自然得到（每个网格体积不同）
                rng = np.random.default_rng(self._random_seed)
                rho_dist = self.rho_pore_dist_by_thickness_bin.get(self.sample_type, {})
                bins_fit = rho_dist.get("bins", [])
                rho_sampling_scope = str(
                    self.geometry_params.get("rho_sampling_scope", "run")
                ).strip().lower()
                if rho_sampling_scope not in {"run", "cell"}:
                    rho_sampling_scope = "run"

                # 第一步：抽样得到 n_cell。
                # run 级抽样用于匹配 Phase2 的子样本/区域级 count-density 方差；
                # cell 级抽样会在一个 400x400 run 内平均多个 cell，显著压窄 run-level 分布。
                n_cells = len(cells)
                V_arr = np.array([float(c["V"]) for c in cells], dtype=float)
                n_cell_arr = np.zeros(n_cells, dtype=int)
                n_cell_raw_arr = np.zeros(n_cells, dtype=int)
                rho_cell_raw_arr = np.full(n_cells, np.nan, dtype=float)
                rho_by_bin_for_run: dict[int, float] = {}
                if rho_sampling_scope == "run":
                    for k in sorted(set(int(x) for x in np.asarray(bin_idx, dtype=int).tolist())):
                        if k < len(bins_fit) and bins_fit[k].get("fit"):
                            s = _sample_rho_from_fit(bins_fit[k], 1, rng)
                            rho_by_bin_for_run[k] = float(s[0]) if s is not None and len(s) > 0 else 0.0
                        else:
                            raise RuntimeError(
                                "pore placement fallback hit: missing per-bin fit for rho_pore; "
                                f"sample_type={self.sample_type}, bin_index={k}, "
                                f"n_bins_fit={len(bins_fit)}"
                            )
                for i, c in enumerate(
                    _tqdm(cells, total=n_cells, desc="按cell抽样孔数量")
                ):
                    k = int(bin_idx[i])
                    if rho_sampling_scope == "run":
                        rho_i = rho_by_bin_for_run.get(k, 0.0)
                    elif k < len(bins_fit) and bins_fit[k].get("fit"):
                        s = _sample_rho_from_fit(bins_fit[k], 1, rng)
                        rho_i = float(s[0]) if s is not None and len(s) > 0 else 0.0
                    else:
                        raise RuntimeError(
                            "pore placement fallback hit: missing per-bin fit for rho_pore; "
                            f"sample_type={self.sample_type}, bin_index={k}, "
                            f"n_bins_fit={len(bins_fit)}, T_cell={float(c['T']):.3f}"
                        )
                    rho_i = max(rho_i, 0.0)
                    rho_cell_raw_arr[i] = float(rho_i)
                    n_raw_i = _stochastic_round_nonneg(rho_i * float(c["V"]), rng)
                    n_cell_raw_arr[i] = int(n_raw_i)
                    n_cell_arr[i] = int(n_raw_i)

                # 第二步：按厚度 bin 做“弱回正”（均值回正）
                # 仅在偏差超过阈值时触发，且 scale 有上下限，避免过度扭曲分布形状。
                weak_corr_rel_tol = float(self.geometry_params.get(
                    "pore_rho_mean_correction_rel_tol",
                    self.geometry_params.get("rho_mean_correction_rel_tol", float("inf")),
                ))
                weak_corr_scale_min = float(self.geometry_params.get(
                    "pore_rho_mean_correction_scale_min",
                    self.geometry_params.get("rho_mean_correction_scale_min", 0.50),
                ))
                weak_corr_scale_max = float(self.geometry_params.get(
                    "pore_rho_mean_correction_scale_max",
                    self.geometry_params.get("rho_mean_correction_scale_max", 2.00),
                ))
                if weak_corr_scale_min > weak_corr_scale_max:
                    weak_corr_scale_min, weak_corr_scale_max = weak_corr_scale_max, weak_corr_scale_min
                scale_arr = np.ones(n_cells, dtype=float)
                n_bins_corrected = 0
                for k in range(n_bins):
                    idx_k = np.where(bin_idx == k)[0]
                    if idx_k.size == 0:
                        continue
                    V_k = V_arr[idx_k]
                    V_sum_k = float(np.sum(V_k))
                    if V_sum_k <= 0:
                        continue
                    mu_sim = float(np.sum(n_cell_arr[idx_k]) / V_sum_k)

                    target_mu = np.nan
                    if k < len(bins_fit):
                        fit_k = bins_fit[k].get("fit", {}) if isinstance(bins_fit[k], dict) else {}
                        target_mu = float(fit_k.get("mean", np.nan))
                    if not np.isfinite(target_mu):
                        t_mid = 0.5 * (float(bin_edges[k]) + float(bin_edges[k + 1]))
                        target_mu = float(
                            _rho_pore_at_thickness(np.array([t_mid], dtype=float), self.density_params, self.sample_type)[0]
                        )
                    if not np.isfinite(target_mu) or target_mu < 0:
                        continue
                    if mu_sim <= 0:
                        continue

                    rel_err = abs(mu_sim - target_mu) / max(target_mu, 1e-12)
                    if rel_err <= weak_corr_rel_tol:
                        continue
                    scale_k = float(np.clip(target_mu / mu_sim, weak_corr_scale_min, weak_corr_scale_max))
                    scale_arr[idx_k] = float(scale_k)
                    n_new = np.round(n_cell_arr[idx_k].astype(float) * scale_k).astype(int)
                    n_cell_arr[idx_k] = np.maximum(n_new, 0)
                    n_bins_corrected += 1

                if n_bins_corrected > 0:
                    print(
                        "孔放置: 已执行按厚度bin弱回正（均值）: "
                        f"{n_bins_corrected}/{n_bins} bins, "
                        f"rel_tol={weak_corr_rel_tol:.3f}, "
                        f"scale_clip=[{weak_corr_scale_min:.3f},{weak_corr_scale_max:.3f}]"
                    )

                # 第三步：根据回正后的 n_cell 实际撒点并记录统计
                coords_list = []
                pore_cell_index = []  # pore i 属于 cell pore_cell_index[i]
                count_per_bin_simulated = np.zeros(n_bins)
                rho_cell_per_bin = [[] for _ in range(n_bins)]
                for i, c in enumerate(
                    _tqdm(cells, total=n_cells, desc="按cell孔数撒点")
                ):
                    n_cell = int(n_cell_arr[i])
                    if n_cell <= 0 or float(c["V"]) <= 0:
                        continue
                    k = int(bin_idx[i])
                    count_per_bin_simulated[k] += n_cell
                    rho_cell_per_bin[k].append(n_cell / float(c["V"]))
                    x_c = rng.uniform(c["x_lo"], c["x_hi"], size=n_cell)
                    y_c = rng.uniform(c["y_lo"], c["y_hi"], size=n_cell)
                    z_c = rng.uniform(0, c["T"], size=n_cell)
                    coords_list.append(np.column_stack([x_c, y_c, z_c]))
                    pore_cell_index.extend([i] * n_cell)

                if coords_list:
                    self.pore_coords = np.vstack(coords_list)
                    self._cells = cells
                    self._pore_cell_index = np.array(pore_cell_index, dtype=int)
                    self._pore_rho_sampling_debug = {
                        "sample_type": self.sample_type,
                        "bin_edges": bin_edges.tolist(),
                        "bin_index_per_cell": bin_idx.astype(int).tolist(),
                        "T_per_cell": np.array([float(c["T"]) for c in cells], dtype=float).tolist(),
                        "V_per_cell": V_arr.astype(float).tolist(),
                        "rho_raw_per_cell": rho_cell_raw_arr.astype(float).tolist(),
                        "rho_sampling_scope": rho_sampling_scope,
                        "n_raw_per_cell": n_cell_raw_arr.astype(int).tolist(),
                        "scale_applied_per_cell": scale_arr.astype(float).tolist(),
                        "n_after_corr_per_cell": n_cell_arr.astype(int).tolist(),
                    }
                    # 记录模拟的孔数量分数（与 Phase2 区间划分一致），供可视化
                    total_sim = count_per_bin_simulated.sum()
                    frac_sim = (count_per_bin_simulated / total_sim).tolist() if total_sim > 0 else [0.0] * n_bins
                    self._pore_count_fraction_simulated = {
                        "bin_edges": bin_edges.tolist(),
                        "count_per_bin": count_per_bin_simulated.tolist(),
                        "fraction_per_bin": frac_sim,
                        "n_pore_total": int(total_sim),
                        "rho_cell_per_bin": [lst for lst in rho_cell_per_bin],  # 每区间内各格的孔数量分数 = 孔数/体积
                        "target_fraction_per_bin": np.asarray(fraction_per_bin).tolist() if fraction_per_bin is not None else None,
                    }
                else:
                    target_n_pores = max(1, int(target_rho_pore * self.volume))
                    x_c = np.random.uniform(0, width, target_n_pores)
                    y_c = np.random.uniform(0, height, target_n_pores)
                    if thickness_interp is not None:
                        T_xy = np.maximum(thickness_interp(x_c, y_c, grid=False).ravel(), 1.0)
                    else:
                        T_xy = np.full(target_n_pores, uniform_cell_T, dtype=float)
                    z_c = np.random.uniform(0, T_xy, size=target_n_pores)
                    self.pore_coords = np.column_stack([x_c, y_c, z_c])
                    self._cells = None
                    self._pore_cell_index = None
                    self._pore_rho_sampling_debug = None
                dist_src = "直接分配rho(n=ρ×V)" if rho_dist.get("bins") else "线性回归rho"
                print(f"孔放置: 网格 {nx_cell}×{ny_cell} (格宽 {cell_nm} nm), {dist_src}, 随机撒点共 {len(self.pore_coords)} 孔")
            else:
                self._cells = None
                self._pore_cell_index = None
                self._pore_rho_sampling_debug = None
                thickness = self.thickness
                target_n_pores = int(target_rho_pore * self.volume)
                print(f"目标孔数: {target_n_pores} (密度 = {target_rho_pore:.2e} 1/nm^3)")
                oversample_factor = 2.0
                n_candidates = int(target_n_pores * oversample_factor)
                x_coords = np.random.uniform(0, width, n_candidates)
                y_coords = np.random.uniform(0, height, n_candidates)
                z_coords = np.random.uniform(0, thickness, n_candidates)
                indices = np.random.choice(n_candidates, size=target_n_pores, replace=False)
                self.pore_coords = np.column_stack([
                    x_coords[indices],
                    y_coords[indices],
                    z_coords[indices],
                ])

        elif self.geometry_type == "sphere":
            radius = self.sphere_radius
            thickness = self.thickness
            target_n_pores = int(target_rho_pore * self.volume)
            print(f"目标孔数: {target_n_pores} (密度 = {target_rho_pore:.2e} 1/nm^3)")
            # 在球面上均匀采样点（使用球坐标）
            # 简化：在球壳内均匀采样
            n_candidates = int(target_n_pores * 2.0)
            coords = []
            for _ in range(n_candidates):
                # 在球壳内随机采样
                r = radius + np.random.uniform(-thickness/2, thickness/2)
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                coords.append([x, y, z])
            
            indices = np.random.choice(n_candidates, size=target_n_pores, replace=False)
            self.pore_coords = np.array([coords[i] for i in indices])
            self._cells = None
            self._pore_cell_index = None

        self.pore_ids = np.arange(1, len(self.pore_coords) + 1)
        print(f"已放置 {len(self.pore_coords)} 个孔心")
    
    def assign_pore_radii(self, n_candidate_groups: int = 8):
        """
        3.3 孔半径赋值：按厚度 bin 从孔半径分布采样，不再施加体积分数约束。
        （原逻辑：多组采样 + frac_pore 权重选择，已注释）
        """
        rng = np.random.default_rng(self._random_seed)
        self.pore_radii = np.zeros(len(self.pore_coords))

        if getattr(self, "_cells", None) is not None and getattr(self, "_pore_cell_index", None) is not None:
            radius_dist = self.pore_radius_dist_by_thickness_bin.get(self.sample_type, {})
            radius_edges = np.array(radius_dist.get("bin_edges", [0, 500]))
            radius_bins = radius_dist.get("bins", [])

            for i, cell in enumerate(
                _tqdm(self._cells, total=len(self._cells), desc="孔半径赋值")
            ):
                idx = np.where(self._pore_cell_index == i)[0]
                n = len(idx)
                if n <= 0:
                    continue
                T = cell["T"]

                # 半径：按厚度 bin 从孔半径分布采样
                k_rad = np.searchsorted(radius_edges, T, side="right") - 1
                k_rad = np.clip(k_rad, 0, len(radius_bins) - 1)
                rad_fit = radius_bins[k_rad] if k_rad < len(radius_bins) else None

                radii = _sample_radius_from_fit(rad_fit, n, rng) if rad_fit and rad_fit.get("fit") else None
                if radii is None or len(radii) != n:
                    mean_r = rad_fit.get("fit", {}).get("mean", 5.0) if rad_fit else 5.0
                    radii = np.maximum(rng.normal(mean_r, 1.0, n), 0.5)
                self.pore_radii[idx] = radii

                # [已注释] 体积分数约束：多组采样 + frac_pore 权重选择
                # frac_dist = self.frac_pore_dist_by_thickness_bin.get(self.sample_type, {})
                # frac_edges = np.array(frac_dist.get("bin_edges", [0, 500]))
                # frac_bins = frac_dist.get("bins", [])
                # k_frac = np.searchsorted(frac_edges, T, side="right") - 1
                # frac_fit = frac_bins[k_frac] if k_frac < len(frac_bins) else None
                # max_frac_pore = 0.75; candidates = []
                # for _ in range(n_candidate_groups): ... if frac_g <= max_frac_pore: candidates.append(...)
                # weights = frac_pore PDF; sel = rng.choice(..., p=weights); self.pore_radii[idx] = candidates[sel][0]

            print(f"孔半径赋值: 按厚度 bin 从孔半径分布采样（无体积分数约束）")
        else:
            # 无厚度场：全局采样，不再按体积分数缩放
            radius_dist = self.radius_params.get("pore", {}).get(self.sample_type, {})
            # 优先用按厚度 bin 的分布（单厚度时取对应 bin）
            radius_bins = self.pore_radius_dist_by_thickness_bin.get(self.sample_type, {}).get("bins", [])
            T_mean = self.geometry_params.get("thickness_mean", 150.0)
            radius_edges = np.array(
                self.pore_radius_dist_by_thickness_bin.get(self.sample_type, {}).get("bin_edges", [0, 500])
            )
            if len(radius_bins) > 0 and len(radius_edges) >= 2:
                k = np.clip(np.searchsorted(radius_edges, T_mean, side="right") - 1, 0, len(radius_bins) - 1)
                rad_fit = radius_bins[k]
                radii = _sample_radius_from_fit(rad_fit, len(self.pore_coords), rng)
                if radii is not None and len(radii) == len(self.pore_coords):
                    self.pore_radii = radii
                else:
                    mean_r = rad_fit.get("fit", {}).get("mean", 5.0)
                    self.pore_radii = np.maximum(rng.normal(mean_r, 1.0, len(self.pore_coords)), 0.5)
            else:
                dist_name = radius_dist.get("distribution", {}).get("name", "lognorm")
                dist_params = radius_dist.get("distribution", {}).get("params", [])
                if dist_name == "lognorm" and len(dist_params) >= 3:
                    self.pore_radii = stats.lognorm.rvs(dist_params[0], loc=dist_params[1], scale=dist_params[2],
                                                        size=len(self.pore_coords), random_state=rng)
                else:
                    mean_r = radius_dist.get("mean", 5.0)
                    std_r = radius_dist.get("std", 1.0)
                    self.pore_radii = np.maximum(rng.normal(mean_r, std_r, len(self.pore_coords)), 0.5)
            # [已注释] 体积分数缩放: alpha = (target_frac_pore * volume / vol)**(1/3); self.pore_radii *= alpha
            vol = np.sum((4.0 / 3.0) * np.pi * self.pore_radii**3)
            print(f"孔半径赋值: 全局分布, 体积分数≈{vol/self.volume:.4f} (未缩放)")

        self.pore_radii = np.maximum(self.pore_radii, 0.5)
        self._apply_soft_z_radius_bias_plane()
        self._apply_top_quintile_spatial_radius_bias_plane()
        self.pore_radii = np.maximum(self.pore_radii, 0.5)
        print(f"孔半径赋值完成: {len(self.pore_coords)} 个孔, 半径范围 [{self.pore_radii.min():.2f}, {self.pore_radii.max():.2f}] nm")

    def _reference_thickness_nm(self) -> float:
        """代表性厚度（nm）：有厚度场时用场网格均值，否则用 generate_geometry 后的 self.thickness。"""
        g = getattr(self, "thickness_field_grid", None)
        if g is not None:
            return float(np.mean(np.asarray(g, dtype=float)))
        return float(getattr(self, "thickness", 0.0))

    def _resolve_pore_radius_z_bias_lambda(self) -> float:
        """
        原：AS 且 T_ref>160 nm 时 λ=1，沿膜厚中面富集大半径（半径多重集不变）。
        现已全局关闭，恒返回 0。
        """
        return 0.0

    def _apply_soft_z_radius_bias_plane(self) -> None:
        """
        膜厚中面大孔软偏置：**当前关闭**（``_resolve_pore_radius_z_bias_lambda`` 恒为 0，立即返回）。
        原逻辑：半径多重集不变，按 |z-T/2| 给靠中面孔更大权重抽到大半径；仅 plane。
        """
        lam = float(self._resolve_pore_radius_z_bias_lambda())
        if lam <= 0.0:
            return
        if self.geometry_type != "plane":
            print("孔半径 Z 软偏置：非 plane 几何，跳过")
            return
        coords = np.asarray(self.pore_coords, dtype=float)
        radii = np.asarray(self.pore_radii, dtype=float)
        n = len(radii)
        if n == 0:
            return
        rng = np.random.default_rng((self._random_seed if self._random_seed is not None else 0) + 9091)
        z_max = float(self.bbox[1, 2]) if getattr(self, "bbox", None) is not None else float(self.thickness)
        z_mid = 0.5 * z_max
        d = np.abs(coords[:, 2] - z_mid)
        d_max = float(np.max(d)) + 1e-9
        u = d / d_max
        order = np.argsort(d)
        remaining = radii.copy().tolist()
        r_med = float(np.median(radii)) + 1e-6
        new_r = np.zeros(n, dtype=float)
        for k in range(n):
            i_pore = int(order[k])
            u_i = float(u[i_pore])
            if len(remaining) == 1:
                new_r[i_pore] = remaining[0]
                break
            w = np.array(
                [np.exp(lam * (1.0 - u_i) * (float(rj) / r_med)) for rj in remaining],
                dtype=float,
            )
            w_sum = float(np.sum(w))
            if not np.isfinite(w_sum) or w_sum <= 0:
                j = int(rng.integers(0, len(remaining)))
            else:
                p = w / w_sum
                j = int(rng.choice(len(remaining), p=p))
            new_r[i_pore] = remaining.pop(j)
        self.pore_radii = new_r
        Tref = self._reference_thickness_nm()
        print(
            f"孔半径 Z 软偏置：λ={lam}（AS 厚膜中面偏置；T_ref={Tref:.1f} nm），"
            f"半径多重集不变，仅重排"
        )

    def _apply_top_quintile_spatial_radius_bias_plane(self) -> None:
        """
        默认开启（``pore_radius_q4_spatial_bias``，CLI 可 ``--no-pore-radius-q4-spatial-bias`` 关闭）。
        在 i.i.d. 采样（及 Z 软偏置）之后，按当前半径做 pd.qcut 五分位；
        仅对**最大档**孔：随机一半保持该子集内随机重排，另一半在**保持该子集半径多重集不变**前提下，
        使较大半径略倾向于落在「随机一半」孔心质心附近（与 phase1.5 观测的大孔局部分组一致，探索用）。

        强度 pore_radius_q4_bias_strength ∈ [0,1]：0 等价于该半档内随机，1 为完全按距离排序匹配；
        中间值为距离优先分与随机分的凸组合（略偏一点点）。
        """
        if not getattr(self, "pore_radius_q4_spatial_bias", False):
            return
        if self.geometry_type != "plane":
            print("孔半径最大五分位空间偏置：非 plane 几何，跳过")
            return
        coords = np.asarray(self.pore_coords, dtype=float)
        radii = np.asarray(self.pore_radii, dtype=float).copy()
        n = len(radii)
        if n < 10:
            print("孔半径最大五分位空间偏置：孔数过少，跳过")
            return
        s = pd.Series(radii)
        try:
            q = pd.qcut(s, q=5, labels=False, duplicates="drop")
        except (ValueError, TypeError):
            print("孔半径最大五分位空间偏置：qcut 失败，跳过")
            return
        q = np.asarray(q, dtype=int)
        if np.any(~np.isfinite(q)):
            print("孔半径最大五分位空间偏置：分位标签无效，跳过")
            return
        top = int(np.nanmax(q))
        idx_q = np.where(q == top)[0]
        if len(idx_q) < 4:
            print(f"孔半径最大五分位空间偏置：最大档孔数={len(idx_q)} 过少，跳过")
            return
        beta = float(getattr(self, "pore_radius_q4_bias_strength", 0.55))
        beta = float(np.clip(beta, 0.0, 1.0))
        rng = np.random.default_rng((self._random_seed if self._random_seed is not None else 0) + 31337)
        rng.shuffle(idx_q)
        n_half = len(idx_q) // 2
        I_rand = idx_q[:n_half]
        I_bias = idx_q[n_half:]
        V = radii[idx_q].copy()
        rng.shuffle(V)
        take_rand = V[: len(I_rand)].copy()
        take_bias = V[len(I_rand) :].copy()
        rng.shuffle(take_rand)
        radii[I_rand] = take_rand
        cent = np.mean(coords[I_rand], axis=0)
        d = np.linalg.norm(coords[I_bias] - cent, axis=1)
        d_n = d / (float(np.max(d)) + 1e-9)
        noise = rng.random(len(I_bias))
        score = (1.0 - beta) * noise + beta * (1.0 - d_n)
        order = np.argsort(-score)
        take_bias_desc = np.sort(take_bias)[::-1]
        radii[I_bias[order]] = take_bias_desc
        self.pore_radii = radii
        print(
            f"孔半径最大五分位空间偏置：已启用（最大档 n={len(idx_q)}，随机半档={len(I_rand)}，"
            f"距离偏置半档={len(I_bias)}，强度 β={beta:.3f}），该档半径多重集不变"
        )

    def reposition_pores_no_overlap(self, max_attempts_per_pore: int = 500):
        """
        [已废弃逻辑保留占位]
        早期版本中，这个函数在每个网格内重新放置孔心以保证“孔与孔不重叠”。
        现在我们允许孔几何上可以重叠，因此此处不再进行重排，只打印提示。
        """
        print("reposition_pores_no_overlap: 已禁用孔不重叠约束，保持原始孔位置。")

    def plot_pores_3d_html(self, output_dir: Path, sample_type: str = None, sample_name: str = None, max_markers: int = 5000):
        """
        撒点完成后 3D 可视化：孔心与半径，导出为可交互 HTML。
        若孔数过多则随机抽样 max_markers 个以保持流畅。
        sample_name 用于文件名，每个 run 独立；缺省时用 sample_type。
        """
        if not _HAS_PLOTLY:
            print("未安装 plotly，跳过 3D HTML 可视化。可运行: pip install plotly")
            return
        coords = np.asarray(self.pore_coords)
        radii = np.asarray(self.pore_radii) if hasattr(self, "pore_radii") and self.pore_radii is not None else np.ones(len(coords)) * 5.0
        if len(coords) == 0:
            print("无孔数据，跳过 3D 可视化")
            return
        st = sample_type or self.sample_type
        file_id = sample_name if sample_name else st
        if len(coords) > max_markers:
            rng = np.random.default_rng(self._random_seed)
            n_choose = min(max_markers, len(coords))
            idx = rng.choice(len(coords), size=n_choose, replace=False)
            coords = coords[idx]
            radii = radii[idx]
        # 散点：marker size 与半径成比例（plotly 的 size 为屏幕像素，需缩放）
        radii_safe = np.where(np.isfinite(radii), radii, np.nanmedian(radii))
        if not np.any(np.isfinite(radii_safe)):
            radii_safe = np.ones_like(radii) * 2.0
        scale = 2.0 / (float(np.median(radii_safe)) + 1e-6)
        sizes = np.clip(radii_safe * scale, 1, 30)
        trace = go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=sizes,
                color=radii_safe,
                colorscale="Viridis",
                colorbar=dict(title="Radius (nm)"),
                line=dict(width=0),
            ),
            text=[f"r={r:.2f} nm" for r in radii_safe],
            hovertemplate="x=%{x:.1f} y=%{y:.1f} z=%{z:.1f}<br>%{text}<extra></extra>",
            name="Pores",
        )
        n_show = len(coords)
        n_total = len(self.pore_coords)
        title = f"Pore centers 3D ({file_id})" + (f" show {n_show}/{n_total}" if n_total > max_markers else f" n={n_total}")
        layout = go.Layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            scene=dict(
                xaxis_title="x (nm)",
                yaxis_title="y (nm)",
                zaxis_title="z (nm)",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig = go.Figure(data=[trace], layout=layout)
        out_path = output_dir / f"pores_3d_{file_id}.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"3D 可视化已保存: {out_path}")

    def connect_throats(self, max_distance: float = None):
        """
        3.4 喉连接（新）：

        1) 先生成所有几何 overlap 喉：
           - d = |x_j - x_i| < R_i + R_j - TOL_NM_OVERLAP
           - intersection_circle_radius_nm(d, R_i, R_j) > 0
           - 这些喉的长度 = d，半径稍后取交界面圆半径 R_cap

        2) 再在「非 overlap 孔对」中，补齐额外喉：
           - 目标总喉数按 phase2 的 rho_throat * 体积近似
           - 剩余喉数 N_extra = max(0, N_target - N_overlap)
           - 在 max_distance 邻域内的非 overlap 孔对里随机抽样补到 N_extra 条
        """
        rng = np.random.default_rng(self._random_seed)
        n_pores = len(self.pore_coords)
        if n_pores == 0:
            self.throat_pore1 = np.array([], dtype=int)
            self.throat_pore2 = np.array([], dtype=int)
            self._throat_cap_radii_geom = np.array([], dtype=float)
            self._throat_kind = np.array([], dtype=object)
            self._throat_is_cross_cell = np.array([], dtype=bool)
            self._throat_cell1 = np.array([], dtype=int)
            self._throat_cell2 = np.array([], dtype=int)
            print("喉连接: 无孔，已连接 0 条喉。")
            return

        # ---------- 1) overlap 喉 ----------
        if max_distance is None:
            max_distance = 30.0  # nm

        coords = np.asarray(self.pore_coords, dtype=float)
        radii = np.asarray(self.pore_radii, dtype=float)

        overlap_i: list[int] = []
        overlap_j: list[int] = []
        overlap_cap: list[float] = []

        tree = cKDTree(coords)
        for i in _tqdm(
            range(n_pores),
            total=n_pores,
            desc=f"构建 overlap 喉（邻域 {max_distance:g} nm）",
        ):
            neighbors = tree.query_ball_point(coords[i], max_distance)
            for j in neighbors:
                if j <= i:
                    continue
                d = float(np.linalg.norm(coords[i] - coords[j]))
                if not np.isfinite(d) or d <= 0:
                    continue
                R1 = float(radii[i])
                R2 = float(radii[j])
                if d >= R1 + R2 - TOL_NM_OVERLAP:
                    continue
                R_cap = _intersection_circle_radius_nm(d, R1, R2)
                if not np.isfinite(R_cap) or R_cap <= 0.0:
                    continue
                overlap_i.append(i)
                overlap_j.append(j)
                overlap_cap.append(R_cap)

        overlap_i = np.asarray(overlap_i, dtype=int)
        overlap_j = np.asarray(overlap_j, dtype=int)
        overlap_cap = np.asarray(overlap_cap, dtype=float)
        overlap_d = np.linalg.norm(coords[overlap_j] - coords[overlap_i], axis=1) if len(overlap_i) > 0 else np.array([], dtype=float)
        n_overlap = len(overlap_i)

        # ---------- 2) 目标喉数量（cell 化抽样 + cell 间 non-overlap 估计） ----------
        target_rho_throat_mean = (
            self.density_params.get("rho_throat", {})
            .get(self.sample_type, {})
            .get("mean", None)
        )

        # 2.1 非 overlap（cell 内基线）：按 cell 厚度分箱，从 rho_throat(T-bin) 分布抽样，再乘 cell 体积
        n_nonoverlap_cell_based = None
        throat_count_target_per_cell = None  # cell内目标总喉数（后续用于cell内分布回正）
        rho_throat_dist = getattr(self, "rho_throat_dist_by_thickness_bin", {}).get(self.sample_type, {})
        if getattr(self, "_cells", None) is not None and len(self._cells) > 0:
            bin_edges_rho = np.array(rho_throat_dist.get("bin_edges", []), dtype=float)
            bins_fit_rho = rho_throat_dist.get("bins", [])
            if len(bin_edges_rho) >= 2 and len(bins_fit_rho) > 0:
                n_cells_local = len(self._cells)
                V_cell = np.array([float(c.get("V", 0.0)) for c in self._cells], dtype=float)
                T_cell = np.array([float(c.get("T", 0.0)) for c in self._cells], dtype=float)
                k_cell = np.searchsorted(bin_edges_rho, T_cell, side="right") - 1
                k_cell = np.clip(k_cell, 0, len(bin_edges_rho) - 2).astype(int)
                throat_count_target_per_cell = np.zeros(n_cells_local, dtype=int)
                throat_count_raw_per_cell = np.zeros(n_cells_local, dtype=int)
                rho_raw_per_cell = np.full(n_cells_local, np.nan, dtype=float)
                rho_sampling_scope = str(
                    self.geometry_params.get("rho_sampling_scope", "run")
                ).strip().lower()
                if rho_sampling_scope not in {"run", "cell"}:
                    rho_sampling_scope = "run"
                rho_by_bin_for_run: dict[int, float] = {}
                if rho_sampling_scope == "run":
                    for k in sorted(set(int(x) for x in np.asarray(k_cell, dtype=int).tolist())):
                        fit_k = bins_fit_rho[k] if k < len(bins_fit_rho) else None
                        if fit_k and fit_k.get("fit"):
                            s = _sample_rho_from_fit(fit_k, 1, rng)
                            rho_by_bin_for_run[k] = float(s[0]) if s is not None and len(s) > 0 else np.nan
                        else:
                            rho_by_bin_for_run[k] = np.nan

                for i, c in enumerate(self._cells):
                    V = float(c.get("V", 0.0))
                    T = float(c.get("T", 0.0))
                    if not (np.isfinite(V) and np.isfinite(T)) or V <= 0:
                        continue
                    k = int(k_cell[i])
                    fit_k = bins_fit_rho[k] if k < len(bins_fit_rho) else None
                    if rho_sampling_scope == "run":
                        rho_k = rho_by_bin_for_run.get(k, np.nan)
                    elif fit_k and fit_k.get("fit"):
                        s = _sample_rho_from_fit(fit_k, 1, rng)
                        rho_k = float(s[0]) if s is not None and len(s) > 0 else np.nan
                    else:
                        rho_k = np.nan
                    if not np.isfinite(rho_k):
                        rho_k = float(target_rho_throat_mean) if target_rho_throat_mean is not None else 0.0
                    rho_k = max(float(rho_k), 0.0)
                    rho_raw_per_cell[i] = float(rho_k)
                    n_raw_i = _stochastic_round_nonneg(rho_k * V, rng)
                    throat_count_raw_per_cell[i] = int(n_raw_i)
                    throat_count_target_per_cell[i] = int(n_raw_i)

                # 按厚度 bin 弱回正（均值）：仅大偏差触发，且限制缩放幅度，保留分布形状
                weak_corr_rel_tol = float(self.geometry_params.get(
                    "throat_rho_mean_correction_rel_tol",
                    self.geometry_params.get("rho_mean_correction_rel_tol", float("inf")),
                ))
                weak_corr_scale_min = float(self.geometry_params.get(
                    "throat_rho_mean_correction_scale_min",
                    self.geometry_params.get("rho_mean_correction_scale_min", 0.50),
                ))
                weak_corr_scale_max = float(self.geometry_params.get(
                    "throat_rho_mean_correction_scale_max",
                    self.geometry_params.get("rho_mean_correction_scale_max", 2.00),
                ))
                if weak_corr_scale_min > weak_corr_scale_max:
                    weak_corr_scale_min, weak_corr_scale_max = weak_corr_scale_max, weak_corr_scale_min
                scale_arr_th = np.ones(n_cells_local, dtype=float)
                n_bins_corrected_th = 0
                for k in range(len(bin_edges_rho) - 1):
                    idx_k = np.where(k_cell == k)[0]
                    if idx_k.size == 0:
                        continue
                    V_k = V_cell[idx_k]
                    V_sum_k = float(np.sum(V_k))
                    if V_sum_k <= 0:
                        continue
                    mu_sim = float(np.sum(throat_count_target_per_cell[idx_k]) / V_sum_k)
                    if mu_sim <= 0:
                        continue
                    fit_k = bins_fit_rho[k] if k < len(bins_fit_rho) else None
                    mu_tar = np.nan
                    if isinstance(fit_k, dict) and fit_k.get("fit"):
                        mu_tar = float((fit_k.get("fit") or {}).get("mean", np.nan))
                    if not np.isfinite(mu_tar):
                        mu_tar = float(target_rho_throat_mean) if target_rho_throat_mean is not None else 0.0
                    if not np.isfinite(mu_tar) or mu_tar < 0:
                        continue
                    rel_err = abs(mu_sim - mu_tar) / max(mu_tar, 1e-12)
                    if rel_err <= weak_corr_rel_tol:
                        continue
                    scale_k = float(np.clip(mu_tar / mu_sim, weak_corr_scale_min, weak_corr_scale_max))
                    scale_arr_th[idx_k] = float(scale_k)
                    throat_count_target_per_cell[idx_k] = np.maximum(
                        np.round(throat_count_target_per_cell[idx_k].astype(float) * scale_k).astype(int),
                        0,
                    )
                    n_bins_corrected_th += 1

                if n_bins_corrected_th > 0:
                    print(
                        "喉连接: 已执行cell内喉数量按厚度bin弱回正（均值）: "
                        f"{n_bins_corrected_th}/{len(bin_edges_rho)-1} bins, "
                        f"rel_tol={weak_corr_rel_tol:.3f}, "
                        f"scale_clip=[{weak_corr_scale_min:.3f},{weak_corr_scale_max:.3f}]"
                    )
                self._throat_rho_sampling_debug = {
                    "sample_type": self.sample_type,
                    "bin_edges": bin_edges_rho.tolist(),
                    "bin_index_per_cell": k_cell.astype(int).tolist(),
                    "T_per_cell": T_cell.astype(float).tolist(),
                    "V_per_cell": V_cell.astype(float).tolist(),
                    "rho_raw_per_cell": rho_raw_per_cell.astype(float).tolist(),
                    "rho_sampling_scope": rho_sampling_scope,
                    "n_raw_per_cell": throat_count_raw_per_cell.astype(int).tolist(),
                    "scale_applied_per_cell": scale_arr_th.astype(float).tolist(),
                    "n_after_corr_per_cell": throat_count_target_per_cell.astype(int).tolist(),
                }
                n_nonoverlap_cell_based = int(np.sum(throat_count_target_per_cell))

        # cell 分布不可用时，回退旧口径（均值 rho_throat * 总体积）
        if n_nonoverlap_cell_based is None:
            self._throat_rho_sampling_debug = None
            if target_rho_throat_mean is None:
                n_nonoverlap_cell_based = 0
            else:
                n_nonoverlap_cell_based = int(np.round(float(target_rho_throat_mean) * float(self.volume)))

        # 2.2 cell 间 non-overlap 喉数估计：唯一内部邻接面（右邻+上邻）累加 rho_cross * A_face
        # 注意：每个共享面只计一次，因此不需要 /2。
        n_nonoverlap_cross_est = 0
        nx_cell_for_cross = None
        ny_cell_for_cross = None
        cross_dist_root = getattr(self, "nonoverlap_cross_density_dist_by_thickness_bin", {}) or {}
        cross_dist = cross_dist_root.get(self.sample_type, {})
        if (
            getattr(self, "_cells", None) is not None
            and len(self._cells) > 0
            and self.geometry_type == "plane"
            and cross_dist
        ):
            bin_edges_cross = np.array(cross_dist.get("bin_edges", []), dtype=float)
            bins_cross = cross_dist.get("bins", [])
            width = float(getattr(self, "_plane_width", self.geometry_params.get("width", 5000.0)))
            height = float(getattr(self, "_plane_height", self.geometry_params.get("height", 5000.0)))
            cell_nm = float(self.geometry_params.get("pore_placement_cell_nm", 50.0))
            nx_cell = max(1, int(np.ceil(width / cell_nm)))
            ny_cell = max(1, int(np.ceil(height / cell_nm)))
            if len(bin_edges_cross) >= 2 and len(bins_cross) > 0 and len(self._cells) == nx_cell * ny_cell:
                nx_cell_for_cross = int(nx_cell)
                ny_cell_for_cross = int(ny_cell)
                sum_cross = 0.0

                def _sample_cross_rho(t_face: float) -> float:
                    kk = int(np.searchsorted(bin_edges_cross, t_face, side="right") - 1)
                    kk = int(np.clip(kk, 0, len(bin_edges_cross) - 2))
                    rec = bins_cross[kk] if kk < len(bins_cross) else None
                    if rec and rec.get("fit"):
                        s = _sample_rho_from_fit(rec, 1, rng)
                        val = float(s[0]) if s is not None and len(s) > 0 else np.nan
                    else:
                        val = np.nan
                    if not np.isfinite(val):
                        # 同 bin 无拟合时，用 raw mean；再不行退到 0
                        val = float(rec.get("mean", 0.0)) if isinstance(rec, dict) else 0.0
                    return max(float(val), 0.0)

                # x 方向内部共享面（右邻）
                for j in range(ny_cell):
                    for i in range(nx_cell - 1):
                        a = j * nx_cell + i
                        b = j * nx_cell + (i + 1)
                        ca = self._cells[a]
                        cb = self._cells[b]
                        t_face = 0.5 * (float(ca.get("T", 0.0)) + float(cb.get("T", 0.0)))
                        face_len = min(
                            float(ca.get("y_hi", 0.0)) - float(ca.get("y_lo", 0.0)),
                            float(cb.get("y_hi", 0.0)) - float(cb.get("y_lo", 0.0)),
                        )
                        area_face = max(face_len, 0.0) * max(t_face, 0.0)
                        if area_face <= 0:
                            continue
                        rho_cross = _sample_cross_rho(t_face)
                        sum_cross += rho_cross * area_face

                # y 方向内部共享面（上邻）
                for j in range(ny_cell - 1):
                    for i in range(nx_cell):
                        a = j * nx_cell + i
                        b = (j + 1) * nx_cell + i
                        ca = self._cells[a]
                        cb = self._cells[b]
                        t_face = 0.5 * (float(ca.get("T", 0.0)) + float(cb.get("T", 0.0)))
                        face_len = min(
                            float(ca.get("x_hi", 0.0)) - float(ca.get("x_lo", 0.0)),
                            float(cb.get("x_hi", 0.0)) - float(cb.get("x_lo", 0.0)),
                        )
                        area_face = max(face_len, 0.0) * max(t_face, 0.0)
                        if area_face <= 0:
                            continue
                        rho_cross = _sample_cross_rho(t_face)
                        sum_cross += rho_cross * area_face

                n_nonoverlap_cross_est = int(np.round(sum_cross))

        # 汇总总目标：overlap（几何确定）+ cell 基线 + cell 间估计
        target_n_throats = int(n_overlap + max(0, n_nonoverlap_cell_based) + max(0, n_nonoverlap_cross_est))
        self._throat_target_debug = {
            "sample_type": self.sample_type,
            "n_overlap_geom": int(n_overlap),
            "n_nonoverlap_cell_based": int(max(0, n_nonoverlap_cell_based)),
            "n_nonoverlap_cross_est": int(max(0, n_nonoverlap_cross_est)),
            "target_n_throats_total": int(target_n_throats),
            "used_cell_based_rho_throat": bool(getattr(self, "_cells", None) is not None and len(self._cells) > 0),
            "used_cross_estimation": bool(n_nonoverlap_cross_est > 0),
        }
        print(
            "喉数量目标: overlap="
            f"{int(n_overlap)}, cell基线={int(max(0, n_nonoverlap_cell_based))}, "
            f"cell间cross估计={int(max(0, n_nonoverlap_cross_est))}, 总目标={int(target_n_throats)}"
        )
        n_extra_target = max(0, target_n_throats - n_overlap)

        # ---------- 3) 构建非 overlap 候选（max_distance 内且非 overlap） ----------
        overlap_set = {(int(a), int(b)) for a, b in zip(overlap_i, overlap_j)}
        non_i: list[int] = []
        non_j: list[int] = []
        non_d: list[float] = []

        if n_extra_target > 0:
            for i in _tqdm(
                range(n_pores),
                total=n_pores,
                desc=f"构建非 overlap 喉候选（邻域 {max_distance:g} nm）",
            ):
                neighbors = tree.query_ball_point(coords[i], max_distance)
                for j in neighbors:
                    if j <= i:
                        continue
                    if (i, j) in overlap_set:
                        continue
                    d = float(np.linalg.norm(coords[i] - coords[j]))
                    if not np.isfinite(d) or d <= 0:
                        continue
                    R1 = float(radii[i])
                    R2 = float(radii[j])
                    # 显式要求非 overlap：d >= R1 + R2 - TOL_NM_OVERLAP
                    if d < R1 + R2 - TOL_NM_OVERLAP:
                        continue
                    non_i.append(i)
                    non_j.append(j)
                    non_d.append(d)

        non_i = np.asarray(non_i, dtype=int)
        non_j = np.asarray(non_j, dtype=int)
        non_d = np.asarray(non_d, dtype=float)
        n_non_candidates = len(non_i)
        extra_kind = np.array([], dtype=object)
        extra_cell1 = np.array([], dtype=int)
        extra_cell2 = np.array([], dtype=int)

        if n_extra_target <= 0 or n_non_candidates == 0:
            extra_i = np.array([], dtype=int)
            extra_j = np.array([], dtype=int)
            if n_extra_target > 0 and n_non_candidates == 0:
                print(
                    f"喉连接: 无非 overlap 候选孔对，只有 overlap 喉 {n_overlap} 条 "
                    f"(目标 {target_n_throats})。"
                )
        else:
            # 目标度数：按 phase2 厚度分箱度数分布采样，并扣除 overlap 已占用度数
            if getattr(self, "_cells", None) is not None and getattr(self, "_pore_cell_index", None) is not None:
                pore_thickness = np.array(
                    [self._cells[self._pore_cell_index[i]]["T"] for i in range(n_pores)], dtype=float
                )
            elif getattr(self, "_thickness_interp", None) is not None:
                pore_thickness = np.maximum(
                    self._thickness_interp(self.pore_coords[:, 0], self.pore_coords[:, 1], grid=False).ravel(),
                    1.0,
                )
            else:
                pore_thickness = np.full(n_pores, self.geometry_params.get("thickness_mean", 150.0))
            if self.sample_type == "AS":
                pore_thickness = np.maximum(pore_thickness, 40.0)

            degree_data = self.degree_dist_by_thickness_bin.get(self.sample_type, {})
            bin_edges_deg = np.array(degree_data.get("bin_edges", []), dtype=float)
            bins_deg = degree_data.get("bins", [])
            n_bins_deg = len(bin_edges_deg) - 1
            deg_target = np.full(n_pores, 3, dtype=int)
            if n_bins_deg > 0 and len(bins_deg) > 0:
                for i in range(n_pores):
                    k = np.searchsorted(bin_edges_deg, pore_thickness[i], side="right") - 1
                    k = np.clip(k, 0, n_bins_deg - 1)
                    fit_deg = bins_deg[k] if k < len(bins_deg) else None
                    if fit_deg and fit_deg.get("fit"):
                        deg_target[i] = int(_sample_degree_from_fit(fit_deg, 1, rng)[0])
                    deg_target[i] = max(0, min(deg_target[i], 20))

            # 你希望“度数要抽取，并让大孔更容易抽到高度数”：
            # 这里保留基于厚度分箱的抽样 deg_target，再做“概率偏置”而非确定性缩放。
            r_mean = float(np.mean(radii))
            r_std = float(np.std(radii) + 1e-8)
            r_z = np.clip((radii - r_mean) / r_std, -2.0, 2.0)
            p_up = np.clip(0.34 + 0.40 * r_z, 0.02, 0.90)
            p_down = np.clip(0.28 - 0.23 * r_z, 0.01, 0.62)
            up = rng.binomial(2, p_up, size=n_pores)
            down = rng.binomial(1, p_down, size=n_pores)
            deg_target = deg_target + up - down
            deg_target = np.maximum(1, np.minimum(20, deg_target.astype(int)))

            deg_used = np.zeros(n_pores, dtype=int)
            if n_overlap > 0:
                np.add.at(deg_used, overlap_i, 1)
                np.add.at(deg_used, overlap_j, 1)
            # deg_rem = deg_target - deg_used_overlap + noise, noise ∈ {-1,0,1}
            noise = rng.integers(-1, 2, size=n_pores)
            deg_rem = deg_target - deg_used + noise

            # 长度权重：phase2 喉长度分布（厚度分箱）
            w_len = np.ones(n_non_candidates, dtype=float)
            throat_len_data = self.throat_length_dist_by_thickness_bin.get(self.sample_type, {})
            bin_edges_len = np.array(throat_len_data.get("bin_edges", []), dtype=float)
            bins_len = throat_len_data.get("bins", [])
            n_bins_len = len(bin_edges_len) - 1
            if n_bins_len > 0 and len(bins_len) > 0:
                T_mid_all = 0.5 * (pore_thickness[non_i] + pore_thickness[non_j])
                k_all = np.searchsorted(bin_edges_len, T_mid_all, side="right") - 1
                k_all = np.clip(k_all, 0, n_bins_len - 1)
                # overlap 喉对应厚度分箱，用于构造“剩余长度分布”
                if n_overlap > 0:
                    T_mid_ov = 0.5 * (pore_thickness[overlap_i] + pore_thickness[overlap_j])
                    k_ov = np.searchsorted(bin_edges_len, T_mid_ov, side="right") - 1
                    k_ov = np.clip(k_ov, 0, n_bins_len - 1)
                else:
                    k_ov = np.array([], dtype=int)

                for k in range(n_bins_len):
                    mask_k = k_all == k
                    if not np.any(mask_k):
                        continue
                    fit_k = bins_len[k] if k < len(bins_len) else None
                    if fit_k and fit_k.get("fit"):
                        d_non_k = non_d[mask_k]
                        pdf_non = _radius_fit_pdf(d_non_k, fit_k)
                        pdf_non = np.where(np.isfinite(pdf_non) & (pdf_non > 0), pdf_non, 0.0)
                        # “剩余长度分布” = base(non-overlap) - overlap 已占据分布（同厚度 bin）
                        ov_mask_k = k_ov == k
                        d_ov_k = overlap_d[ov_mask_k] if np.any(ov_mask_k) else np.array([], dtype=float)
                        if d_ov_k.size > 0 and np.any(pdf_non > 0):
                            # 在当前非overlap候选长度范围上估计 overlap 长度密度
                            # 用直方图估计并映射到每个候选长度点
                            d_min = float(np.min(d_non_k))
                            d_max = float(np.max(d_non_k))
                            if d_max > d_min:
                                bins_hist = np.linspace(d_min, d_max, 32)
                                c_ov, _ = np.histogram(d_ov_k, bins=bins_hist)
                                p_ov = c_ov.astype(float)
                                centers = 0.5 * (bins_hist[:-1] + bins_hist[1:])
                                # 残余分布基底使用 Phase2 总体 PDF（fit_k），不是当前候选 d_non_k 直方图
                                p_total = _radius_fit_pdf(centers, fit_k)
                                p_total = np.where(np.isfinite(p_total) & (p_total > 0), p_total, 0.0)
                                if p_total.sum() <= 0:
                                    continue
                                p_total /= p_total.sum()
                                if p_ov.sum() > 0:
                                    p_ov /= p_ov.sum()
                                # overlap 贡献按占总喉比例 alpha 扣除，而不是按 1:1 归一化后直接相减
                                alpha = float(n_overlap) / float(max(target_n_throats, 1))
                                alpha = float(np.clip(alpha, 0.0, 1.0))
                                p_res = p_total - alpha * p_ov
                                p_res[p_res < 0] = 0.0
                                if p_res.sum() > 0:
                                    p_res /= p_res.sum()
                                    # 将 bin 概率映射到样本点，作为残余修正因子
                                    idx_bin = np.digitize(d_non_k, bins_hist) - 1
                                    idx_bin = np.clip(idx_bin, 0, len(p_res) - 1)
                                    residual_factor = p_res[idx_bin]
                                    pdf_non = pdf_non * residual_factor
                        w_len[mask_k] = np.where(np.isfinite(pdf_non) & (pdf_non > 0), pdf_non, 0.0)

            # 度数权重：使用扣除 overlap 后的剩余度数
            # 你要求基础形式：w_deg = max(deg_rem_i, 0.025) * max(deg_rem_j, 0.025)
            deg_i = np.maximum(deg_rem[non_i].astype(float), 0.025)
            deg_j = np.maximum(deg_rem[non_j].astype(float), 0.025)
            w_deg = 0.8 * (deg_i * deg_j)

            # 为避免度数项压过长度项：先各自归一到均值~1，再给度数项指数降权
            mean_w_len = (
                float(np.mean(w_len[np.isfinite(w_len) & (w_len > 0)]))
                if np.any(np.isfinite(w_len) & (w_len > 0))
                else 1.0
            )
            mean_w_deg = (
                float(np.mean(w_deg[np.isfinite(w_deg) & (w_deg > 0)]))
                if np.any(np.isfinite(w_deg) & (w_deg > 0))
                else 1.0
            )
            w_len_n = w_len / max(mean_w_len, 1e-12)
            w_deg_n = w_deg / max(mean_w_deg, 1e-12)
            beta_deg = 0.45
            # 综合权重（不设下限）
            w = w_len_n * np.power(np.maximum(w_deg_n, 1e-12), beta_deg)
            valid_idx = np.where(np.isfinite(w) & (w > 0))[0]

            if len(valid_idx) == 0:
                extra_i = np.array([], dtype=int)
                extra_j = np.array([], dtype=int)
                print(
                    f"喉连接: 非 overlap 候选 {n_non_candidates} 条，但长度/度数联合权重均为 0，"
                    f"仅保留 overlap 喉 {n_overlap} 条。"
                )
            else:
                n_pick = min(n_extra_target, len(valid_idx))
                w_valid = w[valid_idx]
                p = w_valid / float(np.sum(w_valid))
                pick_local = rng.choice(len(valid_idx), size=n_pick, replace=False, p=p)
                pick_idx = valid_idx[pick_local]
                same_pick_idx = np.array([], dtype=int)
                cross_pick_idx = np.array([], dtype=int)
                pick_kind = ["nonoverlap"] * len(pick_idx)
                pick_cell1 = np.full(len(pick_idx), -1, dtype=int)
                pick_cell2 = np.full(len(pick_idx), -1, dtype=int)
                if (
                    throat_count_target_per_cell is not None
                    and getattr(self, "_pore_cell_index", None) is not None
                    and len(throat_count_target_per_cell) == len(self._cells)
                ):
                    pore_cell = np.asarray(self._pore_cell_index, dtype=int)
                    ci_all = pore_cell[non_i]
                    cj_all = pore_cell[non_j]
                    same_cell_mask_new = (ci_all == cj_all) & (ci_all >= 0)
                    cross_cell_mask = (ci_all != cj_all) & (ci_all >= 0) & (cj_all >= 0)
                    if nx_cell_for_cross is not None and ny_cell_for_cross is not None:
                        xi = ci_all % int(nx_cell_for_cross)
                        yi = ci_all // int(nx_cell_for_cross)
                        xj = cj_all % int(nx_cell_for_cross)
                        yj = cj_all // int(nx_cell_for_cross)
                        cross_cell_mask &= (np.abs(xi - xj) + np.abs(yi - yj)) == 1

                    overlap_same = np.zeros(len(throat_count_target_per_cell), dtype=int)
                    if n_overlap > 0:
                        ci_ov = pore_cell[overlap_i]
                        cj_ov = pore_cell[overlap_j]
                        m_ov_same = (ci_ov == cj_ov) & (ci_ov >= 0)
                        if np.any(m_ov_same):
                            np.add.at(overlap_same, ci_ov[m_ov_same], 1)
                    target_nonov_same = np.maximum(
                        throat_count_target_per_cell.astype(int) - overlap_same,
                        0,
                    )

                    valid_mask = np.zeros(n_non_candidates, dtype=bool)
                    valid_mask[valid_idx] = True
                    same_selected = []
                    same_idx_all_new = np.where(same_cell_mask_new & valid_mask)[0]
                    for c in range(len(throat_count_target_per_cell)):
                        idx_c = same_idx_all_new[ci_all[same_idx_all_new] == c]
                        if idx_c.size == 0:
                            continue
                        need = min(int(target_nonov_same[c]), int(idx_c.size))
                        if need <= 0:
                            continue
                        w_c = w[idx_c]
                        if np.any(np.isfinite(w_c) & (w_c > 0)):
                            p_c = w_c / float(np.sum(w_c))
                            chosen = rng.choice(idx_c, size=need, replace=False, p=p_c)
                        else:
                            chosen = rng.choice(idx_c, size=need, replace=False)
                        same_selected.append(np.asarray(chosen, dtype=int))
                    if same_selected:
                        same_pick_idx = np.concatenate(same_selected).astype(int)

                    cross_pool = np.where(cross_cell_mask & valid_mask)[0]
                    n_cross_target = min(int(max(0, n_nonoverlap_cross_est)), int(cross_pool.size))
                    if n_cross_target > 0:
                        w_cross = w[cross_pool]
                        if np.any(np.isfinite(w_cross) & (w_cross > 0)):
                            p_cross = w_cross / float(np.sum(w_cross))
                            cross_pick_idx = rng.choice(cross_pool, size=n_cross_target, replace=False, p=p_cross)
                        else:
                            cross_pick_idx = rng.choice(cross_pool, size=n_cross_target, replace=False)
                        cross_pick_idx = np.asarray(cross_pick_idx, dtype=int)

                    pick_idx = (
                        np.concatenate([same_pick_idx, cross_pick_idx]).astype(int)
                        if len(same_pick_idx) + len(cross_pick_idx) > 0
                        else np.array([], dtype=int)
                    )
                    pick_kind = (
                        ["same_cell_nonoverlap"] * len(same_pick_idx)
                        + ["cross_cell_nonoverlap"] * len(cross_pick_idx)
                    )
                    pick_cell1 = ci_all[pick_idx].astype(int) if len(pick_idx) > 0 else np.array([], dtype=int)
                    pick_cell2 = cj_all[pick_idx].astype(int) if len(pick_idx) > 0 else np.array([], dtype=int)
                    self._throat_target_debug.update(
                        {
                            "n_same_cell_nonoverlap_target": int(np.sum(target_nonov_same)),
                            "n_same_cell_nonoverlap_selected": int(len(same_pick_idx)),
                            "n_cross_cell_nonoverlap_target": int(max(0, n_nonoverlap_cross_est)),
                            "n_cross_cell_nonoverlap_selected": int(len(cross_pick_idx)),
                            "n_cross_cell_nonoverlap_candidates": int(cross_pool.size),
                        }
                    )
                    # This split selection replaces the older global correction block below.
                    throat_count_target_per_cell = None
                # 可选：cell内喉数量分布按 cell 目标回正（单个模拟网络内部）
                if (
                    throat_count_target_per_cell is not None
                    and getattr(self, "_pore_cell_index", None) is not None
                    and len(throat_count_target_per_cell) == len(self._cells)
                ):
                    sel_mask = np.zeros(n_non_candidates, dtype=bool)
                    sel_mask[pick_idx] = True
                    pore_cell = np.asarray(self._pore_cell_index, dtype=int)
                    ci_all = pore_cell[non_i]
                    same_cell_mask = (ci_all == pore_cell[non_j]) & (ci_all >= 0)

                    # overlap 已占用的 cell内喉数先扣掉
                    overlap_same = np.zeros(len(throat_count_target_per_cell), dtype=int)
                    if n_overlap > 0:
                        ci_ov = pore_cell[overlap_i]
                        cj_ov = pore_cell[overlap_j]
                        m_ov_same = (ci_ov == cj_ov) & (ci_ov >= 0)
                        if np.any(m_ov_same):
                            np.add.at(overlap_same, ci_ov[m_ov_same], 1)
                    target_nonov_same = np.maximum(
                        throat_count_target_per_cell.astype(int) - overlap_same,
                        0,
                    )

                    # 对每个 cell 在 same-cell 候选里做补删，使其更接近目标
                    same_idx_all = np.where(same_cell_mask)[0]
                    for c in range(len(throat_count_target_per_cell)):
                        idx_c = same_idx_all[ci_all[same_idx_all] == c]
                        if idx_c.size == 0:
                            continue
                        cur_idx = idx_c[sel_mask[idx_c]]
                        cur = int(cur_idx.size)
                        need = int(target_nonov_same[c])
                        if cur > need:
                            drop_n = cur - need
                            drop_idx = rng.choice(cur_idx, size=drop_n, replace=False)
                            sel_mask[drop_idx] = False
                        elif cur < need:
                            add_pool = idx_c[~sel_mask[idx_c]]
                            if add_pool.size > 0:
                                add_n = min(need - cur, int(add_pool.size))
                                add_idx = rng.choice(add_pool, size=add_n, replace=False)
                                sel_mask[add_idx] = True

                    pick_idx = np.where(sel_mask)[0]

                extra_i = non_i[pick_idx]
                extra_j = non_j[pick_idx]
                extra_kind = np.asarray(pick_kind, dtype=object)
                extra_cell1 = np.asarray(pick_cell1, dtype=int)
                extra_cell2 = np.asarray(pick_cell2, dtype=int)
                print(
                    f"喉连接: overlap 喉 {n_overlap} 条；非 overlap 候选 {n_non_candidates} 条，"
                    f"按长度+度数权重保留 {len(extra_i)} 条（目标 {n_extra_target}）。"
                )

        # ---------- 4) 汇总 ----------
        if n_overlap == 0 and len(extra_i) == 0:
            self.throat_pore1 = np.array([], dtype=int)
            self.throat_pore2 = np.array([], dtype=int)
            self._throat_cap_radii_geom = np.array([], dtype=float)
            print("喉连接: 最终无喉。")
            return

        all_i = np.concatenate([overlap_i, extra_i]) if n_overlap > 0 else extra_i
        all_j = np.concatenate([overlap_j, extra_j]) if n_overlap > 0 else extra_j
        if getattr(self, "_pore_cell_index", None) is not None:
            pore_cell_meta = np.asarray(self._pore_cell_index, dtype=int)
            overlap_cell1 = pore_cell_meta[overlap_i].astype(int) if n_overlap > 0 else np.array([], dtype=int)
            overlap_cell2 = pore_cell_meta[overlap_j].astype(int) if n_overlap > 0 else np.array([], dtype=int)
        else:
            overlap_cell1 = np.full(n_overlap, -1, dtype=int)
            overlap_cell2 = np.full(n_overlap, -1, dtype=int)
        overlap_kind = np.asarray(["overlap"] * n_overlap, dtype=object)
        all_kind = np.concatenate([overlap_kind, extra_kind]).astype(object) if n_overlap > 0 else np.asarray(extra_kind, dtype=object)
        all_cell1 = np.concatenate([overlap_cell1, extra_cell1]).astype(int) if n_overlap > 0 else np.asarray(extra_cell1, dtype=int)
        all_cell2 = np.concatenate([overlap_cell2, extra_cell2]).astype(int) if n_overlap > 0 else np.asarray(extra_cell2, dtype=int)

        cap_all = np.full(all_i.shape[0], np.nan, dtype=float)
        if n_overlap > 0:
            cap_all[:n_overlap] = overlap_cap

        self.throat_pore1 = np.array([self.pore_ids[i] for i in all_i], dtype=int)
        self.throat_pore2 = np.array([self.pore_ids[j] for j in all_j], dtype=int)
        self._throat_cap_radii_geom = cap_all
        self._throat_kind = all_kind
        self._throat_is_cross_cell = np.asarray(all_kind == "cross_cell_nonoverlap", dtype=bool)
        self._throat_cell1 = all_cell1
        self._throat_cell2 = all_cell2

        print(
            f"喉连接完成: overlap 喉 {n_overlap} 条，非 overlap 喉 {len(all_i) - n_overlap} 条，"
            f"总计 {len(all_i)} 条。"
        )

    

    def _connect_throats_fallback(self, max_distance: float, target_n_throats: int, rng: np.random.Generator):
        """无 Phase2 喉长/度数数据时的简单实现：距离内候选 + 均匀随机选至目标数。"""
        candidate_throats = []
        for i in range(len(self.pore_coords)):
            for j in range(i + 1, len(self.pore_coords)):
                dist = np.linalg.norm(self.pore_coords[i] - self.pore_coords[j])
                if dist < max_distance:
                    candidate_throats.append((i, j, dist))
        if len(candidate_throats) <= target_n_throats:
            selected = candidate_throats
        else:
            idx = rng.choice(len(candidate_throats), size=target_n_throats, replace=False)
            selected = [candidate_throats[k] for k in idx]
        self.throat_pore1 = np.array([self.pore_ids[i] for i, j, _ in selected], dtype=int)
        self.throat_pore2 = np.array([self.pore_ids[j] for i, j, _ in selected], dtype=int)
        n_sel = len(self.throat_pore1)
        self._throat_kind = np.asarray(["fallback"] * n_sel, dtype=object)
        self._throat_is_cross_cell = np.zeros(n_sel, dtype=bool)
        self._throat_cell1 = np.full(n_sel, -1, dtype=int)
        self._throat_cell2 = np.full(n_sel, -1, dtype=int)
        print(f"已连接 {len(self.throat_pore1)} 条喉（回退：均匀随机）")
    
    def assign_throat_radii(
        self,
        max_iterations: int = 50,
        w_thickness: float = 0.4,
        w_pore: float = 0.25,
        w_length: float = 0.03,
        scale_to_target_frac: bool = False,
    ):
        """
        3.5 喉半径赋值：

        - 若 connect_throats 已提供几何交界面圆半径 R_cap（_throat_cap_radii_geom）：
          - overlap：先 r_raw = R_cap×k（k 按 Phase2 R_throat/R_cap 厚度 bin；含小 R_cap 偏置）。
            若加载 throat_gt_min_pore_frac_per_subsample_by_thickness_bin.json：按 Phase2 厚度 bin 抽目标比例 f，
            允许约 n=f×N 条保留 r_raw>min(两端孔半径)，其余重抽 k 直至 r≤min(孔)−eps 或达 5 次，再回退。
            未加载该 JSON 时：每条喉直接重试直至满足上界（旧行为）。
          - 非 overlap：初值来自 Phase2 喉半径按厚度；先做「残余分布直方图」修正（与 overlap 半径联合匹配 Phase2 总喉径）；
            若加载 JSON 的 nonoverlap 段，再按厚度 bin 对 r>min(孔) 做保留 n=f×N + 其余重抽；最后做孔径/喉长相关性加权。

        - 统计写入 _gt_min_pore_frac_report（overlap 与可选 nonoverlap），供 plot_gt_min_pore_frac_raw_vs_phase2 与 Phase2 对比。
        """
        if len(self.throat_pore1) == 0:
            self.throat_radii = np.array([], dtype=float)
            print("喉半径赋值完成: 0 条喉")
            return

        n = len(self.throat_pore1)
        pore_id_to_idx = {int(pid): i for i, pid in enumerate(self.pore_ids)}
        idx1 = np.array([pore_id_to_idx[int(p1)] for p1 in self.throat_pore1], dtype=int)
        idx2 = np.array([pore_id_to_idx[int(p2)] for p2 in self.throat_pore2], dtype=int)

        # 后续统计统一使用孔心距，不再计算有效喉长 L_eff
        diff = self.pore_coords[idx2] - self.pore_coords[idx1]
        throat_lengths = np.linalg.norm(diff, axis=1)
        mean_pore_r = 0.5 * (self.pore_radii[idx1] + self.pore_radii[idx2])

        # 喉中点厚度（用于按厚度 bin 抽样）
        if getattr(self, "_cells", None) is not None and getattr(self, "_pore_cell_index", None) is not None:
            pore_thickness = np.array(
                [self._cells[self._pore_cell_index[i]]["T"] for i in range(len(self.pore_ids))],
                dtype=float,
            )
        elif getattr(self, "_thickness_interp", None) is not None:
            pore_thickness = np.maximum(
                self._thickness_interp(self.pore_coords[:, 0], self.pore_coords[:, 1], grid=False).ravel(),
                1.0,
            )
        else:
            pore_thickness = np.full(
                len(self.pore_ids),
                self.geometry_params.get("thickness_mean", 150.0),
                dtype=float,
            )
        T_mid = 0.5 * (pore_thickness[idx1] + pore_thickness[idx2])

        # 1) 先按 Phase2 喉半径分布 (by 厚度 bin) 给所有喉一个初始 r_pool
        radius_dist = self.radius_params.get("throat", {}).get(self.sample_type, {})
        mean_r_global = radius_dist.get("mean", 5.0)
        rng = np.random.default_rng(self._random_seed)

        r_pool = np.maximum(np.full(n, mean_r_global, dtype=float), 0.5)
        tr_by_t = self.throat_radius_dist_by_thickness_bin.get(self.sample_type, {})
        if tr_by_t and tr_by_t.get("bin_edges") and tr_by_t.get("bins"):
            bin_edges_t = np.array(tr_by_t["bin_edges"], dtype=float)
            bins_t = tr_by_t["bins"]
            n_bins_t = len(bin_edges_t) - 1
            k_t = np.searchsorted(bin_edges_t, T_mid, side="right") - 1
            k_t = np.clip(k_t, 0, n_bins_t - 1)
            for b in range(n_bins_t):
                mask = k_t == b
                if not np.any(mask):
                    continue
                bin_obj = bins_t[b] if b < len(bins_t) else {}
                if bin_obj.get("fit"):
                    n_b = int(np.sum(mask))
                    s = _sample_radius_from_fit(bin_obj, n_b, rng)
                    if s is not None and len(s) == n_b:
                        r_pool[mask] = np.maximum(s, 0.5)

        # 2) overlap：几何 R_cap×k 得 r_raw；若加载 Phase2「超小孔比例」JSON，则按厚度 bin 抽 f，
        #    允许 n=f×N 条保留 r_raw>min(孔)，其余重抽直至 r≤min(孔) 或达上限。
        #    非 overlap：r_pool → 残余分布直方图修正 →（可选）按 Phase2 nonoverlap 超小孔比例保留 n 条 + 重抽 → 相关性。
        throat_radii = r_pool.copy()
        gt_root = self.throat_gt_min_pore_frac_dist_by_thickness_bin or {}
        gt_ov_cfg = gt_root.get("overlap", {}).get(self.sample_type, {})
        gt_non_cfg = gt_root.get("nonoverlap", {}).get(self.sample_type, {})
        bin_edges_gt_ov = np.asarray(gt_ov_cfg.get("bin_edges", []), dtype=float)
        bins_gt_ov = gt_ov_cfg.get("bins", [])
        n_bins_gt_ov = len(bin_edges_gt_ov) - 1
        use_gt_ov = bool(gt_ov_cfg and n_bins_gt_ov > 0 and len(bins_gt_ov) > 0)
        bin_edges_gt_non = np.asarray(gt_non_cfg.get("bin_edges", []), dtype=float)
        bins_gt_non = gt_non_cfg.get("bins", [])
        n_bins_gt_non = len(bin_edges_gt_non) - 1
        use_gt_non = bool(gt_non_cfg and n_bins_gt_non > 0 and len(bins_gt_non) > 0)
        self._gt_min_pore_frac_report = {
            "sample_type": self.sample_type,
            "overlap": {"bins": []},
            "nonoverlap": {"bins": []},
        }

        if hasattr(self, "_throat_cap_radii_geom"):
            cap = np.asarray(self._throat_cap_radii_geom, dtype=float)
            if len(cap) == n:
                eps_nm = 1e-3
                mask_overlap = np.isfinite(cap) & (cap > 0.0)
                cap_ov = cap[mask_overlap]
                n_ov = int(cap_ov.size)
                if n_ov > 0:
                    idx1_ov = idx1[mask_overlap]
                    idx2_ov = idx2[mask_overlap]
                    r1_ov = self.pore_radii[idx1_ov]
                    r2_ov = self.pore_radii[idx2_ov]
                    T_ov = T_mid[mask_overlap]
                    mean_ov = 0.5 * (r1_ov + r2_ov)
                    min_ov = np.minimum(r1_ov, r2_ov)
                    r_max = np.maximum(min_ov - eps_nm, 1e-9)
                    mean_pore_median = float(np.median(mean_ov))

                    def _overlap_resample_one(jo: int, cap_ov_arr, min_ov_arr, r_max_arr, k_t_arr, bins_k_list, use_ratio_f: bool) -> float:
                        cap_j = float(cap_ov_arr[jo])
                        rmin_end_j = float(min_ov_arr[jo])
                        rmax_j = float(r_max_arr[jo])
                        b_k = int(k_t_arr[jo])
                        bin_obj = bins_k_list[b_k] if use_ratio_f and b_k < len(bins_k_list) else {}
                        bias = _overlap_k_bias_for_small_Rcap(cap_j, mean_pore_median)
                        for _attempt in range(5):
                            if use_ratio_f and bin_obj.get("fit"):
                                s = _sample_ratio_from_fit(bin_obj, 1, rng)
                                k_raw = float(s[0]) if s is not None and len(s) >= 1 else 1.0
                            else:
                                k_raw = 1.0
                            k_j = min(k_raw * bias, 20.0)
                            r_try = max(cap_j * k_j, 0.5)
                            if r_try <= rmax_j + 1e-12:
                                return float(r_try)
                        delta = float(rng.uniform(0.01, 0.5))
                        return float(min(max(rmin_end_j - delta, 0.5), rmax_j))

                    kcfg = self.overlap_throat_R_ratio_dist_by_thickness_bin.get(self.sample_type, {})
                    use_ratio = bool(
                        kcfg
                        and kcfg.get("bin_edges")
                        and kcfg.get("bins")
                    )
                    bin_edges_k = (
                        np.asarray(kcfg["bin_edges"], dtype=float)
                        if use_ratio
                        else np.array([], dtype=float)
                    )
                    bins_k = kcfg.get("bins", []) if use_ratio else []
                    n_bins_k = len(bin_edges_k) - 1
                    k_t = np.zeros(n_ov, dtype=int)
                    if use_ratio and n_bins_k > 0:
                        k_t = np.searchsorted(bin_edges_k, T_ov, side="right") - 1
                        k_t = np.clip(k_t, 0, max(n_bins_k - 1, 0))

                    if use_gt_ov:
                        k_t_gt_ov = np.searchsorted(bin_edges_gt_ov, T_ov, side="right") - 1
                        k_t_gt_ov = np.clip(k_t_gt_ov, 0, max(n_bins_gt_ov - 1, 0))
                        r_ov = np.empty(n_ov, dtype=float)
                        for j in range(n_ov):
                            cap_j = float(cap_ov[j])
                            b = int(k_t[j])
                            bin_obj = bins_k[b] if use_ratio and b < len(bins_k) else {}
                            bias = _overlap_k_bias_for_small_Rcap(cap_j, mean_pore_median)
                            if use_ratio and bin_obj.get("fit"):
                                s = _sample_ratio_from_fit(bin_obj, 1, rng)
                                k_raw = float(s[0]) if s is not None and len(s) >= 1 else 1.0
                            else:
                                k_raw = 1.0
                            k_j = min(k_raw * bias, 20.0)
                            r_ov[j] = max(cap_j * k_j, 0.5)

                        for b_gt in range(n_bins_gt_ov):
                            loc = np.where(k_t_gt_ov == b_gt)[0]
                            if loc.size == 0:
                                continue
                            N_b = int(loc.size)
                            bin_obj_gt = bins_gt_ov[b_gt] if b_gt < len(bins_gt_ov) else {}
                            f_b = _sample_frac_metric_from_bin(bin_obj_gt, rng) if bin_obj_gt.get("fit") else 0.0
                            n_allow = int(np.round(f_b * N_b))
                            n_allow = int(np.clip(n_allow, 0, N_b))
                            r_sub = r_ov[loc]
                            min_sub = min_ov[loc]
                            viol_mask = r_sub > min_sub
                            viol_local = np.where(viol_mask)[0]
                            n_viol = int(viol_local.size)
                            frac_raw = float(n_viol / N_b) if N_b > 0 else 0.0
                            n_keep = min(n_allow, n_viol)
                            if n_viol > 0 and n_keep < n_viol:
                                keep_ii = rng.choice(viol_local, size=n_keep, replace=False)
                            else:
                                keep_ii = viol_local
                            keep_set = set(int(x) for x in np.atleast_1d(keep_ii).tolist())
                            t_c = float((bin_edges_gt_ov[b_gt] + bin_edges_gt_ov[b_gt + 1]) * 0.5)
                            self._gt_min_pore_frac_report["overlap"]["bins"].append(
                                {
                                    "t_center": t_c,
                                    "t_min": float(bin_edges_gt_ov[b_gt]),
                                    "t_max": float(bin_edges_gt_ov[b_gt + 1]),
                                    "N": N_b,
                                    "f_sampled": float(f_b),
                                    "n_allow": n_allow,
                                    "n_violators_raw": n_viol,
                                    "frac_raw": frac_raw,
                                    "n_kept_violators": int(len(keep_set)),
                                }
                            )
                            for ii in range(loc.size):
                                jo = int(loc[ii])
                                if not viol_mask[ii]:
                                    continue
                                if ii in keep_set:
                                    continue
                                r_ov[jo] = _overlap_resample_one(
                                    jo, cap_ov, min_ov, r_max, k_t, bins_k, use_ratio
                                )

                        throat_radii[mask_overlap] = r_ov
                    else:
                        r_ov = np.empty(n_ov, dtype=float)
                        for j in range(n_ov):
                            cap_j = float(cap_ov[j])
                            rmin_end_j = float(min_ov[j])
                            rmax_j = float(r_max[j])
                            b = int(k_t[j])
                            bin_obj = bins_k[b] if use_ratio and b < len(bins_k) else {}
                            bias = _overlap_k_bias_for_small_Rcap(cap_j, mean_pore_median)
                            placed = False
                            for _attempt in range(5):
                                if use_ratio and bin_obj.get("fit"):
                                    s = _sample_ratio_from_fit(bin_obj, 1, rng)
                                    k_raw = float(s[0]) if s is not None and len(s) >= 1 else 1.0
                                else:
                                    k_raw = 1.0
                                k_j = min(k_raw * bias, 20.0)
                                r_try = max(cap_j * k_j, 0.5)
                                if r_try <= rmax_j + 1e-12:
                                    r_ov[j] = r_try
                                    placed = True
                                    break
                            if not placed:
                                delta = float(rng.uniform(0.01, 0.5))
                                r_ov[j] = min(max(rmin_end_j - delta, 0.5), rmax_j)

                        throat_radii[mask_overlap] = r_ov

                non_idx = np.where(~mask_overlap)[0]
                if non_idx.size > 0:
                    r_non = throat_radii[non_idx].copy()
                    L_non = throat_lengths[non_idx]
                    mp_non = mean_pore_r[non_idx]
                    T_non = T_mid[non_idx]
                    r_ov = throat_radii[mask_overlap]
                    T_ov = T_mid[mask_overlap]

                    tr_by_t2 = self.throat_radius_dist_by_thickness_bin.get(self.sample_type, {})
                    bin_edges_t2 = (
                        np.array(tr_by_t2["bin_edges"], dtype=float)
                        if (tr_by_t2 and tr_by_t2.get("bin_edges"))
                        else np.array([], dtype=float)
                    )
                    bins_t2 = tr_by_t2.get("bins", []) if tr_by_t2 else []
                    n_bins_t2 = len(bin_edges_t2) - 1

                    if tr_by_t2 and tr_by_t2.get("bin_edges") and tr_by_t2.get("bins"):
                        k_non = np.searchsorted(bin_edges_t2, T_non, side="right") - 1
                        k_non = np.clip(k_non, 0, n_bins_t2 - 1)
                        if T_ov.size > 0:
                            k_ov = np.searchsorted(bin_edges_t2, T_ov, side="right") - 1
                            k_ov = np.clip(k_ov, 0, n_bins_t2 - 1)
                        else:
                            k_ov = np.array([], dtype=int)

                        for b in range(n_bins_t2):
                            m_non = np.where(k_non == b)[0]
                            if m_non.size == 0:
                                continue
                            vals = r_non[m_non]
                            if vals.size < 2:
                                continue
                            m_ov = np.where(k_ov == b)[0] if k_ov.size > 0 else np.array([], dtype=int)
                            if m_ov.size == 0:
                                continue
                            ov_vals = r_ov[m_ov]
                            vmin = float(np.min(vals))
                            vmax = float(np.max(vals))
                            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                                continue
                            bins_hist = np.linspace(vmin, vmax, 32)
                            c_ov, _ = np.histogram(ov_vals, bins=bins_hist)
                            p_ov = c_ov.astype(float)
                            centers = 0.5 * (bins_hist[:-1] + bins_hist[1:])
                            fit_b = None
                            if b < len(bins_t2):
                                fit_b = bins_t2[b]
                            p_total = _radius_fit_pdf(centers, fit_b) if (fit_b and fit_b.get("fit")) else np.ones_like(centers)
                            p_total = np.where(np.isfinite(p_total) & (p_total > 0), p_total, 0.0)
                            if p_total.sum() <= 0:
                                continue
                            p_total /= p_total.sum()
                            if p_ov.sum() > 0:
                                p_ov /= p_ov.sum()
                            alpha_r = float(np.sum(mask_overlap)) / float(max(n, 1))
                            alpha_r = float(np.clip(alpha_r, 0.0, 1.0))
                            p_res = p_total - alpha_r * p_ov
                            p_res[p_res < 0] = 0.0
                            if p_res.sum() <= 0:
                                continue
                            p_res /= p_res.sum()
                            bid = np.digitize(vals, bins_hist) - 1
                            bid = np.clip(bid, 0, len(p_res) - 1)
                            w_res = p_res[bid]
                            if np.sum(w_res) > 0:
                                sel = rng.choice(len(vals), size=len(vals), replace=True, p=w_res / np.sum(w_res))
                                r_non[m_non] = vals[sel]

                    # 非 overlap：在残余分布修正之后，按 Phase2「超小孔比例」各厚度 bin 抽 f，保留约 n=f×N 条 r>min(孔)，其余重抽
                    if use_gt_non and n_bins_t2 > 0:
                        idx1_n = idx1[non_idx]
                        idx2_n = idx2[non_idx]
                        min_nn = np.minimum(self.pore_radii[idx1_n], self.pore_radii[idx2_n])
                        k_gt_non = np.searchsorted(bin_edges_gt_non, T_non, side="right") - 1
                        k_gt_non = np.clip(k_gt_non, 0, max(n_bins_gt_non - 1, 0))

                        for b_gt in range(n_bins_gt_non):
                            loc = np.where(k_gt_non == b_gt)[0]
                            if loc.size == 0:
                                continue
                            N_b = int(loc.size)
                            bin_obj_gt = bins_gt_non[b_gt] if b_gt < len(bins_gt_non) else {}
                            f_b = _sample_frac_metric_from_bin(bin_obj_gt, rng) if bin_obj_gt.get("fit") else 0.0
                            n_allow = int(np.round(f_b * N_b))
                            n_allow = int(np.clip(n_allow, 0, N_b))
                            r_sub = r_non[loc]
                            min_sub = min_nn[loc]
                            viol_mask = r_sub > min_sub
                            viol_local = np.where(viol_mask)[0]
                            n_viol = int(viol_local.size)
                            frac_raw = float(n_viol / N_b) if N_b > 0 else 0.0
                            n_keep = min(n_allow, n_viol)
                            if n_viol > 0 and n_keep < n_viol:
                                keep_ii = rng.choice(viol_local, size=n_keep, replace=False)
                            else:
                                keep_ii = viol_local
                            keep_set = set(int(x) for x in np.atleast_1d(keep_ii).tolist())
                            t_c = float((bin_edges_gt_non[b_gt] + bin_edges_gt_non[b_gt + 1]) * 0.5)
                            self._gt_min_pore_frac_report["nonoverlap"]["bins"].append(
                                {
                                    "t_center": t_c,
                                    "t_min": float(bin_edges_gt_non[b_gt]),
                                    "t_max": float(bin_edges_gt_non[b_gt + 1]),
                                    "N": N_b,
                                    "f_sampled": float(f_b),
                                    "n_allow": n_allow,
                                    "n_violators_raw": n_viol,
                                    "frac_raw": frac_raw,
                                    "n_kept_violators": int(len(keep_set)),
                                }
                            )
                            for ii in range(loc.size):
                                i_loc = int(loc[ii])
                                if not viol_mask[ii]:
                                    continue
                                if ii in keep_set:
                                    continue
                                Tg = float(T_non[i_loc])
                                kb = int(np.searchsorted(bin_edges_t2, Tg, side="right") - 1)
                                kb = int(np.clip(kb, 0, max(n_bins_t2 - 1, 0)))
                                bin_obj_tr = bins_t2[kb] if kb < len(bins_t2) else {}
                                placed = False
                                for _attempt in range(5):
                                    if bin_obj_tr.get("fit"):
                                        s = _sample_radius_from_fit(bin_obj_tr, 1, rng)
                                        r_try = float(s[0]) if s is not None and len(s) >= 1 else r_non[i_loc]
                                    else:
                                        r_try = float(max(mean_r_global, 0.5))
                                    r_try = max(r_try, 0.5)
                                    if r_try <= float(min_nn[i_loc]) - eps_nm + 1e-12:
                                        r_non[i_loc] = r_try
                                        placed = True
                                        break
                                if not placed:
                                    delta = float(rng.uniform(0.01, 0.5))
                                    r_non[i_loc] = float(
                                        min(max(float(min_nn[i_loc]) - delta, 0.5), float(min_nn[i_loc]) - eps_nm)
                                    )

                    z_p = np.clip((mp_non - np.mean(mp_non)) / (np.std(mp_non) + 1e-8), -2.5, 2.5)
                    z_l = np.clip((L_non - np.mean(L_non)) / (np.std(L_non) + 1e-8), -2.5, 2.5)
                    score = 0.1 * ((w_pore * z_p) - (w_length * z_l))
                    order = np.arange(non_idx.size, dtype=int)
                    rng.shuffle(order)
                    remaining = list(np.asarray(r_non, dtype=float))
                    assigned = np.full(non_idx.size, np.nan, dtype=float)
                    gamma = 0.45
                    for pos in order:
                        if len(remaining) == 0:
                            break
                        arr = np.asarray(remaining, dtype=float)
                        mu = float(np.mean(arr))
                        sd = float(np.std(arr) + 1e-8)
                        w_corr = np.exp(gamma * score[pos] * (arr - mu) / sd)
                        w_corr = np.where(np.isfinite(w_corr) & (w_corr > 0), w_corr, 0.0)
                        if np.sum(w_corr) <= 0:
                            w_corr = np.ones_like(arr) / len(arr)
                        else:
                            w_corr = w_corr / np.sum(w_corr)
                        pick = int(rng.choice(len(arr), size=1, p=w_corr)[0])
                        assigned[pos] = arr[pick]
                        remaining.pop(pick)
                    if np.any(~np.isfinite(assigned)):
                        fallback = (
                            float(np.nanmean(r_non)) if np.any(np.isfinite(r_non)) else max(mean_r_global, 0.5)
                        )
                        assigned[~np.isfinite(assigned)] = fallback
                    throat_radii[non_idx] = np.maximum(assigned, 0.5)

        self.throat_radii = np.maximum(throat_radii, 0.5)

        # 简单报告体积分数（不做强制缩放）
        throat_volume = np.sum(np.pi * np.maximum(self.throat_radii, 0.0) ** 2 * np.maximum(throat_lengths, 1e-6))
        current_frac = throat_volume / (float(self.volume) + 1e-20)
        print(
            f"喉半径赋值完成: {len(self.throat_radii)} 条喉, "
            f"体积分数(基于孔心距) ~= {current_frac:.4f} "
            f"(overlap: R_cap×k，若加载 JSON overlap 段则按厚度保留部分超限后重抽；"
            f"非 overlap: 残余直方图 → 可选 JSON nonoverlap 超上限策略 → 相关性)。"
        )

    def plot_gt_min_pore_frac_raw_vs_phase2(self, output_dir: Path) -> list[Path]:
        """
        对比本次合成中「初值超限比例」frac_raw 与 Phase2 拟合（norm 的 mean±std），
        overlap / nonoverlap 各一张图；并保存本 run 的 assignment 报告 JSON。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rep = getattr(self, "_gt_min_pore_frac_report", None)
        gt_root = self.throat_gt_min_pore_frac_dist_by_thickness_bin or {}
        paths: list[Path] = []

        if rep:
            with open(output_dir / "gt_min_pore_frac_synthetic_report.json", "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2, ensure_ascii=False)

        def _phase2_mean_std_for_bin(ph_bins: list, t_min: float, t_max: float) -> tuple[float | None, float | None]:
            for pb in ph_bins:
                if abs(float(pb.get("t_min", 0)) - t_min) < 1e-5 and abs(float(pb.get("t_max", 0)) - t_max) < 1e-5:
                    fit = pb.get("fit") or {}
                    dist = fit.get("distribution") or {}
                    if dist.get("name") == "norm" and len(dist.get("params", [])) >= 2:
                        return float(dist["params"][0]), float(dist["params"][1])
                    return float(fit.get("mean", np.nan)), float(fit.get("std", np.nan))
            return None, None

        for kind in ("overlap", "nonoverlap"):
            sim_bins = (rep or {}).get(kind, {}).get("bins", [])
            ph2 = gt_root.get(kind, {}).get(self.sample_type, {})
            ph_bins = ph2.get("bins", [])
            if not sim_bins:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            t_c = np.array([b["t_center"] for b in sim_bins], dtype=float)
            frac_raw = np.array([b["frac_raw"] for b in sim_bins], dtype=float)
            f_samp = np.array([b["f_sampled"] for b in sim_bins], dtype=float)
            means = []
            stds = []
            for b in sim_bins:
                m, s = _phase2_mean_std_for_bin(ph_bins, float(b["t_min"]), float(b["t_max"]))
                means.append(m if m is not None else np.nan)
                stds.append(s if s is not None else np.nan)
            means = np.asarray(means, dtype=float)
            stds = np.asarray(stds, dtype=float)

            ax0 = axes[0]
            ax0.scatter(t_c, frac_raw, s=45, color="steelblue", label="Synthetic raw frac (violators/N)", zorder=3)
            ax0.scatter(t_c, f_samp, s=30, marker="x", color="darkorange", label="Sampled f (target allow)", zorder=3)
            valid = np.isfinite(means) & np.isfinite(stds)
            if np.any(valid):
                ax0.errorbar(
                    t_c[valid],
                    means[valid],
                    yerr=stds[valid],
                    fmt="none",
                    ecolor="crimson",
                    capsize=3,
                    label="Phase2 norm mean ± std",
                )
                ax0.plot(t_c[valid], means[valid], "r--", alpha=0.6, linewidth=1)
            ax0.set_ylim(0, 1.05)
            ax0.set_xlabel("Thickness (nm, bin center)")
            ax0.set_ylabel("Fraction in [0,1]")
            ax0.set_title(f"{kind}: raw violation fraction vs Phase2 ({self.sample_type})")
            ax0.legend(loc="best", fontsize=8)
            ax0.grid(True, alpha=0.3)

            ax1 = axes[1]
            x = np.arange(len(sim_bins))
            w = 0.35
            ax1.bar(x - w / 2, frac_raw, width=w, label="frac_raw", color="steelblue", alpha=0.8)
            ax1.bar(x + w / 2, np.nan_to_num(means, nan=0.0), width=w, label="Phase2 mean", color="coral", alpha=0.8)
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"{b['t_min']:.0f}–{b['t_max']:.0f}" for b in sim_bins], rotation=45, ha="right", fontsize=7)
            ax1.set_ylabel("Fraction")
            ax1.set_title(f"{kind}: bar comparison per thickness bin")
            ax1.legend(fontsize=8)
            ax1.grid(True, axis="y", alpha=0.3)

            plt.suptitle(
                f"GT min-pore exceedance: synthetic vs Phase2 ({kind}, {self.sample_type})",
                fontsize=12,
                fontweight="bold",
            )
            plt.tight_layout()
            outp = output_dir / f"gt_min_pore_frac_raw_vs_phase2_{kind}_{self.sample_type}.png"
            plt.savefig(outp, dpi=150, bbox_inches="tight")
            plt.close()
            paths.append(outp)
            print(f"  已保存: {outp}")

        return paths
    
    def export_to_xlsx(self, output_path: Path, sample_name: str = "synthetic"):
        """
        3.6 导出与筛过流程对接
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 导出孔表
        pore_radii = np.asarray(self.pore_radii, dtype=float)
        pore_volumes = (4.0 / 3.0) * np.pi * np.maximum(pore_radii, 0.0) ** 3
        pores_df = pd.DataFrame({
            "Pore ID": self.pore_ids,
            "Volume": pore_volumes,
            "EqRadius": pore_radii,
            "X Coord": self.pore_coords[:, 0],
            "Y Coord": self.pore_coords[:, 1],
            "Z Coord": self.pore_coords[:, 2],
        })
        pores_file = output_path / f"{sample_name}_pores.xlsx"
        pores_df.to_excel(pores_file, index=False)
        
        # 导出喉表（Length = |p1 - p2|）
        throat_ids = np.arange(len(self.throat_pore1), dtype=int)
        throat_lengths = []
        pore_id_to_idx = {int(pid): i for i, pid in enumerate(self.pore_ids)}
        throat_radii = np.asarray(self.throat_radii, dtype=float)
        for p1_id, p2_id in zip(self.throat_pore1, self.throat_pore2):
            i1 = pore_id_to_idx[int(p1_id)]
            i2 = pore_id_to_idx[int(p2_id)]
            d = float(np.linalg.norm(self.pore_coords[i2] - self.pore_coords[i1]))
            throat_lengths.append(max(d, 1e-6))
        
        throats_df = pd.DataFrame({
            "Throat ID": throat_ids,
            "Pore ID #1": self.throat_pore1,
            "Pore ID #2": self.throat_pore2,
            "Length": throat_lengths,
            "EqRadius": throat_radii,
        })
        n_throats_export = len(throats_df)
        if len(getattr(self, "_throat_kind", [])) == n_throats_export:
            throats_df["Throat Kind"] = np.asarray(self._throat_kind, dtype=object)
        if len(getattr(self, "_throat_is_cross_cell", [])) == n_throats_export:
            throats_df["Is Cross Cell"] = np.asarray(self._throat_is_cross_cell, dtype=bool)
        if len(getattr(self, "_throat_cell1", [])) == n_throats_export:
            throats_df["Cell ID #1"] = np.asarray(self._throat_cell1, dtype=int)
        if len(getattr(self, "_throat_cell2", [])) == n_throats_export:
            throats_df["Cell ID #2"] = np.asarray(self._throat_cell2, dtype=int)
        throats_file = output_path / f"{sample_name}_throats.xlsx"
        throats_df.to_excel(throats_file, index=False)
        
        print(f"孔/喉表已导出到: {output_path}")
    
    def plot_thickness_field_vs_gmm(self, output_dir: Path) -> Path:
        """
        将模拟厚度场分布与 Phase1 GMM 曲线画在同一张图上对比。
        仅在存在厚度场时有效。返回保存的图片路径。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if getattr(self, "thickness_field_grid", None) is None:
            print("未使用厚度场，跳过厚度分布对比图")
            return None
        gmm = self.thickness_distribution.get(self.sample_type) if self.thickness_distribution else None
        if not gmm or gmm.get("type") != "gmm":
            print("无 GMM 参数，跳过厚度分布对比图")
            return None

        T_flat = self.thickness_field_grid.ravel()
        T_flat = T_flat[np.isfinite(T_flat)]
        if len(T_flat) == 0:
            return None

        vals = T_flat
        t_min, t_max = float(np.min(vals)), float(np.max(vals))
        x_plot = np.linspace(max(1.0, t_min - 10), t_max + 10, 300)
        gmm_pdf = _gmm_pdf(x_plot, gmm)

        fig, ax = plt.subplots(figsize=(8, 5))
        n_bins_hist = min(60, max(20, len(vals) // 100))
        # 直方图：柱高为概率（每个厚度 bin 内网格点占比）
        ax.hist(
            vals,
            bins=n_bins_hist,
            weights=np.ones_like(vals) / len(vals),
            alpha=0.6,
            color="steelblue",
            edgecolor="white",
            label="Simulated thickness field (grid)",
        )
        # GMM PDF 转为概率曲线：P(bin) ≈ PDF(x) * bin_width
        bin_width_approx = (vals.max() - vals.min()) / n_bins_hist if n_bins_hist and vals.max() > vals.min() else 1e-6
        ax.plot(
            x_plot,
            gmm_pdf * bin_width_approx,
            "r-",
            linewidth=2,
            label="Phase1 GMM",
        )
        ax.set_xlabel("Thickness (nm)", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_title(f"Thickness: simulated vs GMM ({self.sample_type})", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = output_dir / f"thickness_field_vs_gmm_{self.sample_type}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"厚度场 vs GMM 对比图已保存: {plot_path}")
        return plot_path

    def plot_pore_count_fraction_simulated_vs_target(self, output_dir: Path) -> Path:
        """
        绘制撒点后各厚度区间内「孔数量分数」(孔数/体积) 的分布，以及 Phase2 目标分数曲线。
        每个厚度区间一个子图，展示该区间内各格孔数量分数 = n_cell/V_cell 的分布；另有一图汇总模拟 vs 目标分数。
        区间划分与 Phase2 一致。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        data = getattr(self, "_pore_count_fraction_simulated", None)
        if not data:
            print("无孔数量分数记录，跳过孔数量分数对比图")
            return None

        bin_edges = np.array(data["bin_edges"])
        rho_cell_per_bin = data.get("rho_cell_per_bin", [])
        frac_sim = np.array(data["fraction_per_bin"])
        frac_target = data.get("target_fraction_per_bin")
        n_bins = len(bin_edges) - 1
        if n_bins <= 0:
            return None

        # 图1：每个厚度区间一个子图，展示该区间内各格孔数量分数 (孔数/体积) 的分布
        bins_with_data = [i for i in range(n_bins) if len(rho_cell_per_bin[i]) > 0]
        if not bins_with_data:
            return None
        rho_dist = self.rho_pore_dist_by_thickness_bin.get(self.sample_type, {})
        bins_fit = rho_dist.get("bins", [])
        n_panels = len(bins_with_data)
        n_cols = min(4, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = np.atleast_2d(axes)
        axes_flat = axes.flatten()
        for idx, k in enumerate(bins_with_data):
            ax = axes_flat[idx]
            vals = np.array(rho_cell_per_bin[k])
            n_bins_hist = min(25, max(5, len(vals) // 3))
            # 柱高 = 每格概率 P(bin) = count/total（与 Phase2 density_frac 一致）
            ax.hist(vals, bins=n_bins_hist, weights=np.ones_like(vals) / len(vals),
                    color="steelblue", alpha=0.8, edgecolor="white", label="Simulated")
            if k < len(bins_fit) and bins_fit[k].get("fit"):
                x_plot = np.linspace(max(1e-10, vals.min() * 0.9), vals.max() * 1.1, 200)
                pdf_vals = _fit_pdf(x_plot, bins_fit[k])
                if np.any(pdf_vals > 0):
                    # PDF 为密度，转为概率：P(bin) ≈ PDF(x)*bin_width（与 Phase2 一致）
                    bin_width_approx = (vals.max() - vals.min()) / n_bins_hist if n_bins_hist and vals.max() > vals.min() else 1e-6
                    ax.plot(x_plot, pdf_vals * bin_width_approx, "r-", linewidth=2, label="Phase2 fit")
            ax.set_xlabel("Pore count fraction (1/nm^3)", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            ax.set_title(f"T in [{bin_edges[k]:.0f},{bin_edges[k+1]:.0f}) nm (n={len(vals)})", fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        for idx in range(len(bins_with_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.suptitle(f"Pore count fraction by thickness bin ({self.sample_type})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plot_path1 = output_dir / f"pore_count_fraction_per_bin_dist_{self.sample_type}.png"
        plt.savefig(plot_path1, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"孔数量分数分布图已保存: {plot_path1}")

        # 图2：汇总 模拟 vs Phase2 目标分数（柱状 + 曲线）
        labels = [f"[{bin_edges[i]:.0f},{bin_edges[i+1]:.0f})" for i in range(n_bins)]
        x_pos = np.arange(n_bins)
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, n_bins * 0.8), 5))
        ax.bar(x_pos - width / 2, frac_sim, width, label="Simulated pore count fraction", color="steelblue", alpha=0.8, edgecolor="white")
        if frac_target is not None and len(frac_target) == n_bins:
            ax.plot(x_pos, frac_target, "ro-", linewidth=2, markersize=8, label="Phase2 target")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Thickness bin (nm)", fontsize=12)
        ax.set_ylabel("Pore count fraction", fontsize=12)
        ax.set_title(f"Pore count fraction: simulated vs Phase2 target ({self.sample_type})", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        plot_path2 = output_dir / f"pore_count_fraction_simulated_vs_target_{self.sample_type}.png"
        plt.savefig(plot_path2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"孔数量分数汇总图已保存: {plot_path2}")
        return plot_path2

    def _compute_same_cell_throat_count_per_cell(self) -> np.ndarray | None:
        """统计每个 cell 内的喉数（仅两端孔都在同一 cell 的喉）。"""
        if getattr(self, "_cells", None) is None or getattr(self, "_pore_cell_index", None) is None:
            return None
        if self.throat_pore1 is None or self.throat_pore2 is None:
            return None
        n_cells = len(self._cells)
        out = np.zeros(n_cells, dtype=int)
        if n_cells <= 0:
            return out
        pore_id_to_idx = {int(pid): i for i, pid in enumerate(self.pore_ids)}
        for p1_id, p2_id in zip(self.throat_pore1, self.throat_pore2):
            i = pore_id_to_idx.get(int(p1_id))
            j = pore_id_to_idx.get(int(p2_id))
            if i is None or j is None:
                continue
            ci = int(self._pore_cell_index[i])
            cj = int(self._pore_cell_index[j])
            if ci == cj and 0 <= ci < n_cells:
                out[ci] += 1
        return out

    def _plot_rho_sampling_panel_by_bin(
        self,
        *,
        values: np.ndarray,
        bin_index: np.ndarray,
        n_per_cell: np.ndarray,
        bin_edges: np.ndarray,
        bins_fit: list,
        metric_label: str,
        figure_label: str,
        output_path: Path,
    ) -> Path | None:
        """将每个 thickness bin 画成一个子图，并叠加 Phase2 拟合曲线。"""
        n_bins = len(bin_edges) - 1
        if n_bins <= 0:
            return None
        bins_with_data = [k for k in range(n_bins) if np.any(bin_index == k)]
        if not bins_with_data:
            return None

        n_panels = len(bins_with_data)
        n_cols = min(4, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.6 * n_rows))
        axes_flat = np.atleast_2d(axes).flatten()

        for pi, k in enumerate(bins_with_data):
            ax = axes_flat[pi]
            idx_k = np.where(bin_index == k)[0]
            vals_k = np.asarray(values[idx_k], dtype=float)
            vals_k = vals_k[np.isfinite(vals_k)]
            if vals_k.size == 0:
                ax.set_visible(False)
                continue
            zero_rate = float(np.mean(np.asarray(n_per_cell[idx_k], dtype=float) <= 0.0)) if idx_k.size > 0 else float("nan")
            x_min = float(np.min(vals_k))
            x_max = float(np.max(vals_k))
            if not np.isfinite(x_min) or not np.isfinite(x_max):
                ax.set_visible(False)
                continue
            if x_max <= x_min:
                x_min = max(0.0, x_min * 0.9)
                x_max = max(x_min + 1e-10, x_min * 1.1 + 1e-10)
            n_hist = min(25, max(5, len(vals_k) // 3))
            ax.hist(
                vals_k,
                bins=n_hist,
                range=(x_min, x_max),
                weights=np.ones_like(vals_k) / len(vals_k),
                color="steelblue",
                alpha=0.82,
                edgecolor="white",
                label="Simulated",
            )
            if k < len(bins_fit) and isinstance(bins_fit[k], dict) and bins_fit[k].get("fit"):
                x_plot = np.linspace(max(1e-12, x_min), x_max, 220)
                pdf_vals = _fit_pdf(x_plot, bins_fit[k])
                if np.any(pdf_vals > 0):
                    bin_w = (x_max - x_min) / n_hist if n_hist > 0 and x_max > x_min else 1e-10
                    ax.plot(x_plot, pdf_vals * bin_w, "r-", linewidth=2, label="Phase2 fit")
            ax.set_xlabel(f"{metric_label} (1/nm^3)", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            ax.set_title(
                f"T in [{bin_edges[k]:.0f},{bin_edges[k+1]:.0f}) nm "
                f"(n={len(vals_k)}, zero_rate={zero_rate:.2f})",
                fontsize=9.5,
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        for pi in range(len(bins_with_data), len(axes_flat)):
            axes_flat[pi].set_visible(False)

        plt.suptitle(
            f"{self.sample_type} {metric_label} by thickness bin - {figure_label}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"{metric_label} {figure_label} 图已保存: {output_path}")
        return output_path

    def plot_sampling_rho_debug_by_thickness_bins(self, output_dir: Path) -> list[Path]:
        """
        采样诊断图：
        - 图A: raw sampled rho
        - 图B: realized rho (= n_generated / V_cell)
        每个 thickness bin 一个子图，并叠加 Phase2 拟合曲线。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_paths: list[Path] = []
        debug_report: dict[str, dict] = {"sample_type": self.sample_type}

        # ---------- 孔 ----------
        pore_dbg = getattr(self, "_pore_rho_sampling_debug", None)
        pore_dist = (self.rho_pore_dist_by_thickness_bin or {}).get(self.sample_type, {})
        if isinstance(pore_dbg, dict) and pore_dist:
            bin_edges = np.asarray(pore_dbg.get("bin_edges", []), dtype=float)
            bins_fit = pore_dist.get("bins", []) or []
            k_cell = np.asarray(pore_dbg.get("bin_index_per_cell", []), dtype=int)
            V_cell = np.asarray(pore_dbg.get("V_per_cell", []), dtype=float)
            rho_raw = np.asarray(pore_dbg.get("rho_raw_per_cell", []), dtype=float)
            n_after = np.asarray(pore_dbg.get("n_after_corr_per_cell", []), dtype=float)
            if (
                len(bin_edges) >= 2
                and len(k_cell) == len(V_cell) == len(rho_raw) == len(n_after)
                and len(k_cell) > 0
            ):
                rho_realized = np.divide(n_after, np.maximum(V_cell, 1e-18))
                pA = output_dir / f"rho_sampling_debug_raw_pore_{self.sample_type}.png"
                pB = output_dir / f"rho_sampling_debug_realized_pore_{self.sample_type}.png"
                x = self._plot_rho_sampling_panel_by_bin(
                    values=rho_raw,
                    bin_index=k_cell,
                    n_per_cell=n_after,
                    bin_edges=bin_edges,
                    bins_fit=bins_fit,
                    metric_label="Pore count fraction",
                    figure_label="Figure A (raw sampled)",
                    output_path=pA,
                )
                y = self._plot_rho_sampling_panel_by_bin(
                    values=rho_realized,
                    bin_index=k_cell,
                    n_per_cell=n_after,
                    bin_edges=bin_edges,
                    bins_fit=bins_fit,
                    metric_label="Pore count fraction",
                    figure_label="Figure B (realized n/V)",
                    output_path=pB,
                )
                if x is not None:
                    out_paths.append(x)
                if y is not None:
                    out_paths.append(y)
                debug_report["pore"] = {
                    "bin_edges": bin_edges.tolist(),
                    "n_cells": int(len(k_cell)),
                    "raw_mean": float(np.nanmean(rho_raw)) if len(rho_raw) > 0 else float("nan"),
                    "realized_mean": float(np.nanmean(rho_realized)) if len(rho_realized) > 0 else float("nan"),
                }

        # ---------- 喉 ----------
        throat_dbg = getattr(self, "_throat_rho_sampling_debug", None)
        throat_dist = (self.rho_throat_dist_by_thickness_bin or {}).get(self.sample_type, {})
        if isinstance(throat_dbg, dict) and throat_dist:
            bin_edges = np.asarray(throat_dbg.get("bin_edges", []), dtype=float)
            bins_fit = throat_dist.get("bins", []) or []
            k_cell = np.asarray(throat_dbg.get("bin_index_per_cell", []), dtype=int)
            V_cell = np.asarray(throat_dbg.get("V_per_cell", []), dtype=float)
            rho_raw = np.asarray(throat_dbg.get("rho_raw_per_cell", []), dtype=float)
            n_after = np.asarray(throat_dbg.get("n_after_corr_per_cell", []), dtype=float)
            n_realized = self._compute_same_cell_throat_count_per_cell()
            if (
                len(bin_edges) >= 2
                and len(k_cell) == len(V_cell) == len(rho_raw) == len(n_after)
                and len(k_cell) > 0
                and n_realized is not None
                and len(n_realized) == len(k_cell)
            ):
                rho_realized = np.divide(np.asarray(n_realized, dtype=float), np.maximum(V_cell, 1e-18))
                pA = output_dir / f"rho_sampling_debug_raw_throat_{self.sample_type}.png"
                pB = output_dir / f"rho_sampling_debug_realized_throat_{self.sample_type}.png"
                x = self._plot_rho_sampling_panel_by_bin(
                    values=rho_raw,
                    bin_index=k_cell,
                    n_per_cell=n_after,
                    bin_edges=bin_edges,
                    bins_fit=bins_fit,
                    metric_label="Throat count fraction",
                    figure_label="Figure A (raw sampled)",
                    output_path=pA,
                )
                y = self._plot_rho_sampling_panel_by_bin(
                    values=rho_realized,
                    bin_index=k_cell,
                    n_per_cell=np.asarray(n_realized, dtype=float),
                    bin_edges=bin_edges,
                    bins_fit=bins_fit,
                    metric_label="Throat count fraction",
                    figure_label="Figure B (realized n/V)",
                    output_path=pB,
                )
                if x is not None:
                    out_paths.append(x)
                if y is not None:
                    out_paths.append(y)
                debug_report["throat"] = {
                    "bin_edges": bin_edges.tolist(),
                    "n_cells": int(len(k_cell)),
                    "raw_mean": float(np.nanmean(rho_raw)) if len(rho_raw) > 0 else float("nan"),
                    "target_mean_after_corr": float(np.nanmean(np.divide(n_after, np.maximum(V_cell, 1e-18))))
                    if len(n_after) > 0
                    else float("nan"),
                    "realized_mean": float(np.nanmean(rho_realized)) if len(rho_realized) > 0 else float("nan"),
                }

        if debug_report.get("pore") or debug_report.get("throat"):
            debug_json = output_dir / f"rho_sampling_debug_summary_{self.sample_type}.json"
            with open(debug_json, "w", encoding="utf-8") as f:
                json.dump(debug_report, f, indent=2, ensure_ascii=False)
            out_paths.append(debug_json)
            print(f"rho 采样诊断汇总已保存: {debug_json}")
        else:
            print("无可用的 rho 采样调试数据，跳过 A/B 诊断图输出")
        return out_paths

    def plot_pore_radius_simulated_vs_fit(self, output_dir: Path) -> Path:
        """
        孔半径：每个厚度区间（10 nm bin）一个子图，直方图 + Phase2 拟合曲线。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if getattr(self, "_cells", None) is None or getattr(self, "_pore_cell_index", None) is None:
            print("无网格信息，跳过孔半径分布图")
            return None
        radius_dist = self.pore_radius_dist_by_thickness_bin.get(self.sample_type, {})
        radius_edges = np.array(radius_dist.get("bin_edges", []))
        radius_bins = radius_dist.get("bins", [])
        if len(radius_edges) < 2:
            return None

        radius_per_bin = [[] for _ in range(len(radius_edges) - 1)]
        for i, cell in enumerate(self._cells):
            idx = np.where(self._pore_cell_index == i)[0]
            if len(idx) == 0:
                continue
            T = cell["T"]
            k = np.searchsorted(radius_edges, T, side="right") - 1
            k = np.clip(k, 0, len(radius_edges) - 2)
            radius_per_bin[k].extend(self.pore_radii[idx].tolist())

        bins_with_data = [i for i in range(len(radius_edges) - 1) if len(radius_per_bin[i]) > 0]
        if not bins_with_data:
            return None
        n_panels = len(bins_with_data)
        n_cols = min(4, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = np.atleast_2d(axes)
        axes_flat = axes.flatten()
        for idx, k in enumerate(bins_with_data):
            ax = axes_flat[idx]
            vals = np.array(radius_per_bin[k], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                ax.set_visible(False)
                continue
            q_lo = max(0.5, float(np.quantile(vals, 0.01)))
            q_hi = float(np.quantile(vals, 0.99))
            if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_hi <= q_lo:
                q_lo = max(0.5, float(np.min(vals)))
                q_hi = max(q_lo + 1e-6, float(np.max(vals)))
            vals_vis = vals[(vals >= q_lo) & (vals <= q_hi)]
            if len(vals_vis) < max(10, int(0.2 * len(vals))):
                vals_vis = vals
                q_lo = max(0.5, float(np.min(vals_vis)))
                q_hi = max(q_lo + 1e-6, float(np.max(vals_vis)))
            n_bins_hist = min(30, max(5, len(vals_vis) // 3))
            # 直方图：柱高为概率（该厚度 bin 内孔半径落入子 bin 的概率）
            ax.hist(
                vals_vis,
                bins=n_bins_hist,
                range=(q_lo, q_hi),
                weights=np.ones_like(vals_vis) / len(vals_vis),
                color="steelblue",
                alpha=0.8,
                edgecolor="white",
                label="Simulated",
            )
            if k < len(radius_bins) and radius_bins[k].get("fit"):
                x_plot = np.linspace(q_lo, q_hi, 200)
                pdf_vals = _radius_fit_pdf(x_plot, radius_bins[k])
                if np.any(pdf_vals > 0):
                    bin_width_approx = (q_hi - q_lo) / n_bins_hist if n_bins_hist and q_hi > q_lo else 1e-6
                    ax.plot(x_plot, pdf_vals * bin_width_approx, "r-", linewidth=2, label="Phase2 fit")
            ax.set_xlim(q_lo, q_hi)
            ax.set_xlabel("Pore radius (nm)", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            ax.set_title(
                f"T in [{radius_edges[k]:.0f},{radius_edges[k+1]:.0f}) nm "
                f"(n={len(vals)}, display p1-p99)",
                fontsize=10,
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        for idx in range(len(bins_with_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.suptitle(f"Pore radius: simulated vs Phase2 fit ({self.sample_type})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plot_path = output_dir / f"pore_radius_simulated_vs_fit_{self.sample_type}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"孔半径分布图已保存: {plot_path}")
        return plot_path

    def plot_pore_frac_simulated_vs_fit(self, output_dir: Path) -> Path:
        """
        孔体积分数：每个厚度区间（20 nm bin）一个子图，直方图 + Phase2 拟合曲线。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if getattr(self, "_cells", None) is None or getattr(self, "_pore_cell_index", None) is None:
            print("无网格信息，跳过孔体积分数分布图")
            return None
        frac_dist = self.frac_pore_dist_by_thickness_bin.get(self.sample_type, {})
        frac_edges = np.array(frac_dist.get("bin_edges", []))
        frac_bins = frac_dist.get("bins", [])
        if len(frac_edges) < 2:
            return None

        frac_per_bin = [[] for _ in range(len(frac_edges) - 1)]
        for i, cell in enumerate(self._cells):
            idx = np.where(self._pore_cell_index == i)[0]
            if len(idx) == 0:
                continue
            T, V = cell["T"], cell["V"]
            if V <= 0:
                continue
            vol_pores = np.sum((4.0 / 3.0) * np.pi * self.pore_radii[idx]**3)
            frac_cell = vol_pores / V
            k = np.searchsorted(frac_edges, T, side="right") - 1
            k = np.clip(k, 0, len(frac_edges) - 2)
            frac_per_bin[k].append(frac_cell)

        bins_with_data = [i for i in range(len(frac_edges) - 1) if len(frac_per_bin[i]) > 0]
        if not bins_with_data:
            return None
        n_panels = len(bins_with_data)
        n_cols = min(4, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = np.atleast_2d(axes)
        axes_flat = axes.flatten()
        for idx, k in enumerate(bins_with_data):
            ax = axes_flat[idx]
            vals = np.array(frac_per_bin[k], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                ax.set_visible(False)
                continue
            q_lo = max(1e-6, float(np.quantile(vals, 0.01)))
            q_hi = min(1.0, float(np.quantile(vals, 0.99)))
            if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_hi <= q_lo:
                q_lo = max(1e-6, float(np.min(vals)))
                q_hi = min(1.0, max(q_lo + 1e-6, float(np.max(vals))))
            vals_vis = vals[(vals >= q_lo) & (vals <= q_hi)]
            if len(vals_vis) < max(10, int(0.2 * len(vals))):
                vals_vis = vals
                q_lo = max(1e-6, float(np.min(vals_vis)))
                q_hi = min(1.0, max(q_lo + 1e-6, float(np.max(vals_vis))))
            n_bins_hist = min(25, max(5, len(vals_vis) // 3))
            ax.hist(vals_vis, bins=n_bins_hist, range=(q_lo, q_hi), weights=np.ones_like(vals_vis) / len(vals_vis),
                    color="steelblue", alpha=0.8, edgecolor="white", label="Simulated")
            if k < len(frac_bins) and frac_bins[k].get("fit"):
                x_plot = np.linspace(q_lo, q_hi, 200)
                pdf_vals = _fit_pdf(x_plot, frac_bins[k])
                if np.any(pdf_vals > 0):
                    bin_width_approx = (q_hi - q_lo) / n_bins_hist if n_bins_hist and q_hi > q_lo else 1e-6
                    ax.plot(x_plot, pdf_vals * bin_width_approx, "r-", linewidth=2, label="Phase2 fit")
            ax.set_xlim(q_lo, q_hi)
            ax.set_xlabel("Pore volume fraction", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            ax.set_title(
                f"T in [{frac_edges[k]:.0f},{frac_edges[k+1]:.0f}) nm "
                f"(n={len(vals)}, display p1-p99)",
                fontsize=10,
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        for idx in range(len(bins_with_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.suptitle(f"Pore volume fraction: simulated vs Phase2 fit ({self.sample_type})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plot_path = output_dir / f"pore_frac_simulated_vs_fit_{self.sample_type}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"孔体积分数分布图已保存: {plot_path}")
        return plot_path

    def plot_throat_length_simulated_vs_fit(self, output_dir: Path) -> Path | None:
        """
        喉长度：每个厚度区间一个子图，模拟喉长度直方图 + Phase2 拟合曲线。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.throat_pore1 is None or len(self.throat_pore1) == 0:
            print("无喉数据，跳过喉长度分布图")
            return None

        throat_len_data = self.throat_length_dist_by_thickness_bin.get(self.sample_type, {})
        bin_edges = np.array(throat_len_data.get("bin_edges", []), dtype=float)
        bins_len = throat_len_data.get("bins", [])
        n_bins = len(bin_edges) - 1
        if n_bins <= 0:
            print("无 Phase2 喉长度按厚度区间的拟合参数，跳过喉长度分布图")
            return None

        # 每个孔的厚度
        n_pores = len(self.pore_coords)
        if getattr(self, "_cells", None) is not None and getattr(self, "_pore_cell_index", None) is not None:
            pore_thickness = np.array([self._cells[self._pore_cell_index[i]]["T"] for i in range(n_pores)], dtype=float)
        elif getattr(self, "_thickness_interp", None) is not None:
            pore_thickness = np.maximum(
                self._thickness_interp(self.pore_coords[:, 0], self.pore_coords[:, 1], grid=False).ravel(),
                1.0,
            )
        else:
            pore_thickness = np.full(n_pores, self.geometry_params.get("thickness_mean", 150.0), dtype=float)

        # 喉长度统一使用孔心距（不使用有效喉长），与导出的 Length 口径一致。
        pore_id_to_idx = {int(pid): i for i, pid in enumerate(self.pore_ids)}
        lengths = []
        T_mid_list = []
        for t_idx, (p1_id, p2_id) in enumerate(zip(self.throat_pore1, self.throat_pore2)):
            i = pore_id_to_idx.get(int(p1_id))
            j = pore_id_to_idx.get(int(p2_id))
            if i is None or j is None:
                continue
            c1 = self.pore_coords[i]
            c2 = self.pore_coords[j]
            d = float(np.linalg.norm(c2 - c1))
            L = max(d, 1e-6)
            if L <= 0:
                continue
            lengths.append(L)
            T_mid_list.append(0.5 * (pore_thickness[i] + pore_thickness[j]))

        if not lengths:
            print("无喉长度数据，跳过喉长度分布图")
            return None

        lengths = np.array(lengths, dtype=float)
        T_mid = np.array(T_mid_list, dtype=float)

        # 按厚度区间分组
        len_per_bin = [[] for _ in range(n_bins)]
        for L, Tm in zip(lengths, T_mid):
            k = np.searchsorted(bin_edges, Tm, side="right") - 1
            if k < 0 or k >= n_bins:
                continue
            len_per_bin[k].append(L)

        bins_with_data = [i for i in range(n_bins) if len(len_per_bin[i]) > 0]
        if not bins_with_data:
            print("所有厚度区间内均无喉长度数据，跳过喉长度分布图")
            return None

        n_panels = len(bins_with_data)
        n_cols = min(4, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = np.atleast_2d(axes)
        axes_flat = axes.flatten()

        for idx, k in enumerate(bins_with_data):
            ax = axes_flat[idx]
            vals = np.array(len_per_bin[k], dtype=float)
            if len(vals) == 0:
                ax.set_visible(False)
                continue
            n_bins_hist = min(30, max(5, len(vals) // 3))
            # 直方图：柱高为概率（该厚度 bin 内喉长度落入子 bin 的概率）
            ax.hist(
                vals,
                bins=n_bins_hist,
                weights=np.ones_like(vals) / len(vals),
                color="steelblue",
                alpha=0.8,
                edgecolor="white",
                label="Simulated",
            )
            # Phase2 fit curve (PDF * bin width -> probability)
            if k < len(bins_len) and bins_len[k].get("fit"):
                x_min = max(0.1, float(vals.min()) * 0.8)
                x_max = float(vals.max()) * 1.2
                x_plot = np.linspace(x_min, x_max, 200)
                pdf_vals = _radius_fit_pdf(x_plot, bins_len[k])
                if np.any(pdf_vals > 0):
                    bin_width_approx = (vals.max() - vals.min()) / n_bins_hist if n_bins_hist and vals.max() > vals.min() else 1e-6
                    ax.plot(x_plot, pdf_vals * bin_width_approx, "r-", linewidth=2, label="Phase2 fit")
            ax.set_xlabel("Throat length (nm)", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            ax.set_title(
                f"T in [{bin_edges[k]:.0f},{bin_edges[k+1]:.0f}) nm (n={len(vals)})",
                fontsize=10,
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(len(bins_with_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.suptitle(f"Throat length: simulated vs Phase2 fit ({self.sample_type})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plot_path = output_dir / f"throat_length_simulated_vs_fit_{self.sample_type}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"喉长度分布图已保存: {plot_path}")
        return plot_path

    def plot_pore_degree_simulated_vs_fit(self, output_dir: Path) -> Path | None:
        """
        孔度数（孔上喉数）：每个厚度区间一个子图，离散分布直方图 + Phase2 拟合 PMF。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.throat_pore1 is None or len(self.throat_pore1) == 0:
            print("无喉数据，跳过孔度数分布图")
            return None

        degree_data = self.degree_dist_by_thickness_bin.get(self.sample_type, {})
        bin_edges = np.array(degree_data.get("bin_edges", []), dtype=float)
        bins_deg = degree_data.get("bins", [])
        n_bins = len(bin_edges) - 1
        if n_bins <= 0:
            print("无 Phase2 孔度数按厚度区间的拟合参数，跳过孔度数分布图")
            return None

        # 每个孔的度数
        pore_ids = np.asarray(self.pore_ids, dtype=int)
        deg = np.zeros(len(pore_ids), dtype=int)
        pid_to_i = {int(pid): i for i, pid in enumerate(pore_ids)}
        for a, b in zip(self.throat_pore1, self.throat_pore2):
            a = int(a)
            b = int(b)
            if a in pid_to_i:
                deg[pid_to_i[a]] += 1
            if b in pid_to_i and b != a:
                deg[pid_to_i[b]] += 1

        # 每个孔的厚度
        n_pores = len(self.pore_coords)
        if getattr(self, "_cells", None) is not None and getattr(self, "_pore_cell_index", None) is not None:
            pore_thickness = np.array([self._cells[self._pore_cell_index[i]]["T"] for i in range(n_pores)], dtype=float)
        elif getattr(self, "_thickness_interp", None) is not None:
            pore_thickness = np.maximum(
                self._thickness_interp(self.pore_coords[:, 0], self.pore_coords[:, 1], grid=False).ravel(),
                1.0,
            )
        else:
            pore_thickness = np.full(n_pores, self.geometry_params.get("thickness_mean", 150.0), dtype=float)

        # 按厚度区间分组
        deg_per_bin = [[] for _ in range(n_bins)]
        for i in range(len(pore_ids)):
            T = pore_thickness[i]
            k = np.searchsorted(bin_edges, T, side="right") - 1
            if k < 0 or k >= n_bins:
                continue
            deg_per_bin[k].append(int(deg[i]))

        bins_with_data = [i for i in range(n_bins) if len(deg_per_bin[i]) > 0]
        if not bins_with_data:
            print("所有厚度区间内均无孔度数数据，跳过孔度数分布图")
            return None

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
            if len(vals) == 0:
                ax.set_visible(False)
                continue
            d_min = int(vals.min())
            d_max = int(vals.max())
            if d_max == d_min:
                d_max = d_min + 1
            # 直方图：离散概率（柱高为该度数出现概率）
            bins_int = np.arange(d_min - 0.5, d_max + 1.5, 1.0)
            ax.hist(
                vals,
                bins=bins_int,
                weights=np.ones_like(vals, dtype=float) / len(vals),
                color="steelblue",
                alpha=0.8,
                edgecolor="white",
                label="Simulated",
            )

            # Phase2: empirical CDF or parametric PMF
            if k < len(bins_deg) and bins_deg[k].get("fit"):
                f = bins_deg[k].get("fit") or bins_deg[k]
                if f.get("fit_type") == "empirical":
                    dv = np.array(f.get("degree_values", []), dtype=int)
                    cnt = np.array(f.get("counts", []), dtype=float)
                    total = float(np.sum(cnt))
                    if total > 0 and len(dv) > 0:
                        pmf = cnt / total
                        ax.plot(dv, pmf, "ro-", linewidth=2, markersize=4, label="Phase2 empirical PMF")
                else:
                    dist_info = f.get("distribution") or {}
                    name = dist_info.get("name", "")
                    params = dist_info.get("params", []) or []
                    d_range = np.arange(d_min, d_max + 1)
                    pmf = None
                    try:
                        if name == "poisson" and len(params) >= 1:
                            lam = max(float(params[0]), 1e-6)
                            pmf = stats.poisson.pmf(d_range, mu=lam)
                        elif name == "nbinom" and len(params) >= 2:
                            n_nb = max(float(params[0]), 1e-3)
                            p_nb = float(params[1])
                            p_nb = min(max(p_nb, 1e-6), 1.0 - 1e-6)
                            pmf = stats.nbinom.pmf(d_range, n_nb, p_nb)
                    except Exception:
                        pmf = None
                    if pmf is None:
                        lam = max(float(f.get("mean", np.mean(vals))), 1e-3)
                        pmf = stats.poisson.pmf(d_range, mu=lam)
                    ax.plot(d_range, pmf, "ro-", linewidth=2, markersize=4, label="Phase2 fit PMF")

            ax.set_xlabel("Pore degree", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            ax.set_title(
                f"T in [{bin_edges[k]:.0f},{bin_edges[k+1]:.0f}) nm (n={len(vals)})",
                fontsize=10,
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(len(bins_with_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.suptitle(f"Pore degree: simulated vs Phase2 fit ({self.sample_type})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plot_path = output_dir / f"pore_degree_simulated_vs_fit_{self.sample_type}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"孔度数分布图已保存: {plot_path}")
        return plot_path

    def plot_throat_count_fraction_simulated_vs_fit(self, output_dir: Path) -> Path | None:
        """
        喉数量分数 (喉数/体积)：每个厚度区间一个子图，cell 级别的喉数量分数分布 + Phase2 拟合曲线。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if getattr(self, "_cells", None) is None or getattr(self, "_pore_cell_index", None) is None:
            print("无网格信息，跳过喉数量分数分布图")
            return None
        if self.throat_pore1 is None or len(self.throat_pore1) == 0:
            print("无喉数据，跳过喉数量分数分布图")
            return None

        rho_throat_dist = getattr(self, "rho_throat_dist_by_thickness_bin", {}).get(self.sample_type, {})
        bin_edges = np.array(rho_throat_dist.get("bin_edges", []), dtype=float)
        bins_fit = rho_throat_dist.get("bins", [])
        n_bins = len(bin_edges) - 1
        if n_bins <= 0:
            print("无 Phase2 喉数量分数按厚度区间的拟合参数，跳过喉数量分数分布图")
            return None

        # 每个 cell 内的喉数：仅统计“非跨网格”的喉（两端孔属于同一 cell）
        n_cells = len(self._cells)
        throat_count_per_cell = np.zeros(n_cells, dtype=int)
        pore_id_to_idx = {int(pid): i for i, pid in enumerate(self.pore_ids)}
        for p1_id, p2_id in zip(self.throat_pore1, self.throat_pore2):
            i = pore_id_to_idx.get(int(p1_id))
            j = pore_id_to_idx.get(int(p2_id))
            if i is None or j is None:
                continue
            cell_i = int(self._pore_cell_index[i])
            cell_j = int(self._pore_cell_index[j])
            # 仅当两端孔在同一网格 cell 内时，才将该喉计入该 cell 的喉数量
            if cell_i == cell_j and 0 <= cell_i < n_cells:
                throat_count_per_cell[cell_i] += 1

        # 按厚度区间统计每个 cell 的喉数量分数 = 喉数/体积
        rho_per_bin = [[] for _ in range(n_bins)]
        for cell_idx, cell in enumerate(self._cells):
            n_cell = throat_count_per_cell[cell_idx]
            V = float(cell.get("V", 0.0))
            if n_cell <= 0 or V <= 0:
                continue
            rho_cell = n_cell / V
            T = float(cell.get("T", 0.0))
            k = np.searchsorted(bin_edges, T, side="right") - 1
            if k < 0 or k >= n_bins:
                continue
            rho_per_bin[k].append(rho_cell)

        bins_with_data = [i for i in range(n_bins) if len(rho_per_bin[i]) > 0]
        if not bins_with_data:
            print("所有厚度区间内均无喉数量分数数据，跳过喉数量分数分布图")
            return None

        n_panels = len(bins_with_data)
        n_cols = min(4, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = np.atleast_2d(axes)
        axes_flat = axes.flatten()

        for idx, k in enumerate(bins_with_data):
            ax = axes_flat[idx]
            vals = np.array(rho_per_bin[k], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                ax.set_visible(False)
                continue
            # 稳健显示区间：默认 p1~p99，避免少量极端值把横轴拉平导致“看起来直线”。
            q_lo = max(1e-12, float(np.quantile(vals, 0.01)))
            q_hi = float(np.quantile(vals, 0.99))
            if not np.isfinite(q_lo) or not np.isfinite(q_hi) or q_hi <= q_lo:
                q_lo = max(1e-12, float(np.min(vals)))
                q_hi = max(q_lo * 1.001, float(np.max(vals)))
            vals_vis = vals[(vals >= q_lo) & (vals <= q_hi)]
            # 小样本时避免几乎全被裁掉；必要时回退全量显示
            if len(vals_vis) < max(5, int(0.3 * len(vals))):
                vals_vis = vals
                q_lo = max(1e-12, float(np.min(vals_vis)))
                q_hi = max(q_lo * 1.001, float(np.max(vals_vis)))
            n_bins_hist = min(20, max(4, len(vals_vis) // 2))
            ax.hist(
                vals_vis,
                bins=n_bins_hist,
                range=(q_lo, q_hi),
                weights=np.ones_like(vals_vis) / len(vals_vis),
                color="steelblue",
                alpha=0.8,
                edgecolor="white",
                label="Simulated",
            )
            if k < len(bins_fit) and bins_fit[k].get("fit"):
                x_plot = np.linspace(q_lo, q_hi, 240)
                pdf_vals = _fit_pdf(x_plot, bins_fit[k])
                if np.any(pdf_vals > 0):
                    bin_width_approx = (q_hi - q_lo) / n_bins_hist if n_bins_hist and q_hi > q_lo else 1e-6
                    ax.plot(
                        x_plot,
                        pdf_vals * bin_width_approx,
                        "r-",
                        linewidth=2,
                        label="Phase2 fit",
                    )
            ax.set_xlim(q_lo, q_hi)
            ax.set_xlabel("Throat count fraction (1/nm^3)", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            ax.set_title(
                f"T in [{bin_edges[k]:.0f},{bin_edges[k+1]:.0f}) nm "
                f"(n={len(vals)}, display p1-p99)",
                fontsize=10,
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(len(bins_with_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.suptitle(f"Throat count fraction: simulated vs Phase2 fit ({self.sample_type})", fontsize=14, fontweight="bold")
        # Note: both experiment and this grid ignore boundary throats; may cause systematic offset on x-axis.
        fig.text(
            0.5,
            0.02,
            "Note: boundary throats are ignored in both experiment and this grid; may cause systematic offset.",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        plt.tight_layout(rect=(0, 0.06, 1, 1))
        plot_path = output_dir / f"throat_count_fraction_simulated_vs_fit_{self.sample_type}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"喉数量分数分布图已保存: {plot_path}")
        return plot_path

    def plot_throat_radius_by_thickness_bins(self, output_dir: Path) -> Path | None:
        """不同厚度区间内喉半径分布 vs Phase2 p(r|T) 曲线。仅统计非边界喉（两端孔在同一 cell）。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if getattr(self, "_cells", None) is None or getattr(self, "_pore_cell_index", None) is None:
            print("无网格信息，跳过喉半径按厚度区间图")
            return None
        if self.throat_radii is None or len(self.throat_radii) == 0:
            return None
        tr_by_t = self.throat_radius_dist_by_thickness_bin.get(self.sample_type, {})
        bin_edges = np.array(tr_by_t.get("bin_edges", []), dtype=float)
        bins_t = tr_by_t.get("bins", [])
        n_bins = len(bin_edges) - 1
        if n_bins <= 0:
            return None
        pore_id_to_idx = {int(pid): i for i, pid in enumerate(self.pore_ids)}
        # 仅非边界喉 + 按该喉所在 cell 的厚度归入区间（两端同 cell，取 cell 的 T）
        radius_per_bin = [[] for _ in range(n_bins)]
        for t, (p1, p2) in enumerate(zip(self.throat_pore1, self.throat_pore2)):
            i, j = pore_id_to_idx.get(int(p1)), pore_id_to_idx.get(int(p2))
            if i is None or j is None:
                continue
            ci, cj = int(self._pore_cell_index[i]), int(self._pore_cell_index[j])
            if ci != cj:
                continue
            T = float(self._cells[ci]["T"])
            k = np.searchsorted(bin_edges, T, side="right") - 1
            k = np.clip(k, 0, n_bins - 1)
            radius_per_bin[k].append(float(self.throat_radii[t]))
        bins_with_data = [b for b in range(n_bins) if len(radius_per_bin[b]) > 0]
        if not bins_with_data:
            return None
        n_panels = len(bins_with_data)
        n_cols = min(4, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes_flat = np.atleast_2d(axes).flatten()
        for idx, k in enumerate(bins_with_data):
            ax = axes_flat[idx]
            vals = np.array(radius_per_bin[k], dtype=float)
            n_hist = min(30, max(5, len(vals) // 3))
            # 直方图：柱高为概率
            ax.hist(
                vals,
                bins=n_hist,
                weights=np.ones_like(vals) / len(vals),
                color="steelblue",
                alpha=0.8,
                edgecolor="white",
                label="Simulated",
            )
            if k < len(bins_t) and bins_t[k].get("fit"):
                x_plot = np.linspace(max(0.5, vals.min() * 0.9), vals.max() * 1.1, 200)
                pdf_vals = _radius_fit_pdf(x_plot, bins_t[k])
                if np.any(pdf_vals > 0):
                    bin_width_approx = (vals.max() - vals.min()) / n_hist if n_hist and vals.max() > vals.min() else 1e-6
                    ax.plot(x_plot, pdf_vals * bin_width_approx, "r-", linewidth=2, label="Phase2 fit")
            ax.set_xlabel("Throat radius (nm)", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            ax.set_title(f"T in [{bin_edges[k]:.0f},{bin_edges[k+1]:.0f}) nm (n={len(vals)})", fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        for idx in range(len(bins_with_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.suptitle(f"Throat radius by thickness: simulated vs Phase2 fit ({self.sample_type}), non-boundary only", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out = output_dir / f"throat_radius_by_thickness_bins_{self.sample_type}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"喉半径按厚度区间图已保存: {out}")
        return out

    def plot_throat_radius_by_length_bins(self, output_dir: Path) -> Path | None:
        """不同长度区间内喉半径分布 vs Phase2 p(r|L) 曲线。包含所有喉。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.throat_radii is None or len(self.throat_radii) == 0:
            return None
        tr_by_l = self.throat_radius_dist_by_length_bin.get(self.sample_type, {})
        bin_edges = np.array(tr_by_l.get("bin_edges_nm", []), dtype=float)
        bins_l = tr_by_l.get("bins", [])
        n_bins = len(bin_edges) - 1
        if n_bins <= 0:
            return None
        pore_id_to_idx = {int(pid): i for i, pid in enumerate(self.pore_ids)}
        lengths = np.array([
            np.linalg.norm(self.pore_coords[pore_id_to_idx[int(p2)]] - self.pore_coords[pore_id_to_idx[int(p1)]])
            for p1, p2 in zip(self.throat_pore1, self.throat_pore2)
        ], dtype=float)
        k_l = np.searchsorted(bin_edges, lengths, side="right") - 1
        k_l = np.clip(k_l, 0, n_bins - 1)
        radius_per_bin = [[] for _ in range(n_bins)]
        for t in range(len(self.throat_radii)):
            radius_per_bin[k_l[t]].append(float(self.throat_radii[t]))
        bins_with_data = [b for b in range(n_bins) if len(radius_per_bin[b]) > 0]
        if not bins_with_data:
            return None
        n_panels = len(bins_with_data)
        n_cols = min(4, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes_flat = np.atleast_2d(axes).flatten()
        for idx, k in enumerate(bins_with_data):
            ax = axes_flat[idx]
            vals = np.array(radius_per_bin[k], dtype=float)
            n_hist = min(30, max(5, len(vals) // 3))
            # 直方图：柱高为概率
            ax.hist(
                vals,
                bins=n_hist,
                weights=np.ones_like(vals) / len(vals),
                color="steelblue",
                alpha=0.8,
                edgecolor="white",
                label="Simulated",
            )
            if k < len(bins_l) and bins_l[k].get("fit"):
                x_plot = np.linspace(max(0.5, vals.min() * 0.9), vals.max() * 1.1, 200)
                pdf_vals = _radius_fit_pdf(x_plot, bins_l[k])
                if np.any(pdf_vals > 0):
                    bin_width_approx = (vals.max() - vals.min()) / n_hist if n_hist and vals.max() > vals.min() else 1e-6
                    ax.plot(x_plot, pdf_vals * bin_width_approx, "r-", linewidth=2, label="Phase2 fit")
            ax.set_xlabel("Throat radius (nm)", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            L_lo, L_hi = bin_edges[k], bin_edges[k + 1]
            ax.set_title(f"L in [{L_lo:.0f},{L_hi:.0f}) nm (n={len(vals)})", fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        for idx in range(len(bins_with_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        plt.suptitle(f"Throat radius by length: simulated vs Phase2 fit ({self.sample_type})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out = output_dir / f"throat_radius_by_length_bins_{self.sample_type}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"喉半径按长度区间图已保存: {out}")
        return out

    def plot_throat_radius_vs_mean_pore(self, output_dir: Path) -> Path | None:
        """Throat radius vs mean pore radius scatter + linear fit, compare with Phase1_5."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.throat_radii is None or len(self.throat_radii) == 0:
            return None
        pore_id_to_idx = {int(pid): i for i, pid in enumerate(self.pore_ids)}
        mean_pore = np.array([
            0.5 * (self.pore_radii[pore_id_to_idx[int(p1)]] + self.pore_radii[pore_id_to_idx[int(p2)]])
            for p1, p2 in zip(self.throat_pore1, self.throat_pore2)
        ], dtype=float)
        r_throat = np.asarray(self.throat_radii, dtype=float)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(mean_pore, r_throat, alpha=0.15, s=4, c="steelblue", label="Simulated")

        # 1) Linear fit to simulated scatter
        slope_sim, intercept_sim, r_sim, _, _ = stats.linregress(mean_pore, r_throat)
        x_min_sim, x_max_sim = float(mean_pore.min()), float(mean_pore.max())

        # 2) Phase1_5 linear relation from phase1_5_linear_fits.json
        x_min, x_max = x_min_sim, x_max_sim
        try:
            phase1_5_dir = OUTPUT_ROOT / "phase1_5_parameter_trends"
            fits_path = phase1_5_dir / "phase1_5_linear_fits.json"
            if fits_path.exists():
                with fits_path.open("r", encoding="utf-8") as f:
                    fits_all = json.load(f)
                fits_rel = fits_all.get("throat_radius_vs_mean_pore_radius", {})
                fit_one = fits_rel.get(self.sample_type)
                if fit_one and "slope" in fit_one and "intercept" in fit_one:
                    slope_p15 = float(fit_one["slope"])
                    intercept_p15 = float(fit_one["intercept"])
                    r_p15 = fit_one.get("pearson_r")
                    x_line = np.linspace(x_min_sim, x_max_sim, 200)
                    label_txt = "Phase1_5 linear"
                    if r_p15 is not None:
                        label_txt += f" r={float(r_p15):.3f}"
                    ax.plot(
                        x_line,
                        slope_p15 * x_line + intercept_p15,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=label_txt,
                    )
        except Exception:
            pass

        # 3) Simulated regression line
        x_line_sim = np.linspace(x_min_sim, x_max_sim, 200)
        ax.plot(
            x_line_sim,
            slope_sim * x_line_sim + intercept_sim,
            color="darkorange",
            linewidth=2,
            label=f"Simulated linear fit r={r_sim:.3f}",
        )
        ax.set_xlabel("Mean pore radius (nm)", fontsize=12)
        ax.set_ylabel("Throat radius (nm)", fontsize=12)
        ax.set_title(f"Throat radius vs mean pore radius ({self.sample_type})", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = output_dir / f"throat_radius_vs_mean_pore_{self.sample_type}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"喉半径 vs 孔平均半径图已保存: {out}")
        return out

    def plot_throat_radius_vs_length(self, output_dir: Path) -> Path | None:
        """喉长度 vs 喉半径 散点图 + 线性拟合，并与 Phase1_5 的线性关系比较。纵轴为喉长度，横轴为喉半径。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.throat_radii is None or len(self.throat_radii) == 0:
            return None
        # 模拟数据：喉长度与半径
        pore_id_to_idx = {int(pid): i for i, pid in enumerate(self.pore_ids)}
        lengths = []
        radii = []
        for p1_id, p2_id, r in zip(self.throat_pore1, self.throat_pore2, self.throat_radii):
            i = pore_id_to_idx.get(int(p1_id))
            j = pore_id_to_idx.get(int(p2_id))
            if i is None or j is None:
                continue
            L = float(np.linalg.norm(self.pore_coords[j] - self.pore_coords[i]))
            if L <= 0:
                continue
            lengths.append(L)
            radii.append(float(r))
        if not lengths:
            print("无喉长度数据，跳过喉半径 vs 喉长度散点图")
            return None
        lengths = np.asarray(lengths, dtype=float)  # 作为 y
        radii = np.asarray(radii, dtype=float)      # 作为 x

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(radii, lengths, alpha=0.15, s=4, c="steelblue", label="Simulated")

        # 1) Linear fit (y = a*x + b, x=radius, y=length)
        slope_sim, intercept_sim, r_sim, _, _ = stats.linregress(radii, lengths)
        x_min_sim, x_max_sim = float(radii.min()), float(radii.max())

        # 2) Phase1_5 L-r linear from phase1_5_linear_fits.json
        try:
            phase1_5_dir = OUTPUT_ROOT / "phase1_5_parameter_trends"
            fits_path = phase1_5_dir / "phase1_5_linear_fits.json"
            if fits_path.exists():
                with fits_path.open("r", encoding="utf-8") as f:
                    fits_all = json.load(f)
                fits_rel = fits_all.get("throat_radius_vs_length", {})
                fit_one = fits_rel.get(self.sample_type)
                if fit_one and "slope" in fit_one and "intercept" in fit_one:
                    slope_p15 = float(fit_one["slope"])
                    intercept_p15 = float(fit_one["intercept"])
                    r_p15 = fit_one.get("pearson_r")
                    x_line = np.linspace(x_min_sim, x_max_sim, 200)
                    label_txt = "Phase1_5 linear"
                    if r_p15 is not None:
                        label_txt += f" r={float(r_p15):.3f}"
                    ax.plot(
                        x_line,
                        slope_p15 * x_line + intercept_p15,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=label_txt,
                    )
        except Exception:
            pass

        # 3) Simulated regression line
        x_line_sim = np.linspace(x_min_sim, x_max_sim, 200)
        ax.plot(
            x_line_sim,
            slope_sim * x_line_sim + intercept_sim,
            color="darkorange",
            linewidth=2,
            label=f"Simulated linear fit r={r_sim:.3f}",
        )

        ax.set_xlabel("Throat radius r (nm)", fontsize=12)
        ax.set_ylabel("Throat length L (nm)", fontsize=12)
        ax.set_title(f"Throat length vs throat radius ({self.sample_type})", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = output_dir / f"throat_radius_vs_length_{self.sample_type}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"喉半径 vs 喉长度图已保存: {out}")
        return out

    def plot_throat_r2L_distribution_vs_phase2(self, output_dir: Path) -> Path | None:
        """
        喉体积因子 π r^2 L 的分布：Phase3 模拟 vs Phase2 联合分布近似。
        这里比较的是整体分布，不按厚度分层。
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.throat_radii is None or len(self.throat_radii) == 0:
            print("无喉数据，跳过 r^2L 分布图")
            return None

        # Phase3: 计算每条喉的 r^2 L（带 π，只作为比例因子）
        pore_id_to_idx = {int(pid): i for i, pid in enumerate(self.pore_ids)}
        lengths = []
        r2L_sim = []
        for p1_id, p2_id, r in zip(self.throat_pore1, self.throat_pore2, self.throat_radii):
            i = pore_id_to_idx.get(int(p1_id))
            j = pore_id_to_idx.get(int(p2_id))
            if i is None or j is None:
                continue
            L = float(np.linalg.norm(self.pore_coords[j] - self.pore_coords[i]))
            if L <= 0:
                continue
            lengths.append(L)
            r2L_sim.append(np.pi * float(r) ** 2 * L)
        if not r2L_sim:
            print("无有效喉长度数据，跳过 r^2L 分布图")
            return None
        r2L_sim = np.asarray(r2L_sim, dtype=float)

        # Phase2: 从联合 GMM p(L, r) 采样，构造对应的 r^2 L 分布
        tr_by_l = self.throat_radius_dist_by_length_bin.get(self.sample_type, {}) or {}
        joint_gmm_len = tr_by_l.get("joint_gmm_length_throat")
        r2L_p2 = None
        if joint_gmm_len:
            rng = np.random.default_rng(self._random_seed)
            n_phase2 = min(100000, max(20000, len(r2L_sim)))
            samples = _sample_joint_from_gmm_2d(joint_gmm_len, n_phase2, rng)
            if samples.size > 0:
                L_p2 = samples[:, 0]
                r_p2 = samples[:, 1]
                r2L_p2 = np.pi * np.asarray(r_p2, dtype=float) ** 2 * np.asarray(L_p2, dtype=float)

        fig, ax = plt.subplots(figsize=(6, 5))
        # 使用对数尺度可以更好地看尾部；避免零值
        vals_sim = r2L_sim[r2L_sim > 0]
        log_sim = np.log10(vals_sim)
        n_bins_hist = min(40, max(10, len(log_sim) // 200))
        ax.hist(
            log_sim,
            bins=n_bins_hist,
            weights=np.ones_like(log_sim) / len(log_sim),
            alpha=0.6,
            color="steelblue",
            edgecolor="white",
            label="Phase3 simulated",
        )
        # Phase2 r^2 L histogram from r2L_log10_hist
        r2L_hist = tr_by_l.get("r2L_log10_hist") or {}

        if r2L_p2 is not None:
            vals_p2 = r2L_p2[r2L_p2 > 0]
            if len(vals_p2) > 0:
                log_p2 = np.log10(vals_p2)
                ax.hist(
                    log_p2,
                    bins=n_bins_hist,
                    weights=np.ones_like(log_p2) / len(log_p2),
                    histtype="step",
                    linewidth=2,
                    color="red",
                    label="Phase2 joint approx",
                )

        if r2L_hist:
            try:
                edges = np.asarray(r2L_hist.get("bin_edges", []), dtype=float)
                prob = np.asarray(r2L_hist.get("prob", []), dtype=float)
                if edges.size >= 2 and prob.size == edges.size - 1 and prob.sum() > 0:
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    ax.step(
                        centers,
                        prob,
                        where="mid",
                        color="green",
                        linewidth=1.8,
                        label="Phase2 empirical histogram",
                    )
            except Exception:
                pass

        ax.set_xlabel("log10(pi r^2 L) (nm^3)", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_title(f"Throat volume factor pi r^2 L: Phase3 vs Phase2 ({self.sample_type})", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = output_dir / f"throat_r2L_distribution_vs_phase2_{self.sample_type}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"喉 r²L 分布图已保存: {out}")
        return out

    def plot_throat_frac_by_thickness_bins(self, output_dir: Path) -> Path | None:
        """不同厚度区间喉的体积分数（仅非边界喉）分布：每个厚度区间一个子图，概率直方图 + Phase2 条件分布曲线。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if getattr(self, "_cells", None) is None or getattr(self, "_pore_cell_index", None) is None:
            print("无网格信息，跳过喉体积分数按厚度区间图")
            return None
        if self.throat_radii is None or len(self.throat_radii) == 0:
            return None
        frac_dist = self.frac_throat_dist_by_thickness_bin.get(self.sample_type, {})
        bin_edges = np.array(frac_dist.get("bin_edges", []), dtype=float)
        bins_fit = frac_dist.get("bins", [])
        n_bins = len(bin_edges) - 1
        if n_bins <= 0:
            return None
        pore_id_to_idx = {int(pid): i for i, pid in enumerate(self.pore_ids)}

        # 先对每个 cell 汇总“非边界喉”的总体积，再折算成 cell 级体积分数并按厚度分 bin
        n_cells = len(self._cells)
        vol_cell = np.zeros(n_cells)
        for t, (p1, p2) in enumerate(zip(self.throat_pore1, self.throat_pore2)):
            i, j = pore_id_to_idx.get(int(p1)), pore_id_to_idx.get(int(p2))
            if i is None or j is None:
                continue
            ci, cj = int(self._pore_cell_index[i]), int(self._pore_cell_index[j])
            # 仅统计非边界喉：两端孔在同一 cell
            if ci != cj or ci < 0 or ci >= n_cells:
                continue
            # 喉体积口径统一使用孔心距（不使用有效喉长）
            L_center = np.linalg.norm(self.pore_coords[j] - self.pore_coords[i])
            L_use = max(float(L_center), 1e-6)
            vol_t = np.pi * (float(self.throat_radii[t]) ** 2) * L_use
            vol_cell[ci] += vol_t

        frac_per_bin = [[] for _ in range(n_bins)]
        for cell_idx, cell in enumerate(self._cells):
            V_cell = float(cell.get("V", 0.0))
            if V_cell <= 0:
                continue
            if vol_cell[cell_idx] <= 0:
                continue
            frac_cell = vol_cell[cell_idx] / V_cell
            T = float(cell.get("T", 0.0))
            k = np.searchsorted(bin_edges, T, side="right") - 1
            if k < 0 or k >= n_bins:
                continue
            frac_per_bin[k].append(frac_cell)

        bins_with_data = [i for i in range(n_bins) if len(frac_per_bin[i]) > 0]
        if not bins_with_data:
            print("所有厚度区间内均无喉体积分数数据，跳过喉体积分数分布图")
            return None

        n_panels = len(bins_with_data)
        n_cols = min(4, n_panels)
        n_rows = (n_panels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = np.atleast_2d(axes)
        axes_flat = axes.flatten()

        for idx, k in enumerate(bins_with_data):
            ax = axes_flat[idx]
            vals = np.array(frac_per_bin[k], dtype=float)
            if len(vals) == 0:
                ax.set_visible(False)
                continue
            n_bins_hist = min(25, max(5, len(vals) // 3))
            # 直方图：柱高为概率（每个 bin 内 cell 数 / 该厚度 bin 的 cell 总数）
            ax.hist(
                vals,
                bins=n_bins_hist,
                weights=np.ones_like(vals) / len(vals),
                color="steelblue",
                alpha=0.8,
                edgecolor="white",
                label="Simulated",
            )
            # Phase2 fit: PDF * bin width -> probability
            if k < len(bins_fit) and bins_fit[k].get("fit"):
                x_min = max(1e-6, float(vals.min()) * 0.9)
                x_max = float(vals.max()) * 1.1
                x_plot = np.linspace(x_min, x_max, 200)
                pdf_vals = _fit_pdf(x_plot, bins_fit[k])
                if np.any(pdf_vals > 0):
                    bin_width_approx = (vals.max() - vals.min()) / n_bins_hist if n_bins_hist and vals.max() > vals.min() else 1e-6
                    ax.plot(
                        x_plot,
                        pdf_vals * bin_width_approx,
                        "r-",
                        linewidth=2,
                        label="Phase2 fit",
                    )
            ax.set_xlabel("Throat volume fraction", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            ax.set_title(
                f"T in [{bin_edges[k]:.0f},{bin_edges[k+1]:.0f}) nm (n={len(frac_per_bin[k])})",
                fontsize=10,
            )
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        for idx in range(len(bins_with_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.suptitle(f"Throat volume fraction: simulated vs Phase2 fit ({self.sample_type})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out = output_dir / f"throat_frac_by_thickness_bins_{self.sample_type}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"喉体积分数分布图已保存: {out}")
        return out

    def plot_throats_3d_html(self, output_dir: Path, sample_type: str = None, sample_name: str = None, max_throats: int = 5000):
        """
        喉 3D 可视化：线段连接两端孔心。喉过多时随机抽样 max_throats 条以保速度（与孔 3D 规则一致）。
        sample_name 用于文件名，每个 run 独立；缺省时用 sample_type。
        """
        if not _HAS_PLOTLY:
            print("未安装 plotly，跳过喉 3D HTML 可视化")
            return
        if self.throat_pore1 is None or len(self.throat_pore1) == 0:
            print("无喉数据，跳过喉 3D 可视化")
            return
        st = sample_type or self.sample_type
        file_id = sample_name if sample_name else st
        pore_id_to_idx = {int(pid): i for i, pid in enumerate(self.pore_ids)}
        n_throats = len(self.throat_pore1)
        if n_throats > max_throats:
            rng = np.random.default_rng(self._random_seed)
            n_choose = min(max_throats, n_throats)
            idx = rng.choice(n_throats, size=n_choose, replace=False)
        else:
            idx = np.arange(n_throats)
        xs, ys, zs = [], [], []
        for t in idx:
            i = pore_id_to_idx.get(int(self.throat_pore1[t]))
            j = pore_id_to_idx.get(int(self.throat_pore2[t]))
            if i is None or j is None:
                continue
            p1 = self.pore_coords[i]
            p2 = self.pore_coords[j]
            xs.extend([p1[0], p2[0], None])
            ys.extend([p1[1], p2[1], None])
            zs.extend([p1[2], p2[2], None])
        if not xs:
            print("无有效喉线段，跳过喉 3D 可视化")
            return
        n_throats_total = len(self.throat_pore1)
        title = f"喉 3D ({file_id})" + (f" 显示 {len(idx)}/{n_throats_total}" if n_throats_total > max_throats else f" n={n_throats_total}")
        trace = go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color="rgba(230,60,50,0.9)", width=3),
            name="喉",
        )
        layout = go.Layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            scene=dict(
                xaxis_title="x (nm)", yaxis_title="y (nm)", zaxis_title="z (nm)",
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        fig = go.Figure(data=[trace], layout=layout)
        out_path = output_dir / f"throats_3d_{file_id}.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        print(f"喉 3D 可视化已保存: {out_path}")

    def generate(self):
        """
        完整生成流程。当前仅完成：厚度模拟 + 孔心撒点。
        后续步骤（孔半径、喉连接、喉半径）为 demo，待优化后再解除注释。
        """
        self.generate_geometry()
        self.place_pores()
        self.assign_pore_radii()
        # 不再强制“孔与孔不重叠”，仅保持原始撒点位置
        # self.reposition_pores_no_overlap()
        self.connect_throats(
            max_distance=float(self.geometry_params.get("throat_neighbor_nm", 30.0))
        )
        self.assign_throat_radii(w_thickness=0.4, w_pore=0.25, w_length=0.03)


def load_phase1_parameters(phase1_dir: Path) -> dict:
    """加载阶段一的厚度分布参数"""
    params_file = phase1_dir / "thickness_distribution_parameters.json"
    if params_file.exists():
        with open(params_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_phase2_parameters(phase2_dir: Path) -> dict:
    """加载阶段二的参数（含孔在不同厚度区间的数量分数，供撒点分布用）"""
    params = {}
    
    # 孔在不同厚度区间的数量分数（Phase2 统计，供 Phase3 按厚度区间分配孔数）
    pore_frac_file = phase2_dir / "pore_count_fraction_by_thickness_bin.json"
    if pore_frac_file.exists():
        with open(pore_frac_file, "r", encoding="utf-8") as f:
            params["pore_count_fraction_by_thickness_bin"] = json.load(f)
    
    # 密度参数
    density_file = phase2_dir / "density_vs_thickness_analysis.json"
    if density_file.exists():
        with open(density_file, "r", encoding="utf-8") as f:
            params["density"] = json.load(f)
    
    # 半径参数
    radius_file = phase2_dir / "radius_distribution_analysis.json"
    if radius_file.exists():
        with open(radius_file, "r", encoding="utf-8") as f:
            params["radius"] = json.load(f)
    
    # 方向性参数
    directionality_file = phase2_dir / "directionality_vs_thickness_analysis.json"
    if directionality_file.exists():
        with open(directionality_file, "r", encoding="utf-8") as f:
            params["directionality"] = json.load(f)

    # 孔数量分数按厚度区间的分布（rho_pore 拟合，供 Phase3 按分布分配每格孔数量分数）
    density_frac_file = phase2_dir / "density_frac_by_thickness_bins_analysis.json"
    if density_frac_file.exists():
        with open(density_frac_file, "r", encoding="utf-8") as f:
            dfrac = json.load(f)
            params["rho_pore_dist_by_thickness_bin"] = dfrac.get("rho_pore", {})
            params["frac_pore_dist_by_thickness_bin"] = dfrac.get("frac_pore", {})
            params["rho_throat_dist_by_thickness_bin"] = dfrac.get("rho_throat", {})
            params["frac_throat_dist_by_thickness_bin"] = dfrac.get("frac_throat", {})

    # 孔/喉半径按厚度区间的分布（10 nm bin，供 Phase3 分配孔半径与喉半径）
    radius_bins_file = phase2_dir / "radius_by_thickness_bins_analysis.json"
    if radius_bins_file.exists():
        with open(radius_bins_file, "r", encoding="utf-8") as f:
            rb = json.load(f)
            params["pore_radius_dist_by_thickness_bin"] = rb.get("pore", {})
            params["throat_radius_dist_by_thickness_bin"] = rb.get("throat", {})

    # 喉半径按喉长度区间的分布（供 Phase3 喉半径三分布加权）
    radius_len_file = phase2_dir / "radius_by_length_bins_throat_analysis.json"
    if radius_len_file.exists():
        with open(radius_len_file, "r", encoding="utf-8") as f:
            params["throat_radius_dist_by_length_bin"] = json.load(f)

    # 喉半径与两端孔平均半径相关（供 Phase3 喉半径线性项 r_pore）
    corr_file = phase2_dir / "pore_throat_radius_correlation.json"
    if corr_file.exists():
        with open(corr_file, "r", encoding="utf-8") as f:
            params["pore_throat_radius_correlation"] = json.load(f)

    # 喉长度按厚度区间的分布（供 Phase3 喉连接长度权重）
    throat_len_file = phase2_dir / "throat_length_analysis.json"
    if throat_len_file.exists():
        with open(throat_len_file, "r", encoding="utf-8") as f:
            params["throat_length_dist_by_thickness_bin"] = json.load(f)

    # overlap 喉 R_throat/R_cap 按厚度区间的分布（供 Phase3 在几何 R_cap 上乘系数）
    overlap_ratio_file = phase2_dir / "overlap_throat_R_ratio_by_thickness_bin.json"
    if overlap_ratio_file.exists():
        with open(overlap_ratio_file, "r", encoding="utf-8") as f:
            params["overlap_throat_R_ratio_dist_by_thickness_bin"] = json.load(f)

    # 子样本级「喉半径超过 min(两端孔)」比例按厚度（overlap / nonoverlap），供 Phase3 保留 n=f×N 条超限 + 重抽
    gt_min_file = phase2_dir / "throat_gt_min_pore_frac_per_subsample_by_thickness_bin.json"
    if gt_min_file.exists():
        with open(gt_min_file, "r", encoding="utf-8") as f:
            params["throat_gt_min_pore_frac_dist_by_thickness_bin"] = json.load(f)

    # cross-face 非 overlap 喉密度按厚度区间分布（预留给 Phase3 跨 cell 喉数量采样）
    nonov_cross_file = phase2_dir / "nonoverlap_cross_density_by_thickness_bin.json"
    if nonov_cross_file.exists():
        with open(nonov_cross_file, "r", encoding="utf-8") as f:
            params["nonoverlap_cross_density_dist_by_thickness_bin"] = json.load(f)

    # 有效喉长 L_raw 按厚度区间的分布（供 Phase3 喉连接时与孔心距一起加权）
    eff_len_file = phase2_dir / "effective_throat_length_analysis.json"
    if eff_len_file.exists():
        with open(eff_len_file, "r", encoding="utf-8") as f:
            params["effective_throat_length_dist_by_thickness_bin"] = json.load(f)

    # 孔度数按厚度区间的分布（供 Phase3 喉连接目标度数）
    degree_bins_file = phase2_dir / "degree_by_thickness_bins_analysis.json"
    if degree_bins_file.exists():
        with open(degree_bins_file, "r", encoding="utf-8") as f:
            params["degree_dist_by_thickness_bin"] = json.load(f)

    return params


def main():
    parser = argparse.ArgumentParser(
        description="阶段三：基底膜几何与合成孔喉网络"
    )
    parser.add_argument(
        "--sample-type",
        type=str,
        choices=("WT", "AS"),
        required=True,
        help="样本类型：WT 或 AS",
    )
    parser.add_argument(
        "--geometry-type",
        type=str,
        choices=("plane", "sphere"),
        default="plane",
        help="几何类型：plane（平面）或 sphere（球面）",
    )
    parser.add_argument(
        "--geometry-width",
        type=float,
        default=500,
        help="平面几何宽度 (nm，默认 500)",
    )
    parser.add_argument(
        "--geometry-height",
        type=float,
        default=500,
        help="平面几何高度 (nm，默认 500)",
    )
    parser.add_argument(
        "--geometry-thickness-mean",
        type=float,
        default=150,
        help="平均厚度 (nm，默认 150)",
    )
    parser.add_argument(
        "--geometry-grid-step-nm",
        type=float,
        default=5,
        help="厚度场网格步长 (nm)，默认 5",
    )
    parser.add_argument(
        "--pore-placement-cell-nm",
        type=float,
        default=50,
        help="孔放置网格格宽/撒点步长 (nm)，每格体积=格宽×格宽×该格厚度，默认 50",
    )
    parser.add_argument(
        "--throat-neighbor-nm",
        type=float,
        default=30.0,
        help="KD-tree neighbor radius for overlap and non-overlap throat candidates (nm), default 30.",
    )
    parser.add_argument(
        "--n-thickness-bins",
        type=int,
        default=20,
        help="厚度区间数，用于按分布分配孔数（默认 20）",
    )
    parser.add_argument(
        "--rho-sampling-scope",
        choices=["run", "cell"],
        default="run",
        help=(
            "rho_pore/rho_throat 的抽样层级。run=每个 synthetic run 的厚度 bin 抽一次，"
            "保留 Phase2 子样本级 count-density 方差；cell=每个 50 nm cell 独立抽样，"
            "run-level 汇总会更窄。默认 run。"
        ),
    )
    parser.add_argument(
        "--rho-mean-correction-rel-tol",
        type=float,
        default=None,
        help=(
            "Legacy shared relative deviation threshold before weak mean correction "
            "is applied to pore/throat count density by thickness bin. If omitted, "
            "pore and throat defaults are used separately."
        ),
    )
    parser.add_argument(
        "--rho-mean-correction-scale-min",
        type=float,
        default=None,
        help="Legacy shared minimum scale for weak mean correction of pore/throat count density.",
    )
    parser.add_argument(
        "--rho-mean-correction-scale-max",
        type=float,
        default=None,
        help="Legacy shared maximum scale for weak mean correction of pore/throat count density.",
    )
    parser.add_argument(
        "--pore-rho-mean-correction-rel-tol",
        type=float,
        default=None,
        help="Pore count-density weak mean-correction trigger threshold. Default inf (disabled).",
    )
    parser.add_argument(
        "--pore-rho-mean-correction-scale-min",
        type=float,
        default=None,
        help="Pore count-density weak mean-correction minimum scale. Default 0.5.",
    )
    parser.add_argument(
        "--pore-rho-mean-correction-scale-max",
        type=float,
        default=None,
        help="Pore count-density weak mean-correction maximum scale. Default 2.0.",
    )
    parser.add_argument(
        "--throat-rho-mean-correction-rel-tol",
        type=float,
        default=None,
        help="Throat count-density weak mean-correction trigger threshold. Default inf (disabled).",
    )
    parser.add_argument(
        "--throat-rho-mean-correction-scale-min",
        type=float,
        default=None,
        help="Throat count-density weak mean-correction minimum scale. Default 0.5.",
    )
    parser.add_argument(
        "--throat-rho-mean-correction-scale-max",
        type=float,
        default=None,
        help="Throat count-density weak mean-correction maximum scale. Default 2.0.",
    )
    parser.add_argument(
        "--phase1-dir",
        type=str,
        default=str(OUTPUT_ROOT / "phase1_thickness"),
        help="阶段一结果目录",
    )
    parser.add_argument(
        "--phase2-dir",
        type=str,
        default=str(OUTPUT_ROOT / "phase2_parameters"),
        help="阶段二结果目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_ROOT / "phase3_synthetic"),
        help="输出目录",
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        default=None,
        help="输出样本名（默认：synthetic_{sample_type}_{geometry_type}）",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="随机种子",
    )
    parser.add_argument(
        "--no-thickness-field",
        action="store_true",
        help="不生成厚度场，使用均一厚度（geometry-thickness-mean），单 cell 无网格",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="跳过 PNG 绘图，仅导出 xlsx（pipeline 模式）",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="生成孔/喉 3D HTML 可视化（可与 --no-plots 同时使用，每个 run 独立文件）",
    )
    parser.add_argument(
        "--pore-radius-q4-spatial-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "最大孔半径五分位（pd.qcut）：该档一半随机重排半径，另一半使较大半径略倾向随机半质心附近。"
            "默认关闭；开启请用 --pore-radius-q4-spatial-bias。"
        ),
    )
    parser.add_argument(
        "--pore-radius-q4-bias-strength",
        type=float,
        default=0.55,
        help="距离偏置强度 0~1（默认 0.55，中等偏置）；0 表示该半档仍近似随机，1 为完全按距质心匹配大小",
    )
    args = parser.parse_args()

    # 加载参数
    phase1_params = load_phase1_parameters(Path(args.phase1_dir)) if not getattr(args, "no_thickness_field", False) else {}
    phase2_params = load_phase2_parameters(Path(args.phase2_dir))
    
    # 准备参数
    geometry_params = {
        "width": args.geometry_width,
        "height": args.geometry_height,
        "thickness_mean": args.geometry_thickness_mean,
        "grid_step_nm": args.geometry_grid_step_nm,
        "pore_placement_cell_nm": args.pore_placement_cell_nm,
        "throat_neighbor_nm": args.throat_neighbor_nm,
        "n_thickness_bins": args.n_thickness_bins,
        "rho_sampling_scope": args.rho_sampling_scope,
        "rho_mean_correction_rel_tol": args.rho_mean_correction_rel_tol,
        "rho_mean_correction_scale_min": args.rho_mean_correction_scale_min,
        "rho_mean_correction_scale_max": args.rho_mean_correction_scale_max,
        "pore_rho_mean_correction_rel_tol": (
            args.pore_rho_mean_correction_rel_tol
            if args.pore_rho_mean_correction_rel_tol is not None
            else (args.rho_mean_correction_rel_tol if args.rho_mean_correction_rel_tol is not None else float("inf"))
        ),
        "pore_rho_mean_correction_scale_min": (
            args.pore_rho_mean_correction_scale_min
            if args.pore_rho_mean_correction_scale_min is not None
            else (args.rho_mean_correction_scale_min if args.rho_mean_correction_scale_min is not None else 0.5)
        ),
        "pore_rho_mean_correction_scale_max": (
            args.pore_rho_mean_correction_scale_max
            if args.pore_rho_mean_correction_scale_max is not None
            else (args.rho_mean_correction_scale_max if args.rho_mean_correction_scale_max is not None else 2.0)
        ),
        "throat_rho_mean_correction_rel_tol": (
            args.throat_rho_mean_correction_rel_tol
            if args.throat_rho_mean_correction_rel_tol is not None
            else (args.rho_mean_correction_rel_tol if args.rho_mean_correction_rel_tol is not None else float("inf"))
        ),
        "throat_rho_mean_correction_scale_min": (
            args.throat_rho_mean_correction_scale_min
            if args.throat_rho_mean_correction_scale_min is not None
            else (args.rho_mean_correction_scale_min if args.rho_mean_correction_scale_min is not None else 0.5)
        ),
        "throat_rho_mean_correction_scale_max": (
            args.throat_rho_mean_correction_scale_max
            if args.throat_rho_mean_correction_scale_max is not None
            else (args.rho_mean_correction_scale_max if args.rho_mean_correction_scale_max is not None else 2.0)
        ),
    }
    
    # 单 cell 均一厚度模式：不传厚度分布，使用 thickness_mean
    if getattr(args, "no_thickness_field", False):
        phase1_params = {}
    
    density_params = phase2_params.get("density", {})
    radius_params = {
        "pore": phase2_params.get("radius", {}).get("pore", {}),
        "throat": phase2_params.get("radius", {}).get("throat", {}),
    }
    directionality_params = phase2_params.get("directionality", {})
    
    pore_count_fraction = phase2_params.get("pore_count_fraction_by_thickness_bin", {})
    rho_pore_dist = phase2_params.get("rho_pore_dist_by_thickness_bin", {})
    frac_pore_dist = phase2_params.get("frac_pore_dist_by_thickness_bin", {})
    pore_radius_dist = phase2_params.get("pore_radius_dist_by_thickness_bin", {})
    rho_throat_dist = phase2_params.get("rho_throat_dist_by_thickness_bin", {})
    frac_throat_dist = phase2_params.get("frac_throat_dist_by_thickness_bin", {})
    throat_length_dist = phase2_params.get("throat_length_dist_by_thickness_bin", {})
    degree_dist = phase2_params.get("degree_dist_by_thickness_bin", {})

    # 生成网络
    generator = GBMSyntheticNetwork(
        sample_type=args.sample_type,
        geometry_type=args.geometry_type,
        geometry_params=geometry_params,
        thickness_distribution=phase1_params,
        density_params=density_params,
        radius_params=radius_params,
        directionality_params=directionality_params,
        pore_count_fraction_by_thickness_bin=pore_count_fraction,
        rho_pore_dist_by_thickness_bin=rho_pore_dist,
        frac_pore_dist_by_thickness_bin=frac_pore_dist,
        pore_radius_dist_by_thickness_bin=pore_radius_dist,
        throat_radius_dist_by_thickness_bin=phase2_params.get("throat_radius_dist_by_thickness_bin", {}),
        throat_radius_dist_by_length_bin=phase2_params.get("throat_radius_dist_by_length_bin", {}),
        pore_throat_radius_correlation=phase2_params.get("pore_throat_radius_correlation", {}),
        rho_throat_dist_by_thickness_bin=rho_throat_dist,
        frac_throat_dist_by_thickness_bin=frac_throat_dist,
        throat_length_dist_by_thickness_bin=throat_length_dist,
        effective_throat_length_dist_by_thickness_bin=phase2_params.get("effective_throat_length_dist_by_thickness_bin", {}),
        overlap_throat_R_ratio_dist_by_thickness_bin=phase2_params.get(
            "overlap_throat_R_ratio_dist_by_thickness_bin", {}
        ),
        throat_gt_min_pore_frac_dist_by_thickness_bin=phase2_params.get(
            "throat_gt_min_pore_frac_dist_by_thickness_bin", {}
        ),
        nonoverlap_cross_density_dist_by_thickness_bin=phase2_params.get(
            "nonoverlap_cross_density_dist_by_thickness_bin", {}
        ),
        degree_dist_by_thickness_bin=degree_dist,
        random_seed=args.random_seed,
        pore_radius_q4_spatial_bias=bool(getattr(args, "pore_radius_q4_spatial_bias", True)),
        pore_radius_q4_bias_strength=float(getattr(args, "pore_radius_q4_bias_strength", 0.55)),
    )
    
    print("=" * 60)
    print(f"阶段三：合成 {args.sample_type} {args.geometry_type} 网络")
    print("=" * 60)
    
    generator.generate()

    output_dir = Path(args.output_dir)
    sample_name = args.sample_name or f"synthetic_{args.sample_type}_{args.geometry_type}"
    if not getattr(args, "no_plots", False):
        # 厚度场 vs GMM 对比图（使用厚度场时）
        generator.plot_thickness_field_vs_gmm(output_dir)
        # 孔半径、孔体积分数：模拟 vs Phase2 拟合（按厚度区间）
        generator.plot_pore_radius_simulated_vs_fit(output_dir)
        generator.plot_pore_frac_simulated_vs_fit(output_dir)
        generator.plot_throat_length_simulated_vs_fit(output_dir)
        generator.plot_pore_degree_simulated_vs_fit(output_dir)
        generator.plot_throat_count_fraction_simulated_vs_fit(output_dir)
        generator.plot_sampling_rho_debug_by_thickness_bins(output_dir)
        generator.plot_throat_radius_by_thickness_bins(output_dir)
        generator.plot_throat_radius_by_length_bins(output_dir)
        generator.plot_throat_radius_vs_mean_pore(output_dir)
        generator.plot_throat_radius_vs_length(output_dir)
        generator.plot_gt_min_pore_frac_raw_vs_phase2(output_dir)
        generator.plot_pores_3d_html(output_dir, sample_name=sample_name)
        generator.plot_throats_3d_html(output_dir, sample_name=sample_name)
    elif getattr(args, "html", False):
        # --no-plots 但 --html：仅生成孔/喉 3D HTML，每个 run 独立文件
        generator.plot_pores_3d_html(output_dir, sample_name=sample_name)
        generator.plot_throats_3d_html(output_dir, sample_name=sample_name)

    # 导出合成网络（供 Phase4 筛分计算使用）
    sample_name = args.sample_name or f"synthetic_{args.sample_type}_{args.geometry_type}"
    generator.export_to_xlsx(output_dir, sample_name)
    
    print("\n" + "=" * 60)
    print("阶段三完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

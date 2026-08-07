"""Run independent GBM sieving sensitivity / ablation experiments.

This script is intentionally separate from the existing phase1-4 pipeline.
It reads the existing phase1/phase2 parameter files, generates small synthetic
networks with selected ablations, calculates sieving, and writes compact CSV
summaries. It does not modify phase1/phase2 JSON files or existing figure code.

Default design:
- uniform single-cell synthetic samples, as in run_uniform_sieving_pipeline.py
- one paired thickness table shared by baseline and most ablations
- phase3 plots/html disabled
- temporary xlsx / phase4 folders removed after each sample unless requested
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

from gbm_sieving.simulation.network_generation.generator import (
    GBMSyntheticNetwork,
    load_phase1_parameters,
    load_phase2_parameters,
)
from gbm_sieving.data_io.aggregation import parse_sieving_results_dataframe, resolve_sieving_results_xlsx
from gbm_sieving.paths import OUTPUT_ROOT, PROJECT_ROOT, SRC_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent
KIDNEY_ROOT = PROJECT_ROOT
DEFAULT_PHASE1_DIR = OUTPUT_ROOT / "phase1_thickness"
DEFAULT_PHASE2_DIR = OUTPUT_ROOT / "phase2_parameters"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "parameter_replacement"
SOLUTE_RADIUS_NM = 3.55
SIEVING_VALID_EPS = 1e-12


DEFAULT_VARIANTS = [
    "baseline",
    "as_to_wt_throat_radius_overall",
    "as_to_wt_throat_density_overall",
    "as_to_wt_throat_length_overall",
    "as_to_wt_degree_overall",
]


VARIANT_SHORT_NAMES = {
    "baseline": "baseline",
    "no_throat_radius_thickness": "no_radius_thick",
    "no_throat_density_thickness": "no_density_thick",
    "no_degree_thickness": "no_degree_thick",
    "no_throat_length_thickness": "no_length_thick",
    "no_radius_correlations": "no_radius_corr",
    "no_all_throat_thickness_dependence": "no_all_throat_dep",
    "thickness_uniform_range": "thick_uniform",
    "binmap_random_other_throat_all": "binmap_rand_all",
    "binmap_random_other_throat_radius": "binmap_rand_radius",
    "binmap_random_other_throat_density": "binmap_rand_density",
    "binmap_random_other_throat_length": "binmap_rand_length",
    "binmap_random_other_degree": "binmap_rand_degree",
    "binmap_random_other_passable_overlap": "binmap_rand_passable",
    "binmap_cycle_throat_all": "binmap_cycle_all",
    "binmap_cycle_throat_radius": "binmap_cycle_radius",
    "binmap_cycle_throat_density": "binmap_cycle_density",
    "binmap_cycle_throat_length": "binmap_cycle_length",
    "binmap_cycle_degree": "binmap_cycle_degree",
    "binmap_cycle_passable_overlap": "binmap_cycle_passable",
    "as_to_wt_throat_radius_overall": "as2wt_radius",
    "as_to_wt_throat_density_overall": "as2wt_density",
    "as_to_wt_throat_length_overall": "as2wt_length",
    "as_to_wt_degree_overall": "as2wt_degree",
    "as_to_wt_passable_overlap_overall": "as2wt_passable",
    "as_to_wt_pore_overall": "as2wt_pore",
    "as_to_wt_pore_radius_overall": "as2wt_pore_radius",
    "as_to_wt_pore_count_overall": "as2wt_pore_count",
    "wt_to_as_pore_overall": "wt2as_pore",
    "wt_to_as_pore_radius_overall": "wt2as_pore_radius",
    "wt_to_as_pore_count_overall": "wt2as_pore_count",
    "wt_to_as_throat_radius_overall": "wt2as_radius",
    "wt_to_as_throat_density_overall": "wt2as_density",
    "wt_to_as_throat_length_overall": "wt2as_length",
    "wt_to_as_degree_overall": "wt2as_degree",
    "wt_to_as_passable_overlap_overall": "wt2as_passable",
}


BIN_MAPPING_VARIANTS: dict[str, tuple[str, str]] = {
    "binmap_random_other_throat_all": ("throat_all", "random_other"),
    "binmap_random_other_throat_radius": ("throat_radius", "random_other"),
    "binmap_random_other_throat_density": ("throat_density", "random_other"),
    "binmap_random_other_throat_length": ("throat_length", "random_other"),
    "binmap_random_other_degree": ("degree", "random_other"),
    "binmap_random_other_passable_overlap": ("passable_overlap", "random_other"),
    "binmap_cycle_throat_all": ("throat_all", "cycle"),
    "binmap_cycle_throat_radius": ("throat_radius", "cycle"),
    "binmap_cycle_throat_density": ("throat_density", "cycle"),
    "binmap_cycle_throat_length": ("throat_length", "cycle"),
    "binmap_cycle_degree": ("degree", "cycle"),
    "binmap_cycle_passable_overlap": ("passable_overlap", "cycle"),
}


BIN_MAPPING_PARAMETER_KEYS: dict[str, tuple[str, ...]] = {
    "throat_radius": (
        "throat_radius_dist_by_thickness_bin",
        "throat_radius_dist_by_length_bin",
    ),
    "passable_overlap": (
        "overlap_throat_R_ratio_dist_by_thickness_bin",
        "throat_gt_min_pore_frac_dist_by_thickness_bin",
    ),
    "throat_density": (
        "rho_throat_dist_by_thickness_bin",
        "nonoverlap_cross_density_dist_by_thickness_bin",
    ),
    "throat_length": (
        "throat_length_dist_by_thickness_bin",
        "effective_throat_length_dist_by_thickness_bin",
    ),
    "degree": ("degree_dist_by_thickness_bin",),
}
BIN_MAPPING_PARAMETER_KEYS["throat_all"] = tuple(
    dict.fromkeys(
        BIN_MAPPING_PARAMETER_KEYS["throat_radius"]
        + BIN_MAPPING_PARAMETER_KEYS["throat_density"]
        + BIN_MAPPING_PARAMETER_KEYS["throat_length"]
        + BIN_MAPPING_PARAMETER_KEYS["degree"]
        + BIN_MAPPING_PARAMETER_KEYS["passable_overlap"]
    )
)


CONDITION_SWAP_VARIANTS: dict[str, str] = {
    "as_to_wt_pore_overall": "pore",
    "as_to_wt_pore_radius_overall": "pore_radius",
    "as_to_wt_pore_count_overall": "pore_count",
    "as_to_wt_throat_radius_overall": "throat_radius",
    "as_to_wt_throat_density_overall": "throat_density",
    "as_to_wt_throat_length_overall": "throat_length",
    "as_to_wt_degree_overall": "degree",
    "as_to_wt_passable_overlap_overall": "passable_overlap",
}


WT_TO_AS_CONDITION_SWAP_VARIANTS: dict[str, str] = {
    "wt_to_as_pore_overall": "pore",
    "wt_to_as_pore_radius_overall": "pore_radius",
    "wt_to_as_pore_count_overall": "pore_count",
    "wt_to_as_throat_radius_overall": "throat_radius",
    "wt_to_as_throat_density_overall": "throat_density",
    "wt_to_as_throat_length_overall": "throat_length",
    "wt_to_as_degree_overall": "degree",
    "wt_to_as_passable_overlap_overall": "passable_overlap",
}


ALL_CONDITION_SWAP_VARIANTS = {
    **CONDITION_SWAP_VARIANTS,
    **WT_TO_AS_CONDITION_SWAP_VARIANTS,
}


CONDITION_SWAP_PORE_COUNT_SUPPORT_KEYS: tuple[str, ...] = (
    "pore_count_fraction_by_thickness_bin",
    "rho_pore_dist_by_thickness_bin",
    "frac_pore_dist_by_thickness_bin",
)


CONDITION_SWAP_PORE_RADIUS_SUPPORT_KEYS: tuple[str, ...] = (
    "pore_radius_dist_by_thickness_bin",
)


CONDITION_SWAP_PARAMETER_KEYS: dict[str, tuple[str, ...]] = {
    "pore": CONDITION_SWAP_PORE_COUNT_SUPPORT_KEYS + CONDITION_SWAP_PORE_RADIUS_SUPPORT_KEYS,
    "pore_count": CONDITION_SWAP_PORE_COUNT_SUPPORT_KEYS,
    "pore_radius": CONDITION_SWAP_PORE_RADIUS_SUPPORT_KEYS,
    "throat_radius": (
        *CONDITION_SWAP_PORE_RADIUS_SUPPORT_KEYS,
        "throat_radius_dist_by_thickness_bin",
        "throat_radius_dist_by_length_bin",
    ),
    "passable_overlap": (
        "overlap_throat_R_ratio_dist_by_thickness_bin",
        "throat_gt_min_pore_frac_dist_by_thickness_bin",
    ),
    "throat_density": (
        *CONDITION_SWAP_PORE_COUNT_SUPPORT_KEYS,
        "rho_throat_dist_by_thickness_bin",
        "frac_throat_dist_by_thickness_bin",
        "nonoverlap_cross_density_dist_by_thickness_bin",
    ),
    "throat_length": (
        "throat_length_dist_by_thickness_bin",
        "effective_throat_length_dist_by_thickness_bin",
    ),
    "degree": (
        "degree_dist_by_thickness_bin",
    ),
}
WT_TO_AS_PARAMETER_KEYS: dict[str, tuple[str, ...]] = {
    "pore": CONDITION_SWAP_PARAMETER_KEYS["pore"],
    "pore_count": CONDITION_SWAP_PARAMETER_KEYS["pore_count"],
    "pore_radius": CONDITION_SWAP_PARAMETER_KEYS["pore_radius"],
    "throat_radius": (
        *CONDITION_SWAP_PORE_RADIUS_SUPPORT_KEYS,
        "throat_radius_dist_by_thickness_bin",
        "throat_radius_dist_by_length_bin",
    ),
    "passable_overlap": (
        "overlap_throat_R_ratio_dist_by_thickness_bin",
        "throat_gt_min_pore_frac_dist_by_thickness_bin",
    ),
    "throat_density": (
        *CONDITION_SWAP_PORE_COUNT_SUPPORT_KEYS,
        "rho_throat_dist_by_thickness_bin",
        "frac_throat_dist_by_thickness_bin",
        "nonoverlap_cross_density_dist_by_thickness_bin",
    ),
    "throat_length": (
        "throat_length_dist_by_thickness_bin",
        "effective_throat_length_dist_by_thickness_bin",
    ),
    "degree": (
        "degree_dist_by_thickness_bin",
    ),
}
GENERATED_PARAMETER_GROUPS: dict[str, tuple[str, ...]] = {
    "pore": (
        "pore_density_count_per_nm3",
        "pore_radius_mean_nm",
        "n_pores",
    ),
    "pore_count": (
        "pore_density_count_per_nm3",
        "n_pores",
    ),
    "pore_radius": (
        "pore_radius_mean_nm",
    ),
    "throat_radius": (
        "pore_radius_mean_nm",
        "throat_radius_mean_nm",
        "throat_radius_median_nm",
        "passable_throat_fraction",
    ),
    "throat_density": (
        "pore_density_count_per_nm3",
        "throat_density_count_per_nm3",
        "n_pores",
        "n_throats",
    ),
    "throat_length": (
        "throat_length_mean_nm",
        "throat_length_median_nm",
    ),
    "degree": (
        "pore_degree_mean",
        "pore_degree_median",
    ),
    "passable_overlap": (
        "passable_throat_fraction",
    ),
}
GENERATED_PARAMETER_SOURCE_KEYS: dict[str, tuple[str, bool]] = {
    "pore_density_count_per_nm3": ("rho_pore_dist_by_thickness_bin", False),
    "pore_radius_mean_nm": ("pore_radius_dist_by_thickness_bin", False),
    "throat_radius_mean_nm": ("throat_radius_dist_by_thickness_bin", False),
    "throat_radius_median_nm": ("throat_radius_dist_by_thickness_bin", False),
    "throat_density_count_per_nm3": ("rho_throat_dist_by_thickness_bin", False),
    "throat_length_mean_nm": ("throat_length_dist_by_thickness_bin", False),
    "throat_length_median_nm": ("throat_length_dist_by_thickness_bin", False),
    "pore_degree_mean": ("degree_dist_by_thickness_bin", True),
    "pore_degree_median": ("degree_dist_by_thickness_bin", True),
    "passable_throat_fraction": ("throat_gt_min_pore_frac_dist_by_thickness_bin", False),
}


CONDITION_SWAP_DISABLE_RADIUS_REWEIGHTING = {
    "as_to_wt_throat_radius_overall",
    "wt_to_as_throat_radius_overall",
}


CONDITION_SWAP_COPY_CORRELATION_GROUPS = {
    "throat_radius",
}


def _variant_output_name(variant_index: int, variant: str) -> str:
    short = VARIANT_SHORT_NAMES.get(variant, variant[:32].strip("_") or "variant")
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in short)
    return f"v{variant_index:02d}_{safe}"


def _json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _safe_float(x: Any, default: float = math.nan) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if math.isfinite(v) else default


def _weighted_pool_fit(bin_records: list[dict], *, degree: bool = False) -> dict:
    rows: list[tuple[float, float, float]] = []
    for rec in bin_records:
        fit = (rec or {}).get("fit") or {}
        if not fit:
            continue
        n = _safe_float(fit.get("n", rec.get("n", 1)), 1.0)
        mean = _safe_float(fit.get("mean"), math.nan)
        std = _safe_float(fit.get("std"), math.nan)
        if not math.isfinite(mean):
            dist = fit.get("distribution") or {}
            params = dist.get("params") or []
            if dist.get("name") == "norm" and len(params) >= 2:
                mean = _safe_float(params[0], math.nan)
                std = _safe_float(params[1], math.nan)
        if not math.isfinite(mean):
            continue
        if not math.isfinite(std) or std < 0:
            std = 0.0
        rows.append((max(n, 1.0), mean, std))

    if not rows:
        mean = 2.0 if degree else 1.0
        std = 1.0
        n_total = 1
    else:
        w = np.array([r[0] for r in rows], dtype=float)
        mu = np.array([r[1] for r in rows], dtype=float)
        sd = np.array([r[2] for r in rows], dtype=float)
        n_total = int(np.sum(w))
        mean = float(np.sum(w * mu) / np.sum(w))
        var = float(np.sum(w * (sd**2 + (mu - mean) ** 2)) / np.sum(w))
        std = math.sqrt(max(var, 1e-20))

    if degree:
        lam = max(mean, 0.0)
        return {
            "fit_type": "parametric",
            "n": n_total,
            "mean": lam,
            "std": std,
            "distribution": {"name": "poisson", "params": [lam]},
        }
    return {
        "fit_type": "parametric",
        "n": n_total,
        "mean": mean,
        "std": max(std, 1e-12),
        "distribution": {"name": "norm", "params": [mean, max(std, 1e-12)]},
    }


def _pool_by_thickness_bins(dist_by_sample: dict, *, degree: bool = False) -> dict:
    """Replace each thickness bin fit by a sample-type pooled fit."""
    out = copy.deepcopy(dist_by_sample or {})
    for sample_type, block in list(out.items()):
        if not isinstance(block, dict):
            continue
        bins = block.get("bins") or []
        if not bins:
            continue
        pooled_fit = _weighted_pool_fit(bins, degree=degree)
        new_bins = []
        for rec in bins:
            rr = copy.deepcopy(rec)
            rr["fit"] = copy.deepcopy(pooled_fit)
            rr["n"] = int(pooled_fit.get("n", rr.get("n", 1)))
            rr["mean"] = float(pooled_fit.get("mean", rr.get("mean", math.nan)))
            rr["std"] = float(pooled_fit.get("std", rr.get("std", 0.0)))
            rr["distribution"] = copy.deepcopy(pooled_fit.get("distribution", {}))
            new_bins.append(rr)
        block["bins"] = new_bins
        block["ablation_note"] = "all thickness bins use the same pooled fit"
    return out


def _pool_nested_kind_sample(dist_nested: dict, *, degree: bool = False) -> dict:
    """Pool structures shaped as kind -> sample_type -> {bin_edges,bins}."""
    out = copy.deepcopy(dist_nested or {})
    for kind, by_sample in list(out.items()):
        if isinstance(by_sample, dict):
            out[kind] = _pool_by_thickness_bins(by_sample, degree=degree)
    return out


def _pooled_fit_for_sample_block(obj: dict, source_sample_type: str, *, degree: bool = False) -> dict | None:
    block = obj.get(source_sample_type)
    if not isinstance(block, dict):
        return None
    bins = block.get("bins") or []
    if not bins:
        return None
    return _weighted_pool_fit(bins, degree=degree)


def _replace_sample_bins_with_fit(
    obj: dict,
    target_sample_type: str,
    pooled_fit: dict,
    *,
    source_sample_type: str,
    key_name: str,
) -> int:
    block = obj.get(target_sample_type)
    if not isinstance(block, dict):
        return 0
    bins = block.get("bins") or []
    if not bins:
        return 0
    new_bins = []
    for rec in bins:
        rr = copy.deepcopy(rec)
        rr["fit"] = copy.deepcopy(pooled_fit)
        rr["n"] = int(pooled_fit.get("n", rr.get("n", 1)))
        rr["mean"] = float(pooled_fit.get("mean", rr.get("mean", math.nan)))
        rr["std"] = float(pooled_fit.get("std", rr.get("std", 0.0)))
        rr["distribution"] = copy.deepcopy(pooled_fit.get("distribution", {}))
        rr["condition_swap_source_sample_type"] = source_sample_type
        rr["condition_swap_key"] = key_name
        new_bins.append(rr)
    block["bins"] = new_bins
    block["condition_swap_note"] = f"{target_sample_type} bins use pooled {source_sample_type} overall fit"
    return 1


def _replace_target_bins_with_source_overall(
    obj: dict,
    *,
    target_sample_type: str,
    source_sample_type: str,
    degree: bool = False,
    key_name: str,
) -> int:
    """Recursively replace target sample bins by a source sample pooled overall fit.

    This covers both direct structures shaped as sample_type -> {bin_edges,bins}
    and nested structures shaped as kind -> sample_type -> {bin_edges,bins}.
    """
    if not isinstance(obj, dict):
        return 0
    changed = 0
    pooled_fit = _pooled_fit_for_sample_block(obj, source_sample_type, degree=degree)
    if pooled_fit is not None:
        changed += _replace_sample_bins_with_fit(
            obj,
            target_sample_type,
            pooled_fit,
            source_sample_type=source_sample_type,
            key_name=key_name,
        )
    for value in obj.values():
        if isinstance(value, dict):
            changed += _replace_target_bins_with_source_overall(
                value,
                target_sample_type=target_sample_type,
                source_sample_type=source_sample_type,
                degree=degree,
                key_name=key_name,
            )
    return changed


def _record_fit_mean_std(rec: dict | None) -> tuple[float, float]:
    fit = (rec or {}).get("fit") or {}
    mean = _safe_float(fit.get("mean"), math.nan)
    std = _safe_float(fit.get("std"), math.nan)
    if not math.isfinite(mean):
        dist = fit.get("distribution") or {}
        params = dist.get("params") or []
        if dist.get("name") == "norm" and len(params) >= 2:
            mean = _safe_float(params[0], math.nan)
            std = _safe_float(params[1], math.nan)
    if not math.isfinite(std) or std < 0:
        std = 0.0
    return mean, std


def _thickness_bin_record(dist_by_sample: dict, sample_type: str, thickness_nm: float) -> dict | None:
    block = (dist_by_sample or {}).get(sample_type, {})
    if not isinstance(block, dict):
        return None
    edges = block.get("bin_edges") or []
    bins = block.get("bins") or []
    if len(edges) < 2 or not bins:
        return None
    k = int(np.searchsorted(np.asarray(edges, dtype=float), float(thickness_nm), side="right") - 1)
    k = int(np.clip(k, 0, min(len(edges) - 2, len(bins) - 1)))
    return bins[k]


def _thickness_group(thickness_nm: float) -> str:
    """Coarse groups used for thickness-structure bin-mapping ablations."""
    t = float(thickness_nm)
    if t < 60.0:
        return "thin"
    if t <= 160.0:
        return "normal"
    return "thick"


def _representative_group_thickness(sample_type: str, group: str) -> float:
    st = str(sample_type).upper()
    if st == "WT":
        reps = {"thin": 60.0, "normal": 110.0, "thick": 165.0}
    else:
        reps = {"thin": 40.0, "normal": 110.0, "thick": 220.0}
    return float(reps[group])


def _mapped_source_thickness(
    sample_type: str,
    thickness_nm: float,
    mode: str,
    rng: np.random.Generator | None = None,
) -> tuple[str, str, float]:
    target_group = _thickness_group(thickness_nm)
    if mode == "cycle":
        # Deliberately break every coarse thickness-structure correspondence:
        # thin -> normal, normal -> thick, thick -> thin.
        source_group = {"thin": "normal", "normal": "thick", "thick": "thin"}[target_group]
    elif mode == "random_other":
        if rng is None:
            rng = np.random.default_rng()
        candidates = [g for g in ("thin", "normal", "thick") if g != target_group]
        source_group = str(rng.choice(candidates))
    elif mode == "swap_thin_thick":
        source_group = {"thin": "thick", "normal": "normal", "thick": "thin"}[target_group]
    else:
        raise ValueError(f"Unknown bin-mapping mode: {mode}")
    return target_group, source_group, _representative_group_thickness(sample_type, source_group)


def _replace_bin_record_with_source(block: dict, target_t: float, source_t: float) -> bool:
    edges = block.get("bin_edges") or []
    bins = block.get("bins") or []
    if len(edges) < 2 or not bins:
        return False
    edges_arr = np.asarray(edges, dtype=float)
    target_idx = int(np.searchsorted(edges_arr, float(target_t), side="right") - 1)
    source_idx = int(np.searchsorted(edges_arr, float(source_t), side="right") - 1)
    max_idx = min(len(edges_arr) - 2, len(bins) - 1)
    target_idx = int(np.clip(target_idx, 0, max_idx))
    source_idx = int(np.clip(source_idx, 0, max_idx))
    if target_idx >= len(bins) or source_idx >= len(bins):
        return False
    source_rec = copy.deepcopy(bins[source_idx])
    target_rec = copy.deepcopy(bins[target_idx])
    for key in ("fit", "n", "mean", "std", "distribution"):
        if key in source_rec:
            target_rec[key] = copy.deepcopy(source_rec[key])
    target_rec["bin_mapping_source_bin_index"] = int(source_idx)
    target_rec["bin_mapping_target_bin_index"] = int(target_idx)
    target_rec["bin_mapping_source_thickness_nm"] = float(source_t)
    bins[target_idx] = target_rec
    return True


def _remap_sample_type_bins_inplace(obj: dict, sample_type: str, target_t: float, source_t: float) -> int:
    """Recursively replace the target thickness-bin fit by a source-bin fit."""
    if not isinstance(obj, dict):
        return 0
    n_changed = 0
    block = obj.get(sample_type)
    if isinstance(block, dict) and "bins" in block and "bin_edges" in block:
        if _replace_bin_record_with_source(block, target_t, source_t):
            n_changed += 1
    for value in obj.values():
        if isinstance(value, dict):
            n_changed += _remap_sample_type_bins_inplace(value, sample_type, target_t, source_t)
    return n_changed


def _condition_swap_config(variant: str) -> tuple[str, str, str] | None:
    """Return property group, source sample type, and target sample type."""
    if variant in CONDITION_SWAP_VARIANTS:
        return CONDITION_SWAP_VARIANTS[variant], "WT", "AS"
    if variant in WT_TO_AS_CONDITION_SWAP_VARIANTS:
        return WT_TO_AS_CONDITION_SWAP_VARIANTS[variant], "AS", "WT"
    return None


def _warn_condition_swap_noops(variants: list[str], sample_types: list[str]) -> None:
    active_types = {str(s).upper() for s in sample_types}
    for variant in variants:
        cfg = _condition_swap_config(variant)
        if cfg is None:
            continue
        property_group, source_sample_type, target_sample_type = cfg
        if target_sample_type not in active_types:
            print(
                "警告: variant "
                f"{variant!r} ({source_sample_type}->{target_sample_type}, {property_group}) "
                f"不会作用于当前 --sample-types={sorted(active_types)}；结果将等价于未替换。"
            )


def _condition_swap_parameter_keys(variant: str, property_group: str) -> tuple[str, ...]:
    if variant in WT_TO_AS_CONDITION_SWAP_VARIANTS:
        return WT_TO_AS_PARAMETER_KEYS[property_group]
    return CONDITION_SWAP_PARAMETER_KEYS[property_group]


def _apply_condition_swap_overall_variant(
    phase2_params: dict,
    variant: str,
    sample_type: str,
) -> dict[str, Any]:
    config = _condition_swap_config(variant)
    if config is None:
        return {}
    property_group, source_sample_type, target_sample_type = config
    if str(sample_type).upper() != target_sample_type:
        return {
            "condition_swap_applied": False,
            "condition_swap_property_group": property_group,
            "condition_swap_source_sample_type": source_sample_type,
            "condition_swap_target_sample_type": target_sample_type,
            "condition_swap_changed_blocks": 0,
            "condition_swap_copied_blocks": 0,
            "condition_swap_disable_radius_reweighting": False,
        }

    changed = 0
    for key in _condition_swap_parameter_keys(variant, property_group):
        obj = phase2_params.get(key)
        if not isinstance(obj, dict):
            continue
        changed += _replace_target_bins_with_source_overall(
            obj,
            target_sample_type=target_sample_type,
            source_sample_type=source_sample_type,
            degree=(key == "degree_dist_by_thickness_bin"),
            key_name=key,
        )
    copied_blocks = 0
    if property_group in CONDITION_SWAP_COPY_CORRELATION_GROUPS:
        corr = phase2_params.get("pore_throat_radius_correlation")
        if isinstance(corr, dict) and source_sample_type in corr:
            corr[target_sample_type] = copy.deepcopy(corr[source_sample_type])
            copied_blocks += 1
    return {
        "condition_swap_applied": True,
        "condition_swap_property_group": property_group,
        "condition_swap_source_sample_type": source_sample_type,
        "condition_swap_target_sample_type": target_sample_type,
        "condition_swap_changed_blocks": int(changed),
        "condition_swap_copied_blocks": int(copied_blocks),
        "condition_swap_disable_radius_reweighting": variant in CONDITION_SWAP_DISABLE_RADIUS_REWEIGHTING,
    }


def _apply_sample_level_variant(
    phase2_params: dict,
    variant: str,
    sample_type: str,
    thickness_nm: float,
    seed: int,
) -> dict[str, Any]:
    condition_meta = _apply_condition_swap_overall_variant(phase2_params, variant, sample_type)
    if condition_meta:
        return condition_meta
    if variant not in BIN_MAPPING_VARIANTS:
        return {}
    property_group, mode = BIN_MAPPING_VARIANTS[variant]
    rng = np.random.default_rng(int(seed) + 917_263)
    target_group, source_group, source_t = _mapped_source_thickness(sample_type, thickness_nm, mode, rng)
    changed = 0
    for key in BIN_MAPPING_PARAMETER_KEYS[property_group]:
        if key in phase2_params and isinstance(phase2_params[key], dict):
            changed += _remap_sample_type_bins_inplace(phase2_params[key], sample_type, thickness_nm, source_t)
    return {
        "bin_mapping_mode": mode,
        "bin_mapping_property_group": property_group,
        "bin_mapping_target_group": target_group,
        "bin_mapping_source_group": source_group,
        "bin_mapping_source_thickness_nm": float(source_t),
        "bin_mapping_changed_blocks": int(changed),
    }


def _apply_uniform_thickness_density_fallback(
    phase2_params: dict,
    sample_type: str,
    thickness_nm: float,
) -> None:
    """Make uniform-thickness samples use the current throat-density bin.

    GBMSyntheticNetwork only uses per-cell thickness-binned density when a
    thickness field exists. The ablation experiment intentionally generates
    fixed-thickness uniform slabs, so we copy the matching throat-density bin
    mean into the global rho_throat fallback before constructing the generator.
    """
    rec = _thickness_bin_record(phase2_params.get("rho_throat_dist_by_thickness_bin", {}), sample_type, thickness_nm)
    mean, std = _record_fit_mean_std(rec)
    if not math.isfinite(mean):
        return
    density_root = phase2_params.setdefault("density", {}).setdefault("rho_throat", {})
    density_rec = density_root.setdefault(sample_type, {})
    density_rec["mean"] = float(max(mean, 0.0))
    density_rec["std"] = float(max(std, 0.0))


def _apply_variant_to_phase2_params(base_params: dict, variant: str) -> tuple[dict, dict[str, float]]:
    params = copy.deepcopy(base_params)
    weights = {"w_thickness": 0.4, "w_pore": 0.25, "w_length": 0.03}

    def pool_throat_radius() -> None:
        params["throat_radius_dist_by_thickness_bin"] = _pool_by_thickness_bins(
            params.get("throat_radius_dist_by_thickness_bin", {})
        )

    def pool_throat_density() -> None:
        params["rho_throat_dist_by_thickness_bin"] = _pool_by_thickness_bins(
            params.get("rho_throat_dist_by_thickness_bin", {})
        )
        params["nonoverlap_cross_density_dist_by_thickness_bin"] = _pool_by_thickness_bins(
            params.get("nonoverlap_cross_density_dist_by_thickness_bin", {})
        )

    def pool_degree() -> None:
        params["degree_dist_by_thickness_bin"] = _pool_by_thickness_bins(
            params.get("degree_dist_by_thickness_bin", {}), degree=True
        )

    def pool_throat_length() -> None:
        params["throat_length_dist_by_thickness_bin"] = _pool_by_thickness_bins(
            params.get("throat_length_dist_by_thickness_bin", {})
        )

    def pool_passable_and_overlap() -> None:
        params["throat_gt_min_pore_frac_dist_by_thickness_bin"] = _pool_nested_kind_sample(
            params.get("throat_gt_min_pore_frac_dist_by_thickness_bin", {})
        )
        params["overlap_throat_R_ratio_dist_by_thickness_bin"] = _pool_by_thickness_bins(
            params.get("overlap_throat_R_ratio_dist_by_thickness_bin", {})
        )

    if (
        variant in {"baseline", "thickness_empirical_bootstrap", "thickness_uniform_range"}
        or variant in BIN_MAPPING_VARIANTS
        or variant in ALL_CONDITION_SWAP_VARIANTS
    ):
        return params, weights
    if variant == "no_throat_radius_thickness":
        pool_throat_radius()
    elif variant == "no_throat_density_thickness":
        pool_throat_density()
    elif variant == "no_degree_thickness":
        pool_degree()
    elif variant == "no_throat_length_thickness":
        pool_throat_length()
    elif variant == "no_radius_correlations":
        weights["w_pore"] = 0.0
        weights["w_length"] = 0.0
    elif variant == "no_all_throat_thickness_dependence":
        pool_throat_radius()
        pool_throat_density()
        pool_degree()
        pool_throat_length()
        pool_passable_and_overlap()
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return params, weights


def _entry_sample(entry: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    kind = entry.get("type")
    if kind == "gmm":
        w = np.asarray(entry["weights"], dtype=float)
        mu = np.asarray(entry["means"], dtype=float)
        cov = np.asarray(entry["covariances"], dtype=float)
        sigma = np.sqrt(np.maximum(cov, 0.0))
        idx = rng.choice(len(w), size=n, p=w / np.sum(w))
        return rng.normal(mu[idx], sigma[idx], size=n)
    if kind == "parametric":
        dist = getattr(stats, str(entry.get("distribution")))
        return np.asarray(dist.rvs(*entry.get("params", []), size=n, random_state=rng), dtype=float)
    if kind == "kde":
        raw = np.asarray(entry.get("kde_data", []), dtype=float)
        raw = raw[np.isfinite(raw)]
        if raw.size >= 2:
            return np.asarray(gaussian_kde(raw).resample(n, seed=rng)).reshape(-1)
    raise ValueError(f"Cannot sample thickness entry of type {kind!r}")


def _raw_thickness_values(phase1_params: dict, phase1_dir: Path, sample_type: str) -> np.ndarray:
    key = f"{sample_type}_kde"
    entry = phase1_params.get(key) or phase1_params.get(sample_type) or {}
    raw = np.asarray(entry.get("kde_data", []), dtype=float)
    raw = raw[np.isfinite(raw)]
    if raw.size:
        return raw

    raw_file = phase1_dir / "thickness_data_raw.xlsx"
    if raw_file.exists():
        df = pd.read_excel(raw_file)
        if "thickness" in df.columns and "is_wt" in df.columns:
            want_wt = sample_type.upper() == "WT"
            arr = df[df["is_wt"] == want_wt]["thickness"].dropna().to_numpy(dtype=float)
            if arr.size:
                return arr
    raise ValueError(f"No raw thickness values found for {sample_type}")


def _resolve_thickness_entry(phase1_params: dict, sample_type: str, wt_fit: str, as_fit: str) -> dict:
    sample_type = sample_type.upper()
    if sample_type == "WT":
        if wt_fit == "kde" and phase1_params.get("WT_kde"):
            return phase1_params["WT_kde"]
        if wt_fit == "norm" and phase1_params.get("WT_norm"):
            return phase1_params["WT_norm"]
        return phase1_params["WT"]
    if sample_type == "AS":
        if as_fit == "kde" and phase1_params.get("AS_kde"):
            return phase1_params["AS_kde"]
        if as_fit == "gmm" and phase1_params.get("AS"):
            return phase1_params["AS"]
        return phase1_params["AS"]
    raise ValueError(f"Unknown sample type: {sample_type}")


def _sample_thickness_table(
    phase1_params: dict,
    phase1_dir: Path,
    sample_type: str,
    n: int,
    rng: np.random.Generator,
    mode: str,
    wt_fit: str,
    as_fit: str,
) -> np.ndarray:
    sample_type = sample_type.upper()
    if mode == "fitted":
        values = _entry_sample(_resolve_thickness_entry(phase1_params, sample_type, wt_fit, as_fit), n, rng)
    elif mode == "empirical_bootstrap":
        raw = _raw_thickness_values(phase1_params, phase1_dir, sample_type)
        values = raw[rng.integers(0, len(raw), size=n)]
    elif mode == "uniform_range":
        raw = _raw_thickness_values(phase1_params, phase1_dir, sample_type)
        values = rng.uniform(float(np.min(raw)), float(np.max(raw)), size=n)
    else:
        raise ValueError(f"Unknown thickness mode: {mode}")

    min_t = 50.0 if sample_type == "WT" else 20.0
    values = np.asarray(values, dtype=float)
    values = np.maximum(values, 1.0)
    bad = ~np.isfinite(values) | (values < min_t)
    guard = 0
    while np.any(bad):
        guard += 1
        if guard > 100:
            values[bad] = min_t
            break
        if mode == "uniform_range":
            raw = _raw_thickness_values(phase1_params, phase1_dir, sample_type)
            repl = rng.uniform(max(float(np.min(raw)), min_t), float(np.max(raw)), size=int(np.sum(bad)))
        elif mode == "empirical_bootstrap":
            raw = _raw_thickness_values(phase1_params, phase1_dir, sample_type)
            raw = raw[np.isfinite(raw) & (raw >= min_t)]
            repl = raw[rng.integers(0, len(raw), size=int(np.sum(bad)))]
        else:
            repl = _entry_sample(_resolve_thickness_entry(phase1_params, sample_type, wt_fit, as_fit), int(np.sum(bad)), rng)
        values[bad] = np.asarray(repl, dtype=float)
        bad = ~np.isfinite(values) | (values < min_t)
    return values


def _build_generator(
    sample_type: str,
    thickness_nm: float,
    base_edge_nm: float,
    phase2_params: dict,
    seed: int,
    pore_cell_nm: float,
    q4_bias: bool,
    q4_bias_strength: float,
    rho_sampling_scope: str,
    pore_rho_mean_correction_rel_tol: float,
    pore_rho_mean_correction_scale_min: float,
    pore_rho_mean_correction_scale_max: float,
    throat_rho_mean_correction_rel_tol: float,
    throat_rho_mean_correction_scale_min: float,
    throat_rho_mean_correction_scale_max: float,
) -> GBMSyntheticNetwork:
    _apply_uniform_thickness_density_fallback(phase2_params, sample_type, thickness_nm)
    geometry_params = {
        "width": float(base_edge_nm),
        "height": float(base_edge_nm),
        "thickness_mean": float(thickness_nm),
        "grid_step_nm": 5.0,
        "pore_placement_cell_nm": float(pore_cell_nm),
        "n_thickness_bins": 20,
        "rho_sampling_scope": str(rho_sampling_scope),
        "pore_rho_mean_correction_rel_tol": float(pore_rho_mean_correction_rel_tol),
        "pore_rho_mean_correction_scale_min": float(pore_rho_mean_correction_scale_min),
        "pore_rho_mean_correction_scale_max": float(pore_rho_mean_correction_scale_max),
        "throat_rho_mean_correction_rel_tol": float(throat_rho_mean_correction_rel_tol),
        "throat_rho_mean_correction_scale_min": float(throat_rho_mean_correction_scale_min),
        "throat_rho_mean_correction_scale_max": float(throat_rho_mean_correction_scale_max),
    }
    return GBMSyntheticNetwork(
        sample_type=sample_type,
        geometry_type="plane",
        geometry_params=geometry_params,
        thickness_distribution={},
        density_params=phase2_params.get("density", {}),
        radius_params={
            "pore": phase2_params.get("radius", {}).get("pore", {}),
            "throat": phase2_params.get("radius", {}).get("throat", {}),
        },
        directionality_params=phase2_params.get("directionality", {}),
        pore_count_fraction_by_thickness_bin=phase2_params.get("pore_count_fraction_by_thickness_bin", {}),
        rho_pore_dist_by_thickness_bin=phase2_params.get("rho_pore_dist_by_thickness_bin", {}),
        frac_pore_dist_by_thickness_bin=phase2_params.get("frac_pore_dist_by_thickness_bin", {}),
        pore_radius_dist_by_thickness_bin=phase2_params.get("pore_radius_dist_by_thickness_bin", {}),
        throat_radius_dist_by_thickness_bin=phase2_params.get("throat_radius_dist_by_thickness_bin", {}),
        throat_radius_dist_by_length_bin=phase2_params.get("throat_radius_dist_by_length_bin", {}),
        pore_throat_radius_correlation=phase2_params.get("pore_throat_radius_correlation", {}),
        rho_throat_dist_by_thickness_bin=phase2_params.get("rho_throat_dist_by_thickness_bin", {}),
        frac_throat_dist_by_thickness_bin=phase2_params.get("frac_throat_dist_by_thickness_bin", {}),
        throat_length_dist_by_thickness_bin=phase2_params.get("throat_length_dist_by_thickness_bin", {}),
        effective_throat_length_dist_by_thickness_bin=phase2_params.get("effective_throat_length_dist_by_thickness_bin", {}),
        overlap_throat_R_ratio_dist_by_thickness_bin=phase2_params.get("overlap_throat_R_ratio_dist_by_thickness_bin", {}),
        throat_gt_min_pore_frac_dist_by_thickness_bin=phase2_params.get("throat_gt_min_pore_frac_dist_by_thickness_bin", {}),
        nonoverlap_cross_density_dist_by_thickness_bin=phase2_params.get("nonoverlap_cross_density_dist_by_thickness_bin", {}),
        degree_dist_by_thickness_bin=phase2_params.get("degree_dist_by_thickness_bin", {}),
        random_seed=seed,
        pore_radius_q4_spatial_bias=q4_bias,
        pore_radius_q4_bias_strength=q4_bias_strength,
    )


def _generate_network(
    generator: GBMSyntheticNetwork,
    networks_dir: Path,
    sample_name: str,
    weights: dict[str, float],
    log_path: Path,
    throat_neighbor_nm: float,
) -> dict[str, Any]:
    networks_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        generator.generate_geometry()
        generator.place_pores()
        generator.assign_pore_radii()
        generator.connect_throats(max_distance=float(throat_neighbor_nm))
        generator.assign_throat_radii(
            w_thickness=float(weights["w_thickness"]),
            w_pore=float(weights["w_pore"]),
            w_length=float(weights["w_length"]),
        )
        generator.export_to_xlsx(networks_dir, sample_name)

    throat_radii = np.asarray(generator.throat_radii if generator.throat_radii is not None else [], dtype=float)
    pore_radii = np.asarray(generator.pore_radii if generator.pore_radii is not None else [], dtype=float)
    throat_kind = np.asarray(getattr(generator, "_throat_kind", []), dtype=object)
    throat_lengths = np.asarray([], dtype=float)
    pore_degrees = np.asarray([], dtype=float)
    if (
        getattr(generator, "throat_pore1", None) is not None
        and getattr(generator, "throat_pore2", None) is not None
        and getattr(generator, "pore_ids", None) is not None
        and getattr(generator, "pore_coords", None) is not None
        and len(generator.throat_pore1) > 0
    ):
        pore_id_to_idx = {int(pid): i for i, pid in enumerate(generator.pore_ids)}
        idx1 = np.array([pore_id_to_idx[int(pid)] for pid in generator.throat_pore1], dtype=int)
        idx2 = np.array([pore_id_to_idx[int(pid)] for pid in generator.throat_pore2], dtype=int)
        coords = np.asarray(generator.pore_coords, dtype=float)
        throat_lengths = np.linalg.norm(coords[idx2] - coords[idx1], axis=1)

        degree = np.zeros(len(generator.pore_ids), dtype=float)
        np.add.at(degree, idx1, 1.0)
        np.add.at(degree, idx2, 1.0)
        pore_degrees = degree

    volume = float(getattr(generator, "volume", math.nan))
    return {
        "n_pores": int(len(pore_radii)),
        "n_throats": int(len(throat_radii)),
        "pore_degree_mean": float(np.nanmean(pore_degrees)) if pore_degrees.size else math.nan,
        "pore_degree_median": float(np.nanmedian(pore_degrees)) if pore_degrees.size else math.nan,
        "pore_radius_mean_nm": float(np.nanmean(pore_radii)) if pore_radii.size else math.nan,
        "pore_density_count_per_nm3": float(len(pore_radii) / volume) if volume and volume > 0 else math.nan,
        "throat_radius_mean_nm": float(np.nanmean(throat_radii)) if throat_radii.size else math.nan,
        "throat_radius_median_nm": float(np.nanmedian(throat_radii)) if throat_radii.size else math.nan,
        "throat_length_mean_nm": float(np.nanmean(throat_lengths)) if throat_lengths.size else math.nan,
        "throat_length_median_nm": float(np.nanmedian(throat_lengths)) if throat_lengths.size else math.nan,
        "passable_throat_fraction": float(np.mean(throat_radii > SOLUTE_RADIUS_NM)) if throat_radii.size else math.nan,
        "overlap_throat_fraction": float(np.mean(throat_kind.astype(str) == "overlap")) if throat_kind.size else math.nan,
        "throat_density_count_per_nm3": float(len(throat_radii) / volume) if volume and volume > 0 else math.nan,
    }


def _run_subprocess(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC_ROOT), env.get("PYTHONPATH", "")) if part
    )
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    text = ""
    if p.stdout:
        text += p.stdout
    if p.stderr:
        text += "\n[stderr]\n" + p.stderr
    if p.returncode != 0:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text, encoding="utf-8", errors="replace")
    return int(p.returncode), text


def _run_phase4(
    sample_name: str,
    sample_dir: Path,
    output_dir: Path,
    log_dir: Path,
    keep_logs: bool,
    solute_radius_nm: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    analyze_log = log_dir / f"{sample_name}_phase4_analyze.log"
    sieve_log = log_dir / f"{sample_name}_phase4_sieve.log"
    cmd_analyze = [
        sys.executable,
        "-m",
        "gbm_sieving.simulation.transport.analyze_network",
        "--sample-name",
        sample_name,
        "--sample-dir",
        str(sample_dir),
        "--output-dir",
        str(output_dir),
        "--no-html",
        "--solute-radius-nm",
        str(float(solute_radius_nm)),
    ]
    rc, txt = _run_subprocess(cmd_analyze, KIDNEY_ROOT, analyze_log)
    if keep_logs and rc == 0:
        analyze_log.write_text(txt, encoding="utf-8", errors="replace")
    if rc != 0:
        return {"status": "phase4_analyze_failed", "sieving_coefficient": math.nan}

    solvent_comp = output_dir / sample_name / f"{sample_name}_solvent_penetration_components.xlsx"
    try:
        if solvent_comp.exists() and len(pd.read_excel(solvent_comp)) == 0:
            return {
                "status": "disconnected",
                "sieving_coefficient": 0.0,
                "Theta_corrected_concentration": 0.0,
                "Q_total_m3s": 0.0,
                "no_exit_nodes": True,
            }
    except Exception:
        pass

    cmd_sieve = [
        sys.executable,
        "-m",
        "gbm_sieving.simulation.transport.sieving_solver",
        "--sample-name",
        sample_name,
        "--sample-dir",
        str(sample_dir),
        "--output-dir",
        str(output_dir),
        "--no-html",
        "--solute-radius-nm",
        str(float(solute_radius_nm)),
    ]
    rc, txt = _run_subprocess(cmd_sieve, KIDNEY_ROOT, sieve_log)
    if keep_logs and rc == 0:
        sieve_log.write_text(txt, encoding="utf-8", errors="replace")

    result_path = resolve_sieving_results_xlsx(output_dir, sample_name)
    if result_path is not None:
        parsed = parse_sieving_results_dataframe(pd.read_excel(result_path)) or {}
        parsed["status"] = (
            "solute_no_exit_zero_sieving"
            if parsed.get("zero_sieving_reason") == "no_solute_exit_nodes"
            else ("ok" if rc == 0 else "ok_with_phase4_warning")
        )
        if rc != 0:
            parsed["phase4_returncode"] = int(rc)
        sc = parsed.get("sieving_coefficient")
        if sc is None:
            sc = parsed.get("Theta_corrected_concentration")
        sc_value = _safe_float(sc, math.nan)
        sc_raw = _safe_float(parsed.get("sieving_coefficient_raw", sc_value), math.nan)
        parsed["sieving_coefficient_raw"] = sc_raw
        if math.isfinite(sc_value) and -SIEVING_VALID_EPS <= sc_value < 0.0:
            sc_value = 0.0
        elif math.isfinite(sc_value) and 1.0 < sc_value <= 1.0 + SIEVING_VALID_EPS:
            sc_value = 1.0
        elif math.isfinite(sc_raw) and (sc_raw < -SIEVING_VALID_EPS or sc_raw > 1.0 + SIEVING_VALID_EPS):
            sc_value = math.nan
            parsed["sieving_invalid_reason"] = "C_out/C_in outside [0, 1]"
        parsed["sieving_coefficient"] = sc_value
        if not math.isfinite(sc_value):
            parsed["status"] = "missing_sieving_value"
            if rc != 0:
                parsed["phase4_returncode"] = int(rc)
        return parsed
    if rc != 0:
        return {"status": "phase4_sieve_failed", "sieving_coefficient": math.nan}
    return {"status": "missing_sieving_result", "sieving_coefficient": math.nan}


def _cleanup_sample_artifacts(
    sample_name: str,
    networks_dir: Path,
    phase4_dir: Path,
    *,
    keep_network_xlsx: bool,
    keep_phase4_sample_dirs: bool,
) -> None:
    if not keep_network_xlsx:
        for suffix in ("pores", "throats"):
            p = networks_dir / f"{sample_name}_{suffix}.xlsx"
            if p.exists():
                p.unlink()
    if not keep_phase4_sample_dirs:
        for p in (phase4_dir / sample_name, phase4_dir / networks_dir.name / sample_name):
            if p.exists() and p.is_dir():
                shutil.rmtree(p, ignore_errors=True)


def _run_one_sample_task(task: dict[str, Any]) -> dict[str, Any]:
    variant = str(task["variant"])
    sample_type = str(task["sample_type"])
    run_index = int(task["run_index"])
    sample_name = str(task["sample_name"])
    networks_dir = Path(task["networks_dir"])
    phase4_dir = Path(task["phase4_dir"])
    logs_dir = Path(task["logs_dir"])
    weights = copy.deepcopy(task["weights"])

    row: dict[str, Any] = {
        "variant": variant,
        "variant_code": str(task.get("variant_code", "")),
        "sample_type": sample_type,
        "run_index": run_index,
        "sample_name": sample_name,
        "seed": int(task["seed"]),
        "thickness_nm": float(task["thickness_nm"]),
        "base_edge_nm": float(task["base_edge_nm"]),
        "thickness_mode": str(task["thickness_mode"]),
        "throat_neighbor_nm": float(task["throat_neighbor_nm"]),
        "pore_rho_mean_correction_rel_tol": float(task["pore_rho_mean_correction_rel_tol"]),
        "pore_rho_mean_correction_scale_min": float(task["pore_rho_mean_correction_scale_min"]),
        "pore_rho_mean_correction_scale_max": float(task["pore_rho_mean_correction_scale_max"]),
        "throat_rho_mean_correction_rel_tol": float(task["throat_rho_mean_correction_rel_tol"]),
        "throat_rho_mean_correction_scale_min": float(task["throat_rho_mean_correction_scale_min"]),
        "throat_rho_mean_correction_scale_max": float(task["throat_rho_mean_correction_scale_max"]),
        "solute_radius_nm": float(task["solute_radius_nm"]),
    }
    try:
        phase2_params = copy.deepcopy(task["phase2_params"])
        sample_variant_meta = _apply_sample_level_variant(
            phase2_params,
            variant,
            sample_type,
            float(task["thickness_nm"]),
            int(task["seed"]),
        )
        row.update(sample_variant_meta)
        if bool(sample_variant_meta.get("condition_swap_disable_radius_reweighting", False)):
            weights["w_pore"] = 0.0
            weights["w_length"] = 0.0
        row["generation_w_thickness"] = float(weights.get("w_thickness", math.nan))
        row["generation_w_pore"] = float(weights.get("w_pore", math.nan))
        row["generation_w_length"] = float(weights.get("w_length", math.nan))
        gen = _build_generator(
            sample_type,
            float(task["thickness_nm"]),
            float(task["base_edge_nm"]),
            phase2_params,
            int(task["seed"]),
            float(task["pore_placement_cell_nm"]),
            bool(task["pore_radius_q4_spatial_bias"]),
            float(task["pore_radius_q4_bias_strength"]),
            str(task["rho_sampling_scope"]),
            float(task["pore_rho_mean_correction_rel_tol"]),
            float(task["pore_rho_mean_correction_scale_min"]),
            float(task["pore_rho_mean_correction_scale_max"]),
            float(task["throat_rho_mean_correction_rel_tol"]),
            float(task["throat_rho_mean_correction_scale_min"]),
            float(task["throat_rho_mean_correction_scale_max"]),
        )
        gen_stats = _generate_network(
            gen,
            networks_dir,
            sample_name,
            weights,
            logs_dir / f"{sample_name}_phase3.log",
            float(task["throat_neighbor_nm"]),
        )
        row.update(gen_stats)
        if bool(task["skip_phase4"]):
            row.update({"status": "phase3_only", "sieving_coefficient": math.nan})
        else:
            res = _run_phase4(
                sample_name,
                networks_dir,
                phase4_dir,
                logs_dir,
                bool(task["keep_logs"]),
                float(task["solute_radius_nm"]),
            )
            row.update(res)
    except Exception as exc:
        row.update({"status": "exception", "error": repr(exc), "sieving_coefficient": math.nan})
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / f"{sample_name}_exception.txt").write_text(repr(exc), encoding="utf-8")
    finally:
        if not bool(task["skip_phase4"]):
            _cleanup_sample_artifacts(
                sample_name,
                networks_dir,
                phase4_dir,
                keep_network_xlsx=bool(task["keep_network_xlsx"]),
                keep_phase4_sample_dirs=bool(task["keep_phase4_sample_dirs"]),
            )
    return row


def _summarize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out_rows = []
    for (variant, st), sub in df.groupby(["variant", "sample_type"], dropna=False):
        sc = pd.to_numeric(sub["sieving_coefficient"], errors="coerce")
        valid = np.isfinite(sc) & (sc >= 0.0) & (sc <= 1.0)
        sc_valid = sc.where(valid)
        zero_sieving = valid & (sc == 0.0)
        positive_sieving = valid & (sc > 0.0)
        out_rows.append(
            {
                "variant": variant,
                "sample_type": st,
                "n_total": int(len(sub)),
                "n_valid": int(valid.sum()),
                "n_disconnected": int(zero_sieving.sum()),
                "n_zero_sieving": int(zero_sieving.sum()),
                "n_positive_sieving": int(positive_sieving.sum()),
                "n_failed": int((~valid).sum()),
                "mean_sieving": float(np.nanmean(sc_valid)) if valid.any() else math.nan,
                "median_sieving": float(np.nanmedian(sc_valid)) if valid.any() else math.nan,
                "min_sieving": float(np.nanmin(sc_valid)) if valid.any() else math.nan,
                "max_sieving": float(np.nanmax(sc_valid)) if valid.any() else math.nan,
                "mean_sieving_positive_only": float(np.nanmean(sc.where(positive_sieving))) if positive_sieving.any() else math.nan,
                "mean_throat_radius_nm": float(np.nanmean(pd.to_numeric(sub["throat_radius_mean_nm"], errors="coerce"))),
                "mean_passable_throat_fraction": float(
                    np.nanmean(pd.to_numeric(sub["passable_throat_fraction"], errors="coerce"))
                ),
            }
        )
    return pd.DataFrame(out_rows)


def _paired_delta(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty or "baseline" not in set(df["variant"]):
        return pd.DataFrame()
    base = df[df["variant"] == "baseline"][
        ["sample_type", "run_index", "thickness_nm", "sieving_coefficient"]
    ].rename(columns={"sieving_coefficient": "baseline_sieving"})
    other = df[df["variant"] != "baseline"].copy()
    merged = other.merge(base, on=["sample_type", "run_index"], how="left", suffixes=("", "_baseline"))
    s = pd.to_numeric(merged["sieving_coefficient"], errors="coerce")
    b = pd.to_numeric(merged["baseline_sieving"], errors="coerce")
    s = s.where(np.isfinite(s) & (s >= 0.0) & (s <= 1.0))
    b = b.where(np.isfinite(b) & (b >= 0.0) & (b <= 1.0))
    merged["delta_sieving"] = s - b
    delta_log = np.full(len(merged), np.nan, dtype=float)
    mask = (s.to_numpy(dtype=float) > 0) & (b.to_numpy(dtype=float) > 0)
    delta_log[mask] = np.log10(s.to_numpy(dtype=float)[mask]) - np.log10(b.to_numpy(dtype=float)[mask])
    merged["delta_log10_sieving"] = delta_log
    return merged


def _collect_bin_records_for_sample(obj: Any, sample_type: str) -> list[dict]:
    records: list[dict] = []
    if not isinstance(obj, dict):
        return records
    block = obj.get(sample_type)
    if isinstance(block, dict) and isinstance(block.get("bins"), list):
        records.extend([rec for rec in block.get("bins", []) if isinstance(rec, dict)])
    for value in obj.values():
        if isinstance(value, dict):
            records.extend(_collect_bin_records_for_sample(value, sample_type))
    return records


def _source_overall_fit_mean(
    phase2_params: dict,
    source_sample_type: str,
    source_key: str,
    *,
    degree: bool = False,
) -> float:
    obj = phase2_params.get(source_key)
    records = _collect_bin_records_for_sample(obj, source_sample_type)
    if not records:
        return math.nan
    fit = _weighted_pool_fit(records, degree=degree)
    return _safe_float(fit.get("mean"), math.nan)


def _paired_generated_parameter_stats(
    df: pd.DataFrame,
    *,
    variant: str,
    target_sample_type: str,
    parameter: str,
) -> dict[str, Any]:
    base = df[df["variant"] == "baseline"][
        ["sample_type", "run_index", parameter]
    ].rename(columns={parameter: "baseline_value"})
    var = df[df["variant"] == variant][
        ["sample_type", "run_index", parameter]
    ].rename(columns={parameter: "variant_value"})
    base = base[base["sample_type"].astype(str).str.upper() == target_sample_type]
    var = var[var["sample_type"].astype(str).str.upper() == target_sample_type]
    merged = base.merge(var, on=["sample_type", "run_index"], how="inner")
    b = pd.to_numeric(merged["baseline_value"], errors="coerce")
    v = pd.to_numeric(merged["variant_value"], errors="coerce")
    finite = np.isfinite(b) & np.isfinite(v)
    b = b[finite]
    v = v[finite]
    out: dict[str, Any] = {
        "n_paired": int(len(b)),
        "baseline_mean": float(np.nanmean(b)) if len(b) else math.nan,
        "variant_mean": float(np.nanmean(v)) if len(v) else math.nan,
        "baseline_median": float(np.nanmedian(b)) if len(b) else math.nan,
        "variant_median": float(np.nanmedian(v)) if len(v) else math.nan,
        "median_delta_variant_minus_baseline": float(np.nanmedian(v - b)) if len(b) else math.nan,
        "test": "none",
        "p_value": math.nan,
    }
    if len(b) >= 3:
        diff = v.to_numpy(dtype=float) - b.to_numpy(dtype=float)
        if np.allclose(diff, 0.0, rtol=1e-12, atol=1e-12):
            out["test"] = "paired_wilcoxon_all_equal"
            out["p_value"] = 1.0
        else:
            try:
                out["p_value"] = float(stats.wilcoxon(v, b, zero_method="wilcox").pvalue)
                out["test"] = "paired_wilcoxon"
            except ValueError:
                out["p_value"] = float(stats.mannwhitneyu(v, b, alternative="two-sided").pvalue)
                out["test"] = "mannwhitney_fallback"
    return out


def _replacement_validation(rows: list[dict[str, Any]], base_phase2_params: dict) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty or "baseline" not in set(df.get("variant", [])):
        return pd.DataFrame()
    df["sample_type"] = df["sample_type"].astype(str).str.upper()
    variants = [v for v in sorted(df["variant"].dropna().unique(), key=str) if v != "baseline"]
    out_rows: list[dict[str, Any]] = []
    for variant in variants:
        cfg = _condition_swap_config(str(variant))
        if cfg is None:
            continue
        property_group, source_sample_type, target_sample_type = cfg
        params = GENERATED_PARAMETER_GROUPS.get(property_group, ())
        for parameter in params:
            if parameter not in df.columns:
                continue
            stats_row = _paired_generated_parameter_stats(
                df,
                variant=str(variant),
                target_sample_type=target_sample_type,
                parameter=parameter,
            )
            source_key, degree = GENERATED_PARAMETER_SOURCE_KEYS.get(parameter, ("", False))
            target_mean = (
                _source_overall_fit_mean(base_phase2_params, source_sample_type, source_key, degree=degree)
                if source_key
                else math.nan
            )
            base_med = _safe_float(stats_row.get("baseline_median"), math.nan)
            var_med = _safe_float(stats_row.get("variant_median"), math.nan)
            base_dist = abs(base_med - target_mean) if math.isfinite(base_med) and math.isfinite(target_mean) else math.nan
            var_dist = abs(var_med - target_mean) if math.isfinite(var_med) and math.isfinite(target_mean) else math.nan
            if math.isfinite(base_dist) and base_dist > 0 and math.isfinite(var_dist):
                distance_reduction = (base_dist - var_dist) / base_dist
            else:
                distance_reduction = math.nan
            if math.isfinite(target_mean) and math.isfinite(base_med) and not math.isclose(target_mean, base_med):
                normalized_shift = (var_med - base_med) / (target_mean - base_med) if math.isfinite(var_med) else math.nan
            else:
                normalized_shift = math.nan
            if math.isfinite(target_mean) and math.isfinite(base_med) and math.isfinite(var_med):
                expected = math.copysign(1.0, target_mean - base_med) if not math.isclose(target_mean, base_med) else 0.0
                observed = math.copysign(1.0, var_med - base_med) if not math.isclose(var_med, base_med) else 0.0
                moved_toward_target = bool(distance_reduction > 0) if math.isfinite(distance_reduction) else False
                direction_agrees = bool(expected == observed) if expected != 0.0 and observed != 0.0 else math.nan
            else:
                moved_toward_target = math.nan
                direction_agrees = math.nan
            out_rows.append(
                {
                    "variant": str(variant),
                    "property_group": property_group,
                    "source_sample_type": source_sample_type,
                    "target_sample_type": target_sample_type,
                    "generated_parameter": parameter,
                    "source_phase2_key": source_key,
                    "source_overall_fit_mean": target_mean,
                    **stats_row,
                    "baseline_distance_to_source_mean": base_dist,
                    "variant_distance_to_source_mean": var_dist,
                    "distance_reduction_fraction": distance_reduction,
                    "normalized_shift_to_source": normalized_shift,
                    "moved_toward_source_mean": moved_toward_target,
                    "median_shift_direction_agrees_with_source": direction_agrees,
                }
            )
    return pd.DataFrame(out_rows)


def _variant_label_for_plot(variant: str) -> str:
    short = VARIANT_SHORT_NAMES.get(str(variant), str(variant))
    return short.replace("as2wt_", "AS->WT ").replace("wt2as_", "WT->AS ").replace("_", " ")


def _plot_replacement_validation(validation: pd.DataFrame, output_dir: Path) -> list[Path]:
    if validation.empty:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = output_dir / "figures_replacement_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    d = validation.copy()
    d["normalized_shift_to_source"] = pd.to_numeric(d["normalized_shift_to_source"], errors="coerce")
    d["distance_reduction_fraction"] = pd.to_numeric(d["distance_reduction_fraction"], errors="coerce")
    d = d[np.isfinite(d["normalized_shift_to_source"])].copy()
    if d.empty:
        return []

    param_labels = {
        "pore_density_count_per_nm3": "pore density",
        "pore_radius_mean_nm": "pore radius",
        "throat_radius_mean_nm": "throat radius mean",
        "throat_radius_median_nm": "throat radius median",
        "throat_density_count_per_nm3": "throat density",
        "throat_length_mean_nm": "throat length mean",
        "throat_length_median_nm": "throat length median",
        "pore_degree_mean": "pore degree mean",
        "pore_degree_median": "pore degree median",
        "passable_throat_fraction": "passable throat fraction",
    }
    group_colors = {
        "pore": "#8C564B",
        "pore_count": "#6B4C9A",
        "pore_radius": "#B07C72",
        "throat_radius": "#D55E00",
        "throat_density": "#E69F00",
        "throat_length": "#56B4E9",
        "degree": "#009E73",
        "passable_overlap": "#CC79A7",
        "all_throat": "#7F3C8D",
    }

    for target_sample_type, sub0 in d.groupby("target_sample_type", dropna=False):
        sub = sub0.copy()
        sub["plot_label"] = [
            f"{_variant_label_for_plot(v)} | {param_labels.get(p, p)}"
            for v, p in zip(sub["variant"], sub["generated_parameter"])
        ]
        sub = sub.sort_values(["property_group", "variant", "generated_parameter"]).reset_index(drop=True)
        n = len(sub)
        fig_h = max(4.2, 0.32 * n + 1.2)
        fig, ax = plt.subplots(figsize=(8.2, fig_h), dpi=260)
        y = np.arange(n)
        vals = sub["normalized_shift_to_source"].to_numpy(float)
        colors = [group_colors.get(str(g), "#6B7280") for g in sub["property_group"]]
        ax.barh(y, vals, color=colors, alpha=0.86, edgecolor="white", linewidth=0.45)
        ax.axvline(0.0, color="#4B5563", linewidth=0.9)
        ax.axvline(1.0, color="#111827", linewidth=1.0, linestyle="--")
        ax.axvspan(0.0, 1.0, color="#EEF2F5", alpha=0.55, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["plot_label"], fontsize=8.2)
        ax.invert_yaxis()
        finite_vals = vals[np.isfinite(vals)]
        lo = min(-0.35, float(np.nanmin(finite_vals)) - 0.15)
        hi = max(1.35, float(np.nanmax(finite_vals)) + 0.15)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Normalized shift toward source distribution\n(baseline = 0, source phase2 overall mean = 1)")
        ax.set_title(f"Replacement validation ({target_sample_type})", fontsize=12.5, pad=8)
        ax.grid(True, axis="x", color="#E5E7EB", linewidth=0.6)
        ax.grid(False, axis="y")
        for spine in ax.spines.values():
            spine.set_color("#CBD5DF")
            spine.set_linewidth(0.7)
        fig.tight_layout()
        stem = f"replacement_validation_normalized_shift_{str(target_sample_type).upper()}"
        for ext in ("png", "svg", "pdf"):
            path = out_dir / f"{stem}.{ext}"
            fig.savefig(path, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    dist = validation.copy()
    dist["distance_reduction_fraction"] = pd.to_numeric(dist["distance_reduction_fraction"], errors="coerce")
    dist = dist[np.isfinite(dist["distance_reduction_fraction"])].copy()
    if not dist.empty:
        for target_sample_type, sub0 in dist.groupby("target_sample_type", dropna=False):
            sub = sub0.sort_values(["property_group", "variant", "generated_parameter"]).reset_index(drop=True)
            labels = [
                f"{_variant_label_for_plot(v)} | {param_labels.get(p, p)}"
                for v, p in zip(sub["variant"], sub["generated_parameter"])
            ]
            n = len(sub)
            fig_h = max(4.2, 0.32 * n + 1.2)
            fig, ax = plt.subplots(figsize=(8.2, fig_h), dpi=260)
            y = np.arange(n)
            vals = sub["distance_reduction_fraction"].to_numpy(float)
            colors = [group_colors.get(str(g), "#6B7280") for g in sub["property_group"]]
            ax.barh(y, vals, color=colors, alpha=0.86, edgecolor="white", linewidth=0.45)
            ax.axvline(0.0, color="#4B5563", linewidth=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8.2)
            ax.invert_yaxis()
            finite_vals = vals[np.isfinite(vals)]
            lo = min(-0.35, float(np.nanmin(finite_vals)) - 0.15)
            hi = max(1.0, float(np.nanmax(finite_vals)) + 0.15)
            ax.set_xlim(lo, hi)
            ax.set_xlabel("Distance reduction relative to source phase2 overall mean")
            ax.set_title(f"Replacement distance-to-source change ({target_sample_type})", fontsize=12.5, pad=8)
            ax.grid(True, axis="x", color="#E5E7EB", linewidth=0.6)
            ax.grid(False, axis="y")
            for spine in ax.spines.values():
                spine.set_color("#CBD5DF")
                spine.set_linewidth(0.7)
            fig.tight_layout()
            stem = f"replacement_validation_distance_reduction_{str(target_sample_type).upper()}"
            for ext in ("png", "svg", "pdf"):
                path = out_dir / f"{stem}.{ext}"
                fig.savefig(path, bbox_inches="tight")
                paths.append(path)
            plt.close(fig)
    return paths


def _run_replacement_plot_scripts(output_dir: Path, sample_types: list[str], figures_root: Path | None = None) -> None:
    log_dir = output_dir / "plot_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "plot_sieving_sensitivity_results.py"),
        "--result-dir",
        str(output_dir),
    ]
    if figures_root is not None:
        cmd.extend(["--figures-root", str(figures_root)])
    rc, _ = _run_subprocess(cmd, KIDNEY_ROOT, log_dir / "plot_sieving_sensitivity_results.log")
    if rc == 0:
        print("[plots] OK: unified sensitivity plotting")
    else:
        print(f"[plots] WARNING: unified sensitivity plotting failed; see {log_dir}")


def run_experiment(args: argparse.Namespace) -> None:
    phase1_dir = Path(args.phase1_dir).resolve()
    phase2_dir = Path(args.phase2_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    sample_types = [s.strip().upper() for s in args.sample_types.split(",") if s.strip()]
    _warn_condition_swap_noops(variants, sample_types)
    phase1_params = load_phase1_parameters(phase1_dir)
    base_phase2_params = load_phase2_parameters(phase2_dir)

    shared_rho_rel_tol = args.rho_mean_correction_rel_tol
    shared_rho_scale_min = args.rho_mean_correction_scale_min
    shared_rho_scale_max = args.rho_mean_correction_scale_max
    pore_rho_rel_tol = (
        args.pore_rho_mean_correction_rel_tol
        if args.pore_rho_mean_correction_rel_tol is not None
        else (shared_rho_rel_tol if shared_rho_rel_tol is not None else float("inf"))
    )
    pore_rho_scale_min = (
        args.pore_rho_mean_correction_scale_min
        if args.pore_rho_mean_correction_scale_min is not None
        else (shared_rho_scale_min if shared_rho_scale_min is not None else 0.5)
    )
    pore_rho_scale_max = (
        args.pore_rho_mean_correction_scale_max
        if args.pore_rho_mean_correction_scale_max is not None
        else (shared_rho_scale_max if shared_rho_scale_max is not None else 2.0)
    )
    throat_rho_rel_tol = (
        args.throat_rho_mean_correction_rel_tol
        if args.throat_rho_mean_correction_rel_tol is not None
        else (shared_rho_rel_tol if shared_rho_rel_tol is not None else float("inf"))
    )
    throat_rho_scale_min = (
        args.throat_rho_mean_correction_scale_min
        if args.throat_rho_mean_correction_scale_min is not None
        else (shared_rho_scale_min if shared_rho_scale_min is not None else 0.5)
    )
    throat_rho_scale_max = (
        args.throat_rho_mean_correction_scale_max
        if args.throat_rho_mean_correction_scale_max is not None
        else (shared_rho_scale_max if shared_rho_scale_max is not None else 2.0)
    )

    config = vars(args).copy()
    config["variants"] = variants
    config["sample_types"] = sample_types
    config["effective_rho_sampling_scope"] = str(args.rho_sampling_scope)
    config["effective_pore_rho_mean_correction_rel_tol"] = float(pore_rho_rel_tol)
    config["effective_pore_rho_mean_correction_scale_min"] = float(pore_rho_scale_min)
    config["effective_pore_rho_mean_correction_scale_max"] = float(pore_rho_scale_max)
    config["effective_throat_rho_mean_correction_rel_tol"] = float(throat_rho_rel_tol)
    config["effective_throat_rho_mean_correction_scale_min"] = float(throat_rho_scale_min)
    config["effective_throat_rho_mean_correction_scale_max"] = float(throat_rho_scale_max)
    _json_write(output_dir / "experiment_config.json", config)

    rng_for_thickness = np.random.default_rng(int(args.seed))
    base_thickness: dict[str, np.ndarray] = {}
    for st in sample_types:
        base_thickness[st] = _sample_thickness_table(
            phase1_params,
            phase1_dir,
            st,
            int(args.n_samples),
            rng_for_thickness,
            "fitted",
            args.wt_thickness_fit,
            args.as_thickness_fit,
        )

    all_rows: list[dict[str, Any]] = []
    for variant_index, variant in enumerate(variants):
        print(f"\n=== Variant: {variant} ===")
        variant_code = _variant_output_name(variant_index, variant)
        variant_dir = output_dir / variant_code
        networks_dir = variant_dir / "networks"
        phase4_dir = variant_dir / "phase4"
        logs_dir = variant_dir / "logs"
        networks_dir.mkdir(parents=True, exist_ok=True)

        phase2_params, weights = _apply_variant_to_phase2_params(base_phase2_params, variant)
        _json_write(variant_dir / "variant_config.json", {"variant": variant, "variant_code": variant_code, "weights": weights})

        thickness_mode = "fitted"
        if variant == "thickness_empirical_bootstrap":
            thickness_mode = "empirical_bootstrap"
        elif variant == "thickness_uniform_range":
            thickness_mode = "uniform_range"

        sample_rows: list[dict[str, Any]] = []
        tasks: list[dict[str, Any]] = []
        for st in sample_types:
            if thickness_mode == "fitted":
                thickness_values = base_thickness[st]
            else:
                thickness_values = _sample_thickness_table(
                    phase1_params,
                    phase1_dir,
                    st,
                    int(args.n_samples),
                    np.random.default_rng(int(args.seed) + 97_531 + variant_index * 1009 + (17 if st == "AS" else 31)),
                    thickness_mode,
                    args.wt_thickness_fit,
                    args.as_thickness_fit,
                )
            for i, thickness_nm in enumerate(thickness_values):
                sample_name = f"sens_v{variant_index:02d}_{st}_r{i:03d}"
                seed = int(args.seed) + (0 if st == "WT" else 1_000_000) + i
                tasks.append(
                    {
                        "variant": variant,
                        "variant_code": variant_code,
                        "sample_type": st,
                        "run_index": int(i),
                        "sample_name": sample_name,
                        "seed": seed,
                        "thickness_nm": float(thickness_nm),
                        "base_edge_nm": float(args.base_edge_nm),
                        "thickness_mode": thickness_mode,
                        "phase2_params": phase2_params,
                        "weights": weights,
                        "networks_dir": networks_dir,
                        "phase4_dir": phase4_dir,
                        "logs_dir": logs_dir,
                        "pore_placement_cell_nm": float(args.pore_placement_cell_nm),
                        "throat_neighbor_nm": float(args.throat_neighbor_nm),
                        "pore_radius_q4_spatial_bias": bool(args.pore_radius_q4_spatial_bias),
                        "pore_radius_q4_bias_strength": float(args.pore_radius_q4_bias_strength),
                        "rho_sampling_scope": str(args.rho_sampling_scope),
                        "pore_rho_mean_correction_rel_tol": float(pore_rho_rel_tol),
                        "pore_rho_mean_correction_scale_min": float(pore_rho_scale_min),
                        "pore_rho_mean_correction_scale_max": float(pore_rho_scale_max),
                        "throat_rho_mean_correction_rel_tol": float(throat_rho_rel_tol),
                        "throat_rho_mean_correction_scale_min": float(throat_rho_scale_min),
                        "throat_rho_mean_correction_scale_max": float(throat_rho_scale_max),
                        "solute_radius_nm": float(args.solute_radius_nm),
                        "skip_phase4": bool(args.skip_phase4),
                        "keep_logs": bool(args.keep_logs),
                        "keep_network_xlsx": bool(args.keep_network_xlsx),
                        "keep_phase4_sample_dirs": bool(args.keep_phase4_sample_dirs),
                    }
                )

        def _record_completed(row: dict[str, Any]) -> None:
            print(
                f"[{row.get('variant')}] {row.get('sample_type')} run{int(row.get('run_index', -1)) + 1} "
                f"T={float(row.get('thickness_nm', math.nan)):.2f} nm -> {row.get('status')}"
            )
            sample_rows.append(row)
            all_rows.append(row)
            sample_rows.sort(key=lambda r: (str(r.get("sample_type")), int(r.get("run_index", 0))))
            all_rows.sort(key=lambda r: (str(r.get("variant")), str(r.get("sample_type")), int(r.get("run_index", 0))))
            pd.DataFrame(sample_rows).to_csv(variant_dir / "per_sample_results.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(all_rows).to_csv(output_dir / "all_per_sample_results.csv", index=False, encoding="utf-8-sig")

        jobs = max(1, int(args.jobs))
        if jobs == 1 or len(tasks) <= 1:
            for task in tasks:
                print(
                    f"[{variant}] {task['sample_type']} {int(task['run_index']) + 1}/{args.n_samples} "
                    f"T={float(task['thickness_nm']):.2f} nm"
                )
                _record_completed(_run_one_sample_task(task))
        else:
            print(f"[{variant}] running {len(tasks)} samples with --jobs {jobs}")
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(_run_one_sample_task, task) for task in tasks]
                for fut in as_completed(futures):
                    _record_completed(fut.result())

        _summarize(sample_rows).to_csv(variant_dir / "variant_summary.csv", index=False, encoding="utf-8-sig")

    summary = _summarize(all_rows)
    summary.to_csv(output_dir / "variant_summary_all.csv", index=False, encoding="utf-8-sig")
    delta = _paired_delta(all_rows)
    if not delta.empty:
        delta.to_csv(output_dir / "paired_delta_vs_baseline.csv", index=False, encoding="utf-8-sig")
        if np.isfinite(pd.to_numeric(delta["delta_log10_sieving"], errors="coerce")).any():
            def _median_abs_delta(s: pd.Series) -> float:
                vals = np.abs(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))
                vals = vals[np.isfinite(vals)]
                return float(np.median(vals)) if vals.size else math.nan

            rank = (
                delta.groupby(["variant", "sample_type"], dropna=False)["delta_log10_sieving"]
                .apply(_median_abs_delta)
                .reset_index(name="median_abs_delta_log10_sieving")
            )
            rank.to_csv(output_dir / "variant_importance_rank.csv", index=False, encoding="utf-8-sig")

    validation = _replacement_validation(all_rows, base_phase2_params)
    if not validation.empty:
        validation.to_csv(output_dir / "replacement_validation_summary.csv", index=False, encoding="utf-8-sig")

    if bool(args.make_plots):
        figures_root = Path(args.figures_root).resolve() if args.figures_root else None
        _run_replacement_plot_scripts(output_dir, sample_types, figures_root)

    print(f"\nDone. Results saved to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independent GBM sieving sensitivity ablation runner.")
    parser.add_argument("--n-samples", type=int, default=60, help="Samples per sample type per variant.")
    parser.add_argument("--base-edge-nm", type=float, default=150.0, help="Square base edge length in nm.")
    parser.add_argument("--sample-types", type=str, default="AS", help="Comma-separated sample types.")
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS), help="Comma-separated variants.")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel sample jobs within each variant.")
    parser.add_argument("--phase1-dir", type=str, default=str(DEFAULT_PHASE1_DIR))
    parser.add_argument("--phase2-dir", type=str, default=str(DEFAULT_PHASE2_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pore-placement-cell-nm", type=float, default=50.0)
    parser.add_argument(
        "--throat-neighbor-nm",
        type=float,
        default=30.0,
        help="KD-tree neighbor radius for overlap and non-overlap throat candidates in Phase3.",
    )
    parser.add_argument("--wt-thickness-fit", choices=("kde", "norm"), default="kde")
    parser.add_argument("--as-thickness-fit", choices=("kde", "gmm", "auto"), default="kde")
    parser.add_argument("--pore-radius-q4-spatial-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pore-radius-q4-bias-strength", type=float, default=0.55)
    parser.add_argument(
        "--rho-sampling-scope",
        choices=("run", "cell"),
        default="run",
        help="Phase3 rho sampling scope. run preserves run-level density variance; cell samples rho per grid cell.",
    )
    parser.add_argument(
        "--solute-radius-nm",
        type=float,
        default=SOLUTE_RADIUS_NM,
        help="Solute hydrated radius in nm used by Phase4 sieving and passable-throat diagnostics.",
    )
    parser.add_argument(
        "--rho-mean-correction-rel-tol",
        type=float,
        default=None,
        help="Legacy shared relative mean-density deviation tolerated before weak pore/throat density correction is applied.",
    )
    parser.add_argument(
        "--rho-mean-correction-scale-min",
        type=float,
        default=None,
        help="Legacy shared lower clipping bound for weak pore/throat density mean-correction scale.",
    )
    parser.add_argument(
        "--rho-mean-correction-scale-max",
        type=float,
        default=None,
        help="Legacy shared upper clipping bound for weak pore/throat density mean-correction scale.",
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
    parser.add_argument("--skip-phase4", action="store_true", help="Generate networks only; do not calculate sieving.")
    parser.add_argument("--keep-network-xlsx", action="store_true", help="Keep temporary phase3 xlsx files.")
    parser.add_argument("--keep-phase4-sample-dirs", action="store_true", help="Keep per-sample phase4 output folders.")
    parser.add_argument("--keep-logs", action="store_true", help="Keep successful subprocess logs too.")
    parser.add_argument(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run replacement-sensitivity plotting scripts after the experiment.",
    )
    parser.add_argument(
        "--figures-root",
        type=str,
        default=None,
        help="Optional separate root directory for replacement-sensitivity figures.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if int(args.n_samples) < 1:
        raise SystemExit("--n-samples must be >= 1")
    if int(args.jobs) < 1:
        raise SystemExit("--jobs must be >= 1")
    run_experiment(args)


if __name__ == "__main__":
    main()

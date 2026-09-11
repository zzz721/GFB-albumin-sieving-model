"""DD2006 cylindrical-pore albumin transport on an existing pore network.

The solvent pressure field is re-solved from the saved geometry because the
legacy ``throat_Q_diff_coeff`` files contain only ``abs(Q)`` and therefore do
not retain flow direction.  Albumin transport uses Dechadilok--Deen (2006)
hindrance factors and the exact steady 1-D advection--diffusion edge flux.  The
steric/electrostatic exclusion radius can be separated from the physical
hydrodynamic radius: the former controls pore accessibility and ``Phi``,
whereas the latter controls ``K_D`` and ``K_C``.

The standardized final cleanup is mandatory: after every concentration solve,
internal connected clusters containing C > 1e4 or C < 1e-6 are removed only
when an inlet-to-outlet solute path remains.  The graph is then reduced to
spanning components, non-boundary leaves are removed, and the field is solved
again until no removable cluster remains.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .hindrance import (
    dd2006_convective_hindrance,
    dd2006_diffusive_hindrance,
)


D0_M2_S = 9.26077952e-11
DEFAULT_EXCLUSION_RADIUS_NM = 3.55
DEFAULT_HYDRODYNAMIC_RADIUS_NM = 3.55
WATER_VISCOSITY_PA_S = 6.91e-4
P_IN_PA = (55.0 - 30.0) * 133.322
P_OUT_PA = 15.0 * 133.322
PRESSURE_TIE_TOL_PA = 1e-9
PRUNE_HIGH = 1e4
PRUNE_LOW = 1e-6
PRUNE_MAX_ROUNDS = 100


@dataclass(frozen=True)
class EdgeTransport:
    throat_id: int
    p1: int
    p2: int
    upstream: int
    downstream: int
    q_abs_m3s: float
    length_nm: float
    radius_nm: float
    lambda_ratio: float
    hydrodynamic_lambda_ratio: float
    phi: float
    h: float
    kd: float
    kc: float
    w: float
    diffusion_conductance_m3s: float
    advection_coefficient_m3s: float
    peclet: float
    flux_coeff_upstream: float
    flux_coeff_downstream: float


def dd2006_factors(lam: float) -> tuple[float, float, float, float, float]:
    """Return ``Phi, H, K_D, K_C, W`` for a cylindrical pore.

    The mappings are ``K_D=H/Phi`` and ``K_C=W/Phi``.  Hence the edge diffusion
    conductance is ``D0*H*A_water/L`` and the advective coefficient is ``W*Q``.
    """
    if not 0.0 <= lam < 1.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    phi = (1.0 - lam) ** 2
    kd = dd2006_diffusive_hindrance(lam)
    h = phi * kd
    kc = dd2006_convective_hindrance(lam)
    w = phi * kc
    return phi, h, kd, kc, w


def dd2006_split_radius_factors(
    exclusion_lam: float, hydrodynamic_lam: float
) -> tuple[float, float, float, float, float]:
    """Return ``Phi, H, K_D, K_C, W`` for separate exclusion/hydrodynamic sizes.

    ``Phi`` is determined by the exclusion radius.  The local DD2006 factors
    ``K_D`` and ``K_C`` are determined by the physical hydrodynamic radius.
    The overall factors used with the water-accessible area and solvent flow
    remain ``H=Phi*K_D`` and ``W=Phi*K_C``.
    """
    if not 0.0 <= exclusion_lam < 1.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    phi = (1.0 - exclusion_lam) ** 2
    _, _, kd, kc, _ = dd2006_factors(hydrodynamic_lam)
    return phi, phi * kd, kd, kc, phi * kc


def exact_flux_coefficients(u: float, g: float) -> tuple[float, float, float]:
    """Return ``a,b,Pe`` for the directed edge flux ``F=a*C_up-b*C_down``."""
    u = max(float(u), 0.0)
    g = max(float(g), 0.0)
    if u == 0.0:
        return g, g, 0.0
    if g == 0.0:
        return u, 0.0, math.inf
    pe = u / g
    b = 0.0 if pe > 700.0 else u / math.expm1(pe)
    return u + b, b, pe


def _component_nodes(path: Path) -> set[int]:
    df = pd.read_excel(path)
    nodes: set[int] = set()
    if "Pore IDs" not in df.columns:
        return nodes
    for raw in df["Pore IDs"]:
        value = str(raw).strip()
        if not value or value.lower() == "nan":
            continue
        nodes.update(int(float(token)) for token in value.split(",") if token.strip())
    return nodes


def _find_analysis_dir(root: Path, sample_name: str, sample_dir: Path) -> Path:
    candidates = [
        root / sample_name,
        root / sample_dir.name / sample_name,
        root,
    ]
    required = f"{sample_name}_pore_classification.xlsx"
    for candidate in candidates:
        if (candidate / required).is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot locate {required} under analysis root {root}"
    )


def _retain_spanning_and_prune_leaves(
    graph: nx.Graph, entrance: set[int], exit_nodes: set[int]
) -> tuple[nx.Graph, set[int], set[int]]:
    keep: set[int] = set()
    for component in nx.connected_components(graph):
        nodes = set(component)
        if nodes & entrance and nodes & exit_nodes:
            keep.update(nodes)
    out = graph.subgraph(keep).copy()
    entrance_out = entrance & set(out.nodes())
    exit_out = exit_nodes & set(out.nodes())
    while True:
        leaves = [
            node
            for node, degree in out.degree()
            if degree == 1 and node not in entrance_out and node not in exit_out
        ]
        if not leaves:
            break
        out.remove_nodes_from(leaves)
    nodes = set(out.nodes())
    return out, entrance_out & nodes, exit_out & nodes


def _has_spanning(
    graph: nx.Graph, entrance: set[int], exit_nodes: set[int]
) -> bool:
    for component in nx.connected_components(graph):
        nodes = set(component)
        if nodes & entrance and nodes & exit_nodes:
            return True
    return False


def _solve_pressure(
    graph: nx.Graph,
    conductance: dict[int, float],
    entrance: set[int],
    exit_nodes: set[int],
    *,
    p_in: float,
    p_out: float,
) -> dict[int, float]:
    boundary = entrance | exit_nodes
    internal = sorted(set(graph.nodes()) - boundary)
    index = {node: i for i, node in enumerate(internal)}
    entries: dict[tuple[int, int], float] = {}
    rhs = np.zeros(len(internal), dtype=float)
    for node in internal:
        i = index[node]
        for neighbor in graph.neighbors(node):
            tid = int(graph[node][neighbor]["throat_id"])
            g = conductance[tid]
            entries[(i, i)] = entries.get((i, i), 0.0) + g
            if neighbor in index:
                j = index[neighbor]
                entries[(i, j)] = entries.get((i, j), 0.0) - g
            elif neighbor in entrance and neighbor in exit_nodes:
                rhs[i] += g * 0.5 * (p_in + p_out)
            elif neighbor in entrance:
                rhs[i] += g * p_in
            elif neighbor in exit_nodes:
                rhs[i] += g * p_out
    pressure: dict[int, float] = {}
    if internal:
        rows, cols, data = zip(
            *((i, j, value) for (i, j), value in entries.items() if value != 0.0)
        )
        matrix = csr_matrix(
            (data, (rows, cols)), shape=(len(internal), len(internal))
        )
        solution = np.asarray(spsolve(matrix, rhs), dtype=float)
        if not np.all(np.isfinite(solution)):
            raise RuntimeError("Non-finite pressure solution")
        pressure.update(
            {node: float(solution[index[node]]) for node in internal}
        )
    for node in graph.nodes():
        if node in entrance and node in exit_nodes:
            pressure[node] = 0.5 * (p_in + p_out)
        elif node in entrance:
            pressure[node] = p_in
        elif node in exit_nodes:
            pressure[node] = p_out
    return pressure


def _edge_transport(
    row: pd.Series,
    coords: dict[int, np.ndarray],
    pressure: dict[int, float],
    solute_radius_nm: float,
    hydrodynamic_radius_nm: float,
) -> EdgeTransport:
    tid = int(row["Throat ID"])
    p1, p2 = int(row["Pore ID #1"]), int(row["Pore ID #2"])
    radius_nm = float(row["EqRadius"])
    length_nm = max(float(np.linalg.norm(coords[p2] - coords[p1])), 1e-3)
    radius_m, length_m = radius_nm * 1e-9, length_nm * 1e-9
    hydraulic_g = math.pi * radius_m**4 / (
        8.0 * WATER_VISCOSITY_PA_S * length_m
    )
    q12 = hydraulic_g * (pressure[p1] - pressure[p2])
    if abs(pressure[p1] - pressure[p2]) <= PRESSURE_TIE_TOL_PA:
        q12 = 0.0
    upstream, downstream = (p1, p2) if q12 >= 0.0 else (p2, p1)
    q_abs = abs(q12)
    exclusion_lam = float(solute_radius_nm) / radius_nm
    hydrodynamic_lam = float(hydrodynamic_radius_nm) / radius_nm
    phi, h, kd, kc, w = dd2006_split_radius_factors(
        exclusion_lam, hydrodynamic_lam
    )
    diffusion_g = D0_M2_S * h * math.pi * radius_m**2 / length_m
    advection_u = w * q_abs
    a, b, pe = exact_flux_coefficients(advection_u, diffusion_g)
    return EdgeTransport(
        throat_id=tid,
        p1=p1,
        p2=p2,
        upstream=upstream,
        downstream=downstream,
        q_abs_m3s=q_abs,
        length_nm=length_nm,
        radius_nm=radius_nm,
        lambda_ratio=exclusion_lam,
        hydrodynamic_lambda_ratio=hydrodynamic_lam,
        phi=phi,
        h=h,
        kd=kd,
        kc=kc,
        w=w,
        diffusion_conductance_m3s=diffusion_g,
        advection_coefficient_m3s=advection_u,
        peclet=pe,
        flux_coeff_upstream=a,
        flux_coeff_downstream=b,
    )


def _outward_coefficients(
    edge: EdgeTransport, node: int, neighbor: int
) -> tuple[float, float]:
    if node == edge.upstream and neighbor == edge.downstream:
        return edge.flux_coeff_upstream, -edge.flux_coeff_downstream
    if node == edge.downstream and neighbor == edge.upstream:
        return edge.flux_coeff_downstream, -edge.flux_coeff_upstream
    raise ValueError(f"Throat {edge.throat_id} does not join {node} and {neighbor}")


def _solve_concentration_at_cbulk(
    graph: nx.Graph,
    edge_map: dict[int, EdgeTransport],
    entrance: set[int],
    exit_nodes: set[int],
    c_bulk: float,
) -> tuple[dict[int, float], float, float]:
    boundary = entrance | exit_nodes
    internal = sorted(set(graph.nodes()) - boundary)
    index = {node: i for i, node in enumerate(internal)}
    entries: dict[tuple[int, int], float] = {}
    rhs = np.zeros(len(internal), dtype=float)
    for node in internal:
        i = index[node]
        for neighbor in graph.neighbors(node):
            edge = edge_map[int(graph[node][neighbor]["throat_id"])]
            diag, off = _outward_coefficients(edge, node, neighbor)
            entries[(i, i)] = entries.get((i, i), 0.0) + diag
            if neighbor in index:
                entries[(i, index[neighbor])] = (
                    entries.get((i, index[neighbor]), 0.0) + off
                )
            else:
                rhs[i] -= off * (1.0 if neighbor in entrance else c_bulk)
    residual = 0.0
    concentration: dict[int, float] = {}
    if internal:
        rows, cols, data = zip(
            *((i, j, value) for (i, j), value in entries.items() if value != 0.0)
        )
        matrix = csr_matrix(
            (data, (rows, cols)), shape=(len(internal), len(internal))
        )
        solution = np.asarray(spsolve(matrix, rhs), dtype=float)
        if not np.all(np.isfinite(solution)):
            raise RuntimeError("Non-finite concentration solution")
        residual = float(np.linalg.norm(matrix @ solution - rhs))
        concentration.update(
            {node: float(solution[index[node]]) for node in internal}
        )
    concentration.update({node: 1.0 for node in entrance})
    concentration.update(
        {node: float(c_bulk) for node in exit_nodes - entrance}
    )
    q_albumin_out = 0.0
    seen: set[int] = set()
    for exit_node in exit_nodes:
        if exit_node not in graph:
            continue
        for neighbor in graph.neighbors(exit_node):
            if neighbor in exit_nodes:
                continue
            tid = int(graph[exit_node][neighbor]["throat_id"])
            if tid in seen:
                continue
            seen.add(tid)
            diag, off = _outward_coefficients(
                edge_map[tid], exit_node, neighbor
            )
            q_albumin_out -= (
                diag * concentration[exit_node] + off * concentration[neighbor]
            )
    return concentration, float(q_albumin_out), residual


def _water_exit_flow(
    graph: nx.Graph,
    conductance: dict[int, float],
    pressure: dict[int, float],
    exit_nodes: set[int],
) -> float:
    total, seen = 0.0, set()
    for exit_node in exit_nodes:
        for neighbor in graph.neighbors(exit_node):
            if neighbor in exit_nodes:
                continue
            tid = int(graph[exit_node][neighbor]["throat_id"])
            if tid in seen:
                continue
            seen.add(tid)
            total += conductance[tid] * (
                pressure[neighbor] - pressure[exit_node]
            )
    return float(total)


def _solve_closed_bulk(
    graph: nx.Graph,
    edge_map: dict[int, EdgeTransport],
    entrance: set[int],
    exit_nodes: set[int],
    q_water_exit: float,
) -> tuple[dict, pd.DataFrame]:
    c0, q0, r0 = _solve_concentration_at_cbulk(
        graph, edge_map, entrance, exit_nodes, 0.0
    )
    c1, q1, r1 = _solve_concentration_at_cbulk(
        graph, edge_map, entrance, exit_nodes, 1.0
    )
    slope = q1 - q0
    denominator = q_water_exit - slope
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError(f"Invalid bulk-closure denominator: {denominator}")
    c_bulk = q0 / denominator
    concentration = {
        node: c0[node] + (c1[node] - c0[node]) * c_bulk for node in c0
    }
    q_albumin = q0 + slope * c_bulk
    values = np.asarray(list(concentration.values()), dtype=float)
    info = {
        "sieving": float(c_bulk),
        "q_albumin_out": float(q_albumin),
        "q_water_exit": float(q_water_exit),
        "balance_error": float(q_albumin - q_water_exit * c_bulk),
        "residual_cbulk0": float(r0),
        "residual_cbulk1": float(r1),
        "concentration_min": float(values.min()),
        "concentration_max": float(values.max()),
    }
    concentration_df = pd.DataFrame(
        {"Pore ID": list(concentration), "Concentration": list(concentration.values())}
    ).sort_values("Pore ID")
    return info, concentration_df


def _prune_oob_clusters(
    graph: nx.Graph,
    concentration_df: pd.DataFrame,
    entrance: set[int],
    exit_nodes: set[int],
    *,
    high: float,
    low: float,
) -> tuple[nx.Graph, set[int], set[int], list[dict], set[int], set[int]]:
    concentration = dict(
        zip(
            concentration_df["Pore ID"].astype(int),
            pd.to_numeric(concentration_df["Concentration"], errors="coerce"),
        )
    )
    oob = {
        node
        for node, value in concentration.items()
        if node in graph
        and node not in (entrance | exit_nodes)
        and np.isfinite(value)
        and (float(value) > high or float(value) < low)
    }
    accepted: set[int] = set()
    records: list[dict] = []
    for cluster_id, component in enumerate(
        nx.connected_components(graph.subgraph(oob)), start=1
    ):
        cluster = set(component)
        trial = graph.copy()
        trial.remove_nodes_from(accepted | cluster)
        touches_boundary = bool(cluster & (entrance | exit_nodes))
        keeps_spanning = _has_spanning(trial, entrance, exit_nodes)
        remove = (not touches_boundary) and keeps_spanning
        if remove:
            accepted.update(cluster)
        vals = [float(concentration[node]) for node in cluster]
        records.append(
            {
                "cluster_id": cluster_id,
                "size": len(cluster),
                "accepted": remove,
                "touches_boundary": touches_boundary,
                "keeps_spanning_after_cumulative_removal": keeps_spanning,
                "min_concentration": min(vals),
                "max_concentration": max(vals),
                "n_above_high": sum(v > high for v in vals),
                "n_below_low": sum(v < low for v in vals),
                "node_ids": ",".join(map(str, sorted(cluster))),
            }
        )
    if not accepted:
        return graph, entrance, exit_nodes, records, oob, accepted
    out = graph.copy()
    out.remove_nodes_from(accepted)
    out, entrance_out, exit_out = _retain_spanning_and_prune_leaves(
        out, entrance, exit_nodes
    )
    return out, entrance_out, exit_out, records, oob, accepted


def calculate_existing_network(
    sample_name: str,
    sample_dir: Path,
    analysis_root: Path,
    output_root: Path,
    *,
    solute_radius_nm: float = DEFAULT_EXCLUSION_RADIUS_NM,
    hydrodynamic_radius_nm: float = DEFAULT_HYDRODYNAMIC_RADIUS_NM,
    p_in_pa: float = P_IN_PA,
    p_out_pa: float = P_OUT_PA,
    prune_high: float = PRUNE_HIGH,
    prune_low: float = PRUNE_LOW,
    prune_max_rounds: int = PRUNE_MAX_ROUNDS,
    save_concentration: bool = True,
    save_edge_transport: bool = False,
    result_tag: str = "dd2006_pruned",
) -> dict:
    """Calculate one saved network and return a one-row summary dictionary."""
    solute_radius_nm = float(solute_radius_nm)
    hydrodynamic_radius_nm = float(hydrodynamic_radius_nm)
    if solute_radius_nm <= 0.0 or hydrodynamic_radius_nm <= 0.0:
        raise ValueError("Solute and hydrodynamic radii must both be positive.")
    sample_dir, analysis_root, output_root = map(
        Path, (sample_dir, analysis_root, output_root)
    )
    analysis_dir = _find_analysis_dir(analysis_root, sample_name, sample_dir)
    out_dir = output_root / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pores = pd.read_excel(sample_dir / f"{sample_name}_pores.xlsx")
    throats = pd.read_excel(sample_dir / f"{sample_name}_throats.xlsx")
    pore_cls = pd.read_excel(
        analysis_dir / f"{sample_name}_pore_classification.xlsx"
    )
    solvent_cls = pd.read_excel(
        analysis_dir / f"{sample_name}_solvent_throat_classification.xlsx"
    )
    component_nodes = _component_nodes(
        analysis_dir / f"{sample_name}_solvent_penetration_components.xlsx"
    )
    coords = {
        int(row["Pore ID"]): np.asarray(
            [row["X Coord"], row["Y Coord"], row["Z Coord"]], dtype=float
        )
        for _, row in pores.iterrows()
    }
    pore_radius = {
        int(row["Pore ID"]): float(row["EqRadius"]) for _, row in pores.iterrows()
    }
    throat_rows = {
        int(row["Throat ID"]): row for _, row in throats.iterrows()
    }
    solvent_tids = set(
        pd.to_numeric(
            solvent_cls.loc[
                solvent_cls["In Solvent Penetration"].fillna(False).astype(bool),
                "Throat ID",
            ],
            errors="coerce",
        ).dropna().astype(int)
    )
    left = set(
        pd.to_numeric(
            pore_cls.loc[
                pore_cls["Is Surface X Left"].fillna(False).astype(bool), "Pore ID"
            ],
            errors="coerce",
        ).dropna().astype(int)
    )
    right = set(
        pd.to_numeric(
            pore_cls.loc[
                pore_cls["Is Surface X Right"].fillna(False).astype(bool), "Pore ID"
            ],
            errors="coerce",
        ).dropna().astype(int)
    )
    entrance_all, exit_all = left & component_nodes, right & component_nodes
    solvent_graph = nx.Graph()
    hydraulic_g: dict[int, float] = {}
    for tid in solvent_tids:
        row = throat_rows.get(tid)
        if row is None:
            continue
        p1, p2 = int(row["Pore ID #1"]), int(row["Pore ID #2"])
        if p1 not in component_nodes or p2 not in component_nodes:
            continue
        length_m = max(float(np.linalg.norm(coords[p2] - coords[p1])), 1e-3) * 1e-9
        radius_m = float(row["EqRadius"]) * 1e-9
        hydraulic_g[tid] = math.pi * radius_m**4 / (
            8.0 * WATER_VISCOSITY_PA_S * length_m
        )
        solvent_graph.add_edge(p1, p2, throat_id=tid)
    if not entrance_all or not exit_all or not _has_spanning(
        solvent_graph, entrance_all, exit_all
    ):
        return _write_disconnected_result(
            sample_name,
            out_dir,
            solute_radius_nm,
            hydrodynamic_radius_nm,
            result_tag,
        )
    solvent_graph, entrance_water, exit_water = _retain_spanning_and_prune_leaves(
        solvent_graph, entrance_all, exit_all
    )
    pressure = _solve_pressure(
        solvent_graph,
        hydraulic_g,
        entrance_water,
        exit_water,
        p_in=p_in_pa,
        p_out=p_out_pa,
    )
    q_water_exit = _water_exit_flow(
        solvent_graph, hydraulic_g, pressure, exit_water
    )
    solute_graph = nx.Graph()
    for tid in solvent_tids:
        row = throat_rows.get(tid)
        if row is None or float(row["EqRadius"]) <= solute_radius_nm:
            continue
        p1, p2 = int(row["Pore ID #1"]), int(row["Pore ID #2"])
        if (
            p1 in pressure
            and p2 in pressure
            and pore_radius.get(p1, 0.0) > solute_radius_nm
            and pore_radius.get(p2, 0.0) > solute_radius_nm
        ):
            solute_graph.add_edge(p1, p2, throat_id=tid)
    graph, entrance, exit_nodes = _retain_spanning_and_prune_leaves(
        solute_graph, entrance_water, exit_water
    )
    if graph.number_of_edges() == 0 or not entrance or not exit_nodes:
        return _write_disconnected_result(
            sample_name,
            out_dir,
            solute_radius_nm,
            hydrodynamic_radius_nm,
            result_tag,
            q_water_exit=q_water_exit,
        )
    initial_nodes = graph.number_of_nodes()
    all_records: list[dict] = []
    removed_cluster_nodes: set[int] = set()
    initial_oob = 0
    final_oob = 0
    info: dict = {}
    concentration_df = pd.DataFrame()
    edge_map: dict[int, EdgeTransport] = {}
    rounds_executed = 0
    rounds_with_removal = 0
    for round_index in range(1, int(prune_max_rounds) + 1):
        rounds_executed = round_index
        edge_map = {
            int(data["throat_id"]): _edge_transport(
                throat_rows[int(data["throat_id"])],
                coords,
                pressure,
                solute_radius_nm,
                hydrodynamic_radius_nm,
            )
            for _, _, data in graph.edges(data=True)
        }
        info, concentration_df = _solve_closed_bulk(
            graph, edge_map, entrance, exit_nodes, q_water_exit
        )
        (
            graph_after,
            entrance_after,
            exit_after,
            records,
            oob_nodes,
            accepted,
        ) = _prune_oob_clusters(
            graph,
            concentration_df,
            entrance,
            exit_nodes,
            high=prune_high,
            low=prune_low,
        )
        if round_index == 1:
            initial_oob = len(oob_nodes)
        all_records.extend(
            {"pruning_round": round_index, **record} for record in records
        )
        if not accepted:
            final_oob = len(oob_nodes)
            break
        rounds_with_removal += 1
        removed_cluster_nodes.update(accepted)
        graph, entrance, exit_nodes = (
            graph_after,
            entrance_after,
            exit_after,
        )
    else:
        raise RuntimeError(
            f"{sample_name}: pruning did not converge in {prune_max_rounds} rounds"
        )
    values = pd.to_numeric(concentration_df["Concentration"], errors="coerce")
    physical = (
        np.isfinite(info["sieving"])
        and -1e-12 <= info["sieving"] <= 1.0 + 1e-12
        and np.all(np.isfinite(values))
    )
    final_records = [
        record
        for record in all_records
        if int(record["pruning_round"]) == rounds_executed
    ]
    protected_low_path = bool(
        physical
        and final_oob > 0
        and final_records
        and all(
            not bool(record["accepted"])
            and not bool(record["touches_boundary"])
            and not bool(record["keeps_spanning_after_cumulative_removal"])
            and int(record["n_above_high"]) == 0
            and int(record["n_below_low"]) == int(record["size"])
            for record in final_records
        )
    )
    if physical and final_oob == 0:
        status = "valid"
    elif protected_low_path:
        status = "valid_protected_low_path"
    else:
        status = "review_required"
    result_rows = [
        ("sieving_coefficient (C_out/C_in, DD2006-pruned)", info["sieving"]),
        ("Total Albumin Flow Rate (Q_alb_total, equiv_C)", info["q_albumin_out"]),
        ("Plasma Concentration (C0)", 1.0),
        ("C_in (mean over entrance nodes)", 1.0),
        ("C_out (mean over exit nodes)", info["sieving"]),
        ("Q_total_full_solvent (m^3/s)", q_water_exit),
        ("Net Outlet Solvent Flow (Q, full solvent network, m^3/s)", q_water_exit),
        ("N entrance nodes in solute graph", len(entrance)),
        ("N exit nodes in solute graph", len(exit_nodes)),
        ("Number of Solute Accessible Throats", graph.number_of_edges()),
        ("new_J solution status", status),
        ("Transport model", "Dechadilok-Deen 2006 + exact 1-D edge flux"),
        ("Solute effective radius (nm)", solute_radius_nm),
        ("Hydrodynamic radius used for K_D and K_C (nm)", hydrodynamic_radius_nm),
        ("C_in normalization", 1.0),
        ("Final standardized pruning mandatory", 1),
        ("Final pruning high threshold (C >)", prune_high),
        ("Final pruning low threshold (C <)", prune_low),
        ("Final pruning rounds executed", rounds_executed),
        ("Final pruning rounds with removal", rounds_with_removal),
        ("Initial out-of-range pore count", initial_oob),
        ("Final out-of-range pore count", final_oob),
        ("Protected low-concentration spanning-path exception", int(protected_low_path)),
        ("Accepted abnormal-cluster pore count", len(removed_cluster_nodes)),
        ("Total pore count removed after cleanup", initial_nodes - graph.number_of_nodes()),
        ("Final concentration minimum", info["concentration_min"]),
        ("Final concentration maximum", info["concentration_max"]),
        ("Albumin mass-balance error (m^3/s at C0=1)", info["balance_error"]),
    ]
    pd.DataFrame(result_rows, columns=["Parameter", "Value"]).to_excel(
        out_dir / f"{sample_name}__sieving_summary.xlsx",
        index=False,
    )
    if save_concentration:
        concentration_df.to_excel(
            out_dir / f"{sample_name}__concentration__{result_tag}.xlsx",
            index=False,
        )
    pd.DataFrame(
        {"Pore ID": list(pressure), "Pressure (Pa)": list(pressure.values())}
    ).sort_values("Pore ID").to_excel(
        out_dir / f"{sample_name}__pressure__{result_tag}.xlsx",
        index=False,
    )
    if all_records:
        pd.DataFrame(all_records).to_excel(
            out_dir / f"{sample_name}__cluster_pruning__{result_tag}.xlsx",
            index=False,
        )
    if save_edge_transport:
        pd.DataFrame([asdict(edge) for edge in edge_map.values()]).to_excel(
            out_dir / f"{sample_name}__edge_transport__{result_tag}.xlsx",
            index=False,
        )
    return {
        "sample_name": sample_name,
        "solute_radius_nm": float(solute_radius_nm),
        "hydrodynamic_radius_nm": float(hydrodynamic_radius_nm),
        "status": status,
        "sieving_coefficient": float(info["sieving"]),
        "Q_alb_total_m3s": float(info["q_albumin_out"]),
        "Q_total_m3s": float(q_water_exit),
        "n_solute_nodes_final": graph.number_of_nodes(),
        "n_solute_throats_final": graph.number_of_edges(),
        "n_entrance_final": len(entrance),
        "n_exit_final": len(exit_nodes),
        "pruning_rounds_executed": rounds_executed,
        "pruning_rounds_with_removal": rounds_with_removal,
        "initial_oob_nodes": initial_oob,
        "final_oob_nodes": final_oob,
        "protected_low_path_exception": protected_low_path,
        "accepted_cluster_nodes_removed": len(removed_cluster_nodes),
        "total_nodes_removed_after_cleanup": initial_nodes - graph.number_of_nodes(),
        "concentration_min": float(info["concentration_min"]),
        "concentration_max": float(info["concentration_max"]),
        "result_file": str(
            out_dir / f"{sample_name}__sieving_summary.xlsx"
        ),
    }


def _write_disconnected_result(
    sample_name: str,
    out_dir: Path,
    solute_radius_nm: float,
    hydrodynamic_radius_nm: float,
    result_tag: str,
    *,
    q_water_exit: float = 0.0,
) -> dict:
    rows = [
        ("sieving_coefficient (C_out/C_in, DD2006-pruned)", 0.0),
        ("Total Albumin Flow Rate (Q_alb_total, equiv_C)", 0.0),
        ("Plasma Concentration (C0)", 1.0),
        ("Q_total_full_solvent (m^3/s)", q_water_exit),
        ("Net Outlet Solvent Flow (Q, full solvent network, m^3/s)", q_water_exit),
        ("N entrance nodes in solute graph", 0),
        ("N exit nodes in solute graph", 0),
        ("new_J solution status", "valid_disconnected_zero"),
        ("Transport model", "Dechadilok-Deen 2006 + exact 1-D edge flux"),
        ("Solute effective radius (nm)", solute_radius_nm),
        ("Hydrodynamic radius used for K_D and K_C (nm)", hydrodynamic_radius_nm),
        ("Final standardized pruning mandatory", 1),
        ("Final pruning rounds executed", 1),
        ("Final out-of-range pore count", 0),
    ]
    result_file = out_dir / f"{sample_name}__sieving_summary.xlsx"
    pd.DataFrame(rows, columns=["Parameter", "Value"]).to_excel(
        result_file, index=False
    )
    return {
        "sample_name": sample_name,
        "solute_radius_nm": float(solute_radius_nm),
        "hydrodynamic_radius_nm": float(hydrodynamic_radius_nm),
        "status": "valid_disconnected_zero",
        "sieving_coefficient": 0.0,
        "Q_alb_total_m3s": 0.0,
        "Q_total_m3s": float(q_water_exit),
        "n_solute_nodes_final": 0,
        "n_solute_throats_final": 0,
        "n_entrance_final": 0,
        "n_exit_final": 0,
        "pruning_rounds_executed": 1,
        "pruning_rounds_with_removal": 0,
        "initial_oob_nodes": 0,
        "final_oob_nodes": 0,
        "accepted_cluster_nodes_removed": 0,
        "total_nodes_removed_after_cleanup": 0,
        "concentration_min": math.nan,
        "concentration_max": math.nan,
        "result_file": str(result_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DD2006 conservative albumin transport on one saved network."
    )
    parser.add_argument("--sample-name", required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--solute-radius-nm",
        type=float,
        default=DEFAULT_EXCLUSION_RADIUS_NM,
        help="Exclusion radius used for accessibility and Phi.",
    )
    parser.add_argument(
        "--hydrodynamic-radius-nm",
        type=float,
        default=DEFAULT_HYDRODYNAMIC_RADIUS_NM,
        help=(
            "Physical albumin radius used only in DD2006 K_D and K_C. "
            "Defaults to the hydrated albumin radius, 3.55 nm."
        ),
    )
    parser.add_argument("--delta-p-pa", type=float, default=P_IN_PA - P_OUT_PA)
    parser.add_argument("--result-tag", default="dd2006_pruned")
    parser.add_argument("--no-concentration-output", action="store_true")
    parser.add_argument("--save-edge-transport", action="store_true")
    args = parser.parse_args()
    result = calculate_existing_network(
        args.sample_name,
        args.sample_dir,
        args.analysis_root,
        args.output_dir,
        solute_radius_nm=args.solute_radius_nm,
        hydrodynamic_radius_nm=args.hydrodynamic_radius_nm,
        p_in_pa=P_OUT_PA + float(args.delta_p_pa),
        p_out_pa=P_OUT_PA,
        save_concentration=not args.no_concentration_output,
        save_edge_transport=args.save_edge_transport,
        result_tag=args.result_tag,
    )
    print(pd.Series(result).to_string())


if __name__ == "__main__":
    main()

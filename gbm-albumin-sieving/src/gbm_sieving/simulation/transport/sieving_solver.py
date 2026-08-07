"""
子样本版本：溶质筛分系数计算（阶段B-F），对齐 solute_network_model_spec_v0.md。

等效浓度 / 纯喉网络：节点自由度为交汇点等效浓度；喉截面 A_eff=π(R_T−r)²；边上 f=1（不乘孔–喉 Φ 比）。
由工作流或实验模块调用；输出使用当前项目的统一文件名。
"""

import pandas as pd
import numpy as np
import networkx as nx
from pandas.core.missing import F
from scipy.sparse import csr_matrix, csc_matrix, diags as sparse_diags, vstack as sparse_vstack
from scipy.sparse.linalg import spsolve, gmres, spilu, LinearOperator
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree
from pathlib import Path
import sys
import argparse
import inspect
import subprocess
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib

from gbm_sieving.data_io.filenames import artifact_filename, decimal_token
from gbm_sieving.simulation.transport.hindrance import (
    dd2006_hindrance_factors,
    is_abnormal_equivalent_concentration,
)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====================================================================
# 参数设置 & 输入输出配置
# ====================================================================
# 溶质参数
SOLUTE_HYDRATED_RADIUS = 3.55  # nm，白蛋白的水合半径
# 溶质通路几何筛选：喉与孔特征半径须严格大于 r_s
SOLUTE_PASSAGE_GEOMETRY_MIN_NM = SOLUTE_HYDRATED_RADIUS * 1.0
SOLUTE_FREE_DIFFUSION = 9.26077952e-11  # m^2/s, Stokes-Einstein estimate at 37°C using r_s=3.55 nm and water viscosity below

# 是否考虑扩散项（默认True）
INCLUDE_DIFFUSION = True  # 如果为False，只考虑对流项

# |Pe| 超过此值时组装退化为纯对流（由 argparse 覆盖；运行默认 0=关闭该分段）
NEW_J_PE_ADV_ONLY_THRESHOLD = 0.0
# |Pe| 低于此值时仅 Fick 扩散（由 argparse 覆盖；运行默认 0=仅 |Pe|<1e-14 走 Fick）
NEW_J_PE_DIFF_ONLY_THRESHOLD = 0.0

# new_J is a historical solver label. The assembled edge quantity is a
# total albumin flow rate (concentration*m^3/s), not an area-normalized flux.
# |Pe|<diff 阈 为 Fick；|Pe|>adv 阈 为纯对流；中间为解析耦合（CLI 可调）。

# 弱连接剪枝：见解析参数 WEAK_LINK_PERCENTILE（默认关闭，需 0<p<100 才启用）。

# 静电作用系数（考虑纤维静电作用使溶质倾向于位于中心位置，增强快车道效应）
ELECTROSTATIC_ENHANCEMENT_FACTOR = 1.0  # K_c的增强系数

# 溶剂参数
SOLVENT_VISCOSITY = 6.91e-4  # Pa*s, dynamic viscosity of water at approximately 37 deg C
#SOLVENT_VISCOSITY = 1.5e-3  # Pa·s，血浆的粘度（37°C）
# 压力边界条件（根据肾小球过滤压力图）
# 根据图片：血液静水压55 mmHg，血液胶体渗透压30 mmHg，囊内静水压15 mmHg
# 净驱动压力 = (血液静水压 - 血液胶体渗透压) - 囊内静水压 = (55 - 30) - 15 = 10 mmHg

# 入口有效压力（毛细血管侧）：血液静水压 - 血液胶体渗透压
P_HYDROSTATIC_IN = 55.0  # mmHg，血液静水压
PI_PLASMA = 30.0  # mmHg，血浆胶体渗透压
P_IN_EFFECTIVE = P_HYDROSTATIC_IN - PI_PLASMA  # mmHg，入口有效压力 = 25 mmHg

# 出口压力（鲍曼囊侧）：囊内静水压
P_HYDROSTATIC_OUT = 15.0  # mmHg，囊内静水压
P_OUT_EFFECTIVE = P_HYDROSTATIC_OUT  # mmHg，出口有效压力 = 15 mmHg

# 转换为Pa（用于计算）
# 1 mmHg = 133.322 Pa
P_IN = P_IN_EFFECTIVE * 133.322  # Pa，入口有效压力（25 mmHg = 3333 Pa）
P_OUT = P_OUT_EFFECTIVE * 133.322  # Pa，出口有效压力（15 mmHg = 2000 Pa）

# 净驱动压力差
DELTA_P = (P_IN_EFFECTIVE - P_OUT_EFFECTIVE) * 133.322  # Pa，净压力差（10 mmHg = 1333 Pa）


class PressurePhysicsViolationError(ValueError):
    """Kirchhoff 解在内部节点上超出 [min(P_IN,P_OUT), max(P_IN,P_OUT)] 容许带；本样本压强场视为无解。"""


# 入口浓度（任意单位，因为输出是比值）
C0 = 1.0


def _throat_solute_diffusion_area_m2(r_throat_nm: float) -> float:
    """
    溶质可及有效横截面积 (m²)，规范 v0：A_eff = π (R_T − r)²（中心可及圆近似，R_T、r 单位 nm）。
    溶剂体积通量 Q 仍用 Poiseuille 几何截面 π R_T²。
    """
    if not np.isfinite(r_throat_nm):
        return 0.0
    r_s = float(SOLUTE_HYDRATED_RADIUS)
    rt = float(r_throat_nm)
    if rt <= r_s:
        return 0.0
    r_eff_nm = rt - r_s
    r_m = r_eff_nm * 1e-9
    return float(np.pi * (r_m ** 2))


def _throat_water_area_m2(r_throat_nm: float) -> float:
    """Solvent hydraulic cross-sectional area pi*R_T^2 (m^2)."""
    if not np.isfinite(r_throat_nm) or float(r_throat_nm) <= 0.0:
        return 0.0
    r_m = float(r_throat_nm) * 1e-9
    return float(np.pi * (r_m ** 2))


def _throat_new_j_adv_coeff_m3s(Kc: float, Q: float, r_throat_nm: float) -> float:
    """
    Advective volume-flow coefficient U = Kc * Q * A_eff / A_water.

    This corresponds to Q_alb,adv = Kc * (Q/A_water) * A_eff * C, while
    diffusion remains G = D_eff * A_eff / L. Therefore Pe = U/G.
    The function name keeps the legacy "new_j" solver label for CLI compatibility.
    """
    if not np.isfinite(Kc) or not np.isfinite(Q):
        return 0.0
    a_solute = _throat_solute_diffusion_area_m2(r_throat_nm)
    a_water = _throat_water_area_m2(r_throat_nm)
    if a_solute <= 0.0 or a_water <= 0.0:
        return 0.0
    return float(Kc) * float(Q) * float(a_solute / a_water)


def _throat_diff_coeff_m3s(r_throat_nm: float, L_nm: float, D_eff: float) -> float:
    """diff_coeff = D_eff * A_solute / L（m³/s），用于 new_J 耦合与 Pe。"""
    if D_eff <= 0 or not np.isfinite(D_eff) or L_nm <= 0:
        return 0.0
    a_s = _throat_solute_diffusion_area_m2(r_throat_nm)
    if a_s <= 0.0:
        return 0.0
    l_m = float(L_nm) * 1e-9
    return float(D_eff * a_s / l_m)


def _throat_advection_diffusion_coeffs_downstream_row(
    Kc: float,
    Q: float,
    diff_coeff: float,
) -> tuple[float, float]:
    """
    稳态一维：Q_alb = Kc*Q*(C_down - C_up*exp(Pe))/(1-exp(Pe))，
    Pe = Kc*Q/diff_coeff（diff_coeff = D_eff*A_solute/L，A_solute=π(R_T−r)²）。
    对下游节点行（净流入 +Q_alb）：Q_alb = c_dd*C_down + c_du*C_up。
    Pe>0：den = 1-e^{-Pe} = -expm1(-Pe)，系数只含 exp(-Pe)，避免 exp(Pe) 溢出；
    Pe<0：expm1(Pe) 与 exp(Pe)（≤1）。不设 |Pe| 切换阈值。
    （等效浓度：孔侧与喉一致，无界面因子。）
    """
    if diff_coeff <= 0 or not np.isfinite(diff_coeff):
        return 0.0, 0.0
    if not np.isfinite(Q) or not np.isfinite(Kc):
        return 0.0, 0.0
    Pe = Kc * float(Q) / float(diff_coeff)
    if not np.isfinite(Pe):
        return 0.0, 0.0
    if abs(Pe) < 1e-11:
        return -float(diff_coeff), float(diff_coeff)
    kq = Kc * float(Q)
    if Pe > 0.0:
        den = -float(np.expm1(-Pe))
        if not np.isfinite(den) or abs(den) < 1e-300:
            return 0.0, float(kq)
        en = float(np.exp(-Pe))
        return float(-kq * en / den), float(kq / den)
    em = float(np.expm1(Pe))
    if abs(em) < 1e-30:
        return -float(diff_coeff), float(diff_coeff)
    ep = float(np.exp(Pe))
    return float(-kq / em), float(kq * ep / em)


def _throat_advection_diffusion_coeffs_upstream_row(
    Kc: float,
    Q: float,
    diff_coeff: float,
) -> tuple[float, float]:
    """
    对上游节点行（净流出 -Q_alb）：-Q_alb = c_uu*C_up + c_ud*C_down。
    若下游行写 Q_alb = c_dd*C_down + c_du*C_up，则 c_uu = -c_du、c_ud = -c_dd（与一维解析总流率一致）。
    一律由已验证的下游系数导出，避免各 Pe 分支符号不一致。
    """
    c_dd, c_du = _throat_advection_diffusion_coeffs_downstream_row(Kc, Q, diff_coeff)
    return float(-c_du), float(-c_dd)


def _throat_coupled_flux_vol(
    Kc: float,
    Q: float,
    diff_coeff: float,
    C_pore_up: float,
    C_pore_down: float,
) -> float:
    """与组装一致的喉白蛋白总流率（沿流动方向：上游 pore -> 下游 pore）。"""
    Cu = float(C_pore_up)
    if diff_coeff <= 0 or not np.isfinite(diff_coeff):
        return float(Kc * Q * Cu)
    Cd = float(C_pore_down)
    Pe = Kc * float(Q) / float(diff_coeff)
    if not np.isfinite(Pe):
        return float(Kc * Q * Cu)
    if abs(Pe) < 1e-14:
        return float(diff_coeff) * (Cu - Cd)
    kq = Kc * float(Q)
    if Pe > 0.0:
        den = -float(np.expm1(-Pe))
        if not np.isfinite(den) or abs(den) < 1e-300:
            return float(kq * Cu)
        en = float(np.exp(-Pe))
        return float(kq * (Cu - Cd * en) / den)
    em = float(np.expm1(Pe))
    if abs(em) < 1e-30:
        return float(diff_coeff) * (Cu - Cd)
    return float(-kq / em * (Cd - Cu * float(np.exp(Pe))))


def _throat_pe_regime(Kc: float, Q: float, diff_coeff: float) -> str:
    """
    无有效扩散 → adv_only；否则 Pe=Kc*Q/diff_coeff。
    |Pe|>NEW_J_PE_ADV_ONLY_THRESHOLD（>0 时）→ adv_only（纯对流）；
    |Pe|<NEW_J_PE_DIFF_ONLY_THRESHOLD（>0 时）→ diff_only（仅 Fick）；若 diff 阈为 0 则仅 |Pe|<1e-14 → diff_only；
    其余 → coupled（解析闭式系数）。与 throat_Pe 统计一致。
    """
    if not INCLUDE_DIFFUSION or diff_coeff <= 0.0 or not np.isfinite(diff_coeff):
        return "adv_only"
    pe = float(Kc) * float(Q) / float(diff_coeff)
    if not np.isfinite(pe):
        return "coupled"
    thr = float(NEW_J_PE_ADV_ONLY_THRESHOLD)
    if thr > 0.0 and abs(pe) > thr:
        return "adv_only"
    dlo = float(NEW_J_PE_DIFF_ONLY_THRESHOLD)
    if dlo > 0.0:
        if abs(pe) < dlo:
            return "diff_only"
    elif abs(pe) < 1e-14:
        return "diff_only"
    return "coupled"


def _throat_fick_row_downstream(diff_coeff: float) -> tuple[float, float]:
    """
    Pe→0 极限：下游节点行（neighbor 为上游 U，当前节点为下游 D）。
    与 Fick 总流率 Q_alb=diff_coeff·(C_U−C_D) 写入离散守恒一致。
    """
    d = float(diff_coeff)
    return -d, d


def _throat_fick_row_upstream(diff_coeff: float) -> tuple[float, float]:
    """
    Pe→0 极限：上游节点行（当前节点为 U，neighbor 为下游 D）。
    对 U：−diff·C_U + diff·C_D（与下游行配对）。
    """
    d = float(diff_coeff)
    return -d, d


def _old_j_adv_dii_dij(
    Kc: float, Q: float, *, flow_from_neighbor: bool
) -> tuple[float, float]:
    """
    old_J 对流在节点 i 行上的 (ΔA_ii, ΔA_ij)。
    Q 为沿 node→neighbor 的 signed Q_ij（与喉 p1→p2 一致）。
    算术平均喉：与「净流入 i」J_in = -Kc*Q_ij*(Ci+Cj)/2 + … 一致，故系数为 −Kc*Q_ij/2。
    纯上风：邻孔为上游时 A_ij=+Kc|Q|（入流正比于 C_j）；本孔为上游时 A_ii=−Kc|Q|（出流）。
    """
    if OLD_J_MEAN_THROAT_CONV:
        h = 0.5 * float(Kc) * float(Q)
        return -h, -h
    qm = abs(float(Q))
    kq = float(Kc) * qm
    if flow_from_neighbor:
        return 0.0, kq
    return -kq, 0.0


def _pore_geometrically_passable_for_solute(r_pore_nm: float) -> bool:
    """孔半径大于 r_s 时可几何进入（通路筛选）。"""
    return float(r_pore_nm) > float(SOLUTE_PASSAGE_GEOMETRY_MIN_NM)


parser = argparse.ArgumentParser(
    description=(
        "Compute sieving coefficient (equiv concentration / pure-throat network, spec v0). "
        "See solute_network_model_spec_v0.md."
    )
)
parser.add_argument(
    "--sample-name",
    type=str,
    default="118",
    help="样本名字，用于输入文件前缀和规范输出文件名。",
)
parser.add_argument(
    "--sample-dir",
    type=str,
    default="118example",
    help="样本所在文件夹，里面应包含 <sample-name>_pores.xlsx 和 <sample-name>_throats.xlsx。",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="results_core_subsamples",
    help=(
        "分析结果根目录或父样本目录。与 run_sub_sample_pipeline 一致时："
        "应指向父样本文件夹，如 results_core_large_subsamples/AS338（内含 AS338_largebox/ 子文件夹）。"
        "若只给根目录，则自动使用 <output-dir>/<sample-dir末级名>/<sample-name>/。"
    ),
)
parser.add_argument(
    "--no-html",
    action="store_true",
    help="跳过压强、出口通量、浓度等 3D HTML 可视化输出；批量模拟时建议开启。",
)
parser.add_argument(
    "--diagnostic-pngs",
    action="store_true",
    help=(
        "Write heavy Phase4 diagnostic PNGs "
        "(throat lambda/Kd/Kc, throat Q/diff, Peclet/matrix). "
        "Default is off for batch simulations."
    ),
)
parser.add_argument(
    "--delta-p-pa",
    type=float,
    default=None,
    help=(
        "Override the net inlet-outlet pressure difference in Pa while keeping "
        "the outlet effective pressure unchanged. Default keeps the built-in "
        "10 mmHg (1333 Pa) pressure difference."
    ),
)
parser.add_argument(
    "--radius-delta",
    type=float,
    default=0.0,
    help="孔和喉半径的调整值（单位：nm），将应用到所有孔和喉的半径上。默认值为0.0（不调整）。",
)
parser.add_argument(
    "--pressure-physics-tol",
    type=float,
    default=10.0,
    help=(
        "压强物理带检验：①内部超节点 Kirchhoff 解 ②映射/补全后各孔最终压强（任一步越界则 exit(1)）。"
        "容差(Pa)，默认 10；设为负值关闭。"
    ),
)
parser.add_argument(
    "--concentration-npz-out",
    type=str,
    default=None,
    help=(
        "若提供路径，将溶质节点浓度写入 npz（pore_ids, C_new, has_new_J）。"
    ),
)
parser.add_argument(
    "--dump-concentration-matrix-xlsx",
    action="store_true",
    help=(
        "强制导出最终浓度线性系统 A/b 到 xlsx（默认仅在溶质节点数<50时自动导出）。"
    ),
)
parser.add_argument(
    "--new-j-stabilization",
    type=float,
    default=0,
    help=(
        "new_J 解析耦合浓度方程：对矩阵 A 施加对角稳定化 A += ε·I（默认 0 关闭），"
        "缓解秩亏/病态；例如 1e-10。"
    ),
)
parser.add_argument(
    "--solute-radius-nm",
    type=float,
    default=SOLUTE_HYDRATED_RADIUS,
    help="Solute hydrated radius in nm; default is albumin radius 3.55 nm.",
)
parser.add_argument(
    "--pressure-flow-tol-pa",
    type=float,
    default=1e-9,
    help=(
        "喉两端压强差 |P_neighbor−P_node| 低于此值(Pa)时视为无压力驱动对流，"
        "仅组装 Fick 扩散（若有）；无扩散则该喉不贡献矩阵元。避免平压/极小压差下浮点误判流向。"
    ),
)
parser.add_argument(
    "--new-j-box-lsq",
    dest="new_j_box_lsq",
    action="store_true",
    help=(
        "启用 new_J 盒约束 LSQ（默认关闭：无约束 spsolve）。"
        "在 [new-j-c-min, new-j-c-max] 上 min ||Ax-b||_2（lsq_linear），非事后 clip。"
    ),
)
parser.add_argument(
    "--no-new-j-box-lsq",
    dest="new_j_box_lsq",
    action="store_false",
    help="显式关闭盒约束（与默认一致）。",
)
parser.set_defaults(new_j_box_lsq=False)
parser.add_argument(
    "--new-j-c-min",
    type=float,
    default=0.0,
    help="仅 --new-j-box-lsq：各节点浓度下界（默认 0）。",
)
parser.add_argument(
    "--new-j-c-max",
    type=float,
    default=1.0,
    help="仅 --new-j-box-lsq：各节点浓度上界（默认 1，与 C0=1 无量纲一致）。",
)
parser.add_argument(
    "--new-j-box-ridge",
    type=float,
    default=1e-10,
    help=(
        "仅 --new-j-box-lsq：岭项 λ（增广最小二乘，见代码）。"
        "需要盒约束时再调；默认 new_J 为无约束 spsolve。"
    ),
)
parser.add_argument(
    "--assembly-row-audit",
    action="store_true",
    help=(
        "组装 new_J 矩阵 A（稳定化前）后：令 n=rank(A)，取全局 |A_ij| 最大的 min(2n,nnz) 个位置；"
        "写出**单表** Excel（每行一个组分，列含父元 A_ij、整格表达式与各组分 Δ 及 Q/Kc/diff/Pe 等）。"
    ),
)
parser.add_argument(
    "--assembly-row-audit-verbose",
    action="store_true",
    help=(
        "与 --assembly-row-audit 配合：额外在控制台打印「按行 ||A[i,:]||_∞ 排序」的旧版逐行分项（"
        "见 --assembly-row-audit-top-rows / --assembly-row-audit-top-entries）。"
    ),
)
parser.add_argument(
    "--assembly-row-audit-top-rows",
    type=int,
    default=25,
    help="仅 --assembly-row-audit-verbose：最多诊断多少行（按行 ∞-范数从大到小）。",
)
parser.add_argument(
    "--assembly-row-audit-top-entries",
    type=int,
    default=14,
    help="仅 --assembly-row-audit-verbose：每行最多列出多少条分项。",
)
parser.add_argument(
    "--new-j-negative-tol",
    type=float,
    default=1e-10,
    help="new_J 有效性判据：若最终节点浓度 min(C) < -tol 则视为无效解（默认 1e-10）。",
)
parser.add_argument(
    "--new-j-rel-residual-invalid-threshold",
    type=float,
    default=1e3,
    help=(
        "new_J 有效性判据：若最终线性系统相对残差 > 阈值则视为无效解（默认 1e3）。"
        "设为 <=0 可关闭该判据。"
    ),
)
parser.add_argument(
    "--disable-auto-prune-oob-concentration",
    action="store_true",
    help=(
        "Disable automatic retry after out-of-bounds new_J concentration clusters. "
        "By default, removable abnormal clusters are excluded only from the solute "
        "concentration graph and the same output files are overwritten by the retry."
    ),
)
parser.add_argument(
    "--auto-prune-oob-high-threshold",
    type=float,
    default=1e4,
    help="Upper concentration threshold for automatic abnormal-cluster pruning (default: 1e4).",
)
parser.add_argument(
    "--auto-prune-oob-low-threshold",
    type=float,
    default=1e-6,
    help="Lower concentration threshold for automatic abnormal-cluster pruning (default: 1e-6).",
)
parser.add_argument(
    "--auto-prune-oob-max-passes",
    type=int,
    default=100,
    help="Maximum automatic abnormal-cluster pruning retries (default: 100).",
)
parser.add_argument("--auto-prune-oob-pass", type=int, default=0, help=argparse.SUPPRESS)
parser.add_argument("--solute-exclude-pore-ids", type=str, default="", help=argparse.SUPPRESS)
parser.add_argument("--solute-exclude-throat-ids", type=str, default="", help=argparse.SUPPRESS)
parser.add_argument(
    "--new-j",
    dest="solve_new_j_deprecated",
    action="store_true",
    help=(
        "已弃用：new_J 现为默认口径，此参数保留兼容旧命令行。"
    ),
)
parser.add_argument(
    "--linear-solver",
    type=str,
    default="direct",
    choices=("direct", "gmres-ilut"),
    help=(
        "线性求解器（用于 new_J 无盒约束）："
        "direct=spsolve（默认），gmres-ilut=GMRES+ILUT 预条件。"
    ),
)
parser.add_argument(
    "--compare-linear-solvers",
    action="store_true",
    help=(
        "同次运行同时计算 direct 与 gmres-ilut 的线性解并打印对比（残差与解差）；"
        "最终采用 --linear-solver 指定的解。"
    ),
)
parser.add_argument(
    "--conc-equilibrate",
    type=str,
    default="none",
    choices=("none", "row", "row-col"),
    help=(
        "浓度方程 Ax=b 求解前等式缩放：none=关闭；row=行平衡 D_r A x=D_r b（每行除以该行 |A| 的 max）；"
        "row-col=再行/列平衡后解 D_r A D_c z=D_r b，再 x=D_c z。"
    ),
)
parser.add_argument(
    "--gmres-rtol",
    type=float,
    default=1e-50,
    help="GMRES 相对容差 rtol（默认 1e-50）。",
)
parser.add_argument(
    "--gmres-atol",
    type=float,
    default=0.0,
    help="GMRES 绝对容差 atol（默认 0）。",
)
parser.add_argument(
    "--gmres-maxiter",
    type=int,
    default=10000,
    help="GMRES 最大迭代次数（默认 10000）。",
)
parser.add_argument(
    "--ilut-drop-tol",
    type=float,
    default=1e-5,
    help="ILUT drop_tol（默认 1e-4，越小预条件越强但更慢/更耗内存）。",
)
parser.add_argument(
    "--ilut-fill-factor",
    type=float,
    default=50.0,
    help="ILUT fill_factor（默认 20，越大预条件越强但更耗内存）。",
)
parser.add_argument(
    "--concentration-clip-refine",
    action="store_true",
    help=(
        "浓度线性方程：先 spsolve(direct)；若有分量越出 [clip-lo,clip-hi]，"
        "则 GMRES 初值为对调越界端（x<clip-lo→clip-hi，x>clip-hi→clip-lo；盒内不变），"
        "GMRES+ILUT 迭代，外循环直至全在盒内、或残差足够小、或达上限；"
        "连续两轮越界模式相同则提前停止。"
    ),
)
parser.add_argument(
    "--concentration-clip-lo",
    type=float,
    default=0.0,
    help="与 --concentration-clip-refine 配合：clip 下界（默认 0）。",
)
parser.add_argument(
    "--concentration-clip-hi",
    type=float,
    default=1.0,
    help="与 --concentration-clip-refine 配合：clip 上界（默认 1，与无量纲 C 一致）。",
)
parser.add_argument(
    "--clip-refine-max-outer",
    type=int,
    default=0,
    help="--concentration-clip-refine：外循环（clip→GMRES）最多次数（默认 10）。",
)
parser.add_argument(
    "--clip-refine-inner-maxiter",
    type=int,
    default=10000,
    help="--concentration-clip-refine：每轮 GMRES 最大迭代数（默认 10000，与全局 --gmres-maxiter 独立）。",
)
parser.add_argument(
    "--weak-link-percentile",
    type=float,
    default=0.0,
    help=(
        "D2.8 弱连接剪枝：在当前溶质可通行喉集合上，分别计算 |Q| 与 D_eff 的 p 分位数（默认 p=0，即关闭）；"
        "仅当该喉同时满足 |Q|<=p 分位 且 D_eff<=p 分位时移除。"
        "INCLUDE_DIFFUSION=False 时 D_eff 全为 0，退化为仅按 |Q| 的弱尾剪枝。"
        "p<=0 或 p>=100 时关闭该剪枝。"
    ),
)
parser.add_argument(
    "--new-j-pe-adv-only-above",
    type=float,
    default=0.0,
    help=(
        "new_J 分段：当 INCLUDE_DIFFUSION 且 diff_coeff>0 时，"
        "若 |Pe|=|Kc*Q/diff_coeff| 大于此值则仅组装纯对流（与无扩散 adv_only 分支一致），不再用解析耦合/Fick。"
        "默认 0（关闭该分段）；设为正数可启用。"
    ),
)
parser.add_argument(
    "--new-j-pe-diff-only-below",
    type=float,
    default=0.0,
    help=(
        "new_J 分段：当 INCLUDE_DIFFUSION 且 diff_coeff>0 时，"
        "若 |Pe|<此值则仅组装 Fick 扩散（不加对流项），不再用解析耦合。"
        "默认 0（关闭该分段）；设为正数可启用。"
    ),
)

_args, _unknown = parser.parse_known_args()
if getattr(_args, "delta_p_pa", None) is not None:
    _delta_p_override = float(_args.delta_p_pa)
    if not np.isfinite(_delta_p_override) or _delta_p_override <= 0.0:
        raise SystemExit("--delta-p-pa must be a positive finite value")
    P_IN = P_OUT + _delta_p_override
    DELTA_P = _delta_p_override
    P_IN_EFFECTIVE = P_IN / 133.322
    P_HYDROSTATIC_IN = P_IN_EFFECTIVE + PI_PLASMA
SOLVE_NEW_J = True
SOLVE_OLD_J = False
def _parse_int_id_csv(value: object) -> set[int]:
    out: set[int] = set()
    text = str(value or "").strip()
    if not text:
        return out
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            out.add(int(float(token)))
        except Exception:
            continue
    return out


if bool(getattr(_args, "solve_new_j_deprecated", False)):
    print("  [兼容] --new-j 已无须显式传入：new_J 现为默认开启。")
# 兼容现有分支判断：旧变量名保留，但语义固定为“非 old-only”。
OLD_J_ONLY = False
OLD_J_MEAN_THROAT_CONV = True
SIEVE_SAMPLE_NAME = _args.sample_name
SIEVE_SAMPLE_DIR = Path(_args.sample_dir)
SOLUTE_HYDRATED_RADIUS = float(getattr(_args, "solute_radius_nm", SOLUTE_HYDRATED_RADIUS))
if SOLUTE_HYDRATED_RADIUS <= 0.0:
    raise SystemExit("--solute-radius-nm must be positive")
SOLUTE_PASSAGE_GEOMETRY_MIN_NM = SOLUTE_HYDRATED_RADIUS * 1.0
RADIUS_DELTA = _args.radius_delta
PRESSURE_PHYSICS_TOL_PA = float(_args.pressure_physics_tol)
NO_HTML = bool(getattr(_args, "no_html", False))
DIAGNOSTIC_PNGS = bool(getattr(_args, "diagnostic_pngs", False))
LINEAR_SOLVER = str(getattr(_args, "linear_solver", "direct"))
COMPARE_LINEAR_SOLVERS = bool(getattr(_args, "compare_linear_solvers", False))
GMRES_RTOL = float(getattr(_args, "gmres_rtol", 1e-50))
GMRES_ATOL = float(getattr(_args, "gmres_atol", 0.0))
GMRES_MAXITER = int(getattr(_args, "gmres_maxiter", 10000))
ILUT_DROP_TOL = float(getattr(_args, "ilut_drop_tol", 1e-4))
ILUT_FILL_FACTOR = float(getattr(_args, "ilut_fill_factor", 50.0))
PRESSURE_FLOW_TOL_PA = float(getattr(_args, "pressure_flow_tol_pa", 1e-9))
NEW_J_NEGATIVE_TOL = max(0.0, float(getattr(_args, "new_j_negative_tol", 1e-10)))
NEW_J_REL_RESIDUAL_INVALID_THRESHOLD = float(
    getattr(_args, "new_j_rel_residual_invalid_threshold", 1e3)
)
CLIP_REFINE_ENABLED = bool(getattr(_args, "concentration_clip_refine", False))
CLIP_REFINE_LO = float(getattr(_args, "concentration_clip_lo", 0.0))
CLIP_REFINE_HI = float(getattr(_args, "concentration_clip_hi", 1.0))
CLIP_REFINE_MAX_OUTER = max(1, int(getattr(_args, "clip_refine_max_outer", 10)))
CLIP_REFINE_INNER_MAXITER = max(1, int(getattr(_args, "clip_refine_inner_maxiter", 10000)))
WEAK_LINK_PERCENTILE = float(getattr(_args, "weak_link_percentile", 0.0))
NEW_J_PE_ADV_ONLY_THRESHOLD = float(getattr(_args, "new_j_pe_adv_only_above", 0.0))
NEW_J_PE_DIFF_ONLY_THRESHOLD = float(getattr(_args, "new_j_pe_diff_only_below", 0.0))
AUTO_PRUNE_OOB_CONCENTRATION = not bool(getattr(_args, "disable_auto_prune_oob_concentration", False))
AUTO_PRUNE_OOB_HIGH_THRESHOLD = float(getattr(_args, "auto_prune_oob_high_threshold", 1e4))
AUTO_PRUNE_OOB_LOW_THRESHOLD = float(getattr(_args, "auto_prune_oob_low_threshold", 1e-6))
if AUTO_PRUNE_OOB_LOW_THRESHOLD >= AUTO_PRUNE_OOB_HIGH_THRESHOLD:
    parser.error("--auto-prune-oob-low-threshold must be smaller than --auto-prune-oob-high-threshold")
AUTO_PRUNE_OOB_MAX_PASSES = max(0, int(getattr(_args, "auto_prune_oob_max_passes", 100)))
AUTO_PRUNE_OOB_PASS = max(0, int(getattr(_args, "auto_prune_oob_pass", 0)))
SOLUTE_EXCLUDE_PORE_IDS = _parse_int_id_csv(getattr(_args, "solute_exclude_pore_ids", ""))
SOLUTE_EXCLUDE_THROAT_IDS = _parse_int_id_csv(getattr(_args, "solute_exclude_throat_ids", ""))
CONC_EQUILIBRATE_MODE = str(getattr(_args, "conc_equilibrate", "none") or "none").strip().lower()
if CONC_EQUILIBRATE_MODE not in ("none", "row", "row-col"):
    CONC_EQUILIBRATE_MODE = "none"

_CONC_EQUILIBRATE_LOGGED = False
if SOLUTE_EXCLUDE_PORE_IDS:
    print(
        "  [new_J auto-prune] Excluding "
        f"{len(SOLUTE_EXCLUDE_PORE_IDS)} pore(s) from the solute concentration graph only: "
        f"{sorted(SOLUTE_EXCLUDE_PORE_IDS)[:20]}"
        + (" ..." if len(SOLUTE_EXCLUDE_PORE_IDS) > 20 else "")
    )
if SOLUTE_EXCLUDE_THROAT_IDS:
    print(
        "  [new_J trial] Excluding "
        f"{len(SOLUTE_EXCLUDE_THROAT_IDS)} throat(s) from the solute concentration graph only: "
        f"{sorted(SOLUTE_EXCLUDE_THROAT_IDS)[:20]}"
        + (" ..." if len(SOLUTE_EXCLUDE_THROAT_IDS) > 20 else "")
    )


def _pressure_tie_neighbor_to_node(neighbor: int, node: int) -> tuple[bool, float]:
    """两侧均有压强时返回 (pressure_tie, dp)，dp=P_neighbor−P_node；否则 (False, 0.0)。"""
    if neighbor not in node_pressures or node not in node_pressures:
        return False, 0.0
    dp = float(node_pressures[neighbor]) - float(node_pressures[node])
    return abs(dp) <= PRESSURE_FLOW_TOL_PA, dp


def _pressure_tie_neighbor_to_exit(neighbor: int, exit_node: int) -> tuple[bool, float]:
    """两侧均有压强时返回 (pressure_tie, dp)，dp=P_neighbor−P_exit；否则 (False, 0.0)。"""
    if neighbor not in node_pressures or exit_node not in node_pressures:
        return False, 0.0
    dp = float(node_pressures[neighbor]) - float(node_pressures[exit_node])
    return abs(dp) <= PRESSURE_FLOW_TOL_PA, dp


# 原始孔喉数据根据样本名/样本文件夹自动拼接
pores_file = SIEVE_SAMPLE_DIR / f"{SIEVE_SAMPLE_NAME}_pores.xlsx"
throats_file = SIEVE_SAMPLE_DIR / f"{SIEVE_SAMPLE_NAME}_throats.xlsx"

# 输出目录（与 pipelines/run_sub_sample_pipeline 一致）：
# - pipeline 传入的 --output-dir 为「父样本」目录，如 .../results_core_large_subsamples/AS338，
#   本子样本（大子样本或切割子样本）产物在 <output-dir>/<sample-name>/。
# - 若只给输出根（如 results_core_subsamples），则落在 <根>/<sample-dir 末级目录名>/<sample-name>/，
#   例如 large_subsamples/AS338 + AS338_largebox -> .../AS338/AS338_largebox/。
output_folder_base = Path(_args.output_dir)
if not output_folder_base.is_absolute():
    output_folder_base = (Path.cwd() / output_folder_base).resolve()
else:
    output_folder_base = output_folder_base.resolve()

_sample_dir_resolved = Path(_args.sample_dir).resolve()
_parent_from_inputs = _sample_dir_resolved.name
_solvent_cls_name = f"{SIEVE_SAMPLE_NAME}_solvent_throat_classification.xlsx"
_p_flat = output_folder_base / SIEVE_SAMPLE_NAME
# 仅当 --output-dir 为「输出根」而非父样本目录时：.../<父目录名>/<子样本名>/
_p_nested_under_root = output_folder_base / _parent_from_inputs / SIEVE_SAMPLE_NAME

if output_folder_base.name == _parent_from_inputs:
    # pipeline：--output-dir 已为父样本目录，如 .../results_core_large_subsamples/AS338
    output_path = output_folder_base / SIEVE_SAMPLE_NAME
elif (_p_flat / _solvent_cls_name).is_file():
    output_path = _p_flat
elif (_p_nested_under_root / _solvent_cls_name).is_file():
    output_path = _p_nested_under_root
else:
    output_path = (
        _p_nested_under_root if _parent_from_inputs != SIEVE_SAMPLE_NAME else _p_flat
    )

output_path.mkdir(parents=True, exist_ok=True)
if output_path == _p_nested_under_root and output_folder_base.name != _parent_from_inputs:
    print(
        f"  输出目录（嵌套，与 pipeline 一致）: {output_path} "
        f"（父目录名={_parent_from_inputs}，子样本={SIEVE_SAMPLE_NAME}）"
    )

# Canonical output naming. Historical readers remain supported in data_io.
def get_output_filename(base_name: str, extension: str = ".xlsx") -> str:
    prefix = f"{SIEVE_SAMPLE_NAME}_"
    artifact = base_name[len(prefix):] if base_name.startswith(prefix) else base_name
    artifact_aliases = {
        "sieving_coefficient_results": "sieving_summary",
        "throat_Q_diff_coeff_distribution": "throat_transport_coefficients",
        "concentration_linear_system_debug": "concentration_system_debug",
    }
    artifact = artifact_aliases.get(artifact, artifact)
    variant = None
    if RADIUS_DELTA != 0.0:
        variant = f"geometry-radius-delta-{decimal_token(RADIUS_DELTA)}nm"
    return artifact_filename(
        SIEVE_SAMPLE_NAME,
        artifact,
        variant=variant,
        extension=extension,
    )


def _check_internal_pressure_in_boundary_band(
    P_internal: np.ndarray,
    p_in: float,
    p_out: float,
    tol_pa: float,
) -> None:
    """
    内部节点上的 Kirchhoff 解应在两侧 Dirichlet 压强之间（离散最大值原理）。
    若超出 [min(P_IN,P_OUT), max(P_IN,P_OUT)] 超过 tol，抛出 PressurePhysicsViolationError，
    由主流程中止本样本（不采用几何插值 fallback）。
    tol_pa < 0 时不检验。
    """
    if tol_pa < 0 or len(P_internal) == 0:
        return
    p_lo = min(p_in, p_out)
    p_hi = max(p_in, p_out)
    bad = (P_internal < p_lo - tol_pa) | (P_internal > p_hi + tol_pa)
    if np.any(bad):
        n_bad = int(np.sum(bad))
        raise PressurePhysicsViolationError(
            f"内部节点压强物理检验失败：{n_bad}/{len(P_internal)} 个内部节点解超出 "
            f"[{p_lo:.4f}, {p_hi:.4f}] Pa（容差 ±{tol_pa:g} Pa）。"
        )


def _finalize_pore_pressure_physics_check(node_pressures: dict) -> None:
    """
    最终用于喉流量计算的各孔压强须在 [min(P_IN,P_OUT), max(P_IN,P_OUT)] ± tol 内。
    映射/几何补全后仍可能越界（与 P_internal 检验无关），此时同样视为本样本压强无解。
    """
    if PRESSURE_PHYSICS_TOL_PA < 0:
        return
    p_lo = min(P_IN, P_OUT)
    p_hi = max(P_IN, P_OUT)
    arr = np.array(list(node_pressures.values()), dtype=float)
    if len(arr) == 0:
        return
    if not np.all(np.isfinite(arr)):
        print("    各孔压强含 NaN/Inf，本样本视为无解，已中止。")
        sys.exit(1)
    bad = (arr < p_lo - PRESSURE_PHYSICS_TOL_PA) | (arr > p_hi + PRESSURE_PHYSICS_TOL_PA)
    if np.any(bad):
        n_bad = int(np.sum(bad))
        pmin, pmax = float(np.min(arr)), float(np.max(arr))
        print(
            f"    各孔压强物理检验失败：{n_bad}/{len(arr)} 个孔超出 [{p_lo:.4f}, {p_hi:.4f}] Pa "
            f"± {PRESSURE_PHYSICS_TOL_PA:g} Pa（当前 min/max={pmin:.4f}/{pmax:.4f} Pa）。"
            f" 虽 Kirchhoff 内部解已通过，各孔压强补全或映射仍产生越界，本样本视为无解，已中止。"
        )
        sys.exit(1)


# 这些文件来自 analyze_pores_and_throats.py，对当前样本名有前缀
solvent_classification_file = output_path / f"{SIEVE_SAMPLE_NAME}_solvent_throat_classification.xlsx"
solvent_components_file = output_path / f"{SIEVE_SAMPLE_NAME}_solvent_penetration_components.xlsx"

print("=" * 70)
print("数据溶质筛分系数计算")
print("=" * 70)
print(f"扩散项计算: {'启用' if INCLUDE_DIFFUSION else '禁用'}")
print(f"静电增强系数: {ELECTROSTATIC_ENHANCEMENT_FACTOR}")
print(
    "等效浓度(v0): 喉-节点界面 f=1; Steric 经 A_eff；D_eff=D0*K_D，v_alb=K_C*v_water（Dechadilok-Deen 2006）。"
)
print("出口边界条件: 入口 C=C0，出口统一 C_bulk（参数化两次求解 + 全局守恒闭合）")
if RADIUS_DELTA != 0.0:
    print(f"半径调整值: {RADIUS_DELTA:.3f} nm (所有孔和喉的半径已调整)")
else:
    print(f"半径调整值: 0.0 nm (未调整)")

# ====================================================================
# 步骤1: 加载数据
# ====================================================================
print("\n步骤1: 加载数据...")

# 加载孔隙和喉道数据
df_pores = pd.read_excel(pores_file)
df_throats = pd.read_excel(throats_file)

pore_ids = df_pores['Pore ID'].values
pore_coords = df_pores[['X Coord', 'Y Coord', 'Z Coord']].values
pore_radii = df_pores['EqRadius'].values  # nm，孔的等效半径
# 应用半径调整值
if RADIUS_DELTA != 0.0:
    pore_radii = pore_radii + RADIUS_DELTA
    # 确保半径不为负
    pore_radii = np.maximum(pore_radii, 1e-6)  # 最小半径1e-6 nm
pore_id_to_idx = {pid: idx for idx, pid in enumerate(pore_ids)}

# 渗透方向轴（真实样本）：与 sub_analyze 中膜厚/法向校准一致，读取 thickness_summary 的 Main Axis Index（0=X,1=Y,2=Z）。
# 合成网络分析器固定 Z 轴；真实网络优先从 thickness_summary 读取法向轴。
PENETRATION_AXIS = 2
_thickness_summary_path = output_path / f"{SIEVE_SAMPLE_NAME}_thickness_summary.xlsx"
_pore_classification_path = output_path / f"{SIEVE_SAMPLE_NAME}_pore_classification.xlsx"
_pen_axis_from_summary = False
_pen_axis_source = "synthetic default"
if _thickness_summary_path.exists():
    try:
        _ts_df = pd.read_excel(_thickness_summary_path)
        if "Main Axis Index" in _ts_df.columns and len(_ts_df) > 0:
            _ax = int(pd.to_numeric(_ts_df["Main Axis Index"].iloc[0], errors="coerce"))
            if _ax in (0, 1, 2):
                PENETRATION_AXIS = _ax
                _pen_axis_from_summary = True
                _pen_axis_source = "thickness_summary"
    except Exception:
        pass
if not _pen_axis_from_summary and _pore_classification_path.exists():
    try:
        _pc_axis_df = pd.read_excel(_pore_classification_path)
        if "Penetration Axis Index" in _pc_axis_df.columns and len(_pc_axis_df) > 0:
            _ax = int(pd.to_numeric(_pc_axis_df["Penetration Axis Index"].iloc[0], errors="coerce"))
            if _ax in (0, 1, 2):
                PENETRATION_AXIS = _ax
                _pen_axis_source = "pore_classification"
    except Exception:
        pass
_AXIS_LABELS = ("X", "Y", "Z")
if _pen_axis_from_summary:
    print(
        f"  渗透方向轴（真实样本，thickness_summary / 法向·厚度校准）: "
        f"{_AXIS_LABELS[PENETRATION_AXIS]} (索引={PENETRATION_AXIS})"
    )
else:
    print(
        f"  渗透方向轴: {_AXIS_LABELS[PENETRATION_AXIS]} (索引={PENETRATION_AXIS}, "
        f"来源={_pen_axis_source})"
    )

# 沿渗透轴坐标跨度与膜厚（nm）：B0 喉长、入口-出口短喉等效长度、与是否有 pore_classification 无关。
_coords_pen = pore_coords[:, PENETRATION_AXIS]
u_min = float(_coords_pen.min())
u_max = float(_coords_pen.max())
span_nm = u_max - u_min
thickness_nm = span_nm
try:
    if _thickness_summary_path.exists():
        _ts_df2 = pd.read_excel(_thickness_summary_path)
        if "Average Thickness (nm)" in _ts_df2.columns and len(_ts_df2) > 0:
            _avg = float(pd.to_numeric(_ts_df2["Average Thickness (nm)"].iloc[0], errors="coerce"))
            if np.isfinite(_avg) and _avg > 0:
                thickness_nm = _avg
except Exception:
    pass
print(
    f"  渗透轴坐标跨度 span={span_nm:.3f} nm；用于电导/跨膜喉长的膜厚 thickness_nm={thickness_nm:.3f} nm"
)


def _permeation_plane_bbox_area_nm2(coords: np.ndarray, penetration_axis: int) -> float:
    """垂直于渗透轴的平面内，全体孔坐标的轴对齐包络矩形面积 (nm²)。"""
    axes = [i for i in (0, 1, 2) if i != penetration_axis]
    s0 = float(np.ptp(coords[:, axes[0]]))
    s1 = float(np.ptp(coords[:, axes[1]]))
    return s0 * s1


PERMEATION_PLANE_AREA_NM2 = _permeation_plane_bbox_area_nm2(pore_coords, PENETRATION_AXIS)
PERMEATION_PLANE_AREA_UM2 = PERMEATION_PLANE_AREA_NM2 / 1e6
print(
    f"  渗透面包络面积（垂直于 {_AXIS_LABELS[PENETRATION_AXIS]} 的平面）: "
    f"{PERMEATION_PLANE_AREA_NM2:.4f} nm² = {PERMEATION_PLANE_AREA_UM2:.6f} µm²"
)

throat_ids = df_throats['Throat ID'].values
throat_radii = df_throats['EqRadius'].values  # nm
# 应用半径调整值
if RADIUS_DELTA != 0.0:
    throat_radii = throat_radii + RADIUS_DELTA
    # 确保半径不为负
    throat_radii = np.maximum(throat_radii, 1e-6)  # 最小半径1e-6 nm
throat_pore1 = df_throats['Pore ID #1'].values
throat_pore2 = df_throats['Pore ID #2'].values

# 加载溶剂筛选结果
solvent_classification_df = pd.read_excel(solvent_classification_file)
solvent_components_df = pd.read_excel(solvent_components_file)

# 筛选出溶剂渗透网络中的喉
solvent_feasible_mask = solvent_classification_df['In Solvent Penetration'].values
solvent_feasible_throat_ids = set(throat_ids[solvent_feasible_mask])

print(f"  总孔隙数: {len(pore_ids)}")
print(f"  总喉道数: {len(throat_ids)}")
print(f"  溶剂渗透网络中的喉数: {len(solvent_feasible_throat_ids)}")

# 获取溶剂渗透网络中的孔隙ID
solvent_penetration_pore_ids = set()
for _, row in solvent_components_df.iterrows():
    pore_ids_str = str(row['Pore IDs']).strip()
    if pore_ids_str and pore_ids_str.lower() != 'nan':
        # 过滤掉空字符串
        pore_id_list = [int(pid.strip()) for pid in pore_ids_str.split(',') if pid.strip()]
        if pore_id_list:
            solvent_penetration_pore_ids.update(pore_id_list)

print(f"  溶剂渗透网络中的孔数: {len(solvent_penetration_pore_ids)}")

# 识别入口和出口节点（左侧和右侧外部孔）
# 从之前的分析结果中读取左右外部孔信息（样本名前缀）
pore_classification_file = output_path / f"{SIEVE_SAMPLE_NAME}_pore_classification.xlsx"
if pore_classification_file.exists():
    pore_classification_df = pd.read_excel(pore_classification_file)
    # 获取左右外部孔
    surface_left_mask = pore_classification_df['Is Surface X Left'].values
    surface_right_mask = pore_classification_df['Is Surface X Right'].values
    all_pore_ids_class = pore_classification_df['Pore ID'].values
    
    surface_left_pore_ids = set(all_pore_ids_class[surface_left_mask])
    surface_right_pore_ids = set(all_pore_ids_class[surface_right_mask])
    
    # 在溶剂渗透网络中的入口和出口节点
    entrance_pore_ids = [pid for pid in solvent_penetration_pore_ids 
                         if pid in surface_left_pore_ids]
    exit_pore_ids = [pid for pid in solvent_penetration_pore_ids 
                     if pid in surface_right_pore_ids]
    # 与 pore_classification / pressure_distribution 对照时的口径说明：
    # 分类表「左/右表面」包含全部几何表面孔；压强 Dirichlet 仅施加在「溶剂渗透网络 ∩ 表面」上。
    _n_tag_left = int(np.sum(surface_left_mask))
    _n_tag_right = int(np.sum(surface_right_mask))
    _left_not_solvent = len(surface_left_pore_ids - solvent_penetration_pore_ids)
    _right_not_solvent = len(surface_right_pore_ids - solvent_penetration_pore_ids)
    print(
        f"  [口径] pore_classification：Is Surface X Left/Right 分别={_n_tag_left}/{_n_tag_right} 个孔；"
        f"其中仅标表面、但不在溶剂渗透网络：左 {_left_not_solvent}、右 {_right_not_solvent} 个"
    )
    print(
        f"  [口径] 上述「仅表面」孔不在溶剂渗透网络中，*_pressure_distribution.xlsx 与压强 3D 图**不列出**；"
        f"边界孔（P_IN/P_OUT）仅：入口={len(entrance_pore_ids)}、出口={len(exit_pore_ids)}"
    )
else:
    # Fallback：沿渗透轴用两端阈值分带（与 analyze 中 main_axis 一致时应先跑 analyze 生成 classification）
    print(
        f"  警告：未找到pore classification文件，沿渗透轴 {_AXIS_LABELS[PENETRATION_AXIS]} 用坐标分带作为fallback"
    )
    u_coords = pore_coords[:, PENETRATION_AXIS]
    u_min_threshold = u_coords.min() + 5.0
    u_max_threshold = u_coords.max() - 5.0
    entrance_pore_ids = [pid for pid in solvent_penetration_pore_ids 
                         if pid in pore_id_to_idx and 
                         pore_coords[pore_id_to_idx[pid], PENETRATION_AXIS] < u_min_threshold]
    exit_pore_ids = [pid for pid in solvent_penetration_pore_ids 
                     if pid in pore_id_to_idx and 
                     pore_coords[pore_id_to_idx[pid], PENETRATION_AXIS] > u_max_threshold]

# 表面孔几何过滤已在 sub_analyze_* 射线判定后写入 classification；此处不再重复。
# 仅当无 classification、用分带 fallback 时，再按心到端面距离做与 analyze 一致的二次筛选。
if not pore_classification_file.exists():
    tol_nm = max(1.0, span_nm * 0.05)
    candidate_surface_pores = set(entrance_pore_ids) | set(exit_pore_ids)
    entrance_pore_ids_before = set(entrance_pore_ids)
    exit_pore_ids_before = set(exit_pore_ids)
    entrance_pore_ids = []
    exit_pore_ids = []
    dual_surface_pores = 0
    for pid in candidate_surface_pores:
        if pid not in pore_id_to_idx:
            continue
        idx = pore_id_to_idx[pid]
        u = float(pore_coords[idx, PENETRATION_AXIS])
        r = float(pore_radii[idx])
        d_left = u - u_min
        d_right = u_max - u
        left_ok = d_left <= (r + tol_nm)
        right_ok = d_right <= (r + tol_nm)
        if left_ok:
            entrance_pore_ids.append(pid)
        if right_ok:
            exit_pore_ids.append(pid)
        if left_ok and right_ok:
            dual_surface_pores += 1
    entrance_pore_ids = sorted(list(set(entrance_pore_ids)))
    exit_pore_ids = sorted(list(set(exit_pore_ids)))
    print("  [表面孔自检·fallback] 无 classification，沿渗透轴 "
          f"{_AXIS_LABELS[PENETRATION_AXIS]}："
          f" span={span_nm:.3f} nm, thickness_nm={thickness_nm:.3f} nm, tol={tol_nm:.3f} nm")
    print("  [表面孔自检·fallback] 候选/双贴面："
          f" candidates={len(candidate_surface_pores)} | dual={dual_surface_pores}")
    print("  [表面孔自检·fallback] 入口/出口："
          f" entrance={len(entrance_pore_ids)} (before={len(entrance_pore_ids_before)})"
          f" | exit={len(exit_pore_ids)} (before={len(exit_pore_ids_before)})")

print(f"  入口节点数（左侧外部孔）: {len(entrance_pore_ids)}")
print(f"  出口节点数（右侧外部孔）: {len(exit_pore_ids)}")

if len(entrance_pore_ids) == 0 or len(exit_pore_ids) == 0:
    print("  错误：未找到入口或出口节点！")
    exit(1)

# ====================================================================
# 阶段B：溶剂流动计算（压力-流量场）
# ====================================================================
print("\n阶段B：溶剂流动计算...")
print("  B1. 计算喉的流导（几何喉长 = 孔心距 d；与 gbm_full_model phase4 口径一致）...")

G_solvent = nx.Graph()
throat_conductances = {}
throat_lengths = {}

for t_id, p1, p2, r in zip(throat_ids, throat_pore1, throat_pore2, throat_radii):
    if t_id not in solvent_feasible_throat_ids:
        continue
    if p1 not in pore_id_to_idx or p2 not in pore_id_to_idx:
        continue
    p1_idx = pore_id_to_idx[p1]
    p2_idx = pore_id_to_idx[p2]
    p1_coord = pore_coords[p1_idx]
    p2_coord = pore_coords[p2_idx]
    center_distance = np.linalg.norm(p2_coord - p1_coord)
    length = center_distance
    if length <= 0:
        length = max(1e-3, center_distance * 0.1)
    r_m = r * 1e-9
    L_m = length * 1e-9
    if L_m > 0:
        conductance = np.pi * (r_m ** 4) / (8 * SOLVENT_VISCOSITY * L_m)
    else:
        conductance = 0.0
    G_solvent.add_edge(p1, p2, throat_id=t_id, radius=r, length=length, conductance=conductance)
    throat_conductances[t_id] = conductance
    throat_lengths[t_id] = length

print(f"    已计算 {len(throat_conductances)} 条喉的流导")
print(f"    入口压强 P_IN = {P_IN:.2f} Pa（{P_IN_EFFECTIVE:.2f} mmHg）")
print(f"    出口压强 P_OUT = {P_OUT:.2f} Pa（{P_OUT_EFFECTIVE:.2f} mmHg）")


def _complete_node_pressures_all_pores(partial: dict) -> dict:
    """仅为溶剂渗透网络 G_solvent 中的孔赋压强：Dirichlet 边界或 partial（Kirchhoff / 求解 fallback）。"""
    solvent_graph_nodes = set(G_solvent.nodes())
    ent_set = set(entrance_pore_ids)
    ex_set = set(exit_pore_ids)
    out = {}
    missing_in_partial: list = []
    for pid in pore_ids:
        if pid not in solvent_graph_nodes:
            continue
        if pid in ent_set and pid not in ex_set:
            out[pid] = P_IN
        elif pid in ex_set and pid not in ent_set:
            out[pid] = P_OUT
        elif pid in ent_set and pid in ex_set:
            out[pid] = (P_IN + P_OUT) / 2.0
        elif pid in partial:
            out[pid] = partial[pid]
        else:
            missing_in_partial.append(pid)
    if missing_in_partial:
        n = len(missing_in_partial)
        preview = missing_in_partial[:24]
        print(
            "  错误：溶剂渗透网络 G_solvent 中存在孔既非入口/出口 Dirichlet，"
            "也未在求解得到的 partial 压强字典中（不应发生，请检查图划分与 partial 键）。"
        )
        print(f"    缺失孔数={n}，示例 Pore ID（至多 24 个）: {preview}")
        sys.exit(1)
    return out


# B2. 建立并求解压力方程组（稀疏矩阵，与 phase4 一致）
print("  B2. 建立并求解压力方程组...")

all_nodes = set(G_solvent.nodes())
internal_nodes = sorted(list(all_nodes - set(entrance_pore_ids) - set(exit_pore_ids)))
n_internal = len(internal_nodes)
node_to_internal_idx = {node: idx for idx, node in enumerate(internal_nodes)}

print(f"    内部节点数: {n_internal}")
print(f"    入口节点数: {len(entrance_pore_ids)}")
print(f"    出口节点数: {len(exit_pore_ids)}")

if n_internal == 0:
    print("    警告：没有内部节点，所有节点都是边界节点")
    node_pressures = {}
    for node in entrance_pore_ids:
        if node in exit_pore_ids:
            node_pressures[node] = (P_IN + P_OUT) / 2.0
        else:
            node_pressures[node] = P_IN
    for node in exit_pore_ids:
        if node not in entrance_pore_ids:
            node_pressures[node] = P_OUT
    node_pressures = _complete_node_pressures_all_pores(node_pressures)
else:
    b = np.zeros(n_internal)
    A_entries = {}
    for node in internal_nodes:
        i = node_to_internal_idx[node]
        for neighbor in G_solvent.neighbors(node):
            edge_data = G_solvent[node][neighbor]
            G_ij = edge_data['conductance']
            if neighbor in internal_nodes:
                j = node_to_internal_idx[neighbor]
                A_entries[(i, j)] = A_entries.get((i, j), 0.0) - G_ij
                A_entries[(i, i)] = A_entries.get((i, i), 0.0) + G_ij
            elif neighbor in entrance_pore_ids:
                b[i] += G_ij * P_IN
                A_entries[(i, i)] = A_entries.get((i, i), 0.0) + G_ij
            elif neighbor in exit_pore_ids:
                b[i] += G_ij * P_OUT
                A_entries[(i, i)] = A_entries.get((i, i), 0.0) + G_ij
    if A_entries:
        rows, cols, data = zip(*[(i, j, val) for (i, j), val in A_entries.items() if val != 0.0])
        A_sparse = csr_matrix((data, (rows, cols)), shape=(n_internal, n_internal))
    else:
        A_sparse = csr_matrix((n_internal, n_internal))

    print("    求解压力方程组（稀疏矩阵）...")
    try:
        P_internal = spsolve(A_sparse, b)
        if not np.all(np.isfinite(P_internal)):
            raise ValueError("压力解含 NaN/Inf（矩阵奇异或病态）")
        _check_internal_pressure_in_boundary_band(P_internal, P_IN, P_OUT, PRESSURE_PHYSICS_TOL_PA)
        if PRESSURE_PHYSICS_TOL_PA >= 0:
            print(
                f"    内部节点压强物理检验通过（须在 [min,max]=[{min(P_IN, P_OUT):.2f}, {max(P_IN, P_OUT):.2f}] Pa "
                f"± {PRESSURE_PHYSICS_TOL_PA:g} Pa）"
            )
        node_pressures = {}
        for node in entrance_pore_ids:
            if node in exit_pore_ids:
                node_pressures[node] = (P_IN + P_OUT) / 2.0
            else:
                node_pressures[node] = P_IN
        for node in exit_pore_ids:
            if node not in entrance_pore_ids:
                node_pressures[node] = P_OUT
        for idx, node in enumerate(internal_nodes):
            node_pressures[node] = float(P_internal[idx])
        node_pressures = _complete_node_pressures_all_pores(node_pressures)
        print(f"    压力求解成功")
        print(f"    内部节点压力范围: [{P_internal.min():.2f}, {P_internal.max():.2f}] Pa")
    except PressurePhysicsViolationError as e:
        print(f"    {e}")
        print(
            "    本样本压强场视为无解（Kirchhoff 解超出物理边界带），已中止；未使用几何插值 fallback、未对压强做截断。"
        )
        sys.exit(1)
    except Exception as e:
        print(f"    压力求解失败: {e}")
        print("    使用线性插值作为fallback...")
        node_pressures = {}
        for node in entrance_pore_ids:
            if node in exit_pore_ids:
                node_pressures[node] = (P_IN + P_OUT) / 2.0
            else:
                node_pressures[node] = P_IN
        for node in exit_pore_ids:
            if node not in entrance_pore_ids:
                node_pressures[node] = P_OUT
        for node in internal_nodes:
            u = pore_coords[pore_id_to_idx[node], PENETRATION_AXIS]
            u_e0 = pore_coords[pore_id_to_idx[entrance_pore_ids[0]], PENETRATION_AXIS]
            u_e1 = pore_coords[pore_id_to_idx[exit_pore_ids[0]], PENETRATION_AXIS]
            if u_e1 > u_e0:
                t = (u - u_e0) / (u_e1 - u_e0)
                node_pressures[node] = P_IN * (1 - t) + P_OUT * t
            else:
                node_pressures[node] = (P_IN + P_OUT) / 2
        node_pressures = _complete_node_pressures_all_pores(node_pressures)

_finalize_pore_pressure_physics_check(node_pressures)

_arr_p = np.array(list(node_pressures.values()), dtype=float)
print(
    f"    各孔压强范围: [{np.nanmin(_arr_p):.4f}, {np.nanmax(_arr_p):.4f}] Pa "
    f"（仅溶剂渗透网络：入口/出口为 P_IN/P_OUT；内部为 Kirchhoff；图外孔不赋压强）"
)

# 出入口压强自检：分类名单在求解过程中未改；此处核对各孔 ID 上是否等于 Dirichlet 给定值
_boundary_p_tol_pa = 1e-3
_ent_set = set(entrance_pore_ids)
_ex_set = set(exit_pore_ids)
_boundary_ids = _ent_set | _ex_set
_n_missing_b = 0
_n_bad_b = 0
_max_err_b = 0.0
_bad_examples = []
for _pid in _boundary_ids:
    if _pid not in node_pressures:
        _n_missing_b += 1
        if len(_bad_examples) < 8:
            _bad_examples.append((_pid, "missing", None, None))
        continue
    _p = float(node_pressures[_pid])
    if _pid in _ent_set and _pid in _ex_set:
        _exp = (float(P_IN) + float(P_OUT)) / 2.0
        _tag = "双侧"
    elif _pid in _ent_set:
        _exp = float(P_IN)
        _tag = "入口"
    else:
        _exp = float(P_OUT)
        _tag = "出口"
    _err = abs(_p - _exp)
    _max_err_b = max(_max_err_b, _err)
    if _err > _boundary_p_tol_pa:
        _n_bad_b += 1
        if len(_bad_examples) < 8:
            _bad_examples.append((_pid, _tag, _p, _exp))
print(
    f"  [出入口压强自检] 入口孔数={len(entrance_pore_ids)} 出口孔数={len(exit_pore_ids)} "
    f"（唯一 ID 数={len(_boundary_ids)}）| 名单未在求解中修改，仅核对 node_pressures"
)
print(
    f"    期望：仅入口→P_IN={P_IN:.4f} Pa；仅出口→P_OUT={P_OUT:.4f} Pa；"
    f"同时标入出口→(P_IN+P_OUT)/2={(P_IN + P_OUT) / 2:.4f} Pa"
)
if _n_missing_b == 0 and _n_bad_b == 0:
    print(
        f"    通过：全部出入口孔压强与给定值一致（容差≤{_boundary_p_tol_pa:g} Pa，max |ΔP|={_max_err_b:.2e} Pa）"
    )
else:
    print(
        f"    警告：缺失压强键={_n_missing_b}，超出容差孔数={_n_bad_b}，max |ΔP|={_max_err_b:.6f} Pa"
    )
    for _t in _bad_examples:
        print(
            f"      Pore ID {_t[0]}: {_t[1]}  P={_t[2]}  期望={_t[3]}"
        )

print(
    f"  [溶质对流方向] 喉两端 |ΔP| ≤ {PRESSURE_FLOW_TOL_PA:g} Pa 时视为无压力驱动对流，"
    "仅组装/累计扩散（若有）；可调 --pressure-flow-tol-pa。"
        )

# B3. 计算喉流量与溶剂速度
print("  B3. 计算喉流量与溶剂速度...")

throat_flows = {}  # 沿 throat 几何 p1→p2 的有符号体积流 (m³/s)，Q = G·(P1−P2)
throat_velocities = {}  # 溶剂速度标量用 |Q|
throat_endpoints: dict[int, tuple[int, int]] = {}

for t_id, p1, p2, r in zip(throat_ids, throat_pore1, throat_pore2, throat_radii):
    if t_id in solvent_feasible_throat_ids:
        if p1 in node_pressures and p2 in node_pressures:
            G_ij = throat_conductances.get(t_id, 0.0)
            P1 = node_pressures[p1]
            P2 = node_pressures[p2]
            Q = G_ij * (float(P1) - float(P2))  # m^3/s，正方向 p1 → p2
            
            r_m = r * 1e-9  # m
            q_abs = abs(Q)
            if r_m > 0:
                v = q_abs / (np.pi * r_m ** 2)  # m/s
            else:
                v = 0.0
            
            tid_i = int(t_id)
            throat_endpoints[tid_i] = (int(p1), int(p2))
            throat_flows[t_id] = Q
            throat_velocities[t_id] = v

print(f"    已计算 {len(throat_flows)} 条喉的流量和速度（有符号：p1→p2 为正）")
if throat_flows:
    total_flow_mag = sum(abs(float(q)) for q in throat_flows.values())
    print(f"    总|Q|（各喉绝对值之和）: {total_flow_mag:.2e} m^3/s")


def _signed_Q_node_to_neighbor(t_id: int, node: int, neighbor: int) -> float:
    """喉 t_id 上从 node 指向 neighbor 的体积流 (m³/s)；与 throat_flows 的 p1→p2 符号一致。"""
    p1, p2 = throat_endpoints[int(t_id)]
    q12 = float(throat_flows[int(t_id)])
    n1, n2 = int(node), int(neighbor)
    if n1 == int(p1) and n2 == int(p2):
        return q12
    if n1 == int(p2) and n2 == int(p1):
        return -q12
    raise RuntimeError(
        f"喉 {t_id} 端点 {(p1, p2)} 与边 ({node}, {neighbor}) 不一致（溶质组装）"
    )

print("\n阶段B完成！")

# ====================================================================
# 压强表与压强三维图（先于浓度计算，避免浓度求解失败时无压强输出）
# ====================================================================
print("\n保存压强分布并生成压强三维图（先于浓度计算）...")
pressure_df = pd.DataFrame({
    "Pore ID": list(node_pressures.keys()),
    "Pressure (Pa)": list(node_pressures.values()),
})
pressure_excel = output_path / get_output_filename(f"{SIEVE_SAMPLE_NAME}_pressure_distribution")
pressure_df.to_excel(pressure_excel, index=False)
print(f"  压力分布已保存: {pressure_excel}")

_has_pressure = np.array([pid in node_pressures for pid in pore_ids], dtype=bool)
_idx_vis = np.flatnonzero(_has_pressure)
pore_coords_vis = pore_coords[_idx_vis]
pore_radii_vis = pore_radii[_idx_vis]
pore_ids_vis = np.array(pore_ids, dtype=object)[_idx_vis]
pore_pressures = np.array([node_pressures[pore_ids[i]] for i in _idx_vis], dtype=float)
print(
    f"  压强三维图仅绘制溶剂渗透网络中有压强的 {len(_idx_vis)} 个孔"
    f"（几何总孔数 {len(pore_ids)}，未入网者不显示）"
)

def _viz_pid_int(p) -> int:
    return int(float(p))


# 与 Kirchhoff Dirichlet 边界一致：仅用 entrance_pore_ids / exit_pore_ids 着色边线，
# 不用 pore_classification 全表「几何左/右表面」（后者可能含未入溶剂渗透网络或非 P_IN/P_OUT 的孔）。
_ent_viz = {_viz_pid_int(x) for x in entrance_pore_ids}
_ex_viz = {_viz_pid_int(x) for x in exit_pore_ids}

_n_full = len(pore_ids)
_il_full = np.zeros(_n_full, dtype=bool)
_ir_full = np.zeros(_n_full, dtype=bool)
for _i in range(_n_full):
    _p = _viz_pid_int(pore_ids[_i])
    _il_full[_i] = _p in _ent_viz
    _ir_full[_i] = _p in _ex_viz
_il_viz = _il_full[_has_pressure]
_ir_viz = _ir_full[_has_pressure]

fig_pressure = go.Figure()
valid_pressure_mask = np.ones(len(pore_pressures), dtype=bool)
_mask_internal = ~(_il_viz | _ir_viz)
_pressure_groups = [
    ("内部", _mask_internal, "rgba(55,55,55,0.4)", 0.65),
    ("入口边界 P_IN（渗透∩左表面）", _il_viz & ~_ir_viz, "#c92a2a", 2.1),
    ("出口边界 P_OUT（渗透∩右表面）", _ir_viz & ~_il_viz, "#1864ab", 2.1),
    ("入口+出口名单重合", _il_viz & _ir_viz, "#7b2cbf", 2.1),
]

_show_cbar = True
for _gname, _gmask, _lcol, _lw in _pressure_groups:
    _m = valid_pressure_mask & _gmask
    if not np.any(_m):
        continue
    _txt = []
    for _pid, _p, _r, _il, _ir in zip(
        pore_ids_vis[_m],
        pore_pressures[_m],
        pore_radii_vis[_m],
        _il_viz[_m],
        _ir_viz[_m],
    ):
        if _il and _ir:
            _st = "入口+出口名单重合"
        elif _il:
            _st = "入口边界 P_IN"
        elif _ir:
            _st = "出口边界 P_OUT"
        else:
            _st = "内部"
        _txt.append(
            f"Pore ID: {_pid}<br>Pressure: {_p:.2f} Pa<br>Radius: {_r:.2f} nm<br>{_st}"
        )
    _marker_kw = dict(
        size=pore_radii_vis[_m] * 2,
        color=pore_pressures[_m],
        colorscale="Viridis",
        line=dict(width=_lw, color=_lcol),
        opacity=0.82,
        showscale=_show_cbar,
    )
    if _show_cbar:
        _marker_kw["colorbar"] = dict(title="Pressure (Pa)", x=1.1)
    fig_pressure.add_trace(
        go.Scatter3d(
            x=pore_coords_vis[_m, 0],
            y=pore_coords_vis[_m, 1],
            z=pore_coords_vis[_m, 2],
            mode="markers",
            marker=_marker_kw,
            name=_gname,
            text=_txt,
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )
    _show_cbar = False

fig_pressure.update_layout(
    title=dict(
        text=(
            f"{SIEVE_SAMPLE_NAME} - Pressure Distribution<br>"
            "<sub>Color = pressure (Pa); edge = Dirichlet 边界（红 P_IN、蓝 P_OUT、紫 双名单、灰 内部），与计算一致"
            "</sub>"
        ),
        x=0.5,
    ),
    scene=dict(
        xaxis_title="X (nm)",
        yaxis_title="Y (nm)",
        zaxis_title="Z (nm)",
        aspectmode="data",
    ),
    width=1200,
    height=800,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255,255,255,0.75)",
        font=dict(size=10),
    ),
)

if NO_HTML:
    print("  压强分布三维 HTML 已跳过（--no-html）")
else:
    pressure_html = output_path / get_output_filename(f"{SIEVE_SAMPLE_NAME}_pressure_distribution_3d", ".html")
    fig_pressure.write_html(str(pressure_html))
    print(f"  压强分布三维图已保存: {pressure_html}")

# ====================================================================
# 阶段C：溶质可进入性分析
# ====================================================================
print("\n阶段C：溶质可进入性分析...")

# C1. 入口边界约束说明（统一浓度口径）
print("  C1. 入口边界将统一施加 Dirichlet：所有入口 C_in = C0（不做孔半径修正）...")

print(f"    入口孔数: {len(entrance_pore_ids)}")
# 所有入口孔都可以进入（筛选已包含在各自的浓度计算中）
open_entrance_pores = list(entrance_pore_ids)

# C3. 确定溶质可通行网络
print("  C3. 确定溶质可通行网络...")
# 从开放入口孔开始，广度优先搜索所有连通的喉
# 注意：溶质可通行网络需要同时考虑孔和喉的半径
# 只有喉半径 > r_s 且两端孔半径均 > r_s 的边才能通过溶质

# 构建溶质可通行网络图（同时检查孔和喉的半径）
G_solute = nx.Graph()
solute_feasible_throat_ids = set()

for t_id, r_throat, p1_id, p2_id in zip(throat_ids, throat_radii, throat_pore1, throat_pore2):
    # 只考虑在溶剂渗透网络中的喉
    if t_id not in solvent_feasible_throat_ids:
        continue
    if int(t_id) in SOLUTE_EXCLUDE_THROAT_IDS:
        continue
    
    # 喉几何可进入：R_T 须严格大于 r_s
    if int(p1_id) in SOLUTE_EXCLUDE_PORE_IDS or int(p2_id) in SOLUTE_EXCLUDE_PORE_IDS:
        continue
    if r_throat <= SOLUTE_PASSAGE_GEOMETRY_MIN_NM:
        continue
    
    # 检查两端孔的半径
    if p1_id not in pore_id_to_idx or p2_id not in pore_id_to_idx:
        continue
    
    r_pore1 = pore_radii[pore_id_to_idx[p1_id]]
    r_pore2 = pore_radii[pore_id_to_idx[p2_id]]
    
    # 两端孔均严格大于水合半径（几何可进入）
    if _pore_geometrically_passable_for_solute(r_pore1) and _pore_geometrically_passable_for_solute(r_pore2):
        G_solute.add_edge(p1_id, p2_id, throat_id=t_id)
        solute_feasible_throat_ids.add(t_id)

# 从开放入口孔开始，找到所有连通的喉（G_solute 中孔/喉已按 r>r_hydrated 过滤）
solute_accessible_throat_ids = set()
solute_accessible_entrance_pores = []

for pore_id in open_entrance_pores:
    # 检查入口孔是否在溶质可通行网络中
    # 如果入口孔在 G_solute 中，说明它的半径 >= r_protein 且有可通行的连接
    if pore_id in G_solute:
        solute_accessible_entrance_pores.append(pore_id)
        # 从该入口孔开始BFS，找到所有连通的喉
        for component in nx.connected_components(G_solute):
            if pore_id in component:
                # 收集这个连通分量中的所有喉
                for n1 in component:
                    for n2 in G_solute.neighbors(n1):
                        if n2 > n1:
                            edge_data = G_solute[n1][n2]
                            solute_accessible_throat_ids.add(edge_data['throat_id'])

print(f"    溶质可进入的入口孔数: {len(solute_accessible_entrance_pores)} (总入口孔数: {len(open_entrance_pores)})")
print(f"    溶质可通行网络中的喉数（初步）: {len(solute_accessible_throat_ids)}")
print(
    f"    溶质几何（通路筛选）：喉 R_T>{SOLUTE_PASSAGE_GEOMETRY_MIN_NM:.4g} nm，"
    f"两端孔 R_node>{SOLUTE_PASSAGE_GEOMETRY_MIN_NM:.4g} nm（=r_s）。"
)

print("\n阶段C完成！")

# ====================================================================
# 阶段D：喉内溶质传输参数
# ====================================================================
print("\n阶段D：喉内溶质传输参数...")

# D1. 计算尺寸比
print("  D1. 计算尺寸比...")
throat_lambdas = {}  # lambda_i = r_s / r_i

for t_id in solute_accessible_throat_ids:
    r_i = throat_radii[throat_ids == t_id][0]
    if r_i > 0:
        lambda_i = SOLUTE_HYDRATED_RADIUS / r_i
        throat_lambdas[t_id] = lambda_i

print(f"    已计算 {len(throat_lambdas)} 条喉的尺寸比")

# D2. Dechadilok-Deen (2006) cylindrical-pore hindrance factors.
print("  D2. 计算 Dechadilok-Deen (2006) 圆柱孔 K_D 与 K_C...")
throat_Kd = {}
throat_Kc = {}

for t_id, lambda_i in throat_lambdas.items():
    Kd, Kc = dd2006_hindrance_factors(lambda_i)
    throat_Kd[t_id] = Kd
    throat_Kc[t_id] = max(0.0, Kc * ELECTROSTATIC_ENHANCEMENT_FACTOR)

print(f"    已计算 {len(throat_Kd)} 条喉的 DD2006 阻碍因子")

# D2.5. 计算有效扩散系数（如果启用扩散）
throat_D_eff = {}  # D_eff,i = D_0 K_D,i
if INCLUDE_DIFFUSION:
    print("  D2.5. 计算有效扩散系数...")
    for t_id in solute_accessible_throat_ids:
        Kd = float(throat_Kd.get(t_id, 0.0))
        throat_D_eff[t_id] = SOLUTE_FREE_DIFFUSION * Kd
    
    print(f"    已计算 {len(throat_D_eff)} 条喉的有效扩散系数")
    if throat_D_eff:
        deff_values = [d for d in throat_D_eff.values() if d > 0]
        if deff_values:
            print(f"    D_eff 范围: [{min(deff_values):.6e}, {max(deff_values):.6e}] m^2/s")
            print(f"    D_eff 平均值: {np.mean(deff_values):.6e} m^2/s")
else:
    print("  D2.5. 跳过扩散系数计算（INCLUDE_DIFFUSION = False）")
    # 初始化为空字典，所有值设为0
    for t_id in solute_accessible_throat_ids:
        throat_D_eff[t_id] = 0.0

# D3. Geometric accessibility; K_C was calculated independently above.
print("  D3. 计算白蛋白质心可及面积比例 η=(1−λ)²...")
throat_partition_eff = {}  # η = (1−λ)²，仅统计/诊断，不进入 K_c 分子

# 确保所有溶质可通行喉都有长度信息（如果缺失则计算；与 phase4 一致：几何喉长 = 孔心距）
for t_id in solute_accessible_throat_ids:
    if t_id not in throat_lengths:
        mask = throat_ids == t_id
        if np.any(mask):
            p1 = throat_pore1[mask][0]
            p2 = throat_pore2[mask][0]
            if p1 in pore_id_to_idx and p2 in pore_id_to_idx:
                p1_idx = pore_id_to_idx[p1]
                p2_idx = pore_id_to_idx[p2]
                p1_coord = pore_coords[p1_idx]
                p2_coord = pore_coords[p2_idx]
                center_distance = np.linalg.norm(p2_coord - p1_coord)
                length = center_distance
                if length <= 0:
                    length = max(1e-3, center_distance * 0.1)
                throat_lengths[t_id] = length

for t_id in solute_accessible_throat_ids:
    if t_id not in throat_lambdas or t_id not in throat_Kd:
        throat_partition_eff[t_id] = 0.0
        continue
    
    lambda_i = throat_lambdas[t_id]
    
    if lambda_i >= 1.0:
        # lambda >= 1 时，溶质无法通过
        throat_Kc[t_id] = 0.0
        throat_partition_eff[t_id] = 0.0
        continue
    
    # 几何空间分配 η = (1−λ)²（steric；与等效浓度孔口 Φ 无关）
    partition_eff = (1.0 - lambda_i) ** 2
    
    throat_partition_eff[t_id] = partition_eff

print(f"    已计算 {len(throat_Kc)} 条喉的对流阻碍因子")
if throat_Kc:
    kc_values = list(throat_Kc.values())
    eta_values = list(throat_partition_eff.values())
    print(f"    Kc 范围: [{min(kc_values):.6f}, {max(kc_values):.6f}]")
    print(f"    Kc 平均值: {np.mean(kc_values):.6f}")
    if eta_values:
        print(f"    η=(1−λ)² 范围: [{min(eta_values):.6f}, {max(eta_values):.6f}]")
        print(f"    η=(1−λ)² 平均值: {np.mean(eta_values):.6f}")

# D3.1. 溶质可通行喉：lambda、Kd、Kc 分布（与 D1–D3 同一批喉）
throat_lambda_dist_n = 0
throat_lambda_min = 0.0
throat_lambda_max = 0.0
throat_lambda_median = 0.0
throat_lambda_mean = 0.0
throat_lambda_bin_lt_02 = 0
throat_lambda_bin_02_04 = 0
throat_lambda_bin_04_06 = 0
throat_lambda_bin_06_08 = 0
throat_lambda_bin_08_1 = 0
throat_lambda_bin_ge_1 = 0

throat_kd_dist_n = 0
throat_kd_min = 0.0
throat_kd_max = 0.0
throat_kd_median = 0.0
throat_kd_mean = 0.0

throat_kc_dist_n = 0
throat_kc_min = 0.0
throat_kc_max = 0.0
throat_kc_median = 0.0
throat_kc_mean = 0.0
throat_kc_bin_eq_0 = 0
throat_kc_bin_0_05 = 0
throat_kc_bin_05_1 = 0
throat_kc_bin_1_2 = 0
throat_kc_bin_gt_2 = 0

_sorted_solute_t_ids = sorted(solute_accessible_throat_ids)
_lam_list: list[float] = []
_kd_list: list[float] = []
_kc_list: list[float] = []
_partition_eff_list: list[float] = []
_tid_list: list = []
for _tid in _sorted_solute_t_ids:
    if _tid not in throat_lambdas or _tid not in throat_Kd or _tid not in throat_Kc:
        continue
    _lam_list.append(float(throat_lambdas[_tid]))
    _kd_list.append(float(throat_Kd[_tid]))
    _kc_list.append(float(throat_Kc[_tid]))
    _partition_eff_list.append(float(throat_partition_eff.get(_tid, 0.0)))
    _tid_list.append(_tid)

if _lam_list:
    print("  D3.1. 溶质可通行喉 lambda、Kd、Kc 分布（lambda=r_s/R_T）...")
    _la = np.asarray(_lam_list, dtype=np.float64)
    _kd = np.asarray(_kd_list, dtype=np.float64)
    _kc = np.asarray(_kc_list, dtype=np.float64)
    throat_lambda_dist_n = int(_la.size)
    throat_lambda_min = float(_la.min())
    throat_lambda_max = float(_la.max())
    throat_lambda_median = float(np.median(_la))
    throat_lambda_mean = float(_la.mean())
    print(f"    lambda 样本数: {throat_lambda_dist_n}")
    print(f"    lambda 范围: [{throat_lambda_min:.6f}, {throat_lambda_max:.6f}]")
    print(f"    lambda 中位数: {throat_lambda_median:.6f}  平均值: {throat_lambda_mean:.6f}")
    _lb = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf]
    throat_lambda_bin_lt_02 = int(np.sum(_la < _lb[1]))
    throat_lambda_bin_02_04 = int(np.sum((_la >= _lb[1]) & (_la < _lb[2])))
    throat_lambda_bin_04_06 = int(np.sum((_la >= _lb[2]) & (_la < _lb[3])))
    throat_lambda_bin_06_08 = int(np.sum((_la >= _lb[3]) & (_la < _lb[4])))
    throat_lambda_bin_08_1 = int(np.sum((_la >= _lb[4]) & (_la < _lb[5])))
    throat_lambda_bin_ge_1 = int(np.sum(_la >= _lb[5]))
    _ltot = max(throat_lambda_dist_n, 1)
    print("    lambda 分段（条喉，占比）：")
    print(
        f"      lambda < 0.2: {throat_lambda_bin_lt_02} ({100.0 * throat_lambda_bin_lt_02 / _ltot:.1f}%)"
    )
    print(
        f"      0.2 <= lambda < 0.4: {throat_lambda_bin_02_04} ({100.0 * throat_lambda_bin_02_04 / _ltot:.1f}%)"
    )
    print(
        f"      0.4 <= lambda < 0.6: {throat_lambda_bin_04_06} ({100.0 * throat_lambda_bin_04_06 / _ltot:.1f}%)"
    )
    print(
        f"      0.6 <= lambda < 0.8: {throat_lambda_bin_06_08} ({100.0 * throat_lambda_bin_06_08 / _ltot:.1f}%)"
    )
    print(
        f"      0.8 <= lambda < 1: {throat_lambda_bin_08_1} ({100.0 * throat_lambda_bin_08_1 / _ltot:.1f}%)"
    )
    print(
        f"      lambda >= 1: {throat_lambda_bin_ge_1} ({100.0 * throat_lambda_bin_ge_1 / _ltot:.1f}%)"
    )

    throat_kd_dist_n = int(_kd.size)
    throat_kd_min = float(_kd.min())
    throat_kd_max = float(_kd.max())
    throat_kd_median = float(np.median(_kd))
    throat_kd_mean = float(_kd.mean())
    print(f"    Kd 样本数: {throat_kd_dist_n}")
    print(f"    Kd 范围: [{throat_kd_min:.6f}, {throat_kd_max:.6f}]")
    print(f"    Kd 中位数: {throat_kd_median:.6f}  平均值: {throat_kd_mean:.6f}")

    throat_kc_dist_n = int(_kc.size)
    throat_kc_min = float(_kc.min())
    throat_kc_max = float(_kc.max())
    throat_kc_median = float(np.median(_kc))
    throat_kc_mean = float(_kc.mean())
    print(f"    Kc 样本数: {throat_kc_dist_n}")
    print(f"    Kc 范围: [{throat_kc_min:.6f}, {throat_kc_max:.6f}]")
    print(f"    Kc 中位数: {throat_kc_median:.6f}  平均值: {throat_kc_mean:.6f}")
    throat_kc_bin_eq_0 = int(np.sum(_kc <= 0.0))
    throat_kc_bin_0_05 = int(np.sum((_kc > 0.0) & (_kc <= 0.5)))
    throat_kc_bin_05_1 = int(np.sum((_kc > 0.5) & (_kc <= 1.0)))
    throat_kc_bin_1_2 = int(np.sum((_kc > 1.0) & (_kc <= 2.0)))
    throat_kc_bin_gt_2 = int(np.sum(_kc > 2.0))
    print("    Kc 分段（条喉，占比）：")
    print(f"      Kc = 0: {throat_kc_bin_eq_0} ({100.0 * throat_kc_bin_eq_0 / _ltot:.1f}%)")
    print(
        f"      0 < Kc <= 0.5: {throat_kc_bin_0_05} ({100.0 * throat_kc_bin_0_05 / _ltot:.1f}%)"
    )
    print(
        f"      0.5 < Kc <= 1: {throat_kc_bin_05_1} ({100.0 * throat_kc_bin_05_1 / _ltot:.1f}%)"
    )
    print(
        f"      1 < Kc <= 2: {throat_kc_bin_1_2} ({100.0 * throat_kc_bin_1_2 / _ltot:.1f}%)"
    )
    print(f"      Kc > 2: {throat_kc_bin_gt_2} ({100.0 * throat_kc_bin_gt_2 / _ltot:.1f}%)")
else:
    _la = np.array([], dtype=np.float64)
    _kd = np.array([], dtype=np.float64)
    _kc = np.array([], dtype=np.float64)
    _tid_list = []
    _lam_list = []
    _kd_list = []
    _kc_list = []
    _partition_eff_list = []

# D2.6. 统计每条喉的 Peclet 数（与 new_J 求解一致：Pe = U/diff_coeff）
throat_Pe = {}
pe_count = 0
pe_min = 0.0
pe_max = 0.0
pe_median = 0.0
pe_mean = 0.0
pe_lt_0_1 = 0
pe_0_1_to_1 = 0
pe_1_to_10 = 0
pe_ge_10 = 0
if INCLUDE_DIFFUSION:
    print(
        "  D2.6. 统计喉道 Peclet 数（new_J: Pe = U/diff_coeff, "
        "U=Kc·Q·A_solute/A_water；diff_coeff = D_eff·A_solute/L）..."
    )
    Pe_values: list[float] = []
    for t_id in solute_accessible_throat_ids:
        Kc = throat_Kc.get(t_id, 0.0)
        if Kc <= 0.0 or not np.isfinite(Kc):
            continue
        Q = throat_flows.get(t_id, 0.0)
        q_abs = abs(float(Q))
        if not np.isfinite(Q) or q_abs == 0.0:
            continue
        D_eff = throat_D_eff.get(t_id, 0.0)
        L_i = throat_lengths.get(t_id, 0.0)
        mask = throat_ids == t_id
        if not np.any(mask):
            continue
        r_i = throat_radii[mask][0]
        diff_coeff = _throat_diff_coeff_m3s(r_i, L_i, D_eff)
        if diff_coeff <= 0.0:
            continue
        U_adv = _throat_new_j_adv_coeff_m3s(Kc, q_abs, r_i)
        Pe = U_adv / diff_coeff
        throat_Pe[t_id] = float(Pe)
        Pe_values.append(float(Pe))

    if Pe_values:
        Pe_arr = np.asarray(Pe_values, dtype=np.float64)
        pe_count = int(len(Pe_arr))
        pe_min = float(Pe_arr.min())
        pe_max = float(Pe_arr.max())
        pe_median = float(np.median(Pe_arr))
        pe_mean = float(Pe_arr.mean())
        print(f"    有效 Pe 样本数: {len(Pe_arr)}")
        print(f"    Pe 范围: [{Pe_arr.min():.3e}, {Pe_arr.max():.3e}]")
        print(f"    Pe 中位数: {np.median(Pe_arr):.3e}")
        print(f"    Pe 平均值: {Pe_arr.mean():.3e}")
        bins = [0.0, 0.1, 1.0, 10.0, np.inf]
        labels = ["Pe < 0.1", "0.1 <= Pe < 1", "1 <= Pe < 10", "Pe >= 10"]
        counts = [0, 0, 0, 0]
        for Pe in Pe_arr:
            if Pe < bins[1]:
                counts[0] += 1
            elif Pe < bins[2]:
                counts[1] += 1
            elif Pe < bins[3]:
                counts[2] += 1
            else:
                counts[3] += 1
        pe_lt_0_1, pe_0_1_to_1, pe_1_to_10, pe_ge_10 = [int(x) for x in counts]
        total = len(Pe_arr)
        print("    Pe 分段统计：")
        for label, c in zip(labels, counts):
            pct = 100.0 * c / total
            print(f"      {label}: {c} 条喉 ({pct:.1f}%)")
    else:
        print("    警告：未能计算任何有效的 Pe（可能 Kc、Q、D_eff、A_solute 或 L 无效）")
else:
    print("  D2.6. 跳过 Peclet 统计（INCLUDE_DIFFUSION = False）")

# D2.7. 溶质可通行喉：Q 与 diff_coeff（与 new_J 组装中 Pe 所用口径一致）
throat_Q_stat = {}
throat_diff_coeff_stat = {}
for _t_dc in sorted(solute_accessible_throat_ids):
    _q = float(throat_flows.get(_t_dc, float("nan")))
    if np.isfinite(_q) and abs(_q) > 0.0:
        throat_Q_stat[_t_dc] = abs(_q)
    _mask_dc = throat_ids == _t_dc
    if not np.any(_mask_dc):
        continue
    _r_dc = float(throat_radii[_mask_dc][0])
    _L_dc = float(throat_lengths.get(_t_dc, 0.0))
    _D_dc = float(throat_D_eff.get(_t_dc, 0.0))
    _dc = float(_throat_diff_coeff_m3s(_r_dc, _L_dc, _D_dc))
    if np.isfinite(_dc):
        throat_diff_coeff_stat[_t_dc] = _dc

print("  D2.7. 喉 Q 与 diff_coeff（溶质可通行；diff_coeff = D_eff·A_solute/L，m³/s）...")
if throat_Q_stat:
    _Q_arr_dc = np.asarray(list(throat_Q_stat.values()), dtype=np.float64)
    print(f"    有流量记录的喉数: {len(throat_Q_stat)}")
    print(
        f"    Q 范围: [{_Q_arr_dc.min():.3e}, {_Q_arr_dc.max():.3e}] m³/s；"
        f"中位 {float(np.median(_Q_arr_dc)):.3e}；均值 {_Q_arr_dc.mean():.3e}"
    )
else:
    print("    警告：无有效 Q 样本")
_dc_pos = np.asarray(
    [v for v in throat_diff_coeff_stat.values() if v > 0.0 and np.isfinite(v)],
    dtype=np.float64,
)
if _dc_pos.size > 0:
    print(
        f"    diff_coeff>0 的喉数: {_dc_pos.size}；"
        f"范围 [{_dc_pos.min():.3e}, {_dc_pos.max():.3e}] m³/s；"
        f"中位 {float(np.median(_dc_pos)):.3e}；均值 {_dc_pos.mean():.3e}"
    )
else:
    print(
        "    diff_coeff>0 的喉数: 0（常见于 INCLUDE_DIFFUSION=False 或 D_eff/A/L 无效）"
    )

# D2.8 弱连接剪枝：|Q| 与 D_eff 同时落在全网络分布最弱 p%%（默认关闭）时移除该喉
print(
    "  D2.8. 弱连接剪枝（|Q| 与 D_eff 同时 ≤ 各自 p 分位数；默认 p=0 关闭）..."
)
_n_weak_before = len(solute_accessible_throat_ids)
_weak_removed_ids: list[int] = []
_wp = float(WEAK_LINK_PERCENTILE)
if _n_weak_before == 0 or _wp <= 0.0 or _wp >= 100.0:
    if _n_weak_before == 0:
        print("    无喉可剪枝。")
    else:
        print(f"    已关闭（weak-link-percentile={_wp:g}，需 0<p<100）。")
else:
    _tids_w = sorted(solute_accessible_throat_ids)
    _q_list: list[float] = []
    _deff_list: list[float] = []
    for _t in _tids_w:
        _qw = float(throat_flows.get(_t, 0.0))
        _q_list.append(abs(_qw) if np.isfinite(_qw) else 0.0)
        _dv = float(throat_D_eff.get(_t, 0.0))
        _deff_list.append(_dv if np.isfinite(_dv) else 0.0)
    _q_arr = np.asarray(_q_list, dtype=np.float64)
    _deff_arr = np.asarray(_deff_list, dtype=np.float64)
    _q_cut = float(np.percentile(_q_arr, _wp))
    _deff_cut = float(np.percentile(_deff_arr, _wp))
    for _k, _tid_w in enumerate(_tids_w):
        if _q_arr[_k] <= _q_cut and _deff_arr[_k] <= _deff_cut:
            _weak_removed_ids.append(int(_tid_w))
            solute_accessible_throat_ids.discard(_tid_w)
    print(
        f"    分位数 p={_wp:g}% → |Q| 分位截断={_q_cut:.6e} m³/s，"
        f"D_eff 分位截断={_deff_cut:.6e} m²/s；"
        f"移除喉数: {len(_weak_removed_ids)}（{_n_weak_before} → {len(solute_accessible_throat_ids)}）"
    )
G_weak_chk = nx.Graph()
for _tid_w, _p1w, _p2w in zip(throat_ids, throat_pore1, throat_pore2):
    if _tid_w in solute_accessible_throat_ids:
        if _p1w in pore_id_to_idx and _p2w in pore_id_to_idx:
            G_weak_chk.add_edge(int(_p1w), int(_p2w), throat_id=int(_tid_w))
if G_weak_chk.number_of_nodes() == 0:
    print("    警告：弱连接剪枝后图中无节点（若后续入口-出口剪枝失败需检查数据）。")
else:
    _degs_w = [d for _, d in G_weak_chk.degree()]
    print(
        f"    剪枝后（检验用）节点数: {G_weak_chk.number_of_nodes()}，"
        f"边数: {G_weak_chk.number_of_edges()}；"
        f"连通分量数: {nx.number_connected_components(G_weak_chk)}"
    )
    print(
        f"    节点度: min={min(_degs_w)}, max={max(_degs_w)}, "
        f"均值={float(np.mean(_degs_w)):.4f}；"
        f"度=0 节点数: {int(np.sum(np.asarray(_degs_w, dtype=np.int64) == 0))}（应为 0）"
    )

print("\n阶段D完成！")

# ====================================================================
# 阶段E：溶质传输计算（对流主导）
# ====================================================================
print("\n阶段E：溶质传输计算...")

# E1. 建立溶质质量守恒方程
print("  E1. 建立溶质质量守恒方程...")
if SOLVE_OLD_J and OLD_J_MEAN_THROAT_CONV:
    print(
        "    [old_J] 对流项：算术平均喉浓度 0.5*(C_i+C_j)；"
        "恢复纯上风请加 --no-old-j-mean-throat-convection。"
    )
elif SOLVE_OLD_J:
    print("    [old_J] 对流项：纯上风向（--no-old-j-mean-throat-convection）。")

# 重新构建只包含连通溶质通路的网络图（用于浓度计算）
G_solute_accessible = nx.Graph()
for t_id, p1, p2 in zip(throat_ids, throat_pore1, throat_pore2):
    if t_id in solute_accessible_throat_ids:
        if p1 in pore_id_to_idx and p2 in pore_id_to_idx:
            G_solute_accessible.add_edge(p1, p2, throat_id=t_id)

# 获取溶质可通行网络中的所有节点
solute_nodes = set()
for t_id in solute_accessible_throat_ids:
    p1 = throat_pore1[throat_ids == t_id][0]
    p2 = throat_pore2[throat_ids == t_id][0]
    solute_nodes.add(p1)
    solute_nodes.add(p2)

solute_nodes = sorted(list(solute_nodes))
n_solute_nodes = len(solute_nodes)

print(f"    入口可达白蛋白可通行网络中的节点数(跨膜剪枝前): {n_solute_nodes}")

if n_solute_nodes == 0:
    print("    错误：溶质可通行网络为空！")
    exit(1)

# ----------------------------------------------------------------
# 剪枝：去除“盲端”节点，只保留同时与入口和出口连通的子网络
# 思路：
#   1) 从入口节点集合出发，找所有可达节点 R_in
#   2) 从出口节点集合出发，找所有可达节点 R_out
#   3) 仅保留 R_keep = R_in ∩ R_out 中的节点及其边
# 这样可以去掉只连通一侧（或完全死区）的“盲端”孔和喉
# ----------------------------------------------------------------

# 入口和出口节点（基于当前溶质网络中的节点）
solute_entrance_nodes = [n for n in solute_nodes if n in entrance_pore_ids]
solute_exit_nodes = [n for n in solute_nodes if n in exit_pore_ids]

print(f"    入口可达白蛋白网络中的入口节点数(跨膜剪枝前): {len(solute_entrance_nodes)}")
print(f"    入口可达白蛋白网络中的出口节点数(跨膜剪枝前): {len(solute_exit_nodes)}")

# 1) 从所有入口出发，找 R_in
reachable_from_entrance = set()
for src in solute_entrance_nodes:
    if src in G_solute_accessible:
        reachable_from_entrance.update(nx.node_connected_component(G_solute_accessible, src))

# 2) 从所有出口出发，找 R_out
reachable_to_exit = set()
for dst in solute_exit_nodes:
    if dst in G_solute_accessible:
        reachable_to_exit.update(nx.node_connected_component(G_solute_accessible, dst))

# 3) 仅保留同时与入口和出口连通的节点
nodes_keep = sorted(list(reachable_from_entrance & reachable_to_exit))

print(f"    与入口连通的节点数: {len(reachable_from_entrance)}")
print(f"    与出口连通的节点数: {len(reachable_to_exit)}")
print(f"    同时与入口和出口连通的节点数(剪枝后): {len(nodes_keep)}")

if len(nodes_keep) == 0:
    print("    警告：没有同时与入口和出口连通的溶质节点，可能所有通路都是盲端")
    print("    为避免出错，暂时使用未剪枝的溶质网络（但结果需要谨慎解释）")
else:
    # 根据 nodes_keep 剪枝 G_solute_accessible 和 solute_accessible_throat_ids
    pruned_G = nx.Graph()
    pruned_throat_ids = set()
    nodes_keep_set = set(nodes_keep)

    for u, v, data in G_solute_accessible.edges(data=True):
        if (u in nodes_keep_set) and (v in nodes_keep_set):
            pruned_G.add_edge(u, v, **data)
            t_id = data.get("throat_id")
            if t_id is not None:
                pruned_throat_ids.add(t_id)

    G_solute_accessible = pruned_G
    solute_accessible_throat_ids = pruned_throat_ids
    solute_nodes = nodes_keep
    n_solute_nodes = len(solute_nodes)

# 重新创建节点ID到索引的映射（可能已经剪枝）
solute_node_to_idx = {node: idx for idx, node in enumerate(solute_nodes)}

print(f"    溶质可通行网络中的节点数(剪枝后): {n_solute_nodes}")

# ----------------------------------------------------------------
# 进一步剪枝：去除拓扑上的“结构盲端”（度=1且不是入口/出口的内部节点）
# 这里的盲端定义与你提到的一致：只有一个可通行喉连接的内部孔
# 反复删除这类叶子节点及其边，直到不存在这样的节点
# ----------------------------------------------------------------

if n_solute_nodes > 0:
    G_tmp = G_solute_accessible.copy()
    # 先基于当前节点集确定入口/出口集合
    solute_entrance_nodes = [n for n in solute_nodes if n in entrance_pore_ids]
    solute_exit_nodes = [n for n in solute_nodes if n in exit_pore_ids]
    entrance_set = set(solute_entrance_nodes)
    exit_set = set(solute_exit_nodes)

    pruned_once = False
    while True:
        # 找到所有度=1且不是入口/出口的内部节点
        leaf_nodes = [
            n for n, deg in G_tmp.degree()
            if deg == 1 and (n not in entrance_set) and (n not in exit_set)
        ]
        if not leaf_nodes:
            break
        pruned_once = True
        G_tmp.remove_nodes_from(leaf_nodes)

    if pruned_once:
        # 更新图和节点/喉集合
        G_solute_accessible = G_tmp
        solute_nodes = sorted(list(G_solute_accessible.nodes()))
        n_solute_nodes = len(solute_nodes)
        solute_node_to_idx = {node: idx for idx, node in enumerate(solute_nodes)}

        # 重新收集仍然存在于图中的喉 ID
        pruned_throat_ids_deg = set()
        for u, v, data in G_solute_accessible.edges(data=True):
            t_id = data.get("throat_id")
            if t_id is not None:
                pruned_throat_ids_deg.add(t_id)
        solute_accessible_throat_ids = solute_accessible_throat_ids & pruned_throat_ids_deg

        print(f"    结构盲端剪枝后节点数: {n_solute_nodes}")
        print(f"    结构盲端剪枝后喉数: {len(solute_accessible_throat_ids)}")

# 构建浓度方程组：默认 new_J。
A_conc_new = np.zeros((n_solute_nodes, n_solute_nodes))
b_conc_new = np.zeros(n_solute_nodes)
A_conc_old = np.zeros((n_solute_nodes, n_solute_nodes))
b_conc_old = np.zeros(n_solute_nodes)

assembly_audit_new = (
    None
    if OLD_J_ONLY
    else ([[] for _ in range(n_solute_nodes)] if _args.assembly_row_audit else None)
)


def _audit_nm_row(
    i_row: int,
    j_col: int,
    delta: float,
    *,
    t_id,
    kind: str,
    Q=None,
    Kc=None,
    diff_coeff=None,
    Pe=None,
    D_eff_m2s=None,
) -> None:
    """记录 new_J 组装时写入 A[i_row,j_col] 的一次增量（供 --assembly-row-audit）。"""
    if OLD_J_ONLY:
        return
    if assembly_audit_new is None:
        return
    rec = {
        "col": int(j_col),
        "delta": float(delta),
        "kind": str(kind),
        "t_id": None if t_id is None else int(t_id),
    }
    if Q is not None and np.isfinite(Q):
        rec["Q_m3s"] = float(Q)
    if Kc is not None and np.isfinite(Kc):
        rec["Kc"] = float(Kc)
    if diff_coeff is not None and float(diff_coeff) > 0.0 and np.isfinite(diff_coeff):
        rec["diff_m3s"] = float(diff_coeff)
    if Pe is not None and np.isfinite(Pe):
        rec["Pe"] = float(Pe)
    if D_eff_m2s is not None and float(D_eff_m2s) > 0.0 and np.isfinite(D_eff_m2s):
        rec["D_eff_m2s"] = float(D_eff_m2s)
    if t_id is not None:
        try:
            _mask_t = throat_ids == t_id
            if np.any(_mask_t):
                rec["r_nm"] = float(throat_radii[_mask_t][0])
            if t_id in throat_lengths:
                rec["L_nm"] = float(throat_lengths[t_id])
        except Exception:
            pass
    assembly_audit_new[i_row].append(rec)

# 标记入口和出口节点（基于最终剪枝后的网络）
solute_entrance_nodes = [n for n in solute_nodes if n in entrance_pore_ids]
solute_exit_nodes = [n for n in solute_nodes if n in exit_pore_ids]

print(f"    入口节点数: {len(solute_entrance_nodes)}")
print(f"    出口节点数: {len(solute_exit_nodes)}")

# 入口边界采用统一等效浓度：所有入口节点 Dirichlet 固定为 C0
def _compute_full_exit_solvent_flow_nls_early() -> float:
    q_total_all = 0.0
    exit_set_all = set(exit_pore_ids)
    seen_tids: set[int] = set()
    for exit_node in exit_pore_ids:
        if exit_node not in G_solvent:
            continue
        for neighbor in G_solvent.neighbors(exit_node):
            if neighbor in exit_set_all:
                continue
            edge_data = G_solvent[exit_node].get(neighbor, None)
            if not edge_data:
                continue
            t_id = edge_data.get("throat_id")
            if not t_id or t_id not in throat_flows:
                continue
            tid_i = int(t_id)
            if tid_i in seen_tids:
                continue
            seen_tids.add(tid_i)
            q_edge = -float(_signed_Q_node_to_neighbor(tid_i, int(exit_node), int(neighbor)))
            q_total_all += q_edge
    return float(q_total_all)


def _write_physical_zero_sieving_and_exit(reason: str) -> None:
    q_total_nls = _compute_full_exit_solvent_flow_nls_early()
    results_df = pd.DataFrame(
        {
            "Parameter": [
                "sieving_coefficient (C_out/C_in, new_J)",
                "Total Albumin Flow Rate (Q_alb_total, equiv_C)",
                "C_in (mean over entrance nodes)",
                "C_out (mean over exit nodes)",
                "C_in std",
                "C_out std",
                "Concentration field used (new_J preferred)",
                "new_J solution status",
                "new_J min concentration",
                "new_J negative tolerance",
                "new_J relative residual",
                "new_J residual invalid threshold",
                "new_J full-compare direct final residual l2",
                "new_J full-compare direct final relative residual",
                "new_J full-compare direct final residual max_abs",
                "new_J full-compare gmres final residual l2",
                "new_J full-compare gmres final relative residual",
                "new_J full-compare gmres final residual max_abs",
                "new_J full-compare final ||x_gmres-x_direct||_2",
                "new_J full-compare final ||x_gmres-x_direct||_2 relative",
                "new_J full-compare final ||x_gmres-x_direct||_inf",
                "N entrance nodes used in concentration stats",
                "N exit nodes used in concentration stats",
                "N entrance nodes in solute graph",
                "N exit nodes in solute graph",
                "Exit C_bulk parameterized boundary enabled",
                "Solved C_bulk (new_J)",
                "Plasma Concentration (C0)",
                "Net Outlet Solvent Flow (Q, full solvent network, m^3/s)",
                "Penetration axis index (0=X, 1=Y, 2=Z)",
                "Permeation plane cross-sectional area (nm^2)",
                "Permeation plane cross-sectional area (um^2)",
                "Number of Solvent Penetration Throats",
                "Number of Solute Accessible Throats",
                "zero_sieving_reason",
            ],
            "Value": [
                0.0,
                0.0,
                float(C0) if solute_entrance_nodes else float("nan"),
                0.0 if solute_exit_nodes else float("nan"),
                0.0 if solute_entrance_nodes else float("nan"),
                0.0 if solute_exit_nodes else float("nan"),
                "physical_zero",
                "physical_no_albumin_accessible_transmembrane_path",
                float("nan"),
                float(NEW_J_NEGATIVE_TOL),
                float("nan"),
                float(NEW_J_REL_RESIDUAL_INVALID_THRESHOLD)
                if NEW_J_REL_RESIDUAL_INVALID_THRESHOLD > 0.0
                else float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                len(solute_entrance_nodes),
                len(solute_exit_nodes),
                len(solute_entrance_nodes),
                len(solute_exit_nodes),
                1,
                0.0,
                C0,
                q_total_nls,
                PENETRATION_AXIS,
                PERMEATION_PLANE_AREA_NM2,
                PERMEATION_PLANE_AREA_UM2,
                len(solvent_feasible_throat_ids),
                len(solute_accessible_throat_ids),
                reason,
            ],
        }
    )
    results_excel = output_path / get_output_filename(f"{SIEVE_SAMPLE_NAME}_sieving_coefficient_results")
    results_df.to_excel(results_excel, index=False)
    concentration_df = pd.DataFrame({"Pore ID": [], "Concentration": []})
    concentration_excel = output_path / get_output_filename(f"{SIEVE_SAMPLE_NAME}_concentration_distribution")
    concentration_df.to_excel(concentration_excel, index=False)
    print(f"    Physical zero sieving: {reason}")
    print(f"    Results saved: {results_excel}")
    print(f"    Concentration table saved: {concentration_excel}")
    print("\n" + "=" * 70)
    print("计算完成：physical zero sieving")
    print("=" * 70)
    sys.exit(0)


if not solute_entrance_nodes or not solute_exit_nodes:
    _write_physical_zero_sieving_and_exit(
        "no albumin-accessible network component connected to both inlet and outlet"
    )

entrance_pore_concentrations = {int(node): {"concentration": float(C0)} for node in solute_entrance_nodes}
if entrance_pore_concentrations:
    print(f"    入口等效浓度统一为 C0={C0:.6f}（入口节点数: {len(entrance_pore_concentrations)}）")
else:
    print("    警告：溶质可通行网络中未找到入口节点。")

# 对每个节点建立质量守恒方程（等效浓度：喉–节点界面 f=1）
# 喉道 Pe：|Pe| 小 → Fick；大 → 纯对流；中间解析耦合（阈值见 CLI）。
_pe_adv_thr = float(NEW_J_PE_ADV_ONLY_THRESHOLD)
_pe_diff_thr = float(NEW_J_PE_DIFF_ONLY_THRESHOLD)
_pe_seg_adv = (
    f"|Pe|>{_pe_adv_thr:g} 纯对流"
    if _pe_adv_thr > 0.0
    else "未启用大|Pe|纯对流分段（--new-j-pe-adv-only-above<=0）"
)
_pe_seg_diff = (
    f"|Pe|<{_pe_diff_thr:g} 仅 Fick"
    if _pe_diff_thr > 0.0
    else "仅 |Pe|<1e-14 走 Fick（--new-j-pe-diff-only-below=0）"
)
print(
    "    new_J 喉组装："
    f"{_pe_seg_diff}；"
    f"{_pe_seg_adv}；"
    "否则解析耦合；无有效扩散时走纯对流（INCLUDE_DIFFUSION="
    f"{INCLUDE_DIFFUSION}）。"
)
# 统计入口 pore 的浓度分布（用于输出）
entrance_concentrations = []
# 入口 Dirichlet 与孤立锚定行先写 A_ii=1、b_i；组装结束后按「其余矩阵元的 max|A|」缩放整行 A 与 b，减轻与 ~1e-22 物理系数的量级差
dirichlet_row_indices = []

# ----------------------------------------------------------------
# 诊断：溶质网络中与组装逻辑一致的「孤立」——无任何邻接喉同时存在于 throat_Kc 与 throat_flows
# （与下方 has_connection 判定一致；缺 Kc/Q 的边单独计数）
# ----------------------------------------------------------------
def _print_solute_network_isolation_diagnostic() -> None:
    def _edge_throat_id(node: int, neighbor: int):
        try:
            edge_data = G_solute_accessible[node][neighbor]
            t_id = edge_data.get("throat_id")
        except KeyError:
            try:
                edge_data = G_solute_accessible[neighbor][node]
                t_id = edge_data.get("throat_id")
            except KeyError:
                t_id = None
        if not t_id:
            for t, p1, p2 in zip(throat_ids, throat_pore1, throat_pore2):
                if t in solute_accessible_throat_ids:
                    if (p1 == node and p2 == neighbor) or (p1 == neighbor and p2 == node):
                        t_id = t
                        break
        return t_id

    exit_set = set(solute_exit_nodes)
    iso_nodes: list[int] = []
    n_deg0 = 0
    n_deg0_exit = 0
    n_edge_no_tid = 0
    n_miss_kc = 0
    n_miss_q = 0
    n_miss_both = 0

    for n in solute_nodes:
        neighbors = [x for x in G_solute_accessible.neighbors(n) if x in solute_node_to_idx]
        if not neighbors:
            n_deg0 += 1
            if n in exit_set:
                n_deg0_exit += 1
            iso_nodes.append(n)
            continue
        has_valid = False
        for nb in neighbors:
            tid = _edge_throat_id(n, nb)
            if not tid:
                n_edge_no_tid += 1
                continue
            in_kc = tid in throat_Kc
            in_q = tid in throat_flows
            if in_kc and in_q:
                has_valid = True
                break
            if not in_kc and not in_q:
                n_miss_both += 1
            elif not in_kc:
                n_miss_kc += 1
            else:
                n_miss_q += 1
        if not has_valid:
            iso_nodes.append(n)

    iso_exit = [x for x in iso_nodes if x in exit_set]
    print("  [诊断] 溶质网络孤立（无至少一条邻接喉同时含 Kc 与 Q，与组装 has_connection 一致）：")
    print(f"    溶质节点总数: {n_solute_nodes}")
    print(f"    孤立节点数: {len(iso_nodes)}（图论度数=0: {n_deg0}，其中出口孔: {n_deg0_exit}）")
    _iso_exit_preview = sorted(iso_exit)[:24]
    print(
        f"    孤立出口孔数: {len(iso_exit)}（pore_id 前 24 个: {_iso_exit_preview}{' …' if len(iso_exit) > 24 else ''}）"
    )
    print(
        "    非零度孤立节点上，各邻接边失败原因计数（同一节点多条边可重复计）: "
        f"无 throat_id={n_edge_no_tid}, 仅缺 Kc={n_miss_kc}, 仅缺 Q={n_miss_q}, 二者皆缺={n_miss_both}"
    )


_print_solute_network_isolation_diagnostic()

for node in solute_nodes:
    i = solute_node_to_idx[node]
    if assembly_audit_new is not None:
        assembly_audit_new[i] = []

    # 入口节点统一 Dirichlet：等效浓度口径下 C_in = C0（不使用孔半径修正）
    if node in solute_entrance_nodes:
        C_pore_inlet = C0
        
        if not OLD_J_ONLY:
            A_conc_new[i, i] = 1.0
        if not OLD_J_ONLY:
            b_conc_new[i] = C_pore_inlet
        A_conc_old[i, i] = 1.0
        b_conc_old[i] = C_pore_inlet
        dirichlet_row_indices.append(i)
        if assembly_audit_new is not None:
            assembly_audit_new[i] = [{"col": i, "delta": 1.0, "kind": "inlet_dirichlet", "t_id": None}]
        entrance_concentrations.append((node, 0.0, C_pore_inlet))
        continue
    
    # 对于非入口节点（出口节点和内部节点），建立质量守恒方程
    # 质量守恒：所有流入的溶质通量 = 所有流出的溶质通量
    has_connection = False
    
    # 使用只包含连通溶质通路的图
    for neighbor in G_solute_accessible.neighbors(node):
        if neighbor not in solute_node_to_idx:
            continue
        
        # 直接从边的属性中获取throat_id（更高效且可靠）
        try:
            edge_data = G_solute_accessible[node][neighbor]
            t_id = edge_data.get('throat_id')
        except KeyError:
            # 如果找不到边，尝试反向查找
            try:
                edge_data = G_solute_accessible[neighbor][node]
                t_id = edge_data.get('throat_id')
            except KeyError:
                t_id = None
        
        # 如果仍然找不到，尝试通过遍历查找（fallback）
        if not t_id:
            for t, p1, p2 in zip(throat_ids, throat_pore1, throat_pore2):
                if t in solute_accessible_throat_ids:
                    if (p1 == node and p2 == neighbor) or (p1 == neighbor and p2 == node):
                        t_id = t
                        break
        
        if not t_id:
            continue
        
        # 检查喉是否在throat_Kc和throat_flows中
        if t_id not in throat_Kc or t_id not in throat_flows:
            # 缺少必要的对流/扩散参数时，跳过该喉
            continue
            
        Kc = throat_Kc[t_id]
        # Q_ij：沿「当前 node → neighbor」的体积流（与 throat p1→p2 一致）。
        # 指向当前节点的体积流可写为 q_in = -Q_ij（入为正）；对流与扩散项仍由下述分支与压强/Pe 共同决定。
        Q_ij = _signed_Q_node_to_neighbor(int(t_id), int(node), int(neighbor))
        Q = abs(float(Q_ij))
        
        # 获取喉道参数用于扩散计算与 new_J 对流可及截面修正
        r_i = throat_radii[throat_ids == t_id][0]
        U_abs = _throat_new_j_adv_coeff_m3s(Kc, Q, r_i)
        diff_coeff = 0.0
        if INCLUDE_DIFFUSION:
            L_i = throat_lengths.get(t_id, 0.0)
            D_eff = throat_D_eff.get(t_id, 0.0)
            
            # 计算扩散系数（如果D_eff和喉道参数有效）
            if D_eff > 0 and r_i > 0 and L_i > 0:
                diff_coeff = _throat_diff_coeff_m3s(r_i, L_i, D_eff)
            
        # 只要该喉参与组装（已有 Kc、Q），即视为连通；不对 |Q| 设「视为 0」下限，避免极小流被误判孤立行覆盖 A
        has_connection = True
        j = solute_node_to_idx[neighbor]
        
        # 溶剂方向：Q_ij>0 为 node→neighbor，neighbor→node 当且仅当 Q_ij<0（与平压/无压回退一致）
        pressure_tie, dp_nb = _pressure_tie_neighbor_to_node(neighbor, node)
        if neighbor in node_pressures and node in node_pressures:
            flow_from_neighbor = (not pressure_tie) and (Q_ij < 0.0)
        else:
            flow_from_neighbor = neighbor in solute_entrance_nodes
        Q_pos = -float(Q_ij) if flow_from_neighbor else float(Q_ij)
        
        if neighbor in solute_entrance_nodes:
            # 溶剂是否沿该喉从入口 neighbor → 当前 node：Q_ij<0（node→neighbor 有符号流为负）；
            # 两侧有压强时与内部一致，平压仅扩散步。
            if neighbor in node_pressures and node in node_pressures:
                flow_from_entrance = (not pressure_tie) and (Q_ij < 0.0)
            else:
                flow_from_entrance = Q_ij < 0.0
            Q_pos_inlet = -float(Q_ij) if flow_from_entrance else float(Q_ij)


            # 入口为统一 Dirichlet（C=C0）；不要回读 b 向量，避免组装顺序依赖。
            C_pore_entrance = C0
            C_throat_from_entrance = C_pore_entrance

            if pressure_tie and neighbor in node_pressures and node in node_pressures:
                if INCLUDE_DIFFUSION and diff_coeff > 0:
                    Pe_nm = float(U_abs / diff_coeff)
                    c_dd, c_du = _throat_fick_row_downstream(diff_coeff)
                    if not OLD_J_ONLY:
                        A_conc_new[i, i] += c_dd
                        A_conc_new[i, j] += c_du
                    _audit_nm_row(
                        i,
                        i,
                        c_dd,
                        t_id=t_id,
                        kind="nj_from_inlet_pressure_tie_fick_dd",
                        Q=Q,
                        Kc=Kc,
                        diff_coeff=diff_coeff,
                        Pe=Pe_nm,
                        D_eff_m2s=D_eff,
                    )
                    _audit_nm_row(
                        i,
                        j,
                        c_du,
                        t_id=t_id,
                        kind="nj_from_inlet_pressure_tie_fick_du",
                        Q=Q,
                        Kc=Kc,
                        diff_coeff=diff_coeff,
                        Pe=Pe_nm,
                        D_eff_m2s=D_eff,
                    )
                    A_conc_old[i, i] -= diff_coeff
                    b_conc_old[i] -= diff_coeff * C_throat_from_entrance
                continue

            if INCLUDE_DIFFUSION and diff_coeff > 0:
                _reg_in = _throat_pe_regime(1.0, U_abs, diff_coeff)
                if _reg_in == "coupled":
                    Pe_nm = (U_abs / diff_coeff) if diff_coeff > 0 else None
                    U_pos_inlet = _throat_new_j_adv_coeff_m3s(Kc, Q_pos_inlet, r_i)
                    c_dd, c_du = _throat_advection_diffusion_coeffs_downstream_row(1.0, U_pos_inlet, diff_coeff)
                    if not OLD_J_ONLY:
                        A_conc_new[i, i] += c_dd
                    _audit_nm_row(
                        i,
                        i,
                        c_dd,
                        t_id=t_id,
                        kind="nj_from_inlet_dd",
                        Q=Q,
                        Kc=Kc,
                        diff_coeff=diff_coeff,
                        Pe=Pe_nm,
                        D_eff_m2s=D_eff,
                    )
                    if not OLD_J_ONLY:
                        A_conc_new[i, j] += c_du
                    _audit_nm_row(
                        i,
                        j,
                        c_du,
                        t_id=t_id,
                        kind="nj_from_inlet_du",
                        Q=Q,
                        Kc=Kc,
                        diff_coeff=diff_coeff,
                        Pe=Pe_nm,
                        D_eff_m2s=D_eff,
                    )
                elif _reg_in == "diff_only":
                    Pe_nm = float(U_abs / diff_coeff)
                    c_dd, c_du = _throat_fick_row_downstream(diff_coeff)
                    if not OLD_J_ONLY:
                        A_conc_new[i, i] += c_dd
                    _audit_nm_row(
                        i,
                        i,
                        c_dd,
                        t_id=t_id,
                        kind="nj_from_inlet_fick_dd",
                        Q=Q,
                        Kc=Kc,
                        diff_coeff=diff_coeff,
                        Pe=Pe_nm,
                        D_eff_m2s=D_eff,
                    )
                    if not OLD_J_ONLY:
                        A_conc_new[i, j] += c_du
                    _audit_nm_row(
                        i,
                        j,
                        c_du,
                        t_id=t_id,
                        kind="nj_from_inlet_fick_du",
                        Q=Q,
                        Kc=Kc,
                        diff_coeff=diff_coeff,
                        Pe=Pe_nm,
                        D_eff_m2s=D_eff,
                    )
                else:
                    # new_J 纯对流：与净流入 i 一致；A_ij=+Kc|Q| 时 b-=Kc|Q|C_in；若溶剂 node→入口则 A_ii=−Kc|Q|。
                    if not OLD_J_ONLY:
                        qm_in = abs(float(Q_ij))
                        U_in = _throat_new_j_adv_coeff_m3s(Kc, qm_in, r_i)
                        if flow_from_entrance:
                            b_conc_new[i] -= U_in * C_throat_from_entrance
                        else:
                            A_conc_new[i, i] -= U_in
            else:
                if not OLD_J_ONLY:
                    qm_in = abs(float(Q_ij))
                    U_in = _throat_new_j_adv_coeff_m3s(Kc, qm_in, r_i)
                    if flow_from_entrance:
                        b_conc_new[i] -= U_in * C_throat_from_entrance
                    else:
                        A_conc_new[i, i] -= U_in

            # old_J 对流：与 _old_j_adv_dii_dij 一致（净流入 i）；平均喉消入口列时 A_ij=−h、h=0.5*Kc*Q_ij → b += h*C_in。
            if OLD_J_MEAN_THROAT_CONV:
                h_conv = 0.5 * float(Kc) * float(Q_ij)
                A_conc_old[i, i] -= h_conv
                b_conc_old[i] += h_conv * C_throat_from_entrance
            else:
                qm_in = abs(float(Q_ij))
                if flow_from_entrance:
                    b_conc_old[i] -= Kc * qm_in * C_throat_from_entrance
                else:
                    A_conc_old[i, i] -= Kc * qm_in
            if INCLUDE_DIFFUSION and diff_coeff > 0:
                A_conc_old[i, i] -= diff_coeff
                b_conc_old[i] -= diff_coeff * C_throat_from_entrance
        else:

            if pressure_tie and neighbor in node_pressures and node in node_pressures:
                if INCLUDE_DIFFUSION and diff_coeff > 0:
                    Pe_nm = float(U_abs / diff_coeff)
                    c_dd, c_du = _throat_fick_row_downstream(diff_coeff)
                    if not OLD_J_ONLY:
                        A_conc_new[i, i] += c_dd
                        A_conc_new[i, j] += c_du
                    _audit_nm_row(
                        i,
                        i,
                        c_dd,
                        t_id=t_id,
                        kind="nj_internal_pressure_tie_fick_dd",
                        Q=Q,
                        Kc=Kc,
                        diff_coeff=diff_coeff,
                        Pe=Pe_nm,
                        D_eff_m2s=D_eff,
                    )
                    _audit_nm_row(
                        i,
                        j,
                        c_du,
                        t_id=t_id,
                        kind="nj_internal_pressure_tie_fick_du",
                        Q=Q,
                        Kc=Kc,
                        diff_coeff=diff_coeff,
                        Pe=Pe_nm,
                        D_eff_m2s=D_eff,
                    )
                    A_conc_old[i, i] -= diff_coeff
                    A_conc_old[i, j] += diff_coeff
                continue

            if INCLUDE_DIFFUSION and diff_coeff > 0:
                _reg = _throat_pe_regime(1.0, U_abs, diff_coeff)
                if _reg == "coupled":
                    Pe_nm = (U_abs / diff_coeff) if diff_coeff > 0 else None
                    if flow_from_neighbor:
                        U_pos = _throat_new_j_adv_coeff_m3s(Kc, Q_pos, r_i)
                        c_dd, c_du = _throat_advection_diffusion_coeffs_downstream_row(1.0, U_pos, diff_coeff)
                        if not OLD_J_ONLY:
                            A_conc_new[i, i] += c_dd
                        _audit_nm_row(
                            i,
                            i,
                            c_dd,
                            t_id=t_id,
                            kind="nj_downstream_dd",
                            Q=Q,
                            Kc=Kc,
                            diff_coeff=diff_coeff,
                            Pe=Pe_nm,
                            D_eff_m2s=D_eff,
                        )
                        if not OLD_J_ONLY:
                            A_conc_new[i, j] += c_du
                        _audit_nm_row(
                            i,
                            j,
                            c_du,
                            t_id=t_id,
                            kind="nj_downstream_du",
                            Q=Q,
                            Kc=Kc,
                            diff_coeff=diff_coeff,
                            Pe=Pe_nm,
                            D_eff_m2s=D_eff,
                        )
                        # old_J 对流 + Fick 扩散
                        _dii, _dij = _old_j_adv_dii_dij(Kc, Q_ij,
                            flow_from_neighbor=True,
                        )
                        if _dii != 0.0:
                            A_conc_old[i, i] += _dii
                        if _dij != 0.0:
                            A_conc_old[i, j] += _dij
                        # 下游行 Fick：与 _throat_fick_row_downstream / new_J 一致（−D 于对角，+D 于邻列）。
                        A_conc_old[i, i] -= diff_coeff
                        A_conc_old[i, j] += diff_coeff
                    else:
                        U_pos = _throat_new_j_adv_coeff_m3s(Kc, Q_pos, r_i)
                        c_uu, c_ud = _throat_advection_diffusion_coeffs_upstream_row(1.0, U_pos, diff_coeff)
                        if not OLD_J_ONLY:
                            A_conc_new[i, i] += c_uu
                        _audit_nm_row(
                            i,
                            i,
                            c_uu,
                            t_id=t_id,
                            kind="nj_upstream_uu",
                            Q=Q,
                            Kc=Kc,
                            diff_coeff=diff_coeff,
                            Pe=Pe_nm,
                            D_eff_m2s=D_eff,
                        )
                        if not OLD_J_ONLY:
                            A_conc_new[i, j] += c_ud
                        _audit_nm_row(
                            i,
                            j,
                            c_ud,
                            t_id=t_id,
                            kind="nj_upstream_ud",
                            Q=Q,
                            Kc=Kc,
                            diff_coeff=diff_coeff,
                            Pe=Pe_nm,
                            D_eff_m2s=D_eff,
                        )
                        # old_J 对流 + Fick 扩散
                        _dii, _dij = _old_j_adv_dii_dij(Kc, Q_ij,
                            flow_from_neighbor=False,
                        )
                        if _dii != 0.0:
                            A_conc_old[i, i] += _dii
                        if _dij != 0.0:
                            A_conc_old[i, j] += _dij
                        # 上游行 Fick：与 _throat_fick_row_upstream / new_J 一致。
                        A_conc_old[i, i] -= diff_coeff
                        A_conc_old[i, j] += diff_coeff
                elif _reg == "diff_only":
                    Pe_nm = float(U_abs / diff_coeff)
                    if flow_from_neighbor:
                        c_dd, c_du = _throat_fick_row_downstream(diff_coeff)
                        if not OLD_J_ONLY:
                            A_conc_new[i, i] += c_dd
                        _audit_nm_row(
                            i,
                            i,
                            c_dd,
                            t_id=t_id,
                            kind="nj_downstream_fick_dd",
                            Q=Q,
                            Kc=Kc,
                            diff_coeff=diff_coeff,
                            Pe=Pe_nm,
                            D_eff_m2s=D_eff,
                        )
                        if not OLD_J_ONLY:
                            A_conc_new[i, j] += c_du
                        _audit_nm_row(
                            i,
                            j,
                            c_du,
                            t_id=t_id,
                            kind="nj_downstream_fick_du",
                            Q=Q,
                            Kc=Kc,
                            diff_coeff=diff_coeff,
                            Pe=Pe_nm,
                            D_eff_m2s=D_eff,
                        )
                        # old_J 对流 + Fick 扩散
                        _dii, _dij = _old_j_adv_dii_dij(Kc, Q_ij,
                            flow_from_neighbor=True,
                        )
                        if _dii != 0.0:
                            A_conc_old[i, i] += _dii
                        if _dij != 0.0:
                            A_conc_old[i, j] += _dij
                        A_conc_old[i, i] -= diff_coeff
                        A_conc_old[i, j] += diff_coeff
                    else:
                        c_uu, c_ud = _throat_fick_row_upstream(diff_coeff)
                        if not OLD_J_ONLY:
                            A_conc_new[i, i] += c_uu
                        _audit_nm_row(
                            i,
                            i,
                            c_uu,
                            t_id=t_id,
                            kind="nj_upstream_fick_uu",
                            Q=Q,
                            Kc=Kc,
                            diff_coeff=diff_coeff,
                            Pe=Pe_nm,
                            D_eff_m2s=D_eff,
                        )
                        if not OLD_J_ONLY:
                            A_conc_new[i, j] += c_ud
                        _audit_nm_row(
                            i,
                            j,
                            c_ud,
                            t_id=t_id,
                            kind="nj_upstream_fick_ud",
                            Q=Q,
                            Kc=Kc,
                            diff_coeff=diff_coeff,
                            Pe=Pe_nm,
                            D_eff_m2s=D_eff,
                        )
                        # old_J 对流 + Fick 扩散
                        _dii, _dij = _old_j_adv_dii_dij(Kc, Q_ij,
                            flow_from_neighbor=False,
                        )
                        if _dii != 0.0:
                            A_conc_old[i, i] += _dii
                        if _dij != 0.0:
                            A_conc_old[i, j] += _dij
                        # 同上：diff_only 上游行 Fick 与 _throat_fick_row_upstream 一致。
                        A_conc_old[i, i] -= diff_coeff
                        A_conc_old[i, j] += diff_coeff
                else:
                    if flow_from_neighbor:
                        if not OLD_J_ONLY:
                            # new_J 纯对流极限（下游行）：J = U_adv*C_up，不含 C_down 项
                            d_off = U_abs
                            A_conc_new[i, j] += d_off
                        _audit_nm_row(
                            i,
                            j,
                            U_abs,
                            t_id=t_id,
                            kind="nj_adv_upwind_downstream",
                            Q=Q,
                            Kc=Kc,
                        )
                        _dii, _dij = _old_j_adv_dii_dij(Kc, Q_ij,
                            flow_from_neighbor=True,
                        )
                        if _dii != 0.0:
                            A_conc_old[i, i] += _dii
                        if _dij != 0.0:
                            A_conc_old[i, j] += _dij
                    else:
                        d_diag = -U_abs
                        if not OLD_J_ONLY:
                            # new_J 纯对流（上游）：净流入 i 约定下对角为 −Kc|Q|
                            A_conc_new[i, i] += d_diag
                        _audit_nm_row(
                            i,
                            i,
                            d_diag,
                            t_id=t_id,
                            kind="nj_adv_upwind_upstream",
                            Q=Q,
                            Kc=Kc,
                        )
                        _dii, _dij = _old_j_adv_dii_dij(Kc, Q_ij,
                            flow_from_neighbor=False,
                        )
                        if _dii != 0.0:
                            A_conc_old[i, i] += _dii
                        if _dij != 0.0:
                            A_conc_old[i, j] += _dij
            else:
                if flow_from_neighbor:
                    if not OLD_J_ONLY:
                        d_off = U_abs
                        A_conc_new[i, j] += d_off
                    _audit_nm_row(
                        i,
                        j,
                        U_abs,
                        t_id=t_id,
                        kind="nj_adv_upwind_downstream",
                        Q=Q,
                        Kc=Kc,
                    )
                    _dii, _dij = _old_j_adv_dii_dij(Kc, Q_ij,
                        flow_from_neighbor=True,
                    )
                    if _dii != 0.0:
                        A_conc_old[i, i] += _dii
                    if _dij != 0.0:
                        A_conc_old[i, j] += _dij
                else:
                    d_diag = -U_abs
                    if not OLD_J_ONLY:
                        A_conc_new[i, i] += d_diag
                    _audit_nm_row(
                        i,
                        i,
                        d_diag,
                        t_id=t_id,
                        kind="nj_adv_upwind_upstream",
                        Q=Q,
                        Kc=Kc,
                    )
                    _dii, _dij = _old_j_adv_dii_dij(Kc, Q_ij,
                        flow_from_neighbor=False,
                    )
                    if _dii != 0.0:
                        A_conc_old[i, i] += _dii
                    if _dij != 0.0:
                        A_conc_old[i, j] += _dij
    
    # 如果节点没有任何连接，说明网络构建有问题，输出警告
    if not has_connection:
        print(f"      警告：节点 {node} 在溶质网络中没有任何有效连接，可能的数据问题")
        # 调试信息：检查该节点在溶质图中的邻接喉及其在参数/流量中的状态
        try:
            neighbors = list(G_solute_accessible.neighbors(node))
        except Exception:
            neighbors = []
        print(f"        调试：该节点在 G_solute_accessible 中的邻居数量 = {len(neighbors)}")
        max_neighbors_to_show = 10
        for nb in neighbors[:max_neighbors_to_show]:
            try:
                edge_data_dbg = G_solute_accessible[node][nb]
                t_id_dbg = edge_data_dbg.get('throat_id')
            except Exception:
                t_id_dbg = None
            print(f"          邻居 {nb}, throat_id={t_id_dbg}")
            if t_id_dbg is not None:
                in_Kc = t_id_dbg in throat_Kc
                in_Q = t_id_dbg in throat_flows
                print(f"            在throat_Kc中: {in_Kc}, 在throat_flows中: {in_Q}")
                if in_Q:
                    print(
                        f"            Q(p1→p2) = {float(throat_flows[t_id_dbg]):.3e} m^3/s"
                    )
                if in_Kc:
                    print(f"            Kc = {throat_Kc[t_id_dbg]:.3f}")
        # 对于孤立节点，如果它是出口节点，可以认为没有溶质流入（浓度可能接近0）
        # 如果是内部节点但没有连接，这不应该发生，需要检查网络构建
        if node in solute_exit_nodes:
            # 出口节点没有连接：可能是边界情况，浓度设为0（无溶质进入）
            if not OLD_J_ONLY:
                A_conc_new[i, i] = 1.0
            if not OLD_J_ONLY:
                b_conc_new[i] = 0.0
            A_conc_old[i, i] = 1.0
            b_conc_old[i] = 0.0
            dirichlet_row_indices.append(i)
            if assembly_audit_new is not None:
                assembly_audit_new[i] = [{"col": i, "delta": 1.0, "kind": "isolated_exit_anchor", "t_id": None}]
        else:
            # 内部节点没有连接：不应该发生，但仍然需要避免矩阵奇异
            # 暂时设为0（但这种情况应该被排除）
            print(f"        错误：内部节点 {node} 没有连接，这不应该发生！")
            if not OLD_J_ONLY:
                A_conc_new[i, i] = 1.0
            if not OLD_J_ONLY:
                b_conc_new[i] = 0.0
            A_conc_old[i, i] = 1.0
            b_conc_old[i] = 0.0
            dirichlet_row_indices.append(i)
            if assembly_audit_new is not None:
                assembly_audit_new[i] = [{"col": i, "delta": 1.0, "kind": "isolated_internal_anchor", "t_id": None}]


exit_equality_row_indices_new: list[int] = []
exit_equality_row_indices_old: list[int] = []

# 新口径：入口/出口均作为边界节点，不在出口节点上施加“内部守恒=0”。
# 做法：出口节点浓度统一为参数 C_bulk_param，先取 0/1 各求一次，利用线性关系与全局守恒闭合求 C_bulk。
_exit_dirichlet_rows: list[int] = sorted(
    {int(solute_node_to_idx[n]) for n in solute_exit_nodes if n in solute_node_to_idx}
)
print(
    "  [边界口径] 内部节点守恒 + 入口/出口聚合边界："
    f"入口 C=C0，出口统一 C_bulk 参数；出口节点行数={len(_exit_dirichlet_rows)}"
)


def _apply_boundary_rows_with_cbulk(
    A_base: np.ndarray,
    b_base: np.ndarray,
    c_bulk_param: float,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray, list[int], float]:
    A = A_base.copy()
    b = b_base.copy()
    for ir in _exit_dirichlet_rows:
        A[ir, :] = 0.0
        A[ir, ir] = 1.0
        b[ir] = float(c_bulk_param)
    constrained_rows = sorted(set(dirichlet_row_indices) | set(_exit_dirichlet_rows))
    A_scale_probe = A.copy()
    if constrained_rows:
        A_scale_probe[constrained_rows, :] = 0.0
    _abs_probe = np.abs(A_scale_probe)
    _nz = _abs_probe[_abs_probe > 0.0]
    _ascale = float(np.mean(_nz)) if _nz.size > 0 else float(np.max(_abs_probe))
    if not np.isfinite(_ascale) or _ascale <= 0.0:
        _ascale = 1.0
    for _ir in constrained_rows:
        A[_ir, :] *= _ascale
        b[_ir] *= _ascale
    print(
        f"    [{label}] 边界参数 C_bulk={float(c_bulk_param):.6e}："
        f"约束行 n={len(constrained_rows)}，max_abs_nz={_ascale:.6e}"
    )
    return A, b, constrained_rows, _ascale

# E2-E3. 求解浓度分布
print("  E2-E3. 求解浓度分布...")
print(
    f"    线性求解器: {LINEAR_SOLVER}"
    + ("（开启 direct vs gmres-ilut 对比）" if COMPARE_LINEAR_SOLVERS else "")
)

# 检查矩阵是否有零行（诊断）
# 注意：由于流量和扩散系数可能非常小（例如 1e-19 量级），
# 零行判断阈值应该更小，避免误判
# 行 L1 范数低于此值才视为「零行」；压低以减少 Q/diff 极小时仍非零的守恒行被误判
zero_row_threshold = 1e-40
zero_rows = []
zero_row_nodes = []
_A_zero_probe = A_conc_old if OLD_J_ONLY else A_conc_new
for i in range(n_solute_nodes):
    row_sum = np.sum(np.abs(_A_zero_probe[i, :]))
    if row_sum < zero_row_threshold:
        zero_rows.append(i)
        zero_row_nodes.append(solute_nodes[i])


def _audit_expr_symbolic(comps):
    """分项 kind 与 throat_id 组成的符号和。"""
    parts = []
    for e in comps:
        k = e.get("kind", "?")
        if e.get("t_id") is not None:
            parts.append(f"{k}[t={e['t_id']}]")
        else:
            parts.append(str(k))
    return " + ".join(parts) if parts else "(no_components)"


def _audit_expr_with_values(comps):
    """表达式 + 各组分 Δ 数值： kind[t](Δ=...) + ..."""
    parts = []
    for e in comps:
        k = e.get("kind", "?")
        d = float(e["delta"])
        if e.get("t_id") is not None:
            parts.append(f"{k}[t={e['t_id']}](Δ={d:+.6e})")
        else:
            parts.append(f"{k}(Δ={d:+.6e})")
    return " + ".join(parts) if parts else ""


def _audit_component_label(e):
    """单行组分符号（不含数值）。"""
    k = e.get("kind", "?")
    if e.get("t_id") is not None:
        return f"{k}[t={e['t_id']}]"
    return str(k)


def _audit_expr_numeric_terms(comps):
    """仅数值展开：Δ1 + Δ2 + ...（与 expression_symbolic 逐项对应）。"""
    if not comps:
        return ""
    return " + ".join(f"{float(e['delta']):+.10e}" for e in comps)


def _audit_equation_line(aij: float, comps, ssum: float, resid: float) -> str:
    """一行可读等式：数值和式、分项和、A_ij、残差。"""
    num = _audit_expr_numeric_terms(comps)
    if not num:
        return f"A_ij={aij:+.10e}; sum(parts)=0; residual={resid:+.10e} (no audit trail)"
    return (
        f"{num} = sum(parts)={ssum:+.10e}; "
        f"A_ij={aij:+.10e}; residual={resid:+.10e}"
    )


# --assembly-row-audit：分项 kind → 传输类型 + 与代码一致的公式说明（中文）
_AUDIT_KIND_CATEGORY_EXPR = {
    "inlet_dirichlet": (
        "边界(Dirichlet)",
        "入口节点：先写 A_ii=1、b_i=C0；组装后约束行可按 amax 整行缩放。",
    ),
    "isolated_exit_anchor": (
        "边界(修复)",
        "孤立出口：强加 A_ii=1、b_i=0 避免奇异矩阵。",
    ),
    "isolated_internal_anchor": (
        "边界(修复)",
        "孤立内部节点：强加 A_ii=1、b_i=0 避免奇异矩阵。",
    ),
    "nj_from_inlet_Q": (
        "纯对流",
        "自入口孔经喉进入：new_J 在质量守恒行对 A_ii 累加体积通量 Q (m³/s)；喉/孔换算在 rhs 已体现。",
    ),
    "nj_adv_Qii": (
        "纯对流",
        "邻孔为流向上游：A_ii += Q。",
    ),
    "nj_adv_KcQfn": (
        "纯对流",
        "邻孔为上游（等效浓度、净流入 i）：A_ij += +Kc·|Q|。",
    ),
    "nj_adv_Qfii": (
        "纯对流",
        "本节点为上游（等效浓度、净流入 i）：A_ii += −Kc·|Q|。",
    ),
    "nj_from_inlet_dd": (
        "对流-扩散耦合",
        "入口邻接、含扩散：下游节点行对角元 c_dd；Pe=Kc·Q/diff_coeff，diff_coeff=D_eff·A_solute/L；"
        "一维稳态 J_vol∝(C_d - C_u·e^Pe)/(1-e^Pe)，实现见 _throat_advection_diffusion_coeffs_downstream_row。",
    ),
    "nj_from_inlet_du": (
        "对流-扩散耦合",
        "入口邻接、含扩散：下游行非对角 c_du（耦合上游孔浓度），Pe 与 diff 同上。",
    ),
    "nj_downstream_dd": (
        "对流-扩散耦合",
        "内部节点、下游行对角 c_dd；公式同 nj_from_inlet_dd（下游行）。",
    ),
    "nj_downstream_du": (
        "对流-扩散耦合",
        "内部节点、下游行非对角 c_du。",
    ),
    "nj_upstream_uu": (
        "对流-扩散耦合",
        "内部节点、上游行对角 c_uu；实现见 _throat_advection_diffusion_coeffs_upstream_row。",
    ),
    "nj_upstream_ud": (
        "对流-扩散耦合",
        "内部节点、上游行非对角 c_ud。",
    ),
    "nj_from_inlet_fick_dd": (
        "纯扩散(Fick)",
        "|Pe|<1e-14：入口下游行对角，Fick 极限 c_dd=-diff。",
    ),
    "nj_from_inlet_fick_du": (
        "纯扩散(Fick)",
        "|Pe|<1e-14：入口下游行非对角 c_du=+diff。",
    ),
    "nj_downstream_fick_dd": (
        "纯扩散(Fick)",
        "|Pe|<1e-14：内部下游行对角（Fick）。",
    ),
    "nj_downstream_fick_du": (
        "纯扩散(Fick)",
        "|Pe|<1e-14：内部下游行非对角（Fick）。",
    ),
    "nj_upstream_fick_uu": (
        "纯扩散(Fick)",
        "|Pe|<1e-14：内部上游行对角（Fick）。",
    ),
    "nj_upstream_fick_ud": (
        "纯扩散(Fick)",
        "|Pe|<1e-14：内部上游行非对角（Fick）。",
    ),
}


def _audit_physics_meta(e: dict) -> dict:
    """表格列：传输类别、公式说明、Q/v/D/L 等组分中文串，以及 v_avg、A_solute 数值。"""
    k = e.get("kind")
    if k is None:
        return {
            "transport_category": "",
            "physics_expression_zh": "",
            "physics_components_zh": "",
            "v_avg_m_s": np.nan,
            "A_solute_m2": np.nan,
        }
    kstr = str(k)
    cat, doc = _AUDIT_KIND_CATEGORY_EXPR.get(
        kstr,
        ("其它/未知", f"分项 kind={kstr}，见本文件中质量守恒组装循环。"),
    )
    parts = []
    q = e.get("Q_m3s")
    rnm = e.get("r_nm")
    v_avg = np.nan
    asol_m2 = np.nan
    if q is not None and np.isfinite(q) and rnm is not None and np.isfinite(rnm) and float(rnm) > 0:
        rm = float(rnm) * 1e-9
        ageo = np.pi * rm * rm
        if ageo > 0:
            v_avg = float(q) / ageo
    if rnm is not None and np.isfinite(rnm):
        _as = float(_throat_solute_diffusion_area_m2(float(rnm)))
        if _as > 0:
            asol_m2 = _as
    if q is not None and np.isfinite(q):
        parts.append(f"Q={float(q):.6e} m³/s")
    if np.isfinite(v_avg):
        parts.append(f"v_avg=Q/(πr²)={float(v_avg):.6e} m/s（溶剂几何截面积 πr²）")
    if e.get("Kc") is not None and np.isfinite(e["Kc"]):
        parts.append(f"Kc={float(e['Kc']):.6g}")
    deff = e.get("D_eff_m2s")
    if deff is not None and np.isfinite(deff) and float(deff) > 0:
        parts.append(f"D_eff={float(deff):.6e} m²/s")
    if e.get("L_nm") is not None and np.isfinite(e["L_nm"]):
        parts.append(f"L={float(e['L_nm']):.6g} nm")
    if rnm is not None and np.isfinite(rnm):
        parts.append(f"r={float(rnm):.6g} nm（喉几何半径）")
    if np.isfinite(asol_m2):
        parts.append(f"A_solute={float(asol_m2):.6e} m²（π(R_T−r)²，与 diff_coeff 一致）")
    if e.get("diff_m3s") is not None and np.isfinite(e["diff_m3s"]) and float(e["diff_m3s"]) > 0:
        parts.append(f"diff_coeff={float(e['diff_m3s']):.6e} m³/s (=D_eff·A_solute/L)")
    if e.get("Pe") is not None and np.isfinite(e["Pe"]):
        parts.append(f"Pe=Kc·Q/diff_coeff={float(e['Pe']):.6g}")
    return {
        "transport_category": cat,
        "physics_expression_zh": doc,
        "physics_components_zh": "; ".join(parts) if parts else "",
        "v_avg_m_s": float(v_avg) if np.isfinite(v_avg) else np.nan,
        "A_solute_m2": float(asol_m2) if np.isfinite(asol_m2) else np.nan,
    }


if _args.assembly_row_audit and assembly_audit_new is not None and n_solute_nodes > 0:
    zset = set(zero_rows)
    nontrivial = [idx for idx in range(n_solute_nodes) if idx not in zset]
    A_nm = A_conc_new.astype(np.float64)
    rank_A = int(np.linalg.matrix_rank(A_nm))

    print(
        f"\n  [assembly-row-audit] rank(A)={rank_A}（稳定化前）；"
        f"取 |A_ij| 最大的 min(2*rank, nnz) 个矩阵元，分项展开为**单表**"
    )

    if rank_A <= 0:
        print("    rank(A)=0，跳过 Top-|A_ij| 表。")
    else:
        all_trip = []
        for ii in range(n_solute_nodes):
            row = A_nm[ii, :]
            for jj in np.flatnonzero(np.abs(row) > 0.0):
                jj = int(jj)
                aij = float(row[jj])
                all_trip.append((abs(aij), ii, jj, aij))
        all_trip.sort(key=lambda x: (-x[0], x[1], x[2]))
        n_top = min(2 * rank_A, len(all_trip))

        integrated_rows = []
        for rnk, (abv, ii, jj, aij) in enumerate(all_trip[:n_top], start=1):
            comps = [e for e in assembly_audit_new[ii] if int(e["col"]) == jj]
            comps.sort(key=lambda e: -abs(e["delta"]))
            ssum = sum(float(e["delta"]) for e in comps)
            resid = float(aij) - ssum
            pid_i = solute_nodes[ii]
            pid_j = solute_nodes[jj]
            expr_sym = _audit_expr_symbolic(comps)
            expr_val = _audit_expr_with_values(comps)
            expr_num = _audit_expr_numeric_terms(comps)
            eq_line = _audit_equation_line(float(aij), comps, float(ssum), float(resid))
            nparts = len(comps)
            if not comps:
                integrated_rows.append(
                    {
                        "rank_A": rank_A,
                        "rank_by_abs": rnk,
                        "row_i": ii,
                        "col_j": jj,
                        "pore_id_row": pid_i,
                        "pore_id_col": pid_j,
                        "A_ij": float(aij),
                        "abs_A_ij": float(abv),
                        "n_parts": 0,
                        "sum_delta": float(ssum),
                        "residual_A_minus_sum": float(resid),
                        "expression_symbolic": expr_sym,
                        "expression_with_deltas": expr_val,
                        "expression_numeric_terms": expr_num,
                        "equation_line": eq_line,
                        "component_label": "(no_audit)",
                        "part": 0,
                        "kind": None,
                        "t_id": None,
                        "delta": np.nan,
                        "component_value": np.nan,
                        "transport_category": "",
                        "physics_expression_zh": "",
                        "physics_components_zh": "",
                        "v_avg_m_s": np.nan,
                        "A_solute_m2": np.nan,
                        "D_eff_m2s": np.nan,
                    }
                )
            else:
                for pk, e in enumerate(comps, start=1):
                    er = {
                        "rank_A": rank_A,
                        "rank_by_abs": rnk,
                        "row_i": ii,
                        "col_j": jj,
                        "pore_id_row": pid_i,
                        "pore_id_col": pid_j,
                        "A_ij": float(aij),
                        "abs_A_ij": float(abv),
                        "n_parts": nparts,
                        "sum_delta": float(ssum),
                        "residual_A_minus_sum": float(resid),
                        "expression_symbolic": expr_sym,
                        "expression_with_deltas": expr_val,
                        "expression_numeric_terms": expr_num,
                        "equation_line": eq_line,
                        "component_label": _audit_component_label(e),
                        "part": pk,
                        "kind": e.get("kind"),
                        "t_id": e.get("t_id"),
                        "delta": float(e["delta"]),
                        "component_value": float(e["delta"]),
                    }
                    for key in (
                        "Q_m3s",
                        "Kc",
                        "diff_m3s",
                        "Pe",
                        "r_nm",
                        "L_nm",
                        "D_eff_m2s",
                    ):
                        if key in e:
                            er[key] = float(e[key])
                    er.update(_audit_physics_meta(e))
                    if "D_eff_m2s" not in er:
                        er["D_eff_m2s"] = np.nan
                    integrated_rows.append(er)

        df_audit = pd.DataFrame(integrated_rows)
        audit_xlsx = output_path / get_output_filename(f"{SIEVE_SAMPLE_NAME}_matrix_top_elements_audit", ".xlsx")
        try:
            with pd.ExcelWriter(str(audit_xlsx), engine="openpyxl") as writer:
                df_audit.to_excel(writer, sheet_name="audit", index=False)
            print(f"    [assembly-row-audit] 已写入: {audit_xlsx}（单表 audit）")
        except Exception as ex:
            audit_csv = output_path / get_output_filename(f"{SIEVE_SAMPLE_NAME}_matrix_top_elements_audit", ".csv")
            df_audit.to_csv(audit_csv, index=False, encoding="utf-8-sig")
            print(f"    [assembly-row-audit] Excel 失败 ({ex})，已写入 CSV: {audit_csv}")

    if _args.assembly_row_audit_verbose and nontrivial:
        tr = max(1, int(_args.assembly_row_audit_top_rows))
        te = max(1, int(_args.assembly_row_audit_top_entries))
        print(
            f"\n  [assembly-row-audit-verbose] 按 ||A[i,:]||_∞ 取前 {tr} 行，每行最多 {te} 条分项"
        )
        scored = sorted(
            ((idx, float(np.max(np.abs(A_conc_new[idx, :])))) for idx in nontrivial),
            key=lambda x: -x[1],
        )
        for rank, (irow, rinf) in enumerate(scored[:tr], start=1):
            pid = solute_nodes[irow]
            r1 = float(np.sum(np.abs(A_conc_new[irow, :])))
            print(f"    #{rank}  row_i={irow}  pore_id={pid}  ||row||_inf={rinf:.6e}  ||row||_1={r1:.6e}")
            rowv = A_conc_new[irow, :]
            topj = np.argsort(-np.abs(rowv))[: min(8, n_solute_nodes)]
            parts_j = [f"col{k}={float(rowv[k]):+.4e}" for k in topj if abs(float(rowv[k])) > 0]
            if parts_j:
                print(f"      行内 |A_ij| 最大若干: " + "  ".join(parts_j))
            ents = list(assembly_audit_new[irow])
            ents.sort(key=lambda e: -abs(e["delta"]))
            for k, e in enumerate(ents[:te], start=1):
                bits = [
                    f"Δ={e['delta']:+.6e}",
                    f"col={e['col']}",
                    f"{e['kind']}",
                ]
                if e.get("t_id") is not None:
                    bits.append(f"t={e['t_id']}")
                for key in ("Q_m3s", "Kc", "diff_m3s", "Pe", "r_nm", "L_nm"):
                    if key in e:
                        bits.append(f"{key}={e[key]:.6e}")
                print(f"      [{k}] " + " ".join(bits))
            comb = {}
            for e in ents:
                key = (e["col"], e.get("t_id"))
                comb[key] = comb.get(key, 0.0) + abs(e["delta"])
            topc = sorted(comb.items(), key=lambda x: -x[1])[:8]
            if topc:
                seg = ", ".join(f"(col={a},t={b}) sum|Δ|={c:.6e}" for (a, b), c in topc)
                print(f"      按 (col, throat_id) 聚合 sum|ΔA|: {seg}")

if zero_rows:
    print(f"    警告：发现 {len(zero_rows)} 个零行")
    print(f"    开始诊断零行产生的原因...")
    
    # 统计零行节点的类型
    zero_entrance = [n for n in zero_row_nodes if n in solute_entrance_nodes]
    zero_exit = [n for n in zero_row_nodes if n in solute_exit_nodes]
    zero_internal = [n for n in zero_row_nodes if n not in solute_entrance_nodes and n not in solute_exit_nodes]
    
    print(f"      零行节点类型统计:")
    print(f"        入口节点: {len(zero_entrance)} 个")
    print(f"        出口节点: {len(zero_exit)} 个")
    print(f"        内部节点: {len(zero_internal)} 个")
    
    # 详细分析前10个零行节点
    print(f"\n      详细分析前10个零行节点:")
    for idx, node_idx in enumerate(zero_rows[:10]):
        node = solute_nodes[node_idx]
        print(f"\n      节点 {idx+1}: Pore ID = {node}")
        
        # 检查节点类型
        if node in solute_entrance_nodes:
            print(f"        类型: 入口节点")
        elif node in solute_exit_nodes:
            print(f"        类型: 出口节点")
        else:
            print(f"        类型: 内部节点")
        
        # 检查该节点在矩阵中的实际方程
        row = _A_zero_probe[node_idx, :]
        row_sum = np.sum(np.abs(row))
        row_nonzero = np.count_nonzero(row)
        print(f"        矩阵行状态: 非零元素数 = {row_nonzero}, 行和 = {row_sum:.3e}")
        if row_nonzero > 0:
            nonzero_indices = np.nonzero(row)[0]
            print(f"        非零元素位置: {nonzero_indices[:10].tolist()}")
            print(f"        非零元素值: {row[nonzero_indices[:10]].tolist()}")
        
        # 检查节点在图中的邻居
        try:
            neighbors = list(G_solute_accessible.neighbors(node))
            print(f"        在图中的邻居数: {len(neighbors)}")
        except Exception:
            neighbors = []
            print(f"        在图中的邻居数: 0 (节点不在图中)")
        
        # 检查每个邻居对应的喉，并模拟方程建立过程
        if neighbors:
            print(f"        邻居详情（模拟方程建立）:")
            total_expected_contrib = 0.0
            tiny_flow_hits = 0
            for nb in neighbors[:5]:  # 只显示前5个
                try:
                    edge_data = G_solute_accessible[node][nb]
                    t_id = edge_data.get('throat_id')
                except Exception:
                    t_id = None
                
                if t_id:
                    in_Kc = t_id in throat_Kc
                    in_Q = t_id in throat_flows
                    in_accessible = t_id in solute_accessible_throat_ids
                    
                    print(f"          邻居 {nb}, throat_id={t_id}:")
                    print(f"            在solute_accessible_throat_ids中: {in_accessible}")
                    print(f"            在throat_Kc中: {in_Kc}")
                    print(f"            在throat_flows中: {in_Q}")
                    
                    if in_Q and in_Kc:
                        Q_ij_dbg = _signed_Q_node_to_neighbor(int(t_id), int(node), int(nb))
                        Q_mag_dbg = abs(float(Q_ij_dbg))
                        Kc_val = throat_Kc[t_id]
                        print(
                            f"            Q(p1→p2)={float(throat_flows[t_id]):.3e}, "
                            f"Q(node→nb)={Q_ij_dbg:.3e} m^3/s, Kc = {Kc_val:.3f}"
                        )
                        
                        # 检查流动方向（与组装一致：Q_ij<0 表示 neighbor→node）
                        p_tie_dbg = False
                        if node in node_pressures and nb in node_pressures:
                            p_tie_dbg, dp_nb = _pressure_tie_neighbor_to_node(nb, node)
                            flow_from_neighbor = (not p_tie_dbg) and (Q_ij_dbg < 0.0)
                            if p_tie_dbg:
                                print("            流动方向: 平压/极小ΔP（无压力驱动对流，仅扩散若有）")
                            else:
                                print(
                                    f"            流动方向: {'从neighbor流向node' if flow_from_neighbor else '从node流向neighbor'}"
                                )
                        else:
                            flow_from_neighbor = (nb in solute_entrance_nodes)
                            print(f"            流动方向: {'从neighbor流向node (假设)' if flow_from_neighbor else '从node流向neighbor (假设)'}")
                        Q_pos_dbg = -float(Q_ij_dbg) if flow_from_neighbor else float(Q_ij_dbg)
                        
                        if abs(Q_ij_dbg) <= 1e-20:
                            tiny_flow_hits += 1
                        
                        # 计算应添加到矩阵的解析耦合项贡献（与当前组装口径一致）
                        diff_coeff_dbg = 0.0
                        r_i = throat_radii[throat_ids == t_id][0]
                        U_mag_dbg = _throat_new_j_adv_coeff_m3s(Kc_val, Q_mag_dbg, r_i)
                        U_pos_dbg = _throat_new_j_adv_coeff_m3s(Kc_val, Q_pos_dbg, r_i)
                        if INCLUDE_DIFFUSION and t_id in throat_D_eff:
                            D_eff_val = throat_D_eff[t_id]
                            L_i = throat_lengths.get(t_id, 0.0)
                            if D_eff_val > 0 and r_i > 0 and L_i > 0:
                                diff_coeff_dbg = _throat_diff_coeff_m3s(r_i, L_i, D_eff_val)
                                print(f"            D_eff = {D_eff_val:.3e} m^2/s, diff_coeff = {diff_coeff_dbg:.3e} m^3/s")

                        _reg_dbg = _throat_pe_regime(1.0, U_mag_dbg, diff_coeff_dbg)
                        if p_tie_dbg:
                            if INCLUDE_DIFFUSION and diff_coeff_dbg > 0:
                                c_ii, c_ij = _throat_fick_row_downstream(diff_coeff_dbg)
                                print(
                                    f"            平压喉（与组装一致）Fick: c_ii={c_ii:.3e}, c_ij={c_ij:.3e}"
                                )
                                total_expected_contrib += abs(c_ii) + abs(c_ij)
                            else:
                                print("            平压喉且无有效扩散: 无对流通量项")
                        elif (
                            _reg_dbg == "coupled"
                            and INCLUDE_DIFFUSION
                            and diff_coeff_dbg > 0
                        ):
                            if flow_from_neighbor:
                                c_ii, c_ij = _throat_advection_diffusion_coeffs_downstream_row(1.0, U_pos_dbg, diff_coeff_dbg)
                            else:
                                c_ii, c_ij = _throat_advection_diffusion_coeffs_upstream_row(1.0, U_pos_dbg, diff_coeff_dbg)
                            print(f"            解析耦合项: c_ii={c_ii:.3e}, c_ij={c_ij:.3e}")
                            total_expected_contrib += abs(c_ii) + abs(c_ij)
                        elif _reg_dbg == "diff_only" and INCLUDE_DIFFUSION and diff_coeff_dbg > 0:
                            if flow_from_neighbor:
                                c_ii, c_ij = _throat_fick_row_downstream(diff_coeff_dbg)
                            else:
                                c_ii, c_ij = _throat_fick_row_upstream(diff_coeff_dbg)
                            print(
                                "            Fick(|Pe|<1e-14): "
                                f"c_ii={c_ii:.3e}, c_ij={c_ij:.3e}"
                            )
                            total_expected_contrib += abs(c_ii) + abs(c_ij)
                        else:
                            if flow_from_neighbor:
                                c_ii = 0.0
                                c_ij = U_mag_dbg
                            else:
                                c_ii = -U_mag_dbg
                                c_ij = 0.0
                            print(
                                "            纯对流(仅无扩散或扩散系数无效): "
                                f"A[i,i] += {c_ii:.3e}, A[i,j] += {c_ij:.3e}"
                            )
                            total_expected_contrib += abs(c_ii) + abs(c_ij)
                    else:
                        print(f"            警告: 缺少Kc或Q (in_Kc={in_Kc}, in_Q={in_Q})")
                else:
                    print(f"          邻居 {nb}: 无法找到throat_id")
            
            print(f"\n        预期总贡献: {total_expected_contrib:.3e}")
            print(f"        实际行和: {row_sum:.3e}")
            # 判断是否为零行（使用与上面相同的阈值）
            is_zero_row = row_sum < zero_row_threshold
            if is_zero_row:
                if total_expected_contrib > 1e-20:
                    print(f"        警告: 预期有贡献 ({total_expected_contrib:.3e}) 但实际行和 ({row_sum:.3e}) 小于阈值 ({zero_row_threshold:.0e})")
                    print(f"        可能原因: 数值精度问题、转换因子过小、或代码逻辑问题")
                else:
                    if tiny_flow_hits > 0:
                        print(f"        原因: 流量过小主导（tiny_flow_hits={tiny_flow_hits}）")
                    else:
                        print(f"        原因: 预期贡献也很小 ({total_expected_contrib:.3e})，可能是流量和扩散系数都很小")
            else:
                print(f"        说明: 虽然贡献很小，但行和 ({row_sum:.3e}) 大于阈值 ({zero_row_threshold:.0e})，不是零行")
        else:
            print(f"        原因: 节点在图中没有邻居（孤立节点）")
        
        # 检查节点的半径
        if node in pore_id_to_idx:
            r_pore = pore_radii[pore_id_to_idx[node]]
            print(f"        孔半径: {r_pore:.2f} nm")
            if r_pore <= SOLUTE_PASSAGE_GEOMETRY_MIN_NM:
                print(
                    f"        警告: 孔半径 ≤ 通路下限 ({SOLUTE_PASSAGE_GEOMETRY_MIN_NM:.4g} nm=r_s)，"
                    "不应该在溶质网络中！"
                )
    
    print(f"\n    零行节点列表（前10个）: {[solute_nodes[i] for i in zero_rows[:10]]}")


_new_j_stab = float(_args.new_j_stabilization)
if (not OLD_J_ONLY) and _new_j_stab > 0.0:
    np.fill_diagonal(A_conc_new, np.diag(A_conc_new) + _new_j_stab)
    print(f"    [new_J] 对角稳定化 A += {_new_j_stab:.3e}·I（仅用于求解，秩诊断基于未稳定矩阵）")

_old_j_stab = float(getattr(_args, "old_j_stabilization", 0.0))
if _old_j_stab > 0.0:
    np.fill_diagonal(A_conc_old, np.diag(A_conc_old) + _old_j_stab)
    print(f"    [old_J] 对角稳定化 A += {_old_j_stab:.3e}·I（仅用于求解）")



def _residual_metrics(A_sp: csr_matrix, b: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    r = np.asarray(A_sp @ x - b, dtype=np.float64).ravel()
    r2 = float(np.linalg.norm(r, ord=2))
    b2 = float(np.linalg.norm(np.asarray(b, dtype=np.float64).ravel(), ord=2))
    rel = r2 / (b2 + 1e-30)
    rmax = float(np.max(np.abs(r))) if r.size > 0 else 0.0
    return r2, rel, rmax


def _csr_row_max_abs(A: csr_matrix) -> np.ndarray:
    """每行非零元绝对值的最大值；全空行记为 0。"""
    A = A.tocsr()
    n = int(A.shape[0])
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo, hi = int(A.indptr[i]), int(A.indptr[i + 1])
        if lo < hi:
            out[i] = float(np.max(np.abs(A.data[lo:hi])))
    return out


def _csr_col_max_abs(A: csr_matrix) -> np.ndarray:
    """每列非零元绝对值的最大值；全空列记为 0。"""
    A = A.tocsc()
    n = int(A.shape[1])
    out = np.zeros(n, dtype=np.float64)
    for j in range(n):
        lo, hi = int(A.indptr[j]), int(A.indptr[j + 1])
        if lo < hi:
            out[j] = float(np.max(np.abs(A.data[lo:hi])))
    return out


def _equilibrate_conc_csr(
    A_sp: csr_matrix,
    b: np.ndarray,
    *,
    mode: str,
) -> tuple[csr_matrix, np.ndarray, object, object]:
    """
    将 Ax=b 变为数值上更均衡的系统等价形式。
    mode=row: D_r A x = D_r b，x 不变。
    mode=row-col: D_r A D_c z = D_r b，x = D_c z。
    返回 (A_eq, b_eq, recover, to_z)，recover(z) 映回原 x；to_z(x_phys) 将物理初值换到 z 空间（供 GMRES x0）。
    """
    b1 = np.asarray(b, dtype=np.float64).ravel()
    n = int(A_sp.shape[0])
    id_x = lambda v: np.asarray(v, dtype=np.float64).ravel()
    if mode not in ("row", "row-col"):
        return A_sp, b1, id_x, id_x

    row_max = _csr_row_max_abs(A_sp)
    row_max = np.where(row_max > 0.0, row_max, 1.0)
    Dr = 1.0 / row_max
    Dr_mat = sparse_diags(Dr, 0, shape=(n, n), format="csr")
    A_r = Dr_mat @ A_sp
    b_r = Dr * b1

    if mode == "row":
        return A_r, b_r, id_x, id_x

    col_max = _csr_col_max_abs(A_r)
    col_max = np.where(col_max > 0.0, col_max, 1.0)
    Dc = 1.0 / col_max
    Dc_mat = sparse_diags(Dc, 0, shape=(n, n), format="csr")
    A_eq = A_r @ Dc_mat

    def _recover(z: np.ndarray) -> np.ndarray:
        zv = np.asarray(z, dtype=np.float64).ravel()
        return Dc * zv

    def _to_z(x_phys: np.ndarray) -> np.ndarray:
        xv = np.asarray(x_phys, dtype=np.float64).ravel()
        return xv / Dc

    return A_eq, b_r, _recover, _to_z


def _log_conc_equilibrate_once() -> None:
    global _CONC_EQUILIBRATE_LOGGED
    if CONC_EQUILIBRATE_MODE == "none" or _CONC_EQUILIBRATE_LOGGED:
        return
    _CONC_EQUILIBRATE_LOGGED = True
    print(
        f"  浓度线性方程：--conc-equilibrate={CONC_EQUILIBRATE_MODE} "
        "（仅影响 spsolve/gmres/clip-refine 内求解，不改变物理组装）。"
    )


def _solve_concentration_direct(A_sp: csr_matrix, b: np.ndarray, label: str) -> np.ndarray | None:
    _log_conc_equilibrate_once()
    try:
        A_eq, b_eq, recover, _ = _equilibrate_conc_csr(A_sp, b, mode=CONC_EQUILIBRATE_MODE)
        z = spsolve(A_eq, b_eq)
        x = recover(z)
        if not np.all(np.isfinite(x)):
            print(f"    [{label}] [direct] spsolve 得到非有限值（NaN/Inf）。")
            return None
        return np.asarray(x, dtype=np.float64).ravel()
    except Exception as e:
        print(f"    [{label}] [direct] 求解失败: {e}")
        return None


def _solve_concentration_gmres_ilut(A_sp: csr_matrix, b: np.ndarray, label: str) -> np.ndarray | None:
    _log_conc_equilibrate_once()
    try:
        A_eq, b_eq, recover, _ = _equilibrate_conc_csr(A_sp, b, mode=CONC_EQUILIBRATE_MODE)
        ilu = spilu(
            csc_matrix(A_eq),
            drop_tol=max(0.0, float(ILUT_DROP_TOL)),
            fill_factor=max(1.0, float(ILUT_FILL_FACTOR)),
        )
        M = LinearOperator(A_eq.shape, matvec=ilu.solve)
    except Exception as e:
        print(f"    [{label}] [gmres-ilut] ILUT 预条件构建失败: {e}")
        return None
    try:
        z, info = gmres(
            A_eq,
            b_eq,
            M=M,
            rtol=max(0.0, float(GMRES_RTOL)),
            atol=max(0.0, float(GMRES_ATOL)),
            maxiter=max(1, int(GMRES_MAXITER)),
        )
        x = recover(z)
        if not np.all(np.isfinite(x)):
            print(f"    [{label}] [gmres-ilut] 得到非有限值（NaN/Inf）。")
            return None
        if info != 0:
            print(f"    [{label}] [gmres-ilut] 未在给定迭代内完全收敛（info={info}）。")
        return np.asarray(x, dtype=np.float64).ravel()
    except Exception as e:
        print(f"    [{label}] [gmres-ilut] 求解失败: {e}")
        return None


def _clip_refine_gmres_initial(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    clip-refine 每轮 GMRES 初值：越界分量不压到盒边界，而是对调到另一端
    （x<lo→hi，x>hi→lo；[lo,hi] 内不变）。默认 lo=0,hi=1 时：负浓度→1，>1→0。
    """
    x0 = np.asarray(x, dtype=np.float64).ravel().copy()
    lo, hi = float(lo), float(hi)
    x0[x0 < lo] = hi
    x0[x0 > hi] = lo
    return x0


def _oob_mask_concentration(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """逐分量是否严格越出 [lo, hi]（用于 clip-refine 与停滞判定）。"""
    xv = np.asarray(x, dtype=np.float64).ravel()
    return (xv < float(lo)) | (xv > float(hi))


def _log_clip_refine_final_concentration_range(x: np.ndarray, label: str) -> None:
    """clip-refine 结束返回解前打印浓度 min/max（无量纲 C 与 direct/gmres 日志一致）。"""
    xv = np.asarray(x, dtype=np.float64).ravel()
    if xv.size == 0:
        print(f"    [{label}] [clip-refine] 最终浓度范围：无分量。")
        return
    if not np.all(np.isfinite(xv)):
        print(f"    [{label}] [clip-refine] 最终浓度范围：含非有限分量，无法给出 min/max。")
        return
    print(
        f"    [{label}] [clip-refine] 最终浓度范围：[{float(xv.min()):.6f}, {float(xv.max()):.6f}]"
    )


def _solve_concentration_clip_refine(
    A_sp: csr_matrix,
    b_vec: np.ndarray,
    x_direct: np.ndarray,
    label: str,
    *,
    x_seed: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    direct 解已得且存在越界：每轮 GMRES 初值由 _clip_refine_gmres_initial 给出（x<lo→hi，x>hi→lo），
    GMRES+ILUT 迭代 Ax=b；外循环重复直至全在界内、或连续两轮越界模式相同、或达上限。

    若提供 x_seed（如 C_bulk 最终场先做的盒约束 LSQ），则以外循环起点 x=x_seed 开始，并至少执行一轮
    GMRES（即便当前已在盒内），以便在盒内初值上继续逼近 Ax=b。
    """
    lo, hi = float(CLIP_REFINE_LO), float(CLIP_REFINE_HI)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        print(f"    [{label}] [clip-refine] 无效盒 [{lo},{hi}]，跳过。")
        return None
    _log_conc_equilibrate_once()
    try:
        A_eq, b_eq, recover, to_z = _equilibrate_conc_csr(
            A_sp, b_vec, mode=CONC_EQUILIBRATE_MODE
        )
        ilu = spilu(
            csc_matrix(A_eq),
            drop_tol=max(0.0, float(ILUT_DROP_TOL)),
            fill_factor=max(1.0, float(ILUT_FILL_FACTOR)),
        )
        M = LinearOperator(A_eq.shape, matvec=ilu.solve)
    except Exception as e:
        print(f"    [{label}] [clip-refine] ILUT 失败: {e}")
        return None

    x = (
        np.asarray(x_seed, dtype=np.float64).ravel()
        if x_seed is not None
        else np.asarray(x_direct, dtype=np.float64).ravel()
    )
    first_gmres_pending = x_seed is not None
    viol_prev: np.ndarray | None = None
    for outer in range(CLIP_REFINE_MAX_OUTER):
        viol = _oob_mask_concentration(x, lo, hi)
        if not np.any(viol):
            r2, rel, rmax = _residual_metrics(A_sp, b_vec, x)
            if not first_gmres_pending:
                print(
                    f"    [{label}] [clip-refine] 外轮 {outer + 1}/{CLIP_REFINE_MAX_OUTER}："
                    f"已全部落在 [{lo:g},{hi:g}]；||Ax-b||_2={r2:.6e}, rel={rel:.6e}, max|res|={rmax:.6e}"
                )
                _log_clip_refine_final_concentration_range(x, label)
                return x
        elif viol_prev is not None and viol.shape == viol_prev.shape and bool(np.array_equal(viol, viol_prev)):
            r2, rel, rmax = _residual_metrics(A_sp, b_vec, x)
            print(
                f"    [{label}] [clip-refine] 越界模式与上一轮相同（仍越界 n={int(np.sum(viol))}），"
                f"停止外循环；||Ax-b||_2={r2:.6e}, rel={rel:.6e}"
            )
            _log_clip_refine_final_concentration_range(x, label)
            return x
        viol_prev = viol.copy()
        x0_phys = _clip_refine_gmres_initial(x, lo, hi)
        z0 = to_z(x0_phys)
        try:
            z_new, info = gmres(
                A_eq,
                b_eq,
                x0=z0,
                M=M,
                rtol=max(0.0, float(GMRES_RTOL)),
                atol=max(0.0, float(GMRES_ATOL)),
                maxiter=max(1, int(CLIP_REFINE_INNER_MAXITER)),
            )
            x_new = recover(z_new)
        except Exception as e:
            print(f"    [{label}] [clip-refine] 外轮 {outer + 1} GMRES 失败: {e}")
            _log_clip_refine_final_concentration_range(x, label)
            return x
        if not np.all(np.isfinite(x_new)):
            print(f"    [{label}] [clip-refine] 外轮 {outer + 1} 得到非有限值，保留上轮解。")
            _log_clip_refine_final_concentration_range(x, label)
            return x
        x = np.asarray(x_new, dtype=np.float64).ravel()
        r2, rel, rmax = _residual_metrics(A_sp, b_vec, x)
        n_bad = int(np.sum(_oob_mask_concentration(x, lo, hi)))
        print(
            f"    [{label}] [clip-refine] 外轮 {outer + 1}/{CLIP_REFINE_MAX_OUTER}："
            f"GMRES info={info}, 越界分量={n_bad}, x∈[{x.min():.6f},{x.max():.6f}], "
            f"||Ax-b||_2={r2:.6e}, rel={rel:.6e}"
        )
        first_gmres_pending = False

    r2, rel, rmax = _residual_metrics(A_sp, b_vec, x)
    print(
        f"    [{label}] [clip-refine] 已达外循环上限 {CLIP_REFINE_MAX_OUTER}；"
        f"仍越界 n={int(np.sum(_oob_mask_concentration(x, lo, hi)))}；"
        f"||Ax-b||_2={r2:.6e}, rel={rel:.6e}"
    )
    _log_clip_refine_final_concentration_range(x, label)
    return x


def _solve_concentration_linear(
    A: np.ndarray,
    b: np.ndarray,
    label: str,
    *,
    clip_refine_x_seed: np.ndarray | None = None,
    skip_clip_refine: bool = False,
    solver_override: str | None = None,
    compare_override: bool | None = None,
):
    """线性浓度求解：direct / gmres-ilut；可选 --concentration-clip-refine 越界后 clip 初值 + GMRES 外循环。

    clip_refine_x_seed 仅应由 C_bulk 最终场路径传入：先盒约束 LSQ 得初值，再进入 clip-refine（见 _finalize_cbulk_concentration_field）。
    skip_clip_refine=True：仅用于 C_bulk=0/1 参数边界解（为求 J0、J1），不进入 clip-refine 外轮，仍按 LINEAR_SOLVER 在 direct 与 gmres-ilut 间选解。
    """
    A_sp = csr_matrix(A)
    b_vec = np.asarray(b, dtype=np.float64).ravel()

    solver_pick = str(solver_override or LINEAR_SOLVER).strip().lower()
    if solver_pick not in ("direct", "gmres-ilut"):
        solver_pick = LINEAR_SOLVER
    compare_flag = COMPARE_LINEAR_SOLVERS if compare_override is None else bool(compare_override)

    x_direct: np.ndarray | None = None
    x_gmres: np.ndarray | None = None
    need_direct = (
        compare_flag
        or solver_pick == "direct"
        or CLIP_REFINE_ENABLED
        or skip_clip_refine
    )
    need_gmres = compare_flag or solver_pick == "gmres-ilut"

    if need_direct:
        x_direct = _solve_concentration_direct(A_sp, b_vec, label)
        if x_direct is not None:
            r2, rel, rmax = _residual_metrics(A_sp, b_vec, x_direct)
            print(
                f"    [{label}] [direct] 成功：x∈[{x_direct.min():.6f},{x_direct.max():.6f}], "
                f"||Ax-b||_2={r2:.6e}, 相对残差={rel:.6e}, max|Ax-b|={rmax:.6e}"
            )
    if need_gmres:
        x_gmres = _solve_concentration_gmres_ilut(A_sp, b_vec, label)
        if x_gmres is not None:
            r2, rel, rmax = _residual_metrics(A_sp, b_vec, x_gmres)
            print(
                f"    [{label}] [gmres-ilut] 成功：x∈[{x_gmres.min():.6f},{x_gmres.max():.6f}], "
                f"||Ax-b||_2={r2:.6e}, 相对残差={rel:.6e}, max|Ax-b|={rmax:.6e}, "
                f"rtol={GMRES_RTOL:.1e}, atol={GMRES_ATOL:.1e}, maxiter={GMRES_MAXITER}, "
                f"drop_tol={ILUT_DROP_TOL:.1e}, fill_factor={ILUT_FILL_FACTOR:.1f}"
            )

    if x_direct is not None and x_gmres is not None:
        d2 = float(np.linalg.norm(x_gmres - x_direct, ord=2))
        d_inf = float(np.linalg.norm(x_gmres - x_direct, ord=np.inf))
        base = float(np.linalg.norm(x_direct, ord=2)) + 1e-30
        print(
            f"    [{label}] [solver-compare] ||x_gmres-x_direct||_2={d2:.6e}, "
            f"相对={d2/base:.6e}, ||·||_inf={d_inf:.6e}"
        )

    x_pick: np.ndarray | None = None
    picked = ""

    if skip_clip_refine and CLIP_REFINE_ENABLED:
        print(
            f"    [{label}] [clip-refine] 本步跳过外轮（Cbulk=0/1 仅用于 J 插值，"
            f"按 solver={solver_pick} 在 direct/gmres 中择一）。"
        )

    if CLIP_REFINE_ENABLED and x_direct is not None and not skip_clip_refine:
        use_cbulk_seed = clip_refine_x_seed is not None
        print(
            f"    [{label}] [clip-refine] 已启用：盒 [{CLIP_REFINE_LO:g},{CLIP_REFINE_HI:g}]，"
            f"GMRES 初值对调越界端（x<lo→hi，x>hi→lo）"
            + ("；最终场已用 box LSQ 作外循环起点" if use_cbulk_seed else "")
            + f"，外循环≤{CLIP_REFINE_MAX_OUTER}，内 GMRES maxiter={CLIP_REFINE_INNER_MAXITER}。"
        )
        viol0 = _oob_mask_concentration(x_direct, CLIP_REFINE_LO, CLIP_REFINE_HI)
        if np.any(viol0) or use_cbulk_seed:
            if use_cbulk_seed:
                print(
                    f"    [{label}] [clip-refine] C_bulk 最终场：以盒约束 LSQ 为初值进入 clip→GMRES 外循环"
                    f"（direct 越界分量数={int(np.sum(viol0))}）。"
                )
            else:
                n0 = int(np.sum(viol0))
                print(f"    [{label}] [clip-refine] direct 解越界分量数={n0}，进入 clip→GMRES 外循环。")
            x_cr = _solve_concentration_clip_refine(
                A_sp,
                b_vec,
                x_direct,
                label,
                x_seed=clip_refine_x_seed,
            )
            if x_cr is not None:
                x_pick = x_cr
                picked = "clip-refine(gmres+ilut)"
            else:
                print(f"    [{label}] [clip-refine] 未得到有效改进，回退为常规线性解选择。")
        elif solver_pick == "direct":
            x_pick = x_direct
            picked = "direct(in-box)"

    if x_pick is None:
        if solver_pick == "gmres-ilut":
            x_pick = x_gmres if x_gmres is not None else x_direct
            picked = "gmres-ilut" if x_gmres is not None else "direct(fallback)"
        else:
            x_pick = x_direct if x_direct is not None else x_gmres
            picked = "direct" if x_direct is not None else "gmres-ilut(fallback)"

    if x_pick is None:
        print(f"    [{label}] 线性求解失败：direct 与 gmres-ilut 均不可用。")
        return None
    print(f"    [{label}] 采用线性解: {picked}")
    return x_pick


def _solve_concentration_box_lsq(
    A: np.ndarray,
    b: np.ndarray,
    lo: float,
    hi: float,
    label: str,
    ridge: float = 0.0,
) -> np.ndarray | None:
    """
    盒约束线性最小二乘：min ||Ax - b||_2 s.t. lo <= x <= hi（逐分量）。
    可选岭项 ridge=λ>0：min ||Ax-b||²+λ||x-x_ref||²，x_ref=clip(spsolve(A,b))，
    在秩亏多解时偏向无约束解在盒内的投影，减轻「残差≈0 但通量全为 0」的退化解。
    """
    n = A.shape[0]
    if n == 0:
        return None
    if not (lo < hi) or not np.isfinite(lo) or not np.isfinite(hi):
        print(f"    [{label}] 盒约束需有限标量且 lo<hi，当前无效，跳过盒约束 LSQ。")
        return None
    try:
        A_orig_sp = csr_matrix(A)
        b_orig = np.asarray(b, dtype=np.float64).ravel()
        if b_orig.shape[0] != n:
            print(f"    [{label}] b 长度与 A 不匹配。")
            return None
        x_ref = np.full(n, 0.5 * (lo + hi), dtype=np.float64)
        x0 = None
        try:
            xc = spsolve(A_orig_sp, b_orig)
            if np.all(np.isfinite(xc)):
                x_ref = np.clip(xc, lo, hi)
                x0 = x_ref.copy()
        except Exception:
            pass
        A_sp = A_orig_sp
        b_1d = b_orig
        lam = float(ridge)
        if lam > 0.0 and np.isfinite(lam):
            s = np.sqrt(lam)
            Iw = sparse_diags(
                np.full(n, s, dtype=np.float64),
                0,
                shape=(n, n),
                format="csr",
            )
            A_sp = sparse_vstack((A_orig_sp, Iw))
            b_1d = np.concatenate((b_orig, s * x_ref))
        _lsq_kw: dict = {
            "bounds": (
                np.full(n, lo, dtype=np.float64),
                np.full(n, hi, dtype=np.float64),
            ),
            "method": "trf",
            "max_iter": max(500, 20 * n),
            "verbose": 0,
        }
        if x0 is not None and "x0" in inspect.signature(lsq_linear).parameters:
            _lsq_kw["x0"] = x0
        res = lsq_linear(A_sp, b_1d, **_lsq_kw)
        x = res.x
        if not np.all(np.isfinite(x)):
            print(f"    [{label}] lsq_linear 得到非有限值（NaN/Inf）。")
            return None
        r_core = float(np.linalg.norm(A_orig_sp @ x - b_orig))
        if lam > 0.0 and np.isfinite(lam):
            r_aug = float(np.linalg.norm(A_sp @ x - b_1d))
            extra = (
                f"，增广残差||·||_2={r_aug:.6e}，岭 λ={lam:.3e}（偏向 clip(spsolve) 参考）"
            )
        else:
            extra = ""
        if res.success:
            print(
                f"    [{label}] 盒约束 LSQ 成功：||Ax-b||_2={r_core:.6e}{extra}，"
                f"x∈[{float(x.min()):.6f},{float(x.max()):.6f}]（盒 [{lo:g},{hi:g}]）"
            )
        else:
            print(
                f"    [{label}] 盒约束 LSQ 未收敛（success=False），采用当前迭代点；"
                f"||Ax-b||_2={r_core:.6e}{extra}，x∈[{float(x.min()):.6f},{float(x.max()):.6f}]"
            )
        return x
    except Exception as e:
        print(f"    [{label}] 盒约束 LSQ 失败: {e}")
        return None


def _compute_q_total_exit_solvent_on_solute_network() -> float:
    q_sum = 0.0
    seen_tids: set[int] = set()
    exit_set = set(solute_exit_nodes)
    for ex in solute_exit_nodes:
        if ex not in G_solute_accessible:
            continue
        for nb in G_solute_accessible.neighbors(ex):
            if nb in exit_set:
                continue
            try:
                edge_data = G_solute_accessible[ex][nb]
                t_id = edge_data.get("throat_id")
            except Exception:
                t_id = None
            if not t_id:
                continue
            tid_i = int(t_id)
            if tid_i in seen_tids:
                continue
            seen_tids.add(tid_i)
            if t_id in throat_flows:
                q_sum += -float(_signed_Q_node_to_neighbor(tid_i, int(ex), int(nb)))
    return float(q_sum)


def _compute_q_total_exit_solvent_on_full_network() -> float:
    """
    全溶剂网络出口净流出通量（m³/s）：以 G_solvent 的出口孔为基准累计
    neighbor -> exit 为正、exit -> neighbor 为负。
    用于 C_bulk 闭合 J = Q_full * C_bulk，避免把对溶质不可通行的溶剂通路漏掉。
    """
    q_sum = 0.0
    seen_tids: set[int] = set()
    exit_set = set(exit_pore_ids)
    for exit_node in exit_pore_ids:
        if exit_node not in G_solvent:
            continue
        for neighbor in G_solvent.neighbors(exit_node):
            if neighbor in exit_set:
                continue
            edge_data = G_solvent[exit_node].get(neighbor, None)
            if not edge_data:
                continue
            t_id = edge_data.get("throat_id")
            if t_id and t_id in throat_flows:
                tid_i = int(t_id)
                if tid_i in seen_tids:
                    continue
                seen_tids.add(tid_i)
                q_sum += -float(_signed_Q_node_to_neighbor(tid_i, int(exit_node), int(neighbor)))
    return float(q_sum)


def _compute_exit_solute_flux_from_map(C_map: dict[int, float], *, use_new_scheme: bool) -> float:
    q_alb_sum = 0.0
    for exit_node in solute_exit_nodes:
        if exit_node not in G_solute_accessible:
            continue
        for neighbor in G_solute_accessible.neighbors(exit_node):
            try:
                edge_data = G_solute_accessible[exit_node][neighbor]
                t_id = edge_data.get("throat_id")
            except KeyError:
                try:
                    edge_data = G_solute_accessible[neighbor][exit_node]
                    t_id = edge_data.get("throat_id")
                except KeyError:
                    t_id = None
            if not t_id:
                continue
            if t_id not in throat_Kc or t_id not in throat_flows:
                continue
            Kc = throat_Kc[t_id]
            Q_exit_nb = _signed_Q_node_to_neighbor(int(t_id), int(exit_node), int(neighbor))
            Q = abs(float(Q_exit_nb))
            C_upstream_pore = float(C_map.get(neighbor, 0.0))
            C_exit_pore = float(C_map.get(exit_node, 0.0))
            C_upstream_throat = C_upstream_pore
            C_exit_throat = C_exit_pore
            r_i = throat_radii[throat_ids == t_id][0]
            U_abs = _throat_new_j_adv_coeff_m3s(Kc, Q, r_i)
            diff_coeff = 0.0
            if INCLUDE_DIFFUSION:
                L_i = throat_lengths.get(t_id, 0.0)
                D_eff = throat_D_eff.get(t_id, 0.0)
                if D_eff > 0 and r_i > 0 and L_i > 0:
                    diff_coeff = _throat_diff_coeff_m3s(r_i, L_i, D_eff)
            pressure_tie_exit, dp_ex = _pressure_tie_neighbor_to_exit(neighbor, exit_node)
            if exit_node in node_pressures and neighbor in node_pressures:
                flow_nb_to_exit = (not pressure_tie_exit) and (Q_exit_nb < 0.0)
            else:
                flow_nb_to_exit = True
            Q_pos_exit = -float(Q_exit_nb) if flow_nb_to_exit else float(Q_exit_nb)
            if pressure_tie_exit:
                if INCLUDE_DIFFUSION and diff_coeff > 0:
                    J_edge = diff_coeff * (C_upstream_throat - C_exit_throat)
                else:
                    J_edge = 0.0
                q_alb_sum += float(J_edge)
                continue
            if use_new_scheme:
                _rf_exit = (
                    _throat_pe_regime(1.0, U_abs, diff_coeff)
                    if (INCLUDE_DIFFUSION and diff_coeff > 0)
                    else "adv_only"
                )
                if INCLUDE_DIFFUSION and diff_coeff > 0:
                    if flow_nb_to_exit:
                        if _rf_exit == "adv_only":
                            J_edge = U_abs * C_upstream_throat
                        elif _rf_exit == "diff_only":
                            J_edge = diff_coeff * (C_upstream_throat - C_exit_throat)
                        else:
                            U_pos_exit = _throat_new_j_adv_coeff_m3s(Kc, Q_pos_exit, r_i)
                            J_edge = _throat_coupled_flux_vol(1.0, U_pos_exit, diff_coeff, C_upstream_pore, C_exit_pore)
                    else:
                        if _rf_exit == "adv_only":
                            J_edge = -U_abs * C_exit_throat
                        elif _rf_exit == "diff_only":
                            J_edge = diff_coeff * (C_upstream_throat - C_exit_throat)
                        else:
                            U_pos_exit = _throat_new_j_adv_coeff_m3s(Kc, Q_pos_exit, r_i)
                            J_edge = -_throat_coupled_flux_vol(1.0, U_pos_exit, diff_coeff, C_exit_pore, C_upstream_pore)
                else:
                    J_edge = U_abs * (C_upstream_throat if flow_nb_to_exit else -C_exit_throat)
            else:
                c_adv_t = (
                    0.5 * (C_upstream_throat + C_exit_throat)
                    if OLD_J_MEAN_THROAT_CONV
                    else (C_upstream_throat if flow_nb_to_exit else C_exit_throat)
                )
                j_conv = (
                    float(Kc) * float(Q_exit_nb) * c_adv_t
                    if OLD_J_MEAN_THROAT_CONV
                    else float(Kc) * float(Q) * c_adv_t
                )
                # 与 A_conc_old 一致：|Pe| 大→adv_only 仅对流，不叠 Fick
                if INCLUDE_DIFFUSION and diff_coeff > 0:
                    _pe_old_ex = _throat_pe_regime(Kc, Q, diff_coeff)
                    if _pe_old_ex == "adv_only":
                        J_edge = j_conv
                    elif flow_nb_to_exit:
                        J_edge = j_conv + diff_coeff * (C_upstream_throat - C_exit_throat)
                    else:
                        J_edge = j_conv + diff_coeff * (C_exit_throat - C_upstream_throat)
                else:
                    J_edge = j_conv
            q_alb_sum += float(J_edge)
    return float(q_alb_sum)


def _finalize_cbulk_concentration_field(
    x_super: np.ndarray,
    A_eval: np.ndarray,
    b_eval: np.ndarray,
    *,
    label: str,
    use_box_lsq: bool,
    solver_override: str | None = None,
    compare_override: bool | None = None,
) -> np.ndarray:
    """
    C_bulk 已定且已由 x0+(x1-x0)*C_bulk 叠加得到 x_super 后，对最终 (A_eval,b_eval)：
    - new_J 且启用盒 LSQ：再做一次盒约束最小二乘（最终场落在 [new-j-c-min,max]）；
    - 否则若启用 clip-refine：先对最终系统做盒约束 LSQ（clip 盒，岭同 --new-j-box-ridge），
      再将该解作为 clip-refine 外循环起点，并走 _solve_concentration_linear（含 direct 与 clip→GMRES）；
    - 否则保持叠加解。
    任一步失败则回退 x_super。
    """
    xs = np.asarray(x_super, dtype=np.float64).ravel()
    if use_box_lsq:
        xf = _solve_concentration_box_lsq(
            A_eval,
            b_eval,
        float(_args.new_j_c_min),
        float(_args.new_j_c_max),
            f"{label}|Cbulk-final-field",
        ridge=float(_args.new_j_box_ridge),
    )
        if xf is not None:
            print(f"    [{label}] 最终浓度场：已对 C_bulk 边界系统做盒约束 LSQ（Cbulk-final-field）。")
            return np.asarray(xf, dtype=np.float64).ravel()
        print(f"    [{label}] 最终场盒约束 LSQ 失败，回退为 C_bulk 叠加解。")
        return xs
    if CLIP_REFINE_ENABLED:
        x_seed = _solve_concentration_box_lsq(
            A_eval,
            b_eval,
            float(CLIP_REFINE_LO),
            float(CLIP_REFINE_HI),
            f"{label}|Cbulk-final-field|box-lsq-seed",
            ridge=float(_args.new_j_box_ridge),
        )
        xf = _solve_concentration_linear(
            A_eval,
            b_eval,
            f"{label}|Cbulk-final-field",
            clip_refine_x_seed=x_seed,
            solver_override=solver_override,
            compare_override=compare_override,
        )
        if xf is not None:
            return np.asarray(xf, dtype=np.float64).ravel()
        print(f"    [{label}] 最终场 linear/clip-refine 失败，回退为 C_bulk 叠加解。")
        return xs
    return xs


def _solve_with_cbulk_parameter(
    A_base: np.ndarray,
    b_base: np.ndarray,
    *,
    label: str,
    use_new_scheme: bool,
    use_box_lsq: bool = False,
    solver_override: str | None = None,
    compare_override: bool | None = None,
) -> tuple[np.ndarray | None, float | None, np.ndarray | None, np.ndarray | None]:
    solver_pick = str(solver_override or LINEAR_SOLVER).strip().lower()
    if solver_pick not in ("direct", "gmres-ilut"):
        solver_pick = LINEAR_SOLVER
    compare_flag = COMPARE_LINEAR_SOLVERS if compare_override is None else bool(compare_override)

    if compare_flag:
        global _FULL_COMPARE_DIRECT_FINAL_R2
        global _FULL_COMPARE_DIRECT_FINAL_REL
        global _FULL_COMPARE_DIRECT_FINAL_RMAX
        global _FULL_COMPARE_GMRES_FINAL_R2
        global _FULL_COMPARE_GMRES_FINAL_REL
        global _FULL_COMPARE_GMRES_FINAL_RMAX
        global _FULL_COMPARE_FINAL_D2
        global _FULL_COMPARE_FINAL_D2_REL
        global _FULL_COMPARE_FINAL_DINF
        x_d, cb_d, A_d, b_d = _solve_with_cbulk_parameter(
            A_base,
            b_base,
            label=label,
            use_new_scheme=use_new_scheme,
            use_box_lsq=use_box_lsq,
            solver_override="direct",
            compare_override=False,
        )
        x_g, cb_g, A_g, b_g = _solve_with_cbulk_parameter(
            A_base,
            b_base,
            label=label,
            use_new_scheme=use_new_scheme,
            use_box_lsq=use_box_lsq,
            solver_override="gmres-ilut",
            compare_override=False,
        )
        if x_d is not None and A_d is not None and b_d is not None:
            rd2, rdrel, rdmax = _residual_metrics(csr_matrix(A_d), np.asarray(b_d, dtype=np.float64), np.asarray(x_d, dtype=np.float64))
            if str(label).strip() == "new_J":
                _FULL_COMPARE_DIRECT_FINAL_R2 = float(rd2)
                _FULL_COMPARE_DIRECT_FINAL_REL = float(rdrel)
                _FULL_COMPARE_DIRECT_FINAL_RMAX = float(rdmax)
            print(f"    [{label}] [full-compare][direct-final] ||Ax-b||_2={rd2:.6e}, 相对残差={rdrel:.6e}, max|Ax-b|={rdmax:.6e}")
        if x_g is not None and A_g is not None and b_g is not None:
            rg2, rgrel, rgmax = _residual_metrics(csr_matrix(A_g), np.asarray(b_g, dtype=np.float64), np.asarray(x_g, dtype=np.float64))
            if str(label).strip() == "new_J":
                _FULL_COMPARE_GMRES_FINAL_R2 = float(rg2)
                _FULL_COMPARE_GMRES_FINAL_REL = float(rgrel)
                _FULL_COMPARE_GMRES_FINAL_RMAX = float(rgmax)
            print(f"    [{label}] [full-compare][gmres-final] ||Ax-b||_2={rg2:.6e}, 相对残差={rgrel:.6e}, max|Ax-b|={rgmax:.6e}")
        if x_d is not None and x_g is not None:
            d2 = float(np.linalg.norm(np.asarray(x_g, dtype=np.float64) - np.asarray(x_d, dtype=np.float64), ord=2))
            d_inf = float(np.linalg.norm(np.asarray(x_g, dtype=np.float64) - np.asarray(x_d, dtype=np.float64), ord=np.inf))
            base = float(np.linalg.norm(np.asarray(x_d, dtype=np.float64), ord=2)) + 1e-30
            if str(label).strip() == "new_J":
                _FULL_COMPARE_FINAL_D2 = float(d2)
                _FULL_COMPARE_FINAL_D2_REL = float(d2 / base)
                _FULL_COMPARE_FINAL_DINF = float(d_inf)
            print(
                f"    [{label}] [full-compare] final ||x_gmres-x_direct||_2={d2:.6e}, 相对={d2/base:.6e}, ||·||_inf={d_inf:.6e}"
            )
        if solver_pick == "gmres-ilut":
            return x_g, cb_g, A_g, b_g
        return x_d, cb_d, A_d, b_d

    if len(_exit_dirichlet_rows) == 0:
        print(f"    [{label}] 无出口节点，回退单次线性求解。")
        x = _solve_concentration_linear(A_base, b_base, label, solver_override=solver_pick, compare_override=False)
        return x, None, A_base, b_base

    A0, b0, _, _ = _apply_boundary_rows_with_cbulk(A_base, b_base, 0.0, label=f"{label}|Cbulk=0")
    A1, b1, _, _ = _apply_boundary_rows_with_cbulk(A_base, b_base, 1.0, label=f"{label}|Cbulk=1")

    if use_box_lsq:
        x0 = _solve_concentration_box_lsq(
            A0, b0, float(_args.new_j_c_min), float(_args.new_j_c_max), f"{label}|Cbulk=0", ridge=float(_args.new_j_box_ridge)
        )
        x1 = _solve_concentration_box_lsq(
            A1, b1, float(_args.new_j_c_min), float(_args.new_j_c_max), f"{label}|Cbulk=1", ridge=float(_args.new_j_box_ridge)
        )
    else:
        x0 = _solve_concentration_linear(
            A0,
            b0,
            f"{label}|Cbulk=0",
            skip_clip_refine=True,
            solver_override=solver_pick,
            compare_override=False,
        )
        x1 = _solve_concentration_linear(
            A1,
            b1,
            f"{label}|Cbulk=1",
            skip_clip_refine=True,
            solver_override=solver_pick,
            compare_override=False,
        )
    if x0 is None or x1 is None:
        print(f"    [{label}] 参数边界双求解失败（Cbulk=0/1 至少一组失败）。")
        return None, None, None, None

    cmap0 = {solute_nodes[k]: float(x0[k]) for k in range(n_solute_nodes)}
    cmap1 = {solute_nodes[k]: float(x1[k]) for k in range(n_solute_nodes)}
    q_alb0 = _compute_exit_solute_flux_from_map(cmap0, use_new_scheme=use_new_scheme)
    q_alb1 = _compute_exit_solute_flux_from_map(cmap1, use_new_scheme=use_new_scheme)
    q_exit_solute = _compute_q_total_exit_solvent_on_solute_network()
    q_exit = _compute_q_total_exit_solvent_on_full_network()
    if np.isfinite(q_exit_solute) and np.isfinite(q_exit):
        print(
            f"    [{label}] C_bulk 闭合通量口径：Q_full={q_exit:.6e}（全溶剂网络），"
            f"Q_solute={q_exit_solute:.6e}（仅溶质通路，诊断）。"
        )
    denom = float(q_exit - (q_alb1 - q_alb0))
    if (not np.isfinite(denom)) or abs(denom) <= 1e-30:
        print(
            f"    [{label}] 计算 C_bulk 分母过小/非有限：denom={denom:.6e}。"
            " 采用 C_bulk=0 回退。"
        )
        c_bulk = 0.0
    else:
        c_bulk = float(q_alb0 / denom)
    if not np.isfinite(c_bulk):
        print(f"    [{label}] C_bulk 非有限，视为无解（本侧浓度场不可用）。")
        return None, None, None, None
    if c_bulk < -NEW_J_NEGATIVE_TOL:
        print(
            f"    [{label}] C_bulk={c_bulk:.6e} 低于负值阈值 -tol={-NEW_J_NEGATIVE_TOL:.6e}，"
            "视为无解（本侧浓度场不可用）。"
        )
        return None, None, None, None
    x_super = np.asarray(x0 + (x1 - x0) * c_bulk, dtype=np.float64)
    A_eval, b_eval, _, _ = _apply_boundary_rows_with_cbulk(A_base, b_base, c_bulk, label=f"{label}|Cbulk=final")
    x = _finalize_cbulk_concentration_field(
        x_super,
        A_eval,
        b_eval,
        label=label,
        use_box_lsq=use_box_lsq,
        solver_override=solver_pick,
        compare_override=False,
    )
    q_alb_final = _compute_exit_solute_flux_from_map(
        {solute_nodes[k]: float(x[k]) for k in range(n_solute_nodes)},
        use_new_scheme=use_new_scheme,
    )
    closure = float(q_alb_final - q_exit * c_bulk)
    print(
        f"    [{label}] C_bulk 求解：Q_alb0={q_alb0:.6e}, Q_alb1={q_alb1:.6e}, Q_exit={q_exit:.6e}, "
        f"denom={denom:.6e}, C_bulk={c_bulk:.6e}, "
        f"闭合误差 Q_alb-Qwater*Cbulk={closure:.6e}"
    )
    return x, c_bulk, A_eval, b_eval


A_eval_new = None
b_eval_new = None
A_eval_old = None
b_eval_old = None
_FULL_COMPARE_DIRECT_FINAL_R2 = float("nan")
_FULL_COMPARE_DIRECT_FINAL_REL = float("nan")
_FULL_COMPARE_DIRECT_FINAL_RMAX = float("nan")
_FULL_COMPARE_GMRES_FINAL_R2 = float("nan")
_FULL_COMPARE_GMRES_FINAL_REL = float("nan")
_FULL_COMPARE_GMRES_FINAL_RMAX = float("nan")
_FULL_COMPARE_FINAL_D2 = float("nan")
_FULL_COMPARE_FINAL_D2_REL = float("nan")
_FULL_COMPARE_FINAL_DINF = float("nan")
if SOLVE_NEW_J:
    conc_new, c_bulk_new, A_eval_new, b_eval_new = _solve_with_cbulk_parameter(
        A_conc_new,
        b_conc_new,
        label="new_J",
        use_new_scheme=True,
        use_box_lsq=bool(_args.new_j_box_lsq),
    )
else:
    conc_new = None
    c_bulk_new = None

if SOLVE_OLD_J:
    conc_old, c_bulk_old, A_eval_old, b_eval_old = _solve_with_cbulk_parameter(
        A_conc_old,
        b_conc_old,
        label="old_J",
        use_new_scheme=False,
        use_box_lsq=False,
    )
else:
    conc_old = None
    c_bulk_old = None


def _print_linear_system_residuals(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray | None,
    label: str,
    *,
    dirichlet_rows: list[int],
    exit_equal_rows: list[int],
    exit_node_rows: list[int],
    exit_up1_rows: list[int],
) -> None:
    """打印 Ax=b 残差与约束行残差（用于阶段 E 数值核对）。"""
    if x is None:
        return
    try:
        r = np.asarray(A @ x - b, dtype=np.float64).ravel()
        if r.size == 0:
            print(f"    [{label}] 线性系统残差: 空系统。")
            return
        r2 = float(np.linalg.norm(r, ord=2))
        b2 = float(np.linalg.norm(np.asarray(b, dtype=np.float64).ravel(), ord=2))
        rel = r2 / (b2 + 1e-30)
        rmax = float(np.max(np.abs(r)))
        print(
            f"    [{label}] 线性系统残差: ||Ax-b||_2={r2:.6e}, "
            f"相对残差={rel:.6e}, max|Ax-b|={rmax:.6e}"
        )
    except Exception as e:
        print(f"    [{label}] 残差计算失败: {e}")


def _exit_up1_rows() -> list[int]:
    """出口节点的一层上游（邻接）节点对应的行索引，排除入口与出口本身。"""
    exit_set = set(solute_exit_nodes)
    in_set = set(solute_entrance_nodes)
    rows: set[int] = set()
    for ex in solute_exit_nodes:
        if ex not in G_solute_accessible:
            continue
        for nb in G_solute_accessible.neighbors(ex):
            if nb in exit_set or nb in in_set:
                continue
            if nb in solute_node_to_idx:
                rows.add(int(solute_node_to_idx[nb]))
    return sorted(rows)


_exit_rows_now = [solute_node_to_idx[n] for n in solute_exit_nodes if n in solute_node_to_idx]
_exit_up1_rows_now = _exit_up1_rows()
if conc_new is not None:
    _print_linear_system_residuals(
        A_eval_new if A_eval_new is not None else A_conc_new,
        b_eval_new if b_eval_new is not None else b_conc_new,
        conc_new,
        "new_J",
        dirichlet_rows=dirichlet_row_indices,
        exit_equal_rows=exit_equality_row_indices_new,
        exit_node_rows=_exit_rows_now,
        exit_up1_rows=_exit_up1_rows_now,
    )
if conc_old is not None:
    _print_linear_system_residuals(
        A_eval_old if A_eval_old is not None else A_conc_old,
        b_eval_old if b_eval_old is not None else b_conc_old,
        conc_old,
        "old_J",
        dirichlet_rows=dirichlet_row_indices,
        exit_equal_rows=exit_equality_row_indices_old,
        exit_node_rows=_exit_rows_now,
        exit_up1_rows=_exit_up1_rows_now,
    )

node_concentrations_new = None
node_concentrations_old = None
if conc_new is not None:
    node_concentrations_new = {solute_nodes[k]: float(conc_new[k]) for k in range(n_solute_nodes)}
if conc_old is not None:
    node_concentrations_old = {solute_nodes[k]: float(conc_old[k]) for k in range(n_solute_nodes)}

NEW_J_INVALID_NEGATIVE = False
NEW_J_MIN_CONCENTRATION = np.nan
NEW_J_INVALID_REL_RESIDUAL = False
NEW_J_REL_RESIDUAL = np.nan
if conc_new is not None:
    try:
        _A_new_chk = csr_matrix(A_eval_new if A_eval_new is not None else A_conc_new)
        _b_new_chk = np.asarray(
            b_eval_new if b_eval_new is not None else b_conc_new,
            dtype=np.float64,
        ).ravel()
        _r2_new, _rel_new, _rmax_new = _residual_metrics(
            _A_new_chk, _b_new_chk, np.asarray(conc_new, dtype=np.float64).ravel()
        )
        NEW_J_REL_RESIDUAL = float(_rel_new)
        if (
            np.isfinite(NEW_J_REL_RESIDUAL)
            and NEW_J_REL_RESIDUAL_INVALID_THRESHOLD > 0.0
            and NEW_J_REL_RESIDUAL > NEW_J_REL_RESIDUAL_INVALID_THRESHOLD
        ):
            NEW_J_INVALID_REL_RESIDUAL = True
            print(
                "    [new_J] 有效性校验失败："
                f"相对残差={NEW_J_REL_RESIDUAL:.6e} > 阈值={NEW_J_REL_RESIDUAL_INVALID_THRESHOLD:.6e}。"
                "将把 new_J 关键筛分结果标记为 invalid 并写 NaN。"
            )
    except Exception as _e_relchk:
        print(f"    [new_J] 相对残差有效性校验失败（跳过该判据）: {_e_relchk}")

if node_concentrations_new is not None:
    _new_vals = np.asarray(
        [float(node_concentrations_new.get(n, np.nan)) for n in solute_nodes],
        dtype=np.float64,
    )
    _new_vals = _new_vals[np.isfinite(_new_vals)]
    if _new_vals.size > 0:
        NEW_J_MIN_CONCENTRATION = float(np.min(_new_vals))
        if NEW_J_MIN_CONCENTRATION < -NEW_J_NEGATIVE_TOL:
            NEW_J_INVALID_NEGATIVE = True
            print(
                "    [new_J] 有效性校验失败："
                f"min(C)={NEW_J_MIN_CONCENTRATION:.6e} < -tol={-NEW_J_NEGATIVE_TOL:.6e}。"
                "将把 new_J 关键筛分结果标记为 invalid 并写 NaN。"
            )
    else:
        NEW_J_MIN_CONCENTRATION = np.nan

NEW_J_INVALID = bool(NEW_J_INVALID_NEGATIVE or NEW_J_INVALID_REL_RESIDUAL)


def _has_inlet_outlet_connection_for_solute_graph(g: nx.Graph) -> bool:
    nodes = set(g.nodes)
    left = set(solute_entrance_nodes) & nodes
    right = set(solute_exit_nodes) & nodes
    if not left or not right:
        return False
    for comp in nx.connected_components(g):
        comp_set = set(comp)
        if comp_set & left and comp_set & right:
            return True
    return False


def _choose_oob_concentration_clusters_to_prune(C_map: dict[int, float] | None) -> tuple[set[int], pd.DataFrame]:
    if C_map is None or G_solute_accessible.number_of_nodes() == 0:
        return set(), pd.DataFrame()
    low_threshold = float(AUTO_PRUNE_OOB_LOW_THRESHOLD)
    high_threshold = float(AUTO_PRUNE_OOB_HIGH_THRESHOLD)
    boundary = set(solute_entrance_nodes) | set(solute_exit_nodes)
    abnormal_nodes: set[int] = set()
    for node, c in C_map.items():
        if node not in G_solute_accessible:
            continue
        try:
            cv = float(c)
        except Exception:
            abnormal_nodes.add(int(node))
            continue
        if is_abnormal_equivalent_concentration(
            cv,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        ):
            abnormal_nodes.add(int(node))
    if not abnormal_nodes:
        return set(), pd.DataFrame()

    remove_nodes: set[int] = set()
    rows: list[dict] = []
    induced = G_solute_accessible.subgraph(abnormal_nodes).copy()
    for cid, comp in enumerate(nx.connected_components(induced), start=1):
        comp_set = {int(x) for x in comp}
        touches_boundary = bool(comp_set & boundary)
        trial_remove = remove_nodes | comp_set
        g_trial = G_solute_accessible.copy()
        g_trial.remove_nodes_from(trial_remove)
        keeps_connection = _has_inlet_outlet_connection_for_solute_graph(g_trial)
        accepted = (not touches_boundary) and keeps_connection
        if accepted:
            remove_nodes.update(comp_set)
        vals = []
        for n in sorted(comp_set):
            try:
                vals.append(float(C_map.get(n, np.nan)))
            except Exception:
                vals.append(float("nan"))
        arr = np.asarray(vals, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        rows.append(
            {
                "cluster_id": int(cid),
                "size": int(len(comp_set)),
                "accepted": bool(accepted),
                "touches_inlet_or_outlet": bool(touches_boundary),
                "keeps_inlet_outlet_connection_after_cumulative_removal": bool(keeps_connection),
                "nodes": ",".join(str(x) for x in sorted(comp_set)),
                "min_C": float(np.min(finite)) if finite.size else np.nan,
                "max_C": float(np.max(finite)) if finite.size else np.nan,
                "max_abs_C": float(np.max(np.abs(finite))) if finite.size else np.nan,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
            }
        )
    return remove_nodes, pd.DataFrame(rows)


def _replace_or_append_cli_option(argv: list[str], option: str, value: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        item = argv[i]
        if item == option:
            i += 2
            continue
        if item.startswith(option + "="):
            i += 1
            continue
        out.append(item)
        i += 1
    out.extend([option, value])
    return out


def _auto_prune_oob_and_rerun_if_needed() -> None:
    if not AUTO_PRUNE_OOB_CONCENTRATION:
        return
    if AUTO_PRUNE_OOB_PASS >= AUTO_PRUNE_OOB_MAX_PASSES:
        return
    if node_concentrations_new is None:
        return
    remove_nodes, cluster_df = _choose_oob_concentration_clusters_to_prune(node_concentrations_new)
    if cluster_df is not None and not cluster_df.empty:
        audit_path = output_path / get_output_filename(
            f"{SIEVE_SAMPLE_NAME}_concentration_oob_auto_prune_clusters"
        )
        cluster_df.to_excel(audit_path, index=False)
        print(f"    [new_J auto-prune] Abnormal concentration cluster audit saved: {audit_path}")
    if not remove_nodes:
        return

    merged_exclude = set(SOLUTE_EXCLUDE_PORE_IDS) | {int(x) for x in remove_nodes}
    print(
        "    [new_J auto-prune] Removing "
        f"{len(remove_nodes)} newly detected pore(s) from the solute concentration graph only "
        f"(pass {AUTO_PRUNE_OOB_PASS + 1}/{AUTO_PRUNE_OOB_MAX_PASSES}) and rerunning. "
        f"Pores: {sorted(remove_nodes)[:30]}"
        + (" ..." if len(remove_nodes) > 30 else "")
    )
    child_args = list(sys.argv[1:])
    child_args = _replace_or_append_cli_option(
        child_args,
        "--solute-exclude-pore-ids",
        ",".join(str(x) for x in sorted(merged_exclude)),
    )
    child_args = _replace_or_append_cli_option(
        child_args,
        "--auto-prune-oob-pass",
        str(AUTO_PRUNE_OOB_PASS + 1),
    )
    cmd = [sys.executable, str(Path(__file__).resolve()), *child_args]
    sys.stdout.flush()
    sys.stderr.flush()
    cp = subprocess.run(cmd, cwd=str(Path.cwd()))
    sys.exit(int(cp.returncode))


_auto_prune_oob_and_rerun_if_needed()

if _args.concentration_npz_out and (conc_new is not None or conc_old is not None):
    _npz_path = Path(_args.concentration_npz_out)
    _npz_path.parent.mkdir(parents=True, exist_ok=True)
    _n = n_solute_nodes
    C_new_arr = (
        np.asarray(conc_new, dtype=np.float64)
        if conc_new is not None
        else np.full(_n, np.nan, dtype=np.float64)
    )
    np.savez_compressed(
        _npz_path,
        pore_ids=np.asarray(solute_nodes, dtype=np.int64),
        C_new=C_new_arr,
        has_new_J=np.bool_(conc_new is not None),
        n_solute_nodes=np.int32(_n),
    )
    print(
        f"    已写入浓度 npz（等效浓度；new_J={'是' if conc_new is not None else '否'}）: {_npz_path}"
    )

_dump_A_auto = n_solute_nodes < 50
_dump_A_force = bool(getattr(_args, "dump_concentration_matrix_xlsx", False))
if _dump_A_auto or _dump_A_force:
    _A_new_final = (A_eval_new if A_eval_new is not None else A_conc_new) if conc_new is not None else None
    _b_new_final = (b_eval_new if b_eval_new is not None else b_conc_new) if conc_new is not None else None
    _A_old_final = (A_eval_old if A_eval_old is not None else A_conc_old) if conc_old is not None else None
    _b_old_final = (b_eval_old if b_eval_old is not None else b_conc_old) if conc_old is not None else None
    _pore_idx = np.asarray(solute_nodes, dtype=np.int64)
    _dbg_xlsx = output_path / get_output_filename(
        f"{SIEVE_SAMPLE_NAME}_concentration_linear_system_debug"
    )
    try:
        with pd.ExcelWriter(str(_dbg_xlsx), engine="openpyxl") as _writer:
            _meta_rows = [
                {"key": "n_solute_nodes", "value": int(n_solute_nodes)},
                {"key": "auto_dump_threshold", "value": 50},
                {"key": "auto_dump_triggered", "value": int(_dump_A_auto)},
                {"key": "force_dump_flag", "value": int(_dump_A_force)},
                {"key": "solve_new_j", "value": int(bool(SOLVE_NEW_J))},
                {"key": "solve_old_j", "value": int(bool(SOLVE_OLD_J))},
                {"key": "new_matrix_source", "value": "A_eval_new" if A_eval_new is not None else "A_conc_new"},
                {"key": "old_matrix_source", "value": "A_eval_old" if A_eval_old is not None else "A_conc_old"},
            ]
            pd.DataFrame(_meta_rows).to_excel(_writer, sheet_name="meta", index=False)

            def _debug_labels(length: int) -> list[str]:
                labels = [str(v) for v in _pore_idx[:length]]
                labels.extend(f"aux_{i + 1}" for i in range(length - len(labels)))
                return labels

            def _write_debug_system(matrix, vector, suffix: str) -> None:
                if matrix is None or vector is None:
                    return
                matrix_arr = np.asarray(matrix, dtype=np.float64)
                vector_arr = np.asarray(vector, dtype=np.float64).ravel()
                row_labels = _debug_labels(matrix_arr.shape[0])
                col_labels = _debug_labels(matrix_arr.shape[1])
                pd.DataFrame(matrix_arr, index=row_labels, columns=col_labels).to_excel(
                    _writer, sheet_name=f"A_final_{suffix}"
                )
                pd.DataFrame(
                    {"row_label": _debug_labels(len(vector_arr)), "b_value": vector_arr}
                ).to_excel(_writer, sheet_name=f"b_final_{suffix}", index=False)

            _write_debug_system(_A_new_final, _b_new_final, "new")
            _write_debug_system(_A_old_final, _b_old_final, "old")
        print(
            f"    已写入浓度线性系统调试矩阵: {_dbg_xlsx} "
            f"（{'自动触发' if _dump_A_auto else '命令强制'}）"
        )
    except Exception as _e_dumpA:
        print(f"    警告：导出浓度线性系统调试矩阵失败: {_e_dumpA}")
else:
    print(
        f"    浓度矩阵调试导出已跳过：n_solute_nodes={n_solute_nodes} >= 50。"
        " 如需导出请加 --dump-concentration-matrix-xlsx"
    )

if node_concentrations_new is None:
    print("    new_J 浓度场不可用；后续 new_J 通量与基于 new_J 的筛分将跳过。")
if SOLVE_OLD_J and node_concentrations_old is None:
    print("    old_J 浓度场不可用；后续 old_J 通量与基于 old_J 的筛分将跳过。")

if node_concentrations_new is None and node_concentrations_old is None:
    print("    两种浓度求解均失败，本样本已中止。")
    sys.exit(1)

node_concentrations = node_concentrations_new if node_concentrations_new is not None else node_concentrations_old


def _concentration_oob_diagnostic_bounds() -> tuple[float, float]:
    """阶段 E 越界诊断用的浓度盒：clip-refine 开启时用其盒，否则用 --new-j-c-min/max（默认 [0,1]）。"""
    if CLIP_REFINE_ENABLED:
        return float(CLIP_REFINE_LO), float(CLIP_REFINE_HI)
    return float(_args.new_j_c_min), float(_args.new_j_c_max)


def _internal_concentration_range(C_map: dict | None) -> tuple[float, float] | None:
    """内部节点（非入口、非出口）浓度 min/max；无有效值返回 None。"""
    if C_map is None:
        return None
    entrance_set = set(solute_entrance_nodes)
    exit_set = set(solute_exit_nodes)
    vals: list[float] = []
    for n in solute_nodes:
        if n in entrance_set or n in exit_set:
            continue
        c = C_map.get(n, np.nan)
        try:
            c_float = float(c)
        except Exception:
            continue
        if np.isfinite(c_float):
            vals.append(c_float)
    if not vals:
        return None
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.min()), float(arr.max())


def _oob_margin(c: float, lo: float, hi: float) -> float:
    """越出 [lo,hi] 的“超出量”；在盒内为 0。"""
    if not np.isfinite(c):
        return 0.0
    if c < lo:
        return float(lo - c)
    if c > hi:
        return float(c - hi)
    return 0.0


def _print_oob_internal_concentration_flux_diagnostic(
    A: np.ndarray | None,
    b: np.ndarray | None,
    x: np.ndarray | None,
    C_map: dict | None,
    label: str,
    *,
    max_nodes: int = 2,
) -> None:
    if A is None or b is None or x is None or C_map is None:
        return
    neg_tol = 1e-12
    entrance_set = set(solute_entrance_nodes)
    exit_set = set(solute_exit_nodes)
    internal_nodes = [n for n in solute_nodes if n not in entrance_set and n not in exit_set]
    oob_nodes: list[tuple[int, float, float]] = []
    for n in internal_nodes:
        c = C_map.get(n, np.nan)
        try:
            c_float = float(c)
        except Exception:
            continue
        if not np.isfinite(c_float):
            continue
        m = float((-neg_tol) - c_float) if c_float < -neg_tol else 0.0
        if m > 0.0:
            oob_nodes.append((int(n), c_float, m))

    if len(oob_nodes) == 0:
        rng = _internal_concentration_range(C_map)
        rng_note = (
            f"，内部 C∈[{rng[0]:.6f},{rng[1]:.6f}]"
            if rng is not None
            else ""
        )
        print(
            f"  [诊断] [{label}] 内部节点浓度低于阈值（C < {-neg_tol:.1e}）: 0 个{rng_note}"
        )
        return

    oob_nodes.sort(key=lambda t: t[2], reverse=True)
    print(
        f"  [诊断] [{label}] 内部节点浓度低于阈值（C < {-neg_tol:.1e}）: {len(oob_nodes)} 个，"
        f"按超出量降序最多展示前 {max_nodes} 个"
    )

    def _edge_throat_id(node: int, neighbor: int):
        try:
            edge_data = G_solute_accessible[node][neighbor]
            t_id = edge_data.get("throat_id")
        except KeyError:
            try:
                edge_data = G_solute_accessible[neighbor][node]
                t_id = edge_data.get("throat_id")
            except KeyError:
                t_id = None
        if not t_id:
            for t, p1, p2 in zip(throat_ids, throat_pore1, throat_pore2):
                if t in solute_accessible_throat_ids:
                    if (p1 == node and p2 == neighbor) or (p1 == neighbor and p2 == node):
                        t_id = t
                        break
        return t_id

    n_show = min(len(oob_nodes), max_nodes)
    for idx, (node, c_node, margin) in enumerate(oob_nodes[:n_show], 1):
        irow = int(solute_node_to_idx[node])
        row = np.asarray(A[irow, :], dtype=np.float64).ravel()
        ax_i = float(np.dot(row, x))
        b_i = float(b[irow])
        r_i = float(ax_i - b_i)
        j_row_net = -r_i
        j_row_diag_out = float(row[irow] * x[irow])
        print(
            f"    [{idx:02d}] pore={node}, C={c_node:.6e}, 低于阈值量={margin:.6e}, "
            f"net_J_row=-r_i={j_row_net:.6e}, r_i={r_i:.6e}, "
            f"b_i={b_i:.6e}, (Ax)_i={ax_i:.6e}"
        )
        try:
            neighbors = [int(nb) for nb in G_solute_accessible.neighbors(node) if nb in solute_node_to_idx]
        except Exception:
            neighbors = []
        if not neighbors:
            print("       无邻居（在溶质图中度数=0）")
            continue

        net_into_node = 0.0
        sum_abs_j = 0.0
        n_valid_edges = 0
        j_row_nb_sum = 0.0
        for nb in neighbors:
            t_id = _edge_throat_id(node, nb)
            if not t_id:
                print(f"       邻居 {nb}: throat_id 缺失，跳过")
                continue
            if t_id not in throat_Kc or t_id not in throat_flows:
                print(
                    f"       邻居 {nb}, throat={t_id}: 缺参数 "
                    f"(has_Kc={t_id in throat_Kc}, has_Q={t_id in throat_flows})，跳过"
                )
                continue

            Kc = float(throat_Kc[t_id])
            Q_ij = _signed_Q_node_to_neighbor(int(t_id), int(node), int(nb))
            Q = abs(float(Q_ij))
            c_nb = float(C_map.get(nb, 0.0))
            jcol = int(solute_node_to_idx[nb])
            j_row_term = float(-row[jcol] * x[jcol])
            j_row_nb_sum += j_row_term

            c_node_throat = c_node
            c_nb_throat = c_nb

            d_eff_m2s = float(throat_D_eff.get(t_id, 0.0))
            diff_coeff = 0.0
            r_throat_nm_dbg = float("nan")
            L_throat_nm_dbg = float("nan")
            a_solute_m2_dbg = float("nan")
            try:
                r_rad = float(throat_radii[throat_ids == t_id][0])
            except Exception:
                r_rad = 0.0
            try:
                L_i = float(throat_lengths.get(t_id, 0.0) or 0.0)
            except Exception:
                L_i = 0.0
            if np.isfinite(r_rad) and r_rad > 0.0:
                r_throat_nm_dbg = r_rad
                a_solute_m2_dbg = float(_throat_solute_diffusion_area_m2(r_rad))
            if np.isfinite(L_i) and L_i > 0.0:
                L_throat_nm_dbg = L_i
            if INCLUDE_DIFFUSION and d_eff_m2s > 0.0 and r_rad > 0.0 and L_i > 0.0:
                diff_coeff = _throat_diff_coeff_m3s(r_rad, L_i, d_eff_m2s)

            pressure_tie_nb, dp_nb = _pressure_tie_neighbor_to_node(nb, node)
            if node in node_pressures and nb in node_pressures:
                flow_nb_to_node = (not pressure_tie_nb) and (Q_ij < 0.0)
            else:
                flow_nb_to_node = nb in solute_entrance_nodes
            Q_pos = -float(Q_ij) if flow_nb_to_node else float(Q_ij)
            U_abs = _throat_new_j_adv_coeff_m3s(Kc, Q, r_rad)
            U_pos = _throat_new_j_adv_coeff_m3s(Kc, Q_pos, r_rad)

            if pressure_tie_nb:
                if INCLUDE_DIFFUSION and diff_coeff > 0:
                    j_into = diff_coeff * (c_nb_throat - c_node_throat)
                else:
                    j_into = 0.0
                regime = "pressure_tie_fick" if (INCLUDE_DIFFUSION and diff_coeff > 0) else "pressure_tie_no_adv"
            elif label == "new_J":
                regime = (
                    _throat_pe_regime(1.0, U_abs, diff_coeff)
                    if (INCLUDE_DIFFUSION and diff_coeff > 0)
                    else "adv_only"
                )
                if INCLUDE_DIFFUSION and diff_coeff > 0:
                    if flow_nb_to_node:
                        if regime == "adv_only":
                            j_into = U_abs * c_nb_throat
                        elif regime == "diff_only":
                            j_into = diff_coeff * (c_nb_throat - c_node_throat)
                        else:
                            j_into = _throat_coupled_flux_vol(
                                1.0, U_pos, diff_coeff, c_nb, c_node
                            )
                    else:
                        if regime == "adv_only":
                            j_into = -(U_abs * c_node_throat)
                        elif regime == "diff_only":
                            j_into = -(diff_coeff * (c_node_throat - c_nb_throat))
                        else:
                            j_into = -_throat_coupled_flux_vol(
                                1.0, U_pos, diff_coeff, c_node, c_nb
                            )
                else:
                    if flow_nb_to_node:
                        j_into = U_abs * c_nb_throat
                    else:
                        j_into = -(U_abs * c_node_throat)
            else:
                # old_J 与组装一致：见 A_conc_old 内非入口邻边分支（约 2468–2818 行）。
                # 含扩散时 _throat_pe_regime==adv_only 的喉在组装中**不加** Fick 项，诊断此前误加导致与矩阵行不一致。
                c_adv_diag = (
                    0.5 * (c_nb_throat + c_node_throat)
                    if OLD_J_MEAN_THROAT_CONV
                    else (c_nb_throat if flow_nb_to_node else c_node_throat)
                )
                if INCLUDE_DIFFUSION and diff_coeff > 0:
                    pe_reg = _throat_pe_regime(Kc, Q, diff_coeff)
                else:
                    pe_reg = "adv_only"
                if pe_reg == "adv_only":
                    regime = (
                        "old_mean_throat_adv_only"
                        if OLD_J_MEAN_THROAT_CONV
                        else "old_upwind_adv_only"
                    )
                    if OLD_J_MEAN_THROAT_CONV:
                        j_into = -Kc * float(Q_ij) * c_adv_diag
                    elif flow_nb_to_node:
                        j_into = Kc * Q * c_adv_diag
                    else:
                        j_into = -(Kc * Q * c_adv_diag)
                else:
                    regime = (
                        "old_mean_throat_fick"
                        if OLD_J_MEAN_THROAT_CONV
                        else "old_upwind_fick"
                    ) + f"|{pe_reg}"
                    if OLD_J_MEAN_THROAT_CONV:
                        j_into = -Kc * float(Q_ij) * c_adv_diag + diff_coeff * (
                            c_nb_throat - c_node_throat
                        )
                    elif flow_nb_to_node:
                        j_into = Kc * Q * c_adv_diag + diff_coeff * (c_nb_throat - c_node_throat)
                    else:
                        j_into = -(Kc * Q * c_adv_diag + diff_coeff * (c_node_throat - c_nb_throat))

            n_valid_edges += 1
            net_into_node += j_into
            sum_abs_j += abs(j_into)
            if pressure_tie_nb:
                dir_tag = "pressure_tie"
            else:
                dir_tag = "nb->node" if flow_nb_to_node else "node->nb"

            # 越界诊断：Pe 与（若可定义）上风对流项 / Fick(diff·ΔC) 分项，澄清 coupled≠|DΔC|≫|KQC|
            diag_pe = float("nan")
            diag_adv_lin = float("nan")
            diag_diff_lin = float("nan")
            diag_ratio_fd_over_adv = float("nan")
            diag_note = ""
            dc_nb_m_node = float(c_nb_throat - c_node_throat)
            if diff_coeff > 0.0 and np.isfinite(diff_coeff):
                diag_pe = float(U_abs if label == "new_J" else Kc * Q) / float(diff_coeff)
            if pressure_tie_nb:
                if INCLUDE_DIFFUSION and diff_coeff > 0:
                    diag_adv_lin, diag_diff_lin = 0.0, float(diff_coeff * dc_nb_m_node)
                    diag_note = "压平:仅Fick"
            elif label == "new_J":
                if regime == "coupled" and INCLUDE_DIFFUSION and diff_coeff > 0:
                    if flow_nb_to_node:
                        diag_adv_lin = float(U_abs * c_nb_throat)
                    else:
                        diag_adv_lin = float(-U_abs * c_node_throat)
                    diag_diff_lin = float(diff_coeff * dc_nb_m_node)
                    diag_note = "new_J 闭式耦合; 上列为上风+Fick参考量级(≠J_edge)"
                elif regime == "diff_only" and INCLUDE_DIFFUSION and diff_coeff > 0:
                    diag_adv_lin, diag_diff_lin = 0.0, float(diff_coeff * dc_nb_m_node)
                elif regime == "adv_only" or not (INCLUDE_DIFFUSION and diff_coeff > 0):
                    diag_adv_lin, diag_diff_lin = float(j_into), 0.0
            else:
                # old_J：与上分支一致重算 c_adv_diag / pe_reg，供分项与说明
                c_adv_diag_dbg = (
                    0.5 * (c_nb_throat + c_node_throat)
                    if OLD_J_MEAN_THROAT_CONV
                    else (c_nb_throat if flow_nb_to_node else c_node_throat)
                )
                if INCLUDE_DIFFUSION and diff_coeff > 0:
                    pe_reg_dbg = _throat_pe_regime(Kc, Q, diff_coeff)
                    diag_diff_lin = float(diff_coeff * dc_nb_m_node)
                    if OLD_J_MEAN_THROAT_CONV:
                        diag_adv_lin = float(-Kc * float(Q_ij) * c_adv_diag_dbg)
                    elif flow_nb_to_node:
                        diag_adv_lin = float(Kc * Q * c_adv_diag_dbg)
                    else:
                        diag_adv_lin = float(-Kc * Q * c_adv_diag_dbg)
                    if pe_reg_dbg == "adv_only":
                        diag_diff_lin = 0.0
                        diag_note = "old Pe 超阈:组装无Fick,分项中Fick=0"
                    else:
                        diag_note = (
                            "old:上风对流+Fick线性叠加; regime 中 coupled 仅表示 Pe 在阈间, "
                            "非|diff·ΔC|≫|Kc·Q·C_up|"
                        )
                else:
                    diag_adv_lin, diag_diff_lin = float(j_into), 0.0
                    diag_note = "无扩散"
            if (
                np.isfinite(diag_adv_lin)
                and np.isfinite(diag_diff_lin)
                and abs(diag_adv_lin) > 0.0
            ):
                diag_ratio_fd_over_adv = abs(diag_diff_lin) / abs(diag_adv_lin)

            print(
                f"   邻居 {nb}, throat={t_id}, dir={dir_tag}, "
                f"J_edge_into={j_into:.6e}, J_row_term={j_row_term:.6e}, "
                f"Kc={Kc:.6e}, Q(node→nb)={float(Q_ij):.6e}, |Q|={Q:.6e}, "
                f"D_eff={d_eff_m2s:.6e} m^2/s, L_nm={L_throat_nm_dbg:.6f}, "
                f"A_solute={a_solute_m2_dbg:.6e} m^2, diff_coeff={diff_coeff:.6e} m^3/s, "
                f"C_node={c_node:.3e}, C_nb={c_nb:.3e}, regime={regime}"
            )
            print(
                f"Pe={'U_adv' if label == 'new_J' else 'Kc|Q|'}/diff={diag_pe:.6e}, "
                f"ΔC(nb−node)={dc_nb_m_node:.6e}, "
                f"分项(线性):adv_like={diag_adv_lin:.6e}, Fick_like=diff·ΔC={diag_diff_lin:.6e}, "
                f"|Fick|/|adv|={diag_ratio_fd_over_adv:.6e}; "
                )
            if diag_note:
                print(f"         … {diag_note}")
        j_row_source = b_i
        j_row_net_rebuild = j_row_nb_sum + j_row_source - j_row_diag_out
        print(
            f"   汇总: 有效邻边={n_valid_edges}, "
            f"net_J_edge={net_into_node:.6e}, sum|J_edge|={sum_abs_j:.6e}, "
            f"net_J_row={j_row_net:.6e}, "
            f"(J_nb_sum={j_row_nb_sum:.6e}, J_diag_out={j_row_diag_out:.6e}, J_src=b={j_row_source:.6e}, rebuild={j_row_net_rebuild:.6e})"
        )


def _print_oob_internal_row_residual_audit(
    A: np.ndarray | None,
    b: np.ndarray | None,
    x: np.ndarray | None,
    C_map: dict | None,
    label: str,
    *,
    max_nodes: int = 2,
    max_terms: int = 5,
) -> None:
    """对内部节点中 C < -1e-12 的节点，打印对应方程行残差与主要 Ax 项。"""
    if A is None or b is None or x is None or C_map is None:
        return
    neg_tol = 1e-12
    entrance_set = set(solute_entrance_nodes)
    exit_set = set(solute_exit_nodes)
    oob_rows: list[tuple[int, int, float, float]] = []
    for n in solute_nodes:
        if n in entrance_set or n in exit_set:
            continue
        idx = solute_node_to_idx.get(n)
        if idx is None:
            continue
        c = C_map.get(n, np.nan)
        try:
            c_float = float(c)
        except Exception:
            continue
        if not np.isfinite(c_float):
            continue
        m = float((-neg_tol) - c_float) if c_float < -neg_tol else 0.0
        if m > 0.0:
            oob_rows.append((int(n), int(idx), c_float, m))
    if len(oob_rows) == 0:
        rng = _internal_concentration_range(C_map)
        rng_note = (
            f"，内部 C∈[{rng[0]:.6f},{rng[1]:.6f}]"
            if rng is not None
            else ""
        )
        print(
            f"  [审计] [{label}] 低于阈值内部节点行残差审计: 0 个"
            f"（阈值 C < {-neg_tol:.1e}{rng_note}）"
        )
        return
    oob_rows.sort(key=lambda t: t[3], reverse=True)
    r = np.asarray(A @ x - b, dtype=np.float64).ravel()
    print(
        f"  [审计] [{label}] 低于阈值内部节点行残差审计: {len(oob_rows)} 个，"
        f"阈值 C < {-neg_tol:.1e}，按超出量降序最多展示前 {max_nodes} 个"
    )
    for k, (pore_id, irow, c_val, margin) in enumerate(oob_rows[:max_nodes], 1):
        row = np.asarray(A[irow, :], dtype=np.float64).ravel()
        ax_i = float(np.dot(row, x))
        b_i = float(b[irow])
        res_i = float(r[irow])
        nz = np.flatnonzero(np.abs(row) > 0.0)
        terms = []
        for j in nz:
            contrib = float(row[j] * x[j])
            terms.append((abs(contrib), int(j), contrib, float(row[j]), float(x[j])))
        terms.sort(key=lambda t: t[0], reverse=True)
        print(
            f"    [{k:02d}] pore={pore_id}, C={c_val:.6e}, 低于阈值量={margin:.6e}, row={irow}, "
            f"Ax={ax_i:.6e}, b={b_i:.6e}, r=Ax-b={res_i:.6e}, "
            f"nnz={int(nz.size)}"
        )
        for _, j, contrib, aij, xj in terms[:max_terms]:
            pj = int(solute_nodes[j]) if 0 <= j < len(solute_nodes) else -1
            print(
                f"       term: A[{irow},{j}]={aij:.3e}, x[{j}]={xj:.3e}, "
                f"A*x={contrib:.3e}, col_pore={pj}"
            )


_print_oob_internal_concentration_flux_diagnostic(
    A_eval_new if A_eval_new is not None else A_conc_new,
    b_eval_new if b_eval_new is not None else b_conc_new,
    conc_new,
    node_concentrations_new,
    "new_J",
)
_print_oob_internal_concentration_flux_diagnostic(
    A_eval_old if A_eval_old is not None else A_conc_old,
    b_eval_old if b_eval_old is not None else b_conc_old,
    conc_old,
    node_concentrations_old,
    "old_J",
)
_print_oob_internal_row_residual_audit(
    A_eval_new if A_eval_new is not None else A_conc_new,
    b_eval_new if b_eval_new is not None else b_conc_new,
    conc_new,
    node_concentrations_new,
    "new_J",
)
_print_oob_internal_row_residual_audit(
    A_eval_old if A_eval_old is not None else A_conc_old,
    b_eval_old if b_eval_old is not None else b_conc_old,
    conc_old,
    node_concentrations_old,
    "old_J",
)

print("\n阶段E完成！")

# ====================================================================
# 阶段F：整体筛分系数计算
# ====================================================================
print("\n阶段F：整体筛分系数计算...")

# F1. 计算出口总溶质通量（并统计每个出口孔的溶质通量）
print("  F1. 计算出口总溶质通量...")
Q_alb_total = 0.0
Q_alb_conv_total = 0.0  # 对流贡献（浓度*m^3/s）
Q_alb_diff_total = 0.0  # 扩散贡献（浓度*m^3/s）
exit_solute_flux = {n: 0.0 for n in solute_exit_nodes}  # 每个出口孔的白蛋白总流率 (浓度*m^3/s)
exit_solute_conv = {n: 0.0 for n in solute_exit_nodes}  # 每个出口孔的对流贡献
exit_solute_diff = {n: 0.0 for n in solute_exit_nodes}  # 每个出口孔的扩散贡献
exit_solute_flux_old = {n: 0.0 for n in solute_exit_nodes}
exit_solute_conv_old = {n: 0.0 for n in solute_exit_nodes}
exit_solute_diff_old = {n: 0.0 for n in solute_exit_nodes}
Q_alb_total_old = 0.0
Q_alb_conv_total_old = 0.0
Q_alb_diff_total_old = 0.0

if node_concentrations_new is not None:
    C_use = node_concentrations_new
    for exit_node in solute_exit_nodes:
        for neighbor in G_solute_accessible.neighbors(exit_node):
            try:
                edge_data = G_solute_accessible[exit_node][neighbor]
                t_id = edge_data.get('throat_id')
            except KeyError:
                try:
                    edge_data = G_solute_accessible[neighbor][exit_node]
                    t_id = edge_data.get('throat_id')
                except KeyError:
                    t_id = None
            if not t_id:
                for t, p1, p2 in zip(throat_ids, throat_pore1, throat_pore2):
                    if t in solute_accessible_throat_ids:
                        if (p1 == exit_node and p2 == neighbor) or (p1 == neighbor and p2 == exit_node):
                            t_id = t
                            break
            if t_id and t_id in throat_Kc and t_id in throat_flows:
                Kc = throat_Kc[t_id]
                Q_exit_nb = _signed_Q_node_to_neighbor(int(t_id), int(exit_node), int(neighbor))
                Q = abs(float(Q_exit_nb))
                C_upstream_pore = C_use.get(neighbor, 0.0)
                C_exit_pore = C_use.get(exit_node, 0.0)
                C_upstream_throat = C_upstream_pore
                C_exit_throat = C_exit_pore
                r_i = throat_radii[throat_ids == t_id][0]
                U_abs = _throat_new_j_adv_coeff_m3s(Kc, Q, r_i)
                diff_coeff = 0.0
                if INCLUDE_DIFFUSION:
                    L_i = throat_lengths.get(t_id, 0.0)
                    D_eff = throat_D_eff.get(t_id, 0.0)
                    if D_eff > 0 and r_i > 0 and L_i > 0:
                        diff_coeff = _throat_diff_coeff_m3s(r_i, L_i, D_eff)
                pressure_tie_exit, dp_ex = _pressure_tie_neighbor_to_exit(neighbor, exit_node)
                if exit_node in node_pressures and neighbor in node_pressures:
                    flow_nb_to_exit = (not pressure_tie_exit) and (Q_exit_nb < 0.0)
                else:
                    flow_nb_to_exit = True
                Q_pos_exit = -float(Q_exit_nb) if flow_nb_to_exit else float(Q_exit_nb)
                _rf_exit = (
                    _throat_pe_regime(1.0, U_abs, diff_coeff)
                    if (INCLUDE_DIFFUSION and diff_coeff > 0)
                    else "adv_only"
                )
                if pressure_tie_exit:
                    if INCLUDE_DIFFUSION and diff_coeff > 0:
                        J_edge = diff_coeff * (C_upstream_throat - C_exit_throat)
                    else:
                        J_edge = 0.0
                    Q_alb_total += J_edge
                    exit_solute_flux[exit_node] += J_edge
                    Q_alb_conv_total += 0.0
                    Q_alb_diff_total += J_edge
                    exit_solute_conv[exit_node] += 0.0
                    exit_solute_diff[exit_node] += J_edge
                    continue
                if INCLUDE_DIFFUSION and diff_coeff > 0:
                    if flow_nb_to_exit:
                        if _rf_exit == "adv_only":
                            J_edge = U_abs * C_upstream_throat
                        elif _rf_exit == "diff_only":
                            J_edge = diff_coeff * (C_upstream_throat - C_exit_throat)
                        else:
                            U_pos_exit = _throat_new_j_adv_coeff_m3s(Kc, Q_pos_exit, r_i)
                            J_edge = _throat_coupled_flux_vol(1.0, U_pos_exit, diff_coeff, C_upstream_pore, C_exit_pore)
                    else:
                        if _rf_exit == "adv_only":
                            J_edge = -U_abs * C_exit_throat
                        elif _rf_exit == "diff_only":
                            J_edge = diff_coeff * (C_upstream_throat - C_exit_throat)
                        else:
                            U_pos_exit = _throat_new_j_adv_coeff_m3s(Kc, Q_pos_exit, r_i)
                            J_edge = -_throat_coupled_flux_vol(1.0, U_pos_exit, diff_coeff, C_exit_pore, C_upstream_pore)
                else:
                    if flow_nb_to_exit:
                        J_edge = U_abs * C_upstream_throat
                    else:
                        J_edge = -U_abs * C_exit_throat
                Q_alb_total += J_edge
                exit_solute_flux[exit_node] += J_edge
                _rj_fl = _throat_pe_regime(1.0, U_abs, diff_coeff) if INCLUDE_DIFFUSION else "adv_only"
                if _rj_fl == "diff_only" and INCLUDE_DIFFUSION and diff_coeff > 0:
                    Q_alb_conv_total += 0.0
                    Q_alb_diff_total += J_edge
                    exit_solute_conv[exit_node] += 0.0
                    exit_solute_diff[exit_node] += J_edge
                elif _rj_fl == "adv_only":
                    Q_alb_conv_total += J_edge
                    Q_alb_diff_total += 0.0
                    exit_solute_conv[exit_node] += J_edge
                    exit_solute_diff[exit_node] += 0.0
                else:
                    if flow_nb_to_exit:
                        J_conv_line = U_abs * C_upstream_throat
                    else:
                        J_conv_line = -U_abs * C_exit_throat
                    Q_alb_conv_total += J_conv_line
                    Q_alb_diff_total += J_edge - J_conv_line
                    exit_solute_conv[exit_node] += J_conv_line
                    exit_solute_diff[exit_node] += J_edge - J_conv_line
else:
    print("    （new_J）无浓度场，跳过解析耦合通量累计。")

if SOLVE_OLD_J and node_concentrations_old is not None:
    C_use = node_concentrations_old
    for exit_node in solute_exit_nodes:
        for neighbor in G_solute_accessible.neighbors(exit_node):
            try:
                edge_data = G_solute_accessible[exit_node][neighbor]
                t_id = edge_data.get('throat_id')
            except KeyError:
                try:
                    edge_data = G_solute_accessible[neighbor][exit_node]
                    t_id = edge_data.get('throat_id')
                except KeyError:
                    t_id = None
            if not t_id:
                for t, p1, p2 in zip(throat_ids, throat_pore1, throat_pore2):
                    if t in solute_accessible_throat_ids:
                        if (p1 == exit_node and p2 == neighbor) or (p1 == neighbor and p2 == exit_node):
                            t_id = t
                            break
            if t_id and t_id in throat_Kc and t_id in throat_flows:
                Kc = throat_Kc[t_id]
                Q_exit_nb = _signed_Q_node_to_neighbor(int(t_id), int(exit_node), int(neighbor))
                Q = abs(float(Q_exit_nb))
                C_upstream_pore = C_use.get(neighbor, 0.0)
                C_exit_pore = C_use.get(exit_node, 0.0)
                C_upstream_throat = C_upstream_pore
                C_exit_throat = C_exit_pore
                diff_coeff = 0.0
                if INCLUDE_DIFFUSION:
                    r_i = throat_radii[throat_ids == t_id][0]
                    L_i = throat_lengths.get(t_id, 0.0)
                    D_eff = throat_D_eff.get(t_id, 0.0)
                    if D_eff > 0 and r_i > 0 and L_i > 0:
                        diff_coeff = _throat_diff_coeff_m3s(r_i, L_i, D_eff)
                pressure_tie_exit, dp_ex = _pressure_tie_neighbor_to_exit(neighbor, exit_node)
                if exit_node in node_pressures and neighbor in node_pressures:
                    flow_nb_to_exit = (not pressure_tie_exit) and (Q_exit_nb < 0.0)
                else:
                    flow_nb_to_exit = True
                # old_J：与矩阵组装一致；adv_only（|Pe| 大）仅对流不加 Fick
                if pressure_tie_exit:
                    if INCLUDE_DIFFUSION and diff_coeff > 0:
                        J_edge_old = diff_coeff * (C_upstream_throat - C_exit_throat)
                    else:
                        J_edge_old = 0.0
                    Q_alb_total_old += J_edge_old
                    exit_solute_flux_old[exit_node] += J_edge_old
                    Q_alb_conv_total_old += 0.0
                    Q_alb_diff_total_old += J_edge_old
                    exit_solute_conv_old[exit_node] += 0.0
                    exit_solute_diff_old[exit_node] += J_edge_old
                    continue
                c_adv_t = (
                    0.5 * (C_upstream_throat + C_exit_throat)
                    if OLD_J_MEAN_THROAT_CONV
                    else (C_upstream_throat if flow_nb_to_exit else C_exit_throat)
                )
                j_conv_edge = (
                    float(Kc) * float(Q_exit_nb) * c_adv_t
                    if OLD_J_MEAN_THROAT_CONV
                    else float(Kc) * float(Q) * c_adv_t
                )
                if INCLUDE_DIFFUSION and diff_coeff > 0:
                    _pe_old_ex = _throat_pe_regime(Kc, Q, diff_coeff)
                    if _pe_old_ex == "adv_only":
                        J_edge_old = j_conv_edge
                    elif flow_nb_to_exit:
                        J_edge_old = j_conv_edge + diff_coeff * (
                            C_upstream_throat - C_exit_throat
                        )
                    else:
                        J_edge_old = j_conv_edge + diff_coeff * (
                            C_exit_throat - C_upstream_throat
                        )
                else:
                    J_edge_old = j_conv_edge
                Q_alb_total_old += J_edge_old
                exit_solute_flux_old[exit_node] += J_edge_old
                J_conv_line_old = j_conv_edge
                Q_alb_conv_total_old += J_conv_line_old
                Q_alb_diff_total_old += J_edge_old - J_conv_line_old
                exit_solute_conv_old[exit_node] += J_conv_line_old
                exit_solute_diff_old[exit_node] += J_edge_old - J_conv_line_old
elif SOLVE_OLD_J:
    print("    （old_J）无浓度场，跳过迎风对流+Fick 通量累计。")


def _print_exit_concentration_stats(C_map: dict | None, label: str) -> None:
    if C_map is None:
        return
    vals = [float(C_map.get(n, np.nan)) for n in solute_exit_nodes]
    arr = np.asarray(vals, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        print(f"    [{label}] 出口浓度统计: 无有限值。")
        return
    af = arr[finite]
    n0 = int(np.sum(af == 0.0))
    n_tiny = int(np.sum((af > 0.0) & (af < 1e-15)))
    print(
        f"    [{label}] 出口浓度统计: n={af.size}, min={float(np.min(af)):.6e}, "
        f"p50={float(np.median(af)):.6e}, max={float(np.max(af)):.6e}, "
        f"==0 个数={n0}, (0,1e-15) 个数={n_tiny}"
    )


def _print_exit_flux_topk(
    total_by_exit: dict,
    conv_by_exit: dict,
    diff_by_exit: dict,
    label: str,
    *,
    k: int = 8,
) -> None:
    if not total_by_exit:
        return
    ranked = sorted(total_by_exit.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
    k_use = min(int(k), len(ranked))
    print(f"    [{label}] 出口孔通量分解 Top-{k_use}（按 |J_total| 降序，单位: 浓度*m^3/s）:")
    for pid, jt in ranked[:k_use]:
        jc = float(conv_by_exit.get(pid, 0.0))
        jd = float(diff_by_exit.get(pid, 0.0))
        print(
            f"      pore {int(pid)}: J_total={float(jt) * C0:+.6e}, "
            f"J_conv={jc * C0:+.6e}, J_diff={jd * C0:+.6e}"
        )


_print_exit_concentration_stats(node_concentrations_new, "new_J")
if SOLVE_OLD_J:
    _print_exit_concentration_stats(node_concentrations_old, "old_J")
if node_concentrations_new is not None:
    print(f"    出口总白蛋白流率(new_J，C_new 场): {Q_alb_total * C0:.6e} (单位: 浓度*m^3/s)")
else:
    print(f"    出口总白蛋白流率(new_J): （未计算）")
if node_concentrations_old is not None:
    print(f"    出口总白蛋白流率(old_J，C_old 场): {Q_alb_total_old * C0:.6e} (单位: 浓度*m^3/s)")
elif SOLVE_OLD_J:
    print(f"    出口总白蛋白流率(old_J): （未计算）")
if INCLUDE_DIFFUSION:
    if node_concentrations_new is not None:
        print(f"    白蛋白对流流率贡献(new_J): {Q_alb_conv_total * C0:.6e} (单位: 浓度*m^3/s)")
        print(f"    白蛋白扩散流率贡献(new_J): {Q_alb_diff_total * C0:.6e} (单位: 浓度*m^3/s)")
    if node_concentrations_old is not None:
        print(f"    白蛋白对流流率贡献(old_J): {Q_alb_conv_total_old * C0:.6e} (单位: 浓度*m^3/s)")
        print(f"    白蛋白扩散流率贡献(old_J): {Q_alb_diff_total_old * C0:.6e} (单位: 浓度*m^3/s)")
    if node_concentrations_new is not None and abs(Q_alb_conv_total) > 1e-20:
        conv_diff_ratio = abs(Q_alb_diff_total / Q_alb_conv_total)
        if abs(conv_diff_ratio) > 1e-5:
            print(f"    扩散/对流比值(new_J): {conv_diff_ratio:.6e}")
            total_flux_abs = abs(Q_alb_conv_total) + abs(Q_alb_diff_total)
            if total_flux_abs > 0:
                print(f"    对流占比(new_J): {abs(Q_alb_conv_total) / total_flux_abs * 100:.2f}%")
                print(f"    扩散占比(new_J): {abs(Q_alb_diff_total) / total_flux_abs * 100:.2f}%")
        else:
            print(f"    警告：扩散/对流比值(new_J)太小")
    elif node_concentrations_new is not None:
        print(f"    警告：总对流贡献(new_J)接近 0，跳过比值计算")
        conv_diff_ratio = 0.0
else:
    if node_concentrations_new is not None:
        print(f"    白蛋白对流流率贡献(new_J): {Q_alb_conv_total * C0:.6e} (单位: 浓度*m^3/s)")
    print(f"    扩散项: 未启用")

# F2. 计算出口净流出溶剂流量
print("  F2. 计算出口净流出溶剂流量...")

# Q_total_network：仅在“溶质可通行网络”上的出口净流出溶剂通量（用于诊断）；同时统计每个出口孔的净流出溶剂通量
Q_total_network = 0.0
exit_solvent_flux = {n: 0.0 for n in solute_exit_nodes}  # 每个出口孔的净流出溶剂通量 (m^3/s)
_solute_exit_set_for_q = set(solute_exit_nodes)
_seen_solute_exit_q_tids: set[int] = set()
for exit_node in solute_exit_nodes:
    for neighbor in G_solute_accessible.neighbors(exit_node):
        if neighbor in _solute_exit_set_for_q:
            continue
        # 直接从边的属性中获取throat_id
        try:
            edge_data = G_solute_accessible[exit_node][neighbor]
            t_id = edge_data.get('throat_id')
        except KeyError:
            try:
                edge_data = G_solute_accessible[neighbor][exit_node]
                t_id = edge_data.get('throat_id')
            except KeyError:
                t_id = None
        
        # Fallback：如果找不到，通过遍历查找
        if not t_id:
            for t, p1, p2 in zip(throat_ids, throat_pore1, throat_pore2):
                if t in solute_accessible_throat_ids:
                    if (p1 == exit_node and p2 == neighbor) or (p1 == neighbor and p2 == exit_node):
                        t_id = t
                        break
        
        if t_id and t_id in throat_flows:
            tid_i = int(t_id)
            if tid_i in _seen_solute_exit_q_tids:
                continue
            _seen_solute_exit_q_tids.add(tid_i)
            q_edge = -float(_signed_Q_node_to_neighbor(tid_i, int(exit_node), int(neighbor)))
            Q_total_network += q_edge
            exit_solvent_flux[exit_node] += q_edge

print(f"    出口净流出溶剂流量(溶质通路): {Q_total_network:.6e} m^3/s")

# Q_total_all：所有溶剂通路在出口侧的净流出通量（包括对溶质不可通行的喉）
Q_total_all = 0.0
_exit_set_all_for_q = set(exit_pore_ids)
_seen_exit_q_tids_all: set[int] = set()
for exit_node in exit_pore_ids:
    if exit_node not in G_solvent:
        continue
    for neighbor in G_solvent.neighbors(exit_node):
        if neighbor in _exit_set_all_for_q:
            continue
        edge_data = G_solvent[exit_node].get(neighbor, None)
        if not edge_data:
            continue
        t_id = edge_data.get('throat_id')
        if t_id and t_id in throat_flows:
            tid_i = int(t_id)
            if tid_i in _seen_exit_q_tids_all:
                continue
            _seen_exit_q_tids_all.add(tid_i)
            q_edge = -float(_signed_Q_node_to_neighbor(tid_i, int(exit_node), int(neighbor)))
            Q_total_all += q_edge

print(f"    出口净流出溶剂通量(所有溶剂通路): {Q_total_all:.6e} m^3/s")

# F2.5/F3 简化口径：仅保留 sieving_coefficient = C_out / C_in
print("  F2.5-F3. 计算唯一筛过系数：sieving_coefficient = C_out / C_in ...")

def _boundary_concentration_stats(C_source: dict | None, nodes: list[int]) -> tuple[float, float, int]:
    if C_source is None:
        return 0.0, 0.0, 0
    vals = [float(C_source[n]) for n in nodes if n in C_source]
    if not vals:
        return 0.0, 0.0, 0
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr)), int(arr.size)


def _sieving_coefficient_cout_over_cin_from_map(C_map: dict | None) -> float | None:
    """对给定节点浓度场计算 C_out_mean/C_in_mean；无场则 None。"""
    if C_map is None:
        return None
    ci_m, _, _ = _boundary_concentration_stats(C_map, solute_entrance_nodes)
    co_m, _, _ = _boundary_concentration_stats(C_map, solute_exit_nodes)
    return (co_m / ci_m) if ci_m > 0.0 else 0.0


sc_cout_cin_new = _sieving_coefficient_cout_over_cin_from_map(node_concentrations_new)
sc_cout_cin_old = _sieving_coefficient_cout_over_cin_from_map(node_concentrations_old)
if NEW_J_INVALID:
    sc_cout_cin_new = None

if node_concentrations_new is not None and not NEW_J_INVALID:
    _c_source = node_concentrations_new
    _c_source_label = "new_J"
elif node_concentrations_new is None and node_concentrations_old is not None:
    _c_source = node_concentrations_old
    _c_source_label = "old_J"
else:
    _c_source = None
    _c_source_label = "invalid"
C_in_mean, C_in_std, n_in_eval = _boundary_concentration_stats(_c_source, solute_entrance_nodes)
C_out_mean, C_out_std, n_out_eval = _boundary_concentration_stats(_c_source, solute_exit_nodes)
if _c_source is None:
    C_in_mean = np.nan
    C_out_mean = np.nan
    C_in_std = np.nan
    C_out_std = np.nan
sieving_coefficient = (C_out_mean / C_in_mean) if C_in_mean > 0.0 else np.nan

print(f"    采用浓度场: {_c_source_label}")
print(f"    C_in(入口平均): {C_in_mean:.6e}（std={C_in_std:.3e}, n={n_in_eval}）")
print(f"    C_out(出口平均): {C_out_mean:.6e}（std={C_out_std:.3e}, n={n_out_eval}）")
print(f"    sieving_coefficient = C_out/C_in = {sieving_coefficient:.6e}")
if sc_cout_cin_new is not None:
    print(f"    sieving_coefficient (new_J, C_out/C_in) = {sc_cout_cin_new:.6e}")
elif NEW_J_INVALID:
    print("    sieving_coefficient (new_J, C_out/C_in) = NaN（new_J invalid）")
if sc_cout_cin_old is not None:
    print(f"    sieving_coefficient (old_J, C_out/C_in) = {sc_cout_cin_old:.6e}")

print("\n阶段F完成！")

# ====================================================================
# 出口孔溶质/溶剂通量分布统计图
# ====================================================================
print("\n绘制出口孔溶质/溶剂通量分布图...")
if solute_exit_nodes and exit_solvent_flux and exit_solute_flux:
    Q_per_pore_m3s = [exit_solvent_flux[n] for n in solute_exit_nodes]
    Js_per_pore = [exit_solute_flux[n] * C0 for n in solute_exit_nodes]     # 白蛋白总流率 (浓度*m^3/s)
    Q_per_pore_m3s = np.array(Q_per_pore_m3s)
    Js_per_pore = np.array(Js_per_pore)
    # 过滤掉两者都接近 0 的点，避免 log 或小值影响显示（可选：不过滤，直接画）
    valid = (np.abs(Q_per_pore_m3s) > 1e-20) | (np.abs(Js_per_pore) > 1e-20)
    if np.sum(valid) == 0:
        valid = np.ones(len(Q_per_pore_m3s), dtype=bool)
    Q_plot = Q_per_pore_m3s[valid]
    Js_plot = Js_per_pore[valid]

    fig_flux = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "出口孔溶剂通量分布 (m^3/s)",
            "出口孔白蛋白流率分布 (浓度*m^3/s)",
            "出口孔：溶剂流量 vs 白蛋白流率"
        ),
        specs=[[{"type": "histogram"}, {"type": "histogram"}], [{"type": "scatter", "colspan": 2}, None]],
    )
    fig_flux.add_trace(go.Histogram(x=Q_plot, name="溶剂净流出通量", nbinsx=min(50, max(10, len(Q_plot) // 5))), row=1, col=1)
    fig_flux.add_trace(go.Histogram(x=Js_plot, name="白蛋白流率", nbinsx=min(50, max(10, len(Js_plot) // 5))), row=1, col=2)
    fig_flux.add_trace(
        go.Scatter(x=Q_plot, y=Js_plot, mode="markers", name="出口孔", marker=dict(size=6, opacity=0.7)),
        row=2, col=1
    )
    fig_flux.update_xaxes(title_text="溶剂净流出通量 (m^3/s)", row=1, col=1)
    fig_flux.update_xaxes(title_text="白蛋白流率 (浓度*m^3/s)", row=1, col=2)
    fig_flux.update_xaxes(title_text="溶剂净流出通量 (m^3/s)", row=2, col=1)
    fig_flux.update_yaxes(title_text="白蛋白流率 (浓度*m^3/s)", row=2, col=1)
    fig_flux.update_layout(
        title=dict(text=f"{SIEVE_SAMPLE_NAME} - 出口孔溶质/溶剂净流出通量统计", x=0.5),
        height=600,
        showlegend=True,
    )
    if NO_HTML:
        print("  出口孔通量分布 HTML 已跳过（--no-html）")
    else:
        flux_html = output_path / get_output_filename(f"{SIEVE_SAMPLE_NAME}_exit_flux_distribution", ".html")
        fig_flux.write_html(str(flux_html))
        print(f"  出口孔通量分布图已保存: {flux_html}")
else:
    print("  出口节点为空或通量未计算，跳过出口通量分布图")

# ====================================================================
# 保存结果
# ====================================================================
print("\n保存结果...")

_nan = float("nan")
results_df = pd.DataFrame(
    {
        "Parameter": [
            "sieving_coefficient (C_out/C_in, new_J)",
            "Total Albumin Flow Rate (Q_alb_total, equiv_C)",
            "C_in (mean over entrance nodes)",
            "C_out (mean over exit nodes)",
            "C_in std",
            "C_out std",
            "Concentration field used (new_J preferred)",
            "new_J solution status",
            "new_J min concentration",
            "new_J negative tolerance",
            "new_J relative residual",
            "new_J residual invalid threshold",
            "new_J full-compare direct final residual l2",
            "new_J full-compare direct final relative residual",
            "new_J full-compare direct final residual max_abs",
            "new_J full-compare gmres final residual l2",
            "new_J full-compare gmres final relative residual",
            "new_J full-compare gmres final residual max_abs",
            "new_J full-compare final ||x_gmres-x_direct||_2",
            "new_J full-compare final ||x_gmres-x_direct||_2 relative",
            "new_J full-compare final ||x_gmres-x_direct||_inf",
            "N entrance nodes used in concentration stats",
            "N exit nodes used in concentration stats",
            "N entrance nodes in solute graph",
            "N exit nodes in solute graph",
            "Exit C_bulk parameterized boundary enabled",
            "Solved C_bulk (new_J)",
            "Plasma Concentration (C0)",
            "Net Outlet Solvent Flow (Q, full solvent network, m^3/s)",
            "Penetration axis index (0=X, 1=Y, 2=Z)",
            "Permeation plane cross-sectional area (nm^2)",
            "Permeation plane cross-sectional area (um^2)",
            "Number of Solvent Penetration Throats",
            "Number of Solute Accessible Throats",
            "Auto-pruned OOB concentration enabled",
            "Auto-pruned OOB concentration pass",
            "Auto-pruned OOB concentration low threshold",
            "Auto-pruned OOB concentration high threshold",
            "Auto-pruned OOB concentration excluded pore count",
            "Auto-pruned OOB concentration excluded pore IDs",
        ],
        "Value": [
            sc_cout_cin_new if sc_cout_cin_new is not None else _nan,
            (Q_alb_total * C0) if (node_concentrations_new is not None and not NEW_J_INVALID) else _nan,
            C_in_mean,
            C_out_mean,
            C_in_std,
            C_out_std,
            _c_source_label,
            (
                "invalid_negative_concentration+high_relative_residual"
                if (NEW_J_INVALID_NEGATIVE and NEW_J_INVALID_REL_RESIDUAL)
                else (
                    "invalid_negative_concentration"
                    if NEW_J_INVALID_NEGATIVE
                    else (
                        "invalid_high_relative_residual"
                        if NEW_J_INVALID_REL_RESIDUAL
                        else ("valid" if node_concentrations_new is not None else "not_solved")
                    )
                )
            ),
            float(NEW_J_MIN_CONCENTRATION) if np.isfinite(NEW_J_MIN_CONCENTRATION) else _nan,
            float(NEW_J_NEGATIVE_TOL),
            float(NEW_J_REL_RESIDUAL) if np.isfinite(NEW_J_REL_RESIDUAL) else _nan,
            (
                float(NEW_J_REL_RESIDUAL_INVALID_THRESHOLD)
                if NEW_J_REL_RESIDUAL_INVALID_THRESHOLD > 0.0
                else _nan
            ),
            float(_FULL_COMPARE_DIRECT_FINAL_R2) if np.isfinite(_FULL_COMPARE_DIRECT_FINAL_R2) else _nan,
            float(_FULL_COMPARE_DIRECT_FINAL_REL) if np.isfinite(_FULL_COMPARE_DIRECT_FINAL_REL) else _nan,
            float(_FULL_COMPARE_DIRECT_FINAL_RMAX) if np.isfinite(_FULL_COMPARE_DIRECT_FINAL_RMAX) else _nan,
            float(_FULL_COMPARE_GMRES_FINAL_R2) if np.isfinite(_FULL_COMPARE_GMRES_FINAL_R2) else _nan,
            float(_FULL_COMPARE_GMRES_FINAL_REL) if np.isfinite(_FULL_COMPARE_GMRES_FINAL_REL) else _nan,
            float(_FULL_COMPARE_GMRES_FINAL_RMAX) if np.isfinite(_FULL_COMPARE_GMRES_FINAL_RMAX) else _nan,
            float(_FULL_COMPARE_FINAL_D2) if np.isfinite(_FULL_COMPARE_FINAL_D2) else _nan,
            float(_FULL_COMPARE_FINAL_D2_REL) if np.isfinite(_FULL_COMPARE_FINAL_D2_REL) else _nan,
            float(_FULL_COMPARE_FINAL_DINF) if np.isfinite(_FULL_COMPARE_FINAL_DINF) else _nan,
            n_in_eval,
            n_out_eval,
            len(solute_entrance_nodes),
            len(solute_exit_nodes),
            1,
            float(c_bulk_new) if c_bulk_new is not None and np.isfinite(c_bulk_new) else _nan,
            C0,
            Q_total_all,
        PENETRATION_AXIS,
        PERMEATION_PLANE_AREA_NM2,
        PERMEATION_PLANE_AREA_UM2,
        len(solvent_feasible_throat_ids),
        len(solute_accessible_throat_ids),
        int(bool(AUTO_PRUNE_OOB_CONCENTRATION)),
        int(AUTO_PRUNE_OOB_PASS),
        float(AUTO_PRUNE_OOB_LOW_THRESHOLD),
        float(AUTO_PRUNE_OOB_HIGH_THRESHOLD),
        int(len(SOLUTE_EXCLUDE_PORE_IDS)),
        ",".join(str(x) for x in sorted(SOLUTE_EXCLUDE_PORE_IDS)),
        ],
    }
)
_old_j_output_mask = results_df["Parameter"].astype(str).str.contains(
    "old_J|\\(old_J\\)", na=False, regex=True
)
if bool(_old_j_output_mask.any()):
    results_df = results_df.loc[~_old_j_output_mask].reset_index(drop=True)

results_excel = output_path / get_output_filename(f'{SIEVE_SAMPLE_NAME}_sieving_coefficient_results')
results_df.to_excel(results_excel, index=False)
print(f"  结果已保存: {results_excel}")

# 保存详细的浓度分布
if node_concentrations:
    concentration_df = pd.DataFrame({
        'Pore ID': list(node_concentrations.keys()),
        'Concentration': list(node_concentrations.values())
    })
    concentration_excel = output_path / get_output_filename(f'{SIEVE_SAMPLE_NAME}_concentration_distribution')
    concentration_df.to_excel(concentration_excel, index=False)
    print(f"  浓度分布已保存: {concentration_excel}")
    _cvals = pd.to_numeric(concentration_df.get("Concentration", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    _mask_finite = np.isfinite(_cvals)
    _cvals_f = _cvals[_mask_finite]
    _n_total_c = int(_cvals.size)
    _n_finite_c = int(_cvals_f.size)
    _n_zero_c = int(np.sum(_cvals_f == 0.0)) if _n_finite_c > 0 else 0
    _n_small_c = int(np.sum(np.abs(_cvals_f) < 1e-12)) if _n_finite_c > 0 else 0
    print(
        "  [new_J] 浓度统计: "
        f"total={_n_total_c}, finite={_n_finite_c}, "
        f"C==0: {_n_zero_c}, |C|<1e-12: {_n_small_c}"
    )

if throat_Pe:
    pe_df = pd.DataFrame({
        'Throat ID': list(throat_Pe.keys()),
        'Peclet Number (Pe)': list(throat_Pe.values())
    })
    pe_excel = output_path / get_output_filename(f'{SIEVE_SAMPLE_NAME}_peclet_distribution')
    pe_df.to_excel(pe_excel, index=False)
print(f"  Peclet 分布已保存: {pe_excel}")

if _tid_list:
    hindrance_df = pd.DataFrame({
        'Throat ID': _tid_list,
        'Lambda (r_s/R_T)': _lam_list,
        'Kd (DD2006 diffusive hindrance)': _kd_list,
        'Kc (DD2006 convective hindrance)': _kc_list,
        'eta (1-lambda)^2': _partition_eff_list,
    })
    hindrance_excel = output_path / get_output_filename(f'{SIEVE_SAMPLE_NAME}_throat_hindrance_factors')
    hindrance_df.to_excel(hindrance_excel, index=False)
    print(f"  喉 lambda / Kd / Kc / η 逐喉表已保存: {hindrance_excel}")

_qdc_sorted = sorted(solute_accessible_throat_ids)
if _qdc_sorted:
    qdc_df = pd.DataFrame(
        {
            "Throat ID": _qdc_sorted,
            "Q (m3/s)": [float(throat_Q_stat.get(t, float("nan"))) for t in _qdc_sorted],
            "diff_coeff (m3/s)": [
                float(throat_diff_coeff_stat.get(t, float("nan"))) for t in _qdc_sorted
            ],
        }
    )
    qdc_excel = output_path / get_output_filename(
        f"{SIEVE_SAMPLE_NAME}_throat_Q_diff_coeff_distribution"
    )
    qdc_df.to_excel(qdc_excel, index=False)
    print(f"  喉 Q / diff_coeff 逐喉表已保存: {qdc_excel}")


def _nonzero_finite_ravel(a: np.ndarray):
    """展平后仅保留有限且非零的矩阵元（全部参与直方图，不子采样）。"""
    v = np.asarray(a, dtype=np.float64).ravel()
    v = v[np.isfinite(v) & (v != 0.0)]
    return v if v.size > 0 else None


def _log10_abs_for_hist(v):
    """非零有限矩阵元 -> log10(|.|+1e-300)，用于柱状图分箱（与 Pe 右图一致）。"""
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return None
    return np.log10(np.abs(v) + 1e-300)


# 与 Peclet&A 诊断图同步：列出「远弱于主流」的矩阵元（默认 |A|< 非零|A| 的 1% 分位 ×1e-4）
_SMALL_A_ABS_PERCENTILE = 1.0
_SMALL_A_SCALE = 1e-4


def _df_small_matrix_entries(
    A_mat: np.ndarray,
    *,
    matrix_label: str,
    solute_nodes_list: list,
    assembly_audit,
    percentile: float,
    scale: float,
) -> pd.DataFrame:
    """
    在当前矩阵的非零有限元上取 |A_ij| 的 `percentile` 分位 p_cut，阈值 thr=p_cut*scale；
    输出所有满足 0<|A_ij|<thr 的位置；若提供 new_J 组装分项则附 Q/Kc/diff/Pe 等说明。
    """
    A = np.asarray(A_mat, dtype=np.float64)
    if A.size == 0:
        return pd.DataFrame()
    mask = np.isfinite(A) & (A != 0.0)
    abs_vals = np.abs(A[mask])
    if abs_vals.size == 0:
        return pd.DataFrame()
    p_cut = float(np.percentile(abs_vals, float(percentile)))
    if not np.isfinite(p_cut) or p_cut <= 0.0:
        return pd.DataFrame()
    thr = p_cut * float(scale)
    if not np.isfinite(thr) or thr <= 0.0:
        return pd.DataFrame()
    rows: list[dict] = []
    ri, cj = np.nonzero(mask)
    for ii, jj in zip(ri, cj):
        ii = int(ii)
        jj = int(jj)
        aij = float(A[ii, jj])
        ab = abs(aij)
        if not (ab < thr):
            continue
        pid_i = solute_nodes_list[ii] if 0 <= ii < len(solute_nodes_list) else ii
        pid_j = solute_nodes_list[jj] if 0 <= jj < len(solute_nodes_list) else jj
        rec: dict = {
            "matrix": matrix_label,
            "row_i": ii,
            "col_j": jj,
            "pore_id_row": pid_i,
            "pore_id_col": pid_j,
            "A_ij": aij,
            "abs_A_ij": ab,
            "p_abs_nonzero_pctile": p_cut,
            "threshold_pctile_x_scale": thr,
            "percentile_used": float(percentile),
            "scale_applied": float(scale),
        }
        comps = []
        if assembly_audit is not None and ii < len(assembly_audit):
            comps = [e for e in assembly_audit[ii] if int(e["col"]) == jj]
        if comps:
            ssum = sum(float(e["delta"]) for e in comps)
            resid = float(aij) - float(ssum)
            kinds = ";".join(str(e.get("kind", "")) for e in comps)
            t_ids = ";".join(
                (str(int(e["t_id"])) if e.get("t_id") is not None else "")
                for e in comps
            )
            part_labels = [_audit_component_label(e) for e in comps]
            phys_zh = [_audit_physics_meta(e).get("physics_components_zh", "") for e in comps]
            rec["n_parts"] = len(comps)
            rec["kinds"] = kinds
            rec["throat_ids"] = t_ids
            rec["component_labels"] = " + ".join(part_labels)
            rec["sum_delta"] = float(ssum)
            rec["residual_A_minus_sum"] = float(resid)
            rec["physics_components_zh"] = " || ".join(phys_zh)
        else:
            rec["n_parts"] = np.nan
            rec["kinds"] = ""
            rec["throat_ids"] = ""
            rec["component_labels"] = ""
            rec["sum_delta"] = np.nan
            rec["residual_A_minus_sum"] = np.nan
            rec["physics_components_zh"] = ""
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["abs_A_ij"], ascending=True, ignore_index=True)


def _enrich_small_A_df_with_graph(
    df: pd.DataFrame,
    G,
    *,
    throat_flows_map: dict,
    throat_Kc_map: dict,
    throat_diff_map: dict,
    throat_Pe_map: dict,
) -> pd.DataFrame:
    """
    在极小 A 表上按 (pore_id_row, pore_id_col) 在 G 上查边：非对角取唯一邻接喉的 Q/Kc/diff/Pe；
    对角元列出本孔所有邻接喉的物理量摘要（A_ii 常为多项之和，不单属一条喉）。
    """
    if df is None or len(df) == 0:
        return df
    if G is None:
        return df

    def _scalar_tid(tid):
        if tid is None:
            return np.nan
        try:
            return int(tid)
        except (TypeError, ValueError):
            return np.nan

    ltid: list = []
    lq: list = []
    lkc: list = []
    ldiff: list = []
    lpe: list = []
    lnote: list = []

    for _, row in df.iterrows():
        pi = int(row["pore_id_row"])
        pj = int(row["pore_id_col"])
        ri = int(row["row_i"])
        rj = int(row["col_j"])

        if ri != rj:
            if G.has_edge(pi, pj):
                tid = _scalar_tid(G[pi][pj].get("throat_id"))
                ltid.append(tid)
                if np.isfinite(tid):
                    t = int(tid)
                    lq.append(float(throat_flows_map.get(t, float("nan"))))
                    lkc.append(float(throat_Kc_map.get(t, float("nan"))))
                    ldiff.append(float(throat_diff_map.get(t, float("nan"))))
                    lpe.append(float(throat_Pe_map.get(t, float("nan"))))
                else:
                    lq.append(float("nan"))
                    lkc.append(float("nan"))
                    ldiff.append(float("nan"))
                    lpe.append(float("nan"))
                lnote.append("offdiag_direct_edge")
            else:
                ltid.append(np.nan)
                lq.append(float("nan"))
                lkc.append(float("nan"))
                ldiff.append(float("nan"))
                lpe.append(float("nan"))
                lnote.append("offdiag_no_graph_edge")
        else:
            ltid.append(np.nan)
            lq.append(float("nan"))
            lkc.append(float("nan"))
            ldiff.append(float("nan"))
            lpe.append(float("nan"))
            segs = []
            try:
                for nb in G.neighbors(pi):
                    ed = G[pi][nb]
                    tid = _scalar_tid(ed.get("throat_id"))
                    if not np.isfinite(tid):
                        continue
                    t = int(tid)
                    _q = throat_flows_map.get(t, float("nan"))
                    _kc = throat_Kc_map.get(t, float("nan"))
                    _dc = throat_diff_map.get(t, float("nan"))
                    _pe = throat_Pe_map.get(t, float("nan"))
                    segs.append(
                        f"t={t} Q={_q:.6e} Kc={_kc:.6g} diff={_dc:.6e} Pe={_pe:.6g}"
                    )
            except Exception:
                pass
            lnote.append(
                "diag_all_incident_throats: " + (" | ".join(segs) if segs else "no_neighbors")
            )

    out = df.copy()
    out["lookup_throat_id"] = ltid
    out["lookup_Q_m3s"] = lq
    out["lookup_Kc"] = lkc
    out["lookup_diff_m3s"] = ldiff
    out["lookup_Pe"] = lpe
    out["lookup_note"] = lnote
    return out


def _bar_hist_ax(ax, values, bins, title, xlabel, color, empty_msg="No data"):
    """PNG 诊断图仅用 ASCII/英文，避免 DejaVu 缺 CJK 字元警告。"""
    ax.set_title(title, fontsize=10)
    if values is None or np.asarray(values).size == 0:
        ax.text(0.5, 0.5, empty_msg, ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        return
    counts, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = np.diff(edges)
    ax.bar(centers, counts, width=w * 0.92, color=color, edgecolor="white", linewidth=0.35)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")


# 喉 lambda / Kd / Kc / η：PNG 柱状图（溶质可通行喉，与 D3.1 一致）
if DIAGNOSTIC_PNGS and _tid_list:
    print("\n生成喉 lambda / Kd / Kc / η=(1−λ)² 分布图（PNG）...")
    fig_lk, axes_lk = plt.subplots(2, 2, figsize=(12, 9))
    ax_lk00, ax_lk01 = axes_lk[0, 0], axes_lk[0, 1]
    ax_lk10, ax_lk11 = axes_lk[1, 0], axes_lk[1, 1]
    _la_np = np.asarray(_lam_list, dtype=float)
    _kd_np = np.asarray(_kd_list, dtype=float)
    _kc_np = np.asarray(_kc_list, dtype=float)
    _eta_np = np.asarray(_partition_eff_list, dtype=float)
    _bar_hist_ax(ax_lk00, _la_np, 80, "Throat lambda (r_s/R_T)", "lambda", "#6a3d9a")
    _bar_hist_ax(
        ax_lk01,
        _kd_np,
        80,
        "Throat Kd (DD2006)",
        "Kd",
        "#c44e52",
    )
    _bar_hist_ax(ax_lk10, _kc_np, 80, "Throat Kc", "Kc", "#2f9e44")
    _bar_hist_ax(ax_lk11, _eta_np, 80, "Throat eta (1-lambda)^2", "eta", "#1864ab")
    fig_lk.suptitle(
        f"{SIEVE_SAMPLE_NAME} - throat lambda, Kd, Kc, eta (solute-accessible)",
        fontsize=11,
    )
    fig_lk.tight_layout(rect=[0, 0, 1, 0.96])
    hindrance_png = output_path / get_output_filename(
        f"{SIEVE_SAMPLE_NAME}_throat_hindrance_factors", ".png"
    )
    fig_lk.savefig(str(hindrance_png), dpi=150, bbox_inches="tight")
    plt.close(fig_lk)
    print(f"  喉 lambda / Kd / Kc / η 分布图已保存: {hindrance_png}")


# 喉 Q 与 diff_coeff：PNG 柱状图（溶质可通行喉，D2.7）
if DIAGNOSTIC_PNGS and (throat_Q_stat or throat_diff_coeff_stat):
    print("\n生成喉 Q 与 diff_coeff 分布图（PNG）...")
    fig_qdc, axes_qdc = plt.subplots(2, 2, figsize=(12, 9))
    ax_q0, ax_q1 = axes_qdc[0, 0], axes_qdc[0, 1]
    ax_d0, ax_d1 = axes_qdc[1, 0], axes_qdc[1, 1]
    _Qv = np.asarray(list(throat_Q_stat.values()), dtype=float)
    _bar_hist_ax(
        ax_q0,
        _Qv if _Qv.size > 0 else None,
        80,
        "Throat Q (m3/s)",
        "Q (m3/s)",
        "#e67700",
        "No Q",
    )
    _bar_hist_ax(
        ax_q1,
        np.log10(np.maximum(_Qv, 1e-300)) if _Qv.size > 0 else None,
        80,
        "log10(Q+1e-300)",
        "log10(Q+1e-300)",
        "#e67700",
        "No Q",
    )
    _dc_all = np.asarray(list(throat_diff_coeff_stat.values()), dtype=float)
    _dc_p = _dc_all[(_dc_all > 0.0) & np.isfinite(_dc_all)]
    _bar_hist_ax(
        ax_d0,
        _dc_p if _dc_p.size > 0 else None,
        80,
        "Throat diff_coeff (m3/s), >0 only",
        "diff_coeff (m3/s)",
        "#087f5b",
        "No positive diff_coeff",
    )
    _bar_hist_ax(
        ax_d1,
        np.log10(_dc_p) if _dc_p.size > 0 else None,
        80,
        "log10(diff_coeff)",
        "log10(diff_coeff)",
        "#087f5b",
        "No positive diff_coeff",
    )
    fig_qdc.suptitle(
        f"{SIEVE_SAMPLE_NAME} - throat Q & diff_coeff (solute-accessible; diff=D_eff*A_solute/L)",
        fontsize=11,
    )
    fig_qdc.tight_layout(rect=[0, 0, 1, 0.96])
    qdc_png = output_path / get_output_filename(
        f"{SIEVE_SAMPLE_NAME}_throat_Q_diff_coeff_diagnostics", ".png"
    )
    fig_qdc.savefig(str(qdc_png), dpi=150, bbox_inches="tight")
    plt.close(fig_qdc)
    print(f"  喉 Q 与 diff_coeff 分布图已保存: {qdc_png}")


# Peclet 与浓度方程矩阵 A：PNG 柱状图；矩阵子图对非零元的 log10(|A_ij|+1e-300) 分箱
if DIAGNOSTIC_PNGS and (throat_Pe or n_solute_nodes > 0):
    print("\n生成 Peclet 与浓度矩阵元分布图（PNG 柱状图）...")
    fig_m, axes_m = plt.subplots(2, 2, figsize=(12, 9))
    ax00, ax01, ax10, ax11 = axes_m[0, 0], axes_m[0, 1], axes_m[1, 0], axes_m[1, 1]
    if throat_Pe:
        pe_arr = np.array(list(throat_Pe.values()), dtype=float)
        _bar_hist_ax(ax00, pe_arr, 80, "Throat Peclet Pe", "Pe", "#7f7f7f")
        _bar_hist_ax(
            ax01,
            np.log10(np.abs(pe_arr) + 1e-300),
            80,
            "log10(|Pe|+1e-300)",
            "log10(|Pe|+1e-300)",
            "#7f7f7f",
        )
    else:
        _bar_hist_ax(
            ax00,
            None,
            80,
            "Throat Peclet Pe",
            "Pe",
            "#7f7f7f",
            "No Pe (e.g. INCLUDE_DIFFUSION=False)",
        )
        _bar_hist_ax(ax01, None, 80, "log10(|Pe|+1e-300)", "log10(|Pe|+1e-300)", "#7f7f7f", "No Pe")
    if n_solute_nodes > 0:
        # 与线性求解一致：优先用 C_bulk 边界处理后的 A_eval（约束行已按 max|A_transport|×10 整行缩放 A、b）；
        # 若求解未产出 A_eval，则回退为组装直出 A_conc_*（入口/出口锚定行常为 1，与传输系数可差许多个数量级）。
        _A_hist_new = A_eval_new if A_eval_new is not None else A_conc_new
        _A_hist_old = A_eval_old if A_eval_old is not None else A_conc_old
        v_new = (
            np.array([], dtype=float)
            if OLD_J_ONLY
            else _nonzero_finite_ravel(_A_hist_new)
        )
        v_old = _nonzero_finite_ravel(_A_hist_old)
        _hist_src_new = (
            "new_J skipped"
            if OLD_J_ONLY
            else ("A_eval (scaled rows)" if A_eval_new is not None else "A_conc raw")
        )
        _hist_src_old = "A_eval (scaled rows)" if A_eval_old is not None else "A_conc raw"
        _bar_hist_ax(
            ax10,
            _log10_abs_for_hist(v_new),
            100,
            (
                f"A |log10| ({_hist_src_new}, n={n_solute_nodes})"
            ),
            "log10(|A_ij|+1e-300)",
            "#2f9e44",
            "No nonzero entries" if not OLD_J_ONLY else "old_J only (default)",
        )
        _bar_hist_ax(
            ax11,
            _log10_abs_for_hist(v_old),
            100,
            f"A |log10| ({_hist_src_old}, n={n_solute_nodes})",
            "log10(|A_ij|+1e-300)",
            "#1864ab",
            "No nonzero entries",
        )
    else:
        ax10.text(0.5, 0.5, "n_solute_nodes=0", ha="center", va="center", transform=ax10.transAxes, fontsize=11)
        ax11.text(0.5, 0.5, "n_solute_nodes=0", ha="center", va="center", transform=ax11.transAxes, fontsize=11)
        ax10.set_title("A nonzero log10|.| (new_J)", fontsize=10)
        ax11.set_title("A nonzero log10|.| (old_J)", fontsize=10)
    fig_m.suptitle(
        f"{SIEVE_SAMPLE_NAME} - Peclet & matrix A\n"
        "Bottom: log10(|A_ij|+1e-300) over nonzero entries; "
        "when A_eval is available, rows match the linear solve (constraint rows scaled with A and b).",
        fontsize=11,
    )
    fig_m.tight_layout(rect=[0, 0, 1, 0.96])
    pe_mat_png = output_path / get_output_filename(
        f"{SIEVE_SAMPLE_NAME}_peclet_and_concentration_matrix_diagnostics", ".png"
    )
    fig_m.savefig(str(pe_mat_png), dpi=150, bbox_inches="tight")
    plt.close(fig_m)
    print(f"  Peclet 与浓度矩阵元分布图已保存: {pe_mat_png}")

    # 极小 |A_ij|：与上图同一矩阵口径（优先 A_eval），阈值 = 非零|A| 的 p% 分位 × scale（默认 p=1, scale=1e-4）
    if n_solute_nodes > 0:
        _A_small_new = (
            (A_eval_new if A_eval_new is not None else A_conc_new)
            if not OLD_J_ONLY
            else None
        )
        _A_small_old = A_eval_old if A_eval_old is not None else A_conc_old
        _pct = _SMALL_A_ABS_PERCENTILE
        _sc = _SMALL_A_SCALE
        df_sn = (
            _df_small_matrix_entries(
                _A_small_new,
                matrix_label="new_J",
                solute_nodes_list=solute_nodes,
                assembly_audit=assembly_audit_new,
                percentile=_pct,
                scale=_sc,
            )
            if _A_small_new is not None
            else pd.DataFrame()
        )
        df_so = _df_small_matrix_entries(
            _A_small_old,
            matrix_label="old_J",
            solute_nodes_list=solute_nodes,
            assembly_audit=None,
            percentile=_pct,
            scale=_sc,
        )
        df_sn = _enrich_small_A_df_with_graph(
            df_sn,
            G_solute_accessible,
            throat_flows_map=throat_flows,
            throat_Kc_map=throat_Kc,
            throat_diff_map=throat_diff_coeff_stat,
            throat_Pe_map=throat_Pe,
        )
        df_so = _enrich_small_A_df_with_graph(
            df_so,
            G_solute_accessible,
            throat_flows_map=throat_flows,
            throat_Kc_map=throat_Kc,
            throat_diff_map=throat_diff_coeff_stat,
            throat_Pe_map=throat_Pe,
        )
        _n_sn = len(df_sn)
        _n_so = len(df_so)
        print(
            f"  极小矩阵元（|A_ij| < 非零|A| 的 {_pct:g}% 分位 × {_sc:g}；"
            f"与上图矩阵口径一致）: new_J 条数={_n_sn}，old_J 条数={_n_so}。"
            " 已附 lookup_Q/Kc/diff/Pe（非对角=该边喉；对角=本孔各邻喉摘要）。"
            + (
                " new_J 矩阵分项分解另需 --assembly-row-audit。"
                if (not OLD_J_ONLY) and assembly_audit_new is None and _n_sn > 0
                else ""
            )
        )
        if _n_sn > 0 or _n_so > 0:
            small_a_path = output_path / get_output_filename(
                f"{SIEVE_SAMPLE_NAME}_matrix_small_A_entries", ".xlsx"
            )
            try:
                with pd.ExcelWriter(str(small_a_path), engine="openpyxl") as writer:
                    if _n_sn > 0:
                        df_sn.to_excel(writer, sheet_name="new_J", index=False)
                    if _n_so > 0:
                        df_so.to_excel(writer, sheet_name="old_J", index=False)
                print(f"  极小矩阵元表已保存: {small_a_path}")
            except Exception as ex:
                small_a_csv = output_path / get_output_filename(
                    f"{SIEVE_SAMPLE_NAME}_matrix_small_A_entries", ".csv"
                )
                parts_csv = []
                if _n_sn > 0:
                    parts_csv.append(df_sn.assign(matrix_plane="new_J"))
                if _n_so > 0:
                    parts_csv.append(df_so.assign(matrix_plane="old_J"))
                pd.concat(parts_csv, ignore_index=True).to_csv(
                    small_a_csv, index=False, encoding="utf-8-sig"
                )
                print(f"  Excel 失败 ({ex})，极小矩阵元已写入 CSV: {small_a_csv}")

# ====================================================================
# 浓度三维可视化（压强表与压强三维图已在阶段 B 后写出）
# ====================================================================
def _pore_conc_array_for_3d_viz(c_src):
    """与 pore_ids 同长；无浓度键的孔为 NaN。坐标用全几何 pore_coords（与压强子集 viz 区分）。"""
    return np.array(
        [
            float(c_src[pid])
            if c_src and pid in c_src
            else np.nan
            for pid in pore_ids
        ],
        dtype=float,
    )


def _write_concentration_3d_html(c_src, file_suffix):
    """
    file_suffix: \"\" 汇总（node_concentrations，与历史文件名一致） | \"_new_J\" | \"_old_J\"
    输出: <sample>_concentration_distribution_3d{suffix}.html
    着色：log10(max(|C|, eps))，跨多个数量级仍可分辨；符号仅体现在 hover 的真实 C。
    """
    if NO_HTML:
        tag = file_suffix.strip("_") or "default"
        print(f"  浓度分布三维 HTML 已跳过 ({tag}, --no-html)")
        return
    arr = _pore_conc_array_for_3d_viz(c_src)
    mask = ~np.isnan(arr)
    if not np.any(mask):
        return
    _c_viz_floor = 1e-30
    c_raw = arr[mask]
    color_log = np.log10(np.maximum(np.abs(c_raw), _c_viz_floor))
    tag = file_suffix.lstrip("_") if file_suffix else "汇总"
    if file_suffix == "":
        title_text = (
            f"{SIEVE_SAMPLE_NAME} - Concentration Distribution<br>"
            "<sub>Color = log10(max(|C|,1e-30))；hover 为真实 C（含符号）</sub>"
        )
    else:
        title_text = (
            f"{SIEVE_SAMPLE_NAME} - Concentration Distribution — {tag}<br>"
            "<sub>Color = log10(max(|C|,1e-30))；hover 为真实 C（含符号）</sub>"
        )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=pore_coords[mask, 0],
            y=pore_coords[mask, 1],
            z=pore_coords[mask, 2],
            mode="markers",
            marker=dict(
                size=pore_radii[mask] * 2,
                color=color_log,
                colorscale="Plasma",
                colorbar=dict(title="log10 max(|C|,1e-30)", x=1.1),
                line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
                opacity=0.8,
                showscale=True,
            ),
            name=f"Pores ({tag})",
            text=[
                f"Pore ID: {pid}<br>Concentration: {c:.6g}<br>log10 max(|C|,1e-30): {lg:.4f}<br>Radius: {r:.2f} nm"
                for pid, c, lg, r in zip(
                    np.array(pore_ids, dtype=object)[mask],
                    c_raw,
                    color_log,
                    pore_radii[mask],
                )
            ],
            hovertemplate="<b>%{text}</b><extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title_text, x=0.5),
        scene=dict(
            xaxis_title="X (nm)",
            yaxis_title="Y (nm)",
            zaxis_title="Z (nm)",
            aspectmode="data",
        ),
        width=1200,
        height=800,
    )
    html_path = output_path / get_output_filename(
        f"{SIEVE_SAMPLE_NAME}_concentration_distribution_3d{file_suffix}", ".html"
    )
    fig.write_html(str(html_path))
    print(f"  浓度分布三维图已保存 ({tag}): {html_path}")


if node_concentrations_new is not None or node_concentrations_old is not None or node_concentrations:
    print("\n创建浓度分布三维图...")
if node_concentrations_new is not None:
    _write_concentration_3d_html(node_concentrations_new, "_new_J")
if node_concentrations_old is not None:
    _write_concentration_3d_html(node_concentrations_old, "_old_J")
if node_concentrations:
    _write_concentration_3d_html(node_concentrations, "")

print("\n" + "=" * 70)
print("计算完成！")
print("=" * 70)
print("\n主要结果（唯一口径）:")
print(f"  使用浓度场: {_c_source_label}")
print(f"  C_in(入口平均) = {C_in_mean:.6e} (std={C_in_std:.3e}, n={n_in_eval})")
print(f"  C_out(出口平均) = {C_out_mean:.6e} (std={C_out_std:.3e}, n={n_out_eval})")
print(f"  sieving_coefficient = C_out/C_in = {sieving_coefficient:.6e}")
print(f"  出口全溶剂网络净流出通量 Q_tot = {Q_total_all:.6e} m^3/s")
print(f"\n边界约束:")
print(f"  入口浓度统一 Dirichlet: C_in = C0 = {C0:.6e}")
print("  出口边界条件: 统一 C_bulk（由两次参数化求解与全局守恒闭合）")
print(f"\n压力参数:")
print(f"  血液静水压: {P_HYDROSTATIC_IN:.1f} mmHg")
print(f"  血浆胶体渗透压: {PI_PLASMA:.1f} mmHg")
print(f"  囊内静水压: {P_HYDROSTATIC_OUT:.1f} mmHg")
print(f"  入口有效压力: {P_IN_EFFECTIVE:.1f} mmHg ({P_IN:.1f} Pa)")
print(f"  出口有效压力: {P_OUT_EFFECTIVE:.1f} mmHg ({P_OUT:.1f} Pa)")
print(f"  净驱动压力差: {P_IN_EFFECTIVE - P_OUT_EFFECTIVE:.1f} mmHg ({DELTA_P:.1f} Pa)")

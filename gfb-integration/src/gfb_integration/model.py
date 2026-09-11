"""Self-consistent four-layer GFB transport with conserved flow rates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SerialGFBParameters:
    """Parameters for bulk 1 -> GBM -> bulk 2 -> SD transport."""

    q_water_m3_s: float = 7.8584e-20
    albumin_diffusion_m2_s: float = 9.2608e-11
    bulk1_length_nm: float = 30.0
    bulk1_area_nm2: float = 16000.0
    bulk2_length_nm: float = 30.0
    bulk2_area_nm2: float = 16000.0
    s_gbm: float = 0.005000577408884041
    s_sd: float = 0.47690897161621393
    c_plasma: float = 1.0


def _bulk_exponential(q_water_m3_s: float, length_nm: float, area_nm2: float, diffusion_m2_s: float) -> tuple[float, float]:
    if q_water_m3_s <= 0 or length_nm < 0 or area_nm2 <= 0 or diffusion_m2_s <= 0:
        raise ValueError("Water flow, bulk area, and diffusion must be positive; length cannot be negative")
    peclet = q_water_m3_s * (length_nm * 1e-9) / ((area_nm2 * 1e-18) * diffusion_m2_s)
    return peclet, math.exp(peclet)


def solve_serial_gfb(parameters: SerialGFBParameters) -> dict[str, float]:
    """Solve the self-consistent concentration chain and report conservation checks.

    Concentrations are equivalent albumin concentrations. ``c_star`` is the
    conserved albumin-to-water flow ratio, Q_albumin/Q_water. The same water
    and albumin flow rates are used in all four layers.
    """

    p = parameters
    if not 0 <= p.s_gbm <= 1 or not 0 <= p.s_sd <= 1 or p.c_plasma <= 0:
        raise ValueError("Sieving coefficients must be in [0,1] and c_plasma must be positive")
    pe1, exp1 = _bulk_exponential(p.q_water_m3_s, p.bulk1_length_nm, p.bulk1_area_nm2, p.albumin_diffusion_m2_s)
    pe2, exp2 = _bulk_exponential(p.q_water_m3_s, p.bulk2_length_nm, p.bulk2_area_nm2, p.albumin_diffusion_m2_s)

    def state(c_star: float) -> tuple[float, float, float]:
        c_gbm_in = c_star + (p.c_plasma - c_star) * exp1
        c_gbm_out = p.s_gbm * c_gbm_in
        c_sd_in = c_star + (c_gbm_out - c_star) * exp2
        return c_gbm_in, c_gbm_out, c_sd_in

    def closure(c_star: float) -> float:
        return c_star - p.s_sd * state(c_star)[2]

    f0 = closure(0.0)
    f1 = closure(p.c_plasma)
    denominator = f1 - f0
    if abs(denominator) < 1e-15:
        raise RuntimeError("Serial-layer closure is singular")
    c_star = -f0 * p.c_plasma / denominator
    c_gbm_in, c_gbm_out, c_sd_in = state(c_star)
    c_sd_out = c_star
    q_albumin = p.q_water_m3_s * c_star

    residuals = {
        "bulk1_equation_residual": c_gbm_in - (c_star + (p.c_plasma - c_star) * exp1),
        "gbm_equation_residual": c_gbm_out - p.s_gbm * c_gbm_in,
        "bulk2_equation_residual": c_sd_in - (c_star + (c_gbm_out - c_star) * exp2),
        "sd_equation_residual": c_sd_out - p.s_sd * c_sd_in,
        "albumin_flow_definition_residual_m3_s": q_albumin - p.q_water_m3_s * c_sd_out,
    }
    result = {
        **asdict(p),
        "pe_bulk1": pe1,
        "pe_bulk2": pe2,
        "c_gbm_in_over_c0": c_gbm_in / p.c_plasma,
        "c_gbm_out_over_c0": c_gbm_out / p.c_plasma,
        "c_sd_in_over_c0": c_sd_in / p.c_plasma,
        "c_bowman_over_c0": c_sd_out / p.c_plasma,
        "gfb_sieving": c_sd_out / p.c_plasma,
        "plasma_to_gbm_transfer": c_gbm_in / p.c_plasma,
        "gbm_transfer": c_gbm_out / c_gbm_in if c_gbm_in else 0.0,
        "gbm_to_sd_transfer": c_sd_in / c_gbm_out if c_gbm_out else 0.0,
        "sd_transfer": c_sd_out / c_sd_in if c_sd_in else 0.0,
        "q_albumin_at_c0_1_m3_s": q_albumin / p.c_plasma,
        "q_water_bulk1_m3_s": p.q_water_m3_s,
        "q_water_gbm_m3_s": p.q_water_m3_s,
        "q_water_bulk2_m3_s": p.q_water_m3_s,
        "q_water_sd_m3_s": p.q_water_m3_s,
        "q_albumin_bulk1_at_c0_1_m3_s": q_albumin / p.c_plasma,
        "q_albumin_gbm_at_c0_1_m3_s": q_albumin / p.c_plasma,
        "q_albumin_bulk2_at_c0_1_m3_s": q_albumin / p.c_plasma,
        "q_albumin_sd_at_c0_1_m3_s": q_albumin / p.c_plasma,
        **residuals,
        "max_abs_equation_residual": max(abs(value) for value in residuals.values()),
    }
    return result

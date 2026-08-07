"""Hydrodynamic hindrance factors for spherical solutes in cylindrical pores."""

from __future__ import annotations

import math


def dd2006_diffusive_hindrance(lambda_ratio: float) -> float:
    """Return the Dechadilok-Deen (2006) diffusive hindrance factor ``K_D``.

    The polynomial/logarithmic approximation is used for ``0 <= lambda <= 0.95``
    and the near-occlusion asymptote is used for ``0.95 < lambda < 1``. A solute
    with ``lambda >= 1`` is geometrically inaccessible and therefore has zero
    mobility.
    """
    lam = float(lambda_ratio)
    if not (0.0 <= lam < 1.0):
        return 0.0

    if lam <= 0.95:
        lambda_log_lambda = 0.0 if lam == 0.0 else lam * math.log(lam)
        numerator = (
            1.0
            + (9.0 / 8.0) * lambda_log_lambda
            - 1.56034 * lam
            + 0.528155 * lam**2
            + 1.91521 * lam**3
            - 2.81903 * lam**4
            + 0.270788 * lam**5
            + 1.10115 * lam**6
            - 0.435933 * lam**7
        )
        kd = numerator / (1.0 - lam) ** 2
    else:
        kd = 0.984 * ((1.0 - lam) / lam) ** 2.5

    return min(1.0, max(0.0, float(kd)))


def dd2006_convective_hindrance(lambda_ratio: float) -> float:
    """Return the Dechadilok-Deen (2006) convective hindrance factor ``K_C``."""
    lam = float(lambda_ratio)
    if not (0.0 <= lam < 1.0):
        return 0.0

    numerator = 1.0 + 3.867 * lam - 1.907 * lam**2 - 0.834 * lam**3
    denominator = 1.0 + 1.867 * lam - 0.741 * lam**2
    kc = numerator / denominator
    return max(0.0, float(kc))


def dd2006_hindrance_factors(lambda_ratio: float) -> tuple[float, float]:
    """Return ``(K_D, K_C)`` for the given solute-to-pore radius ratio."""
    return (
        dd2006_diffusive_hindrance(lambda_ratio),
        dd2006_convective_hindrance(lambda_ratio),
    )


def is_abnormal_equivalent_concentration(
    concentration: float,
    *,
    low_threshold: float = 1e-6,
    high_threshold: float = 1e4,
) -> bool:
    """Return whether a solved equivalent concentration triggers cluster pruning."""
    value = float(concentration)
    return (
        not math.isfinite(value)
        or value < float(low_threshold)
        or value > float(high_threshold)
    )

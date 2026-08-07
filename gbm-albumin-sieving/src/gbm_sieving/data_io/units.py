"""Authoritative unit conversions used by GBM analysis and simulation code."""

from __future__ import annotations

import numpy as np

NM_TO_M = 1e-9
NM2_TO_M2 = 1e-18
NM3_TO_M3 = 1e-27
NL_TO_M3 = 1e-12
MMHG_TO_PA = 133.322387415


def nm_to_m(value):
    """Convert nanometres to metres."""
    return np.asarray(value) * NM_TO_M


def nm2_to_m2(value):
    """Convert square nanometres to square metres."""
    return np.asarray(value) * NM2_TO_M2


def nl_per_s_to_m3_per_s(value):
    """Convert nanolitres per second to cubic metres per second."""
    return np.asarray(value) * NL_TO_M3


def m3_per_s_to_nl_per_s(value):
    """Convert cubic metres per second to nanolitres per second."""
    return np.asarray(value) / NL_TO_M3


def mmhg_to_pa(value):
    """Convert millimetres of mercury to pascals."""
    return np.asarray(value) * MMHG_TO_PA


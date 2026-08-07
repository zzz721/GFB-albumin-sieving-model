"""Shared filesystem locations for the cleaned project copy.

Input defaults may point to the surrounding research workspace so the copied
code can reuse existing data without duplicating it. New outputs always stay
inside this project.
"""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
WORKSPACE_ROOT = PROJECT_ROOT.parent


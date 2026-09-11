"""Run the manuscript DD2006 solver on one classified GBM network."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gbm_sieving.simulation.transport.dd2006_conservative_solver import main


if __name__ == "__main__":
    main()

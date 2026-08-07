from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sd_albumin_sieving.visualization.manuscript.figure3_sd_sieving.plot_fig3G import main


if __name__ == "__main__":
    main()

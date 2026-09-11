#!/usr/bin/env python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galactic_evolution.coefficient_grid import main


if __name__ == "__main__":
    main()

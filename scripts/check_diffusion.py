#!/usr/bin/env python
"""Deterministic parity check for both diffusion-distribution routes."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galactic_evolution.diffusion import (
    empirical_particle_coefficients,
    thermal_particle_coefficients,
)


ENERGY_X = 1.0
ANGULAR_MOMENTUM_J = 0.02
DISTRIBUTION_INDEX = 0
EXPECTED_EMPIRICAL = np.array(
    [
        -0.116639691700217,
        11.158607262798611,
        8.50127230388594,
        0.3402448267394774,
        0.006034470579100654,
    ]
)
EXPECTED_THERMAL = np.array(
    [
        -0.4635991174020548,
        105.10416858704765,
        8.529894739740334,
        0.3412675928146399,
        0.006619552453203601,
    ]
)


def main():
    empirical = empirical_particle_coefficients(
        ENERGY_X,
        ANGULAR_MOMENTUM_J,
        DISTRIBUTION_INDEX,
    )
    thermal = thermal_particle_coefficients(
        ENERGY_X,
        ANGULAR_MOMENTUM_J,
        DISTRIBUTION_INDEX,
    )

    np.testing.assert_allclose(empirical, EXPECTED_EMPIRICAL, rtol=1.0e-12)
    np.testing.assert_allclose(thermal, EXPECTED_THERMAL, rtol=1.0e-12)
    print(f"empirical={empirical.tolist()}")
    print(f"thermal={thermal.tolist()}")
    print("diffusion parity check passed")


if __name__ == "__main__":
    main()

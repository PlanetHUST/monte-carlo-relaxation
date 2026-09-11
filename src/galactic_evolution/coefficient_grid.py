"""Generate diffusion-coefficient grids for either supported distribution."""

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .diffusion import (
    DEFAULT_RUNTIME_DATA_DIR,
    empirical_particle_coefficients,
    thermal_particle_coefficients,
)


GRID_POINT_COUNT = 300
ANGULAR_MOMENTUM_MIN = 3.0e-4
ENERGY_MIN_X = 1.0
ENERGY_MAX_X = 1.0e4


def _compute_point(task):
    energy_x, angular_momentum_j, distribution_index, distribution, data_dir = task
    coefficient_function = {
        "empirical": empirical_particle_coefficients,
        "thermal": thermal_particle_coefficients,
    }[distribution]
    return coefficient_function(
        energy_x,
        angular_momentum_j,
        distribution_index,
        data_dir,
    )


def generate_coefficient_grids(
    distribution,
    data_dir=DEFAULT_RUNTIME_DATA_DIR,
    output_dir=DEFAULT_RUNTIME_DATA_DIR,
):
    energy_values_x = np.linspace(ENERGY_MIN_X, ENERGY_MAX_X, GRID_POINT_COUNT)
    angular_momentum_values = np.linspace(
        ANGULAR_MOMENTUM_MIN,
        1 - ANGULAR_MOMENTUM_MIN,
        GRID_POINT_COUNT,
    )
    energy_grid_x, angular_momentum_grid = np.meshgrid(
        energy_values_x,
        angular_momentum_values,
    )
    tasks = [
        (energy_x, angular_momentum_j, 0, distribution, data_dir)
        for energy_x, angular_momentum_j in zip(
            energy_grid_x.ravel(),
            angular_momentum_grid.ravel(),
        )
    ]

    with Pool() as pool:
        coefficients = pool.map(_compute_point, tasks)

    coefficient_grid = np.asarray(coefficients).reshape(
        GRID_POINT_COUNT,
        GRID_POINT_COUNT,
        5,
    )
    output_prefix = "given" if distribution == "empirical" else "thermal"
    output_dir = Path(output_dir)
    for coefficient_index in range(4):
        output_path = output_dir / (
            f"{output_prefix}_result{coefficient_index + 1}.csv"
        )
        pd.DataFrame(coefficient_grid[:, :, coefficient_index]).to_csv(
            output_path,
            index=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate the 300-by-300 diffusion-coefficient grids."
    )
    parser.add_argument("distribution", choices=("empirical", "thermal"))
    args = parser.parse_args()
    generate_coefficient_grids(args.distribution)


if __name__ == "__main__":
    main()

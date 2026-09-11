# Model, data, and execution notes

## Model routes

`galactic_evolution.diffusion` owns the shared diffusion integrals. Its two
public routes differ only in the distribution supplied to those integrals:

- `empirical_particle_coefficients` constructs a Gaussian KDE from the
  `xvalue` column of `data/runtime/copy_data_<index>.csv`.
- `thermal_particle_coefficients` uses the fixed piecewise thermal
  distribution. The annihilation workflow uses this route.

The variables `x` (energy) and `j` (angular momentum) are the established
dimensionless model coordinates. Output coefficient ordering remains
`Dx, Dxx, Dj, Djj, Dxj`.

`angular_momentum_grid.py` is a separate alternative calculation of `Dj`; it
is not another implementation of the five shared diffusion coefficients.

## Data flow

The initialization cells in
`notebooks/initial_conditions_and_analysis.ipynb` write
`data/runtime/copy_data_0.csv`. The two cells generate different unseeded
initial distributions and overwrite the same file, so run only the intended
cell.

The annihilation workflow reads `copy_data_<index>.csv` from `data/runtime/`
and writes later populations plus `energytrack.csv` and the established,
misspelled `angulertrack.csv` there. Grid generators also write their CSV files
to `data/runtime/`. Generated runtime files are ignored by Git; the initial
`copy_data_0.csv` remains tracked.

`data/snapshots/archived/` contains the original `datas/` result set.
`data/snapshots/new_annihilation/` contains the distinct, newer annihilation
result set. These are analysis inputs and are not overwritten by scripts.

Preserved plots and equation references are under `results/figures/`. Their
filenames identify the empirical/thermal coefficient grids, distribution
evolution, or equation shown; the cleanup changed names only, not image bytes.

## Cost and known limits

- `scripts/run_annihilation.py` creates a process pool in each of 1,000
  stochastic iterations.
- Both coefficient distributions use numerical quadrature. Full coefficient
  grids evaluate 90,000 points.
- The alternative angular-momentum grid evaluates 90,000 points, each with one
  quadrature and four double integrals.
- A full annihilation run still has the pre-existing trajectory-index mismatch:
  `ind_track` includes 1000 while a 1,000-row population ends at position 999.
  The structural cleanup intentionally does not change that model behavior.
- Initialization and evolution remain unseeded and are not reproducible by
  default.

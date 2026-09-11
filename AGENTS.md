# Repository Guide

## Runtime and Verification

- Run commands from the repository root: runtime CSV paths are relative to the working directory.
- Use the Conda environment `sevnpy_env` (`conda activate sevnpy_env`). It supplies Python 3.12 plus NumPy, pandas, SciPy, and Matplotlib. The PyCharm setting naming Python 3.14 is stale.
- There is no dependency manifest, test suite, CI, linter, or formatter configuration. Do not invent project-wide checks.
- Safe syntax check: `python -m py_compile src/galactic_evolution/*.py scripts/*.py`.
- Safe import check: `PYTHONPATH=src python -c "import galactic_evolution; import galactic_evolution.annihilation; import galactic_evolution.angular_momentum_grid; import galactic_evolution.coefficient_grid"`.
- Focused numerical check: `python scripts/check_diffusion.py`. It reads `data/runtime/copy_data_0.csv`, compares empirical and thermal coefficient vectors with the pre-refactor baseline, and does not write data.

## Execution Flow

- `notebooks/initial_conditions_and_analysis.ipynb` creates `data/runtime/copy_data_0.csv` and analyzes snapshots. Its two initialization cells generate different distributions and both overwrite the same file; run only the intended one.
- `src/galactic_evolution/diffusion.py` owns the shared diffusion equations. It exposes empirical-KDE and fixed-thermal coefficient routes; annihilation intentionally uses the thermal route.
- `scripts/generate_diffusion_grid.py` selects either distribution and preserves the established `given_result*.csv` and `thermal_result*.csv` output names.
- `src/galactic_evolution/angular_momentum_grid.py` is the parallel 300-by-300 alternative `Dj` calculation.
- `scripts/run_annihilation.py` is the runnable stochastic evolution implementation. Superseded notebook and text copies have been removed.

## Data and Cost

- `data/runtime/` holds the current `copy_data_0.csv` and mutable run outputs. The misspelling `angulertrack.csv` is part of the established interface.
- `data/snapshots/archived/` and `data/snapshots/new_annihilation/` hold distinct result snapshots used by notebook analysis; scripts do not write there. Do not replace them unless the task requires a new run.
- Do not use full generators as routine verification. The annihilation workflow runs 1,000 stochastic iterations and creates a process pool each iteration; each coefficient-grid route evaluates a 300-by-300 integration grid; the alternative angular-momentum grid evaluates 90,000 points, each with one quadrature and four double integrals. All default to the available CPU count.
- `scripts/run_annihilation.slurm` is not a test: it launches `python -u scripts/run_annihilation.py` on the `cpu-pgmf` partition and writes `out.txt`/`err.txt`.
- A full annihilation run currently cannot record its first trajectory row with the checked-in 1,000-row input: `ind_track` includes index 1000, while valid positions end at 999. Do not report a full run as viable without addressing this mismatch.
- Simulation initialization and evolution use unseeded random draws, so regenerated CSVs are not reproducible by default.

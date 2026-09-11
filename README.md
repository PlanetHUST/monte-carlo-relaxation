# Galactic Evolution

This repository models diffusion-driven stellar evolution in dimensionless
energy (`x`) and angular momentum (`j`). It retains two scientifically distinct
distribution choices—an empirical KDE and a fixed thermal distribution—while
sharing one implementation of the diffusion equations.

## Layout

- `src/galactic_evolution/`: diffusion physics and evolution workflows
- `scripts/`: runnable entry points, the Slurm launcher, and a focused check
- `notebooks/`: initialization and archived-result analysis
- `data/runtime/`: current input and generated run files
- `data/snapshots/`: preserved result sets used by the notebook
- `results/figures/`: preserved figures
- `docs/`: model, data, and execution notes

Run every command from the repository root in the `sevnpy_env` Conda
environment. The source layout does not require installing the package because
the entry-point scripts add `src/` to their import path.

## Commands

```bash
conda activate sevnpy_env
python scripts/check_diffusion.py
python scripts/run_annihilation.py
python scripts/generate_diffusion_grid.py thermal
python scripts/generate_diffusion_grid.py empirical
python scripts/generate_angular_momentum_grid.py
```

Only the first command is a routine check. The annihilation workflow performs
1,000 stochastic iterations, and each grid generator evaluates a 300-by-300
integration grid with multiprocessing. See [the execution notes](docs/model-and-data.md)
before launching a full run.

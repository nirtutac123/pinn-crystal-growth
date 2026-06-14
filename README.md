# Crystal Growth PINN (Clean Public Version)

A clean, modular Physics-Informed Neural Network (PINN) project for PDE-driven crystal growth and thermal-fluid experiments.

This public package contains:
- Core training pipeline (`PINN`, `GNPINN`, `AgenticPINN`)
- Equation modules for Heat, KdV, and Navier-Stokes-style coupling
- Model architectures (`MLP`, `SIREN`)
- Visualization and utility helpers
- Synthetic Czochralski-style dataset used by this project

## Included Data

The same project dataset is included at:
- `Data/cz_synthetic_data.csv`

## Quick Start

1. Create and activate a virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run training:

```bash
python main.py --equation heat --epochs 2000 --use_gn
```

## Project Structure

- `main.py` – CLI entry point
- `models/` – neural networks and PINN trainers
- `equations/` – PDE formulations and residuals
- `examples/` – runnable examples
- `utils/` – training and IO helpers
- `visualization/` – plotting utilities
- `Data/` – synthetic dataset and generator

## License

MIT License (see `LICENSE`).
# pinn-crystal-growth

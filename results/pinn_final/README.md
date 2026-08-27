# Final PINN Result Inputs

This folder contains the compact numerical inputs used to regenerate the
publication-ready SVG result figures.

- `summary.csv` contains the final total, PDE, and boundary/initial-condition
  losses for the nine thesis runs.
- `*_history.json` contains the saved training histories for the baseline,
  gradient-normalized, and agentic MLP runs used in the convergence and
  controller-dynamics figures.

The data correspond to the final thesis experiment matrix. Regenerate the SVG
collection from the repository root with:

```bash
python tools/generate_svg_figures.py
```

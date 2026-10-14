# neuroblast-lineage-ode-model

ODE models of *Drosophila* neuroblast (NB) lineage growth and figure generation code for the manuscript "Cell size asymmetry regulates neural stem cell proliferation and lineage expansion in vivo and in silico models".

## Models

| Model | Mechanism |
|---|---|
| base | Constant division and maturation rates (counts only) |
| 1 | Volume-dependent NB/GMC division (saturating power law), no feedback |
| 2 | Symmetric NB divisions lower the NB division threshold (no recovery) |
| 3 | NB growth rate scales with average NB volume (exponent α) |
| 4 | Models 2 + 3 combined: volume-scaled growth and threshold pulled by symmetric divisions |
| 5 | NB division rate self-repressed by NB count (Hill exponent β) |

Each model is run for three genotypes: WT (0% symmetric NB divisions), *mud* mutant (15% symmetric) and nanobody (15% symmetric, NB growth × 0.8).

## Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/bagherilab/neuroblast-lineage-ode-model.git
cd neuroblast-lineage-ode-model
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact package versions used to generate the figures in `figures/`. To use the models with your own (newer) package versions instead, run `pip install .`.

## Generating figures

```bash
python make_figures.py                         # all figures → figures/
python make_figures.py --only manuscript       # manuscript | supplemental
python make_figures.py --out other_dir
```

Output is SVG (text kept as text) and EPS. With the versions in `requirements.txt`, timestamps and SVG ids are fixed, so re-running produces byte-identical files.

## Figures

| File | Contents |
|---|---|
| `figures/models_grid` | Models 1–5 × genotypes, cell counts over 48 h |
| `figures/base_model` | Base model, WT vs. mutant |
| `figures/NB_cellcycle_duration` | NB cell-cycle duration for all models |
| `figures/Model3_alpha_sweep`, `Model4_alpha_sweep` | Models 3 and 4 for α = 1, 2, 3, 4, 10 |
| `figures/Model5_beta_sweep` | Model 5 for β = 1, 2, 3, 4, 10 |

## Layout

```
neuroblast_ode/
  models.py     ODE right-hand sides
  params.py     parameters, initial conditions, genotypes
  solvers.py    solve_ivp wrappers per model
  ccd.py        cell-cycle duration from solutions
  plotting.py   styling, end-of-curve count labels, SVG/EPS export
  figures.py    one function per figure
make_figures.py command-line entry point
```

## License

Copyright (C) 2026 Sophia Kehn Jannetty and Neda Bagheri

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE).

## AI Disclosure

OpenAI's ChatGPT and Anthropic's Claude were used for code refactoring and code documentation.

# Research Overview

## Working Title

Adaptive Agentic Physics-Informed Neural Networks for CFD-Oriented Silicon Single-Crystal Growth Simulation

## Motivation

Silicon single-crystal growth involves coupled heat transfer and fluid-flow dynamics. Conventional CFD simulations are established but can be expensive when repeated parameter studies are required. Physics-Informed Neural Networks (PINNs) provide a scientific machine-learning route because they embed PDE residuals directly into the training objective.

The main training difficulty is that PINN objectives contain several competing components: PDE residual loss, boundary-condition loss, and initial-condition loss. If one component dominates, the model may appear to train well while still failing to satisfy part of the physical problem.

## Research Question

Can adaptive and agentic training control improve PINN training behavior for crystal-growth-oriented PDE problems compared with baseline and gradient-normalized PINN training?

## Implemented Methods

This codebase implements and compares:

- Baseline PINN training
- Gradient-Normalized PINN training
- AgenticPINN, a rule-based adaptive controller that adjusts loss weights using observed training signals

The implementation supports:

- Heat equation benchmark
- Simplified Navier-Stokes-style thermal-fluid crystal-growth model
- MLP and SIREN network architectures
- Gaussian Process Regression surrogate evaluation on corrected Czochralski sweep datasets

## Thesis Contribution

The contribution is not a full industrial CFD replacement. Instead, the project provides a reproducible experimental framework for studying how adaptive loss balancing affects PINN training behavior in a crystal-growth-inspired PDE setting.

The most defensible thesis claims are:

- Adaptive loss control can strongly reduce PDE residuals in selected settings.
- Training behavior depends on the equation, network architecture, and PDE-vs-boundary loss balance.
- For crystal-growth proxy experiments, adaptive methods should be evaluated with both total weighted loss and individual PDE/BC/IC components.
- Corrected Czochralski CFD sweep data can support supervised surrogate validation, uncertainty visualization, and physical interpretation.

## Reproducibility Strategy

Each run should record:

- Equation
- Network type
- Trainer type
- Random seed
- Epoch count
- Collocation points
- Device
- Final total, PDE, and BC/IC losses
- Generated plots and model artifacts

Smoke-test and validation commands are listed in `README.md` and `REPRODUCIBILITY.md`.

## Limitations

- The crystal-growth model is a simplified research proxy.
- Physical superiority should not be claimed from weighted loss alone.
- Runtime comparison requires explicit wall-clock logging during experiment runs.
- The corrected CFD surrogate evaluation uses one-parameter sweeps and does not yet capture coupled multi-parameter process changes.

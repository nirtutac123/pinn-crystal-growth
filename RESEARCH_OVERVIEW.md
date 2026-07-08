# Research Overview

## Working Title

Physics-Based AI Models for Accelerating and Improving CFD Simulations in Silicon Single-Crystal Growth

## Motivation

Silicon single-crystal growth involves coupled heat transfer and fluid-flow dynamics. Conventional CFD simulations can be computationally expensive, especially when repeated parameter studies or control-oriented workflows are needed. Physics-Informed Neural Networks (PINNs) provide a possible surrogate modeling direction because they embed PDE residuals directly into the learning objective.

## Research Question

Can adaptive physics-informed training improve optimization stability and residual minimization for crystal-growth thermal-fluid simulations compared with baseline PINN training?

## Implemented Methods

This codebase implements and compares:

- Baseline PINN training
- Gradient-Normalized PINN training
- AgenticPINN, a rule-based adaptive controller that adjusts loss weights using observed training signals

The implementation supports:

- Heat equation benchmark
- KdV benchmark
- Simplified Navier-Stokes-style thermal-fluid crystal-growth model
- MLP and SIREN network architectures

## Expected Thesis Contribution

The contribution is not a full industrial CFD replacement. Instead, the project provides a reproducible experimental framework for studying how adaptive loss balancing affects PINN training behavior in a crystal-growth-inspired PDE setting.

The most defensible thesis claims are:

- Adaptive loss control can strongly reduce PDE residuals in selected settings.
- Training behavior depends on the equation, network architecture, and PDE-vs-boundary loss balance.
- For crystal-growth proxy experiments, adaptive methods should be evaluated with both total weighted loss and individual PDE/BC/IC components.
- Future work should validate the approach against higher-fidelity CFD or experimental reference data.

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

Smoke-test commands are listed in `README.md`.

## Limitations

- The crystal-growth model is a simplified research proxy.
- Synthetic data is included for demonstration and reproducibility.
- Physical superiority should not be claimed from weighted loss alone.
- Runtime comparison requires explicit wall-clock logging during experiment runs.

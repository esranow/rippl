# ripple

ripple is a PyTorch library for physics-informed neural networks (PINNs) and operator learning. It provides structured components for modeling differential equations and dynamical systems with neural networks that incorporate differential operators, residual constraints, and spectral transformations.

Repository: https://github.com/esranow/rippleml

---

## Features

- **Models**: MLP, Fourier-feature MLP, SIREN, Fourier Neural Operator (FNO)
- **Physics Core**: PDE specification, autograd-based residuals, Conservative Flux wrappers (StreamFunction, VectorPotential), BCs (Dirichlet, Neumann, Periodic)
- **Physics Blocks**: Modular physics-aware neural layers combining fixed operators with learnable corrections
- **Solvers**: Finite-difference and spectral solvers for controlled validation
- **Training**: Causal weighting (binned/continuous), Adaptive loss weighting (NTK, GradNorm), and PINN/Operator learning engine
- **Diagnostics**: Physics Validator (residual/constraint auditing), energy functionals, spectral-domain analysis
- **IO**: Checkpointing, TorchScript/ONNX export, structured experiment metadata

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/esranow/rippleml.git
```

Or for local development:

```bash
git clone https://github.com/esranow/rippleml.git
cd rippleml
pip install -e .
```

## Quick Start

### Training a PINN

```bash
python -m ripple.cli --config ripple/configs/demo_pinn_1d.yaml
```

### Running Tests

```bash
pytest ripple/tests
```

### Example: Prediction and Plotting

```bash
python ripple/examples/predict_and_plot.py
```

## Project Structure

```
ripple/
├── models/          # Neural network architectures
├── physics/         # PDE specifications and residual construction
├── physics_blocks/  # Modular physics-aware neural layers
├── datasets/        # Data generators
├── solvers/         # Numerical solvers
├── training/        # Training engine and callbacks
├── operators/       # Operator learning utilities
├── io/              # Checkpointing and export utilities
├── diagnostics/     # Metrics and analysis tools
├── configs/         # Example configurations
├── examples/        # Example scripts
└── tests/           # Unit tests
```

## Configuration

Example YAML config for PINN training:

```yaml
name: demo_pinn
task: physics_informed_neural_network
model:
  type: mlp
  input_dim: 2
  output_dim: 1
  hidden_layers: [50, 50, 50]
  activation: tanh
training:
  epochs: 100
  learning_rate: 0.001
  save_dir: ./checkpoints
  save_freq: 10
```

## Testing

All modules include unit tests. Run the full test suite:

```bash
pytest ripple/tests -v
```

Run specific test modules:

```bash
pytest ripple/tests/test_models.py
pytest ripple/tests/test_physics.py
```

## Known Limitations

**CRITICAL: High-Fidelity Physics Warnings**

- **Shock Capturing**: No native support for WENO, TVD, or flux limiters. Inviscid hyperbolic PDEs (e.g., inviscid Burgers) will exhibit Gibbs oscillations.
- **Non-Dimensionalization**: No automated scaling for disparate physical constants. Users must manually normalize systems with high stiffness or high Reynolds numbers.
- **Dynamic Topologies**: Hard constraints are currently limited to static geometries. Time-dependent distance fields ($B(t)$) are unsupported.
- **Uncertainty Quantification**: Predictions are deterministic. No native support for Bayesian PINNs or ensembles.

For full technical details, see [LIMITATIONS.md](ripple/LIMITATIONS.md).

## License

MIT License


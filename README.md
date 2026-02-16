# TensorWAV

TensorWAV is a PyTorch library for physics-informed neural networks (PINNs) and operator learning. It provides structured components for modeling differential equations and dynamical systems with neural networks that incorporate differential operators, residual constraints, and spectral transformations.

Repository: https://github.com/esranow/tensorwav

---

## Features

- **Models**: MLP, Fourier-feature MLP, SIREN, Fourier Neural Operator (FNO)
- **Physics Core**: PDE specification, automatic differentiation–based residuals, boundary conditions (Dirichlet, Neumann, Periodic)
- **Physics Blocks**: Modular physics-aware neural layers combining fixed operators with learnable corrections
- **Solvers**: Finite-difference and spectral solvers for controlled validation
- **Training**: Unified engine supporting PINN and operator learning with mixed precision
- **Diagnostics**: Error metrics, energy functionals, spectral-domain analysis
- **IO**: Checkpointing, TorchScript/ONNX export, structured experiment metadata

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/esranow/tensorwav.git#subdirectory=TensorWAV

Or for local development:

```bash
git clone https://github.com/esranow/tensorwav.git
cd tensorwav/TensorWAV
pip install -e .
```

## Quick Start

### Training a PINN

```bash
python -m TensorWAV.cli --config TensorWAV/configs/demo_pinn_1d.yaml
```

### Running Tests

```bash
pytest TensorWAV/tests
```

### Example: Prediction and Plotting

```bash
python TensorWAV/examples/predict_and_plot.py
```

## Project Structure

```
TensorWAV/
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
pytest TensorWAV/tests -v
```

Run specific test modules:

```bash
pytest TensorWAV/tests/test_models.py
pytest TensorWAV/tests/test_physics.py
```

## License

MIT License


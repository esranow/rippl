"""
rippl: Physics Package
Defines PDE specifications, residuals, and boundary conditions.
"""
from rippl.physics.pde import PDESpec
from rippl.physics.residuals import build_residual_fn
from rippl.physics.boundary import BoundaryCondition, DirichletBC, NeumannBC, PeriodicBC

__all__ = [
    "PDESpec",
    "build_residual_fn",
    "BoundaryCondition",
    "DirichletBC",
    "NeumannBC",
    "PeriodicBC",
]

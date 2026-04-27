"""
rippl.physics.equation — Generic PDE residual from a list of (coeff, Operator) terms.
Re-exports Equation from rippl.core.equation for backward compatibility.
"""
from rippl.core.equation import Equation

__all__ = ["Equation"]

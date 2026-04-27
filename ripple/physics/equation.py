"""
ripple.physics.equation — Generic PDE residual from a list of (coeff, Operator) terms.
Re-exports Equation from ripple.core.equation for backward compatibility.
"""
from ripple.core.equation import Equation

__all__ = ["Equation"]

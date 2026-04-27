import warnings
from ripple.models.registry import build_model as _original_build_model, register_model
from ripple.models import mlp, siren, fourier_mlp, fno  # noqa: F401

HIGH_FREQUENCY_OPERATORS = {
    "schrodinger_kinetic", "schrodinger_time",
    "wave", "burgers_advection"
}

class PhysicsModelWarning(UserWarning):
    """Warning for potential spectral bias issues in physics models."""
    pass

def build_model(model_type, config, equation=None):
    """Constructs a model and issues warnings if spectral bias is likely."""
    if model_type == "mlp" and equation is not None:
        # Extract operator types from the equation
        from ripple.core.equation_system import EquationSystem
        if isinstance(equation, EquationSystem):
            operators = [op for eq in equation.equations for _, op in eq.terms]
        else:
            operators = [op for _, op in equation.terms]
            
        op_types = {op.signature()["type"] for op in operators}
        if op_types & HIGH_FREQUENCY_OPERATORS:
            warnings.warn(
                f"Model type 'mlp' may suffer from spectral bias for operators {op_types & HIGH_FREQUENCY_OPERATORS}. "
                "Consider using 'siren' or 'fourier_mlp'.",
                PhysicsModelWarning,
                stacklevel=2
            )
            
    return _original_build_model(model_type, config)

__all__ = ["build_model", "register_model", "PhysicsModelWarning"]

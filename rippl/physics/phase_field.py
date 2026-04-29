"""
Phase-field models for interface dynamics.
Cahn-Hilliard: ∂φ/∂t = M∇²μ, μ = f'(φ) - ε²∇²φ
Allen-Cahn: ∂φ/∂t = -M[f'(φ) - ε²∇²φ]
"""
import torch
from rippl.physics.operators import Operator
from rippl.core.equation import Equation
from rippl.core.equation_system import EquationSystem

class CahnHilliardOperator(Operator):
    """
    Cahn-Hilliard equation:
    ∂φ/∂t = M * ∇²[f'(φ) - ε²∇²φ]
    
    Split into system:
    ∂φ/∂t = M * ∇²μ          (transport equation)
    μ = f'(φ) - ε²∇²φ         (chemical potential)
    
    Double-well: f(φ) = φ²(1-φ)²/4, f'(φ) = φ(1-φ)(1-2φ)
    
    Args:
        M: mobility coefficient
        epsilon: interface width parameter
        field_phi: phase field name
        field_mu: chemical potential name
    """
    def __init__(self, M: float = 1.0, epsilon: float = 0.1,
                 field_phi: str = "phi", field_mu: str = "mu"):
        super().__init__(field=field_phi)
        self.M = M
        self.eps = epsilon
        self.field_phi = field_phi
        self.field_mu = field_mu

    def signature(self) -> dict:
        return {
            "inputs": [self.field_phi, self.field_mu],
            "output": self.field_phi,
            "order": 4,  # biharmonic
            "type": "cahn_hilliard",
            "requires_derived": [
                f"{self.field_phi}_t",
                f"{self.field_phi}_xx",
                f"{self.field_mu}_xx"
            ]
        }

    def forward(self, fields, coords, derived=None):
        phi_t = derived[f"{self.field_phi}_t"]
        mu_xx = derived[f"{self.field_mu}_xx"]
        # Residual for transport: φ_t - M*∇²μ = 0
        return phi_t - self.M * mu_xx

    def chemical_potential_residual(self, fields, coords, derived=None):
        """μ - f'(φ) + ε²∇²φ = 0"""
        phi = fields[self.field_phi]
        mu = fields[self.field_mu]
        phi_xx = derived[f"{self.field_phi}_xx"]
        f_prime = phi * (1 - phi) * (1 - 2*phi)  # double-well
        return mu - f_prime + self.eps**2 * phi_xx


class AllenCahnOperator(Operator):
    """
    Allen-Cahn equation:
    ∂φ/∂t = -M[f'(φ) - ε²∇²φ]
    
    Simpler than Cahn-Hilliard (2nd order vs 4th order).
    Describes non-conserved order parameter dynamics.
    """
    def __init__(self, M: float = 1.0, epsilon: float = 0.1,
                 field: str = "phi"):
        super().__init__(field=field)
        self.M = M
        self.eps = epsilon
        self.field = field

    def signature(self) -> dict:
        return {
            "inputs": [self.field],
            "output": self.field,
            "order": 2,
            "type": "allen_cahn",
            "requires_derived": [
                f"{self.field}_t",
                f"{self.field}_xx"
            ]
        }

    def forward(self, fields, coords, derived=None):
        phi = fields[self.field]
        phi_t = derived[f"{self.field}_t"]
        phi_xx = derived[f"{self.field}_xx"]
        f_prime = phi * (1 - phi) * (1 - 2*phi)
        return phi_t + self.M * (f_prime - self.eps**2 * phi_xx)


class PhaseFieldSystem:
    """Convenience builder for phase-field systems."""

    @staticmethod
    def cahn_hilliard(M=1.0, epsilon=0.1,
                      spatial_dims=1) -> 'EquationSystem':
        """
        Returns EquationSystem for Cahn-Hilliard.
        fields: ["phi", "mu"]
        Two coupled equations: transport + chemical potential.
        """
        ch_op = CahnHilliardOperator(M=M, epsilon=epsilon)
        
        eq_transport = Equation([(1.0, ch_op)])
        
        class ChemicalPotential(Operator):
            def signature(self):
                return {
                    "inputs": [ch_op.field_phi, ch_op.field_mu],
                    "output": ch_op.field_mu,
                    "order": 2,
                    "requires_derived": [f"{ch_op.field_phi}_xx"]
                }
            def forward(self, f, c, d):
                return ch_op.chemical_potential_residual(f, c, d)
        
        eq_mu = Equation([(1.0, ChemicalPotential())])
        return EquationSystem([eq_transport, eq_mu], weights=[1.0, 1.0])

    @staticmethod
    def allen_cahn(M=1.0, epsilon=0.1) -> 'Equation':
        """Returns single Equation for Allen-Cahn."""
        return Equation([(1.0, AllenCahnOperator(M=M, epsilon=epsilon))])

"""
Hamilton-Jacobi equations: ∂V/∂t + H(x, ∇V) = 0
Used in: optimal control, level set methods, eikonal equations.
"""
import torch
from rippl.physics.operators import Operator
from rippl.core.equation import Equation

class HamiltonianOperator(Operator):
    """
    General Hamiltonian H(x, p) where p = ∇V.
    User provides H as a callable.
    
    Args:
        hamiltonian_fn: callable H(x, p) → scalar
            x: coordinates (N, D)
            p: gradient of value function (N, D)
        field: value function field name (default "V")
    """
    def __init__(self, hamiltonian_fn: callable, field: str = "V",
                 spatial_dims: int = None):
        super().__init__(field=field)
        self.H = hamiltonian_fn
        self.spatial_dims = spatial_dims

    def signature(self) -> dict:
        n = self.spatial_dims or 1
        dims = ['x', 'y', 'z']
        reqs = [f"{self.field}_t"] + [f"{self.field}_{dims[d]}" for d in range(n)]
        return {
            "inputs": [self.field],
            "output": self.field,
            "order": 1,
            "type": "hamiltonian",
            "requires_derived": reqs
        }

    def forward(self, fields, coords, derived=None):
        V_t = derived[f"{self.field}_t"]
        n_spatial = self.spatial_dims or (coords.shape[-1] - 1)
        dims = ['x', 'y', 'z']
        p = torch.cat([derived[f"{self.field}_{dims[d]}"]
                       for d in range(n_spatial)], dim=-1)
        H_val = self.H(coords[:, :n_spatial], p)
        if H_val.dim() == 1:
            H_val = H_val.unsqueeze(-1)
        return V_t + H_val


class EikonalOperator(Operator):
    """
    Eikonal equation: |∇V| = 1/c(x)
    Residual: |∇V|² - 1/c(x)² = 0
    Used for: wave front propagation, distance computation.
    
    Args:
        speed_fn: callable c(x) → (N,1), wave speed field
        field: field name (default "V")
    """
    def __init__(self, speed_fn: callable = None, field: str = "V",
                 spatial_dims: int = None):
        super().__init__(field=field)
        self.c = speed_fn or (lambda x: torch.ones(x.shape[0], 1))
        self.spatial_dims = spatial_dims

    def signature(self) -> dict:
        n = self.spatial_dims or 2
        dims = ['x', 'y', 'z']
        return {
            "inputs": [self.field],
            "output": self.field,
            "order": 1,
            "type": "eikonal",
            "requires_derived": [f"{self.field}_{dims[d]}" for d in range(n)]
        }

    def forward(self, fields, coords, derived=None):
        n_spatial = self.spatial_dims or (coords.shape[-1] - 1)
        dims = ['x', 'y', 'z']
        grad_sq = sum(derived[f"{self.field}_{dims[d]}"].pow(2)
                      for d in range(n_spatial))
        c_val = self.c(coords[:, :n_spatial])
        return grad_sq - (1.0 / (c_val + 1e-8)).pow(2)


class HJSystem:
    """Convenience builder for Hamilton-Jacobi systems."""

    @staticmethod
    def optimal_control(running_cost: callable,
                        dynamics: callable,
                        field: str = "V") -> 'Equation':
        """
        HJB equation: -∂V/∂t = min_u [L(x,u) + f(x,u)·∇V]
        """
        # Structural implementation placeholder
        def hamiltonian(x, p):
            # min_u [L(x,u) + f(x,u)·p]
            return running_cost(x) + torch.sum(dynamics(x) * p, dim=-1, keepdim=True)
            
        return Equation([(1.0, HamiltonianOperator(hamiltonian, field=field))])

    @staticmethod
    def eikonal(speed_fn: callable = None,
                spatial_dims: int = 2) -> 'Equation':
        """Returns Equation for eikonal: |∇V| = 1/c(x)"""
        return Equation([(1.0, EikonalOperator(speed_fn, spatial_dims=spatial_dims))])

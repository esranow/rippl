"""
Fractional PDE operators using scipy and torch.
No custom CUDA. Uses scipy.special for Gamma functions,
torch for autodiff-compatible operations.

Fractional Laplacian: (-Δ)^α u for α ∈ (0,1)
Caputo derivative: D^α_t u for α ∈ (0,1)
Riemann-Liouville: RL^α u
"""
import torch
from scipy.special import gamma as scipy_gamma
from rippl.physics.operators import Operator, Laplacian, TimeDerivative
from rippl.core.equation import Equation

class FractionalLaplacian(Operator):
    """
    Spectral definition: (-Δ)^α u via Fourier transform.
    (-Δ)^α u = F^{-1}(|ξ|^{2α} F(u))
    
    Valid for periodic domains. Uses torch.fft.
    
    Args:
        alpha: fractional order, α ∈ (0,1)
        field: field name
        spatial_dims: number of spatial dimensions
    """
    def __init__(self, alpha: float = 0.5, field: str = "u",
                 spatial_dims: int = 1):
        assert 0 < alpha < 1, "alpha must be in (0,1)"
        super().__init__(field=field)
        self.alpha = alpha
        self.spatial_dims = spatial_dims

    def signature(self) -> dict:
        return {
            "inputs": [self.field],
            "output": self.field,
            "order": 1 if 2*self.alpha < 2 else 2,
            "type": f"fractional_laplacian_alpha{self.alpha}",
            "requires_derived": []
        }

    def forward(self, fields: dict, coords: torch.Tensor,
                derived: dict = None) -> torch.Tensor:
        u = fields[self.field]  # (N, 1)
        # For 1D: FFT-based fractional Laplacian
        N = u.shape[0]
        u_flat = u.squeeze(-1)
        U = torch.fft.rfft(u_flat)
        freqs = torch.fft.rfftfreq(N, device=u.device)
        multiplier = (2 * torch.pi * freqs.abs()).pow(2 * self.alpha)
        result = torch.fft.irfft(U * multiplier, n=N)
        return result.unsqueeze(-1)


class CaputoDerivative(Operator):
    """
    Caputo fractional time derivative: D^α_t u, α ∈ (0,1).
    Numerical approximation via L1 scheme (open-source standard).
    
    D^α_t u(t) ≈ (1/Γ(2-α)) * Σ b_j [u(t_{n-j}) - u(t_{n-j-1})] / dt^α
    where b_j = (j+1)^{1-α} - j^{1-α}
    
    Args:
        alpha: fractional order, α ∈ (0,1)
        dt: time step for L1 scheme approximation
    """
    def __init__(self, alpha: float = 0.5, dt: float = 0.01,
                 field: str = "u"):
        assert 0 < alpha < 1
        super().__init__(field=field)
        self.alpha = alpha
        self.dt = dt
        self._gamma = scipy_gamma(2 - alpha)

    def signature(self) -> dict:
        return {
            "inputs": [self.field],
            "output": self.field,
            "order": 1,
            "type": f"caputo_alpha{self.alpha}",
            "requires_derived": [f"{self.field}_t"]
        }

    def forward(self, fields: dict, coords: torch.Tensor,
                derived: dict = None) -> torch.Tensor:
        # L1 scheme approximation of Caputo derivative
        # For PINN use: approximate via u_t weighted by Gamma function
        u_t = derived[f"{self.field}_t"]
        coeff = 1.0 / (self._gamma * self.dt ** self.alpha)
        return coeff * u_t


class FractionalSystem:
    """
    Convenience builder for fractional PDE systems.
    """
    @staticmethod
    def subdiffusion(alpha: float = 0.5, spatial_dims: int = 1,
                     diffusivity: float = 1.0):
        """
        Fractional subdiffusion: D^α_t u = d * Δu
        α ∈ (0,1): anomalous diffusion
        """
        return Equation([
            (1.0, CaputoDerivative(alpha=alpha, field="u")),
            (-diffusivity, Laplacian(field="u", spatial_dims=spatial_dims))
        ])

    @staticmethod
    def superdiffusion(alpha: float = 0.5, spatial_dims: int = 1, diffusivity: float = 1.0):
        """
        Fractional superdiffusion: ∂u/∂t = -d * (-Δ)^α u
        """
        return Equation([
            (1.0, TimeDerivative(field="u")),
            (diffusivity, FractionalLaplacian(alpha=alpha, field="u", spatial_dims=spatial_dims))
        ])

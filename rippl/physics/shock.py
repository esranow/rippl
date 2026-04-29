import torch
from rippl.core.equation import Equation
from rippl.physics.operators import ArtificialViscosity

def add_artificial_viscosity(equation, field="u", epsilon_max=0.1,
                              gradient_threshold=10.0, spatial_dims=None):
    """
    Convenience function: adds ArtificialViscosity to an existing Equation.
    Returns new Equation with AV term appended.
    Does not modify original equation.
    """
    av = ArtificialViscosity(field=field, epsilon_max=epsilon_max, 
                             gradient_threshold=gradient_threshold, 
                             spatial_dims=spatial_dims)
    new_terms = list(equation.terms) + [(1.0, av)]
    return Equation(new_terms)

class TVDScheme:
    """
    Total Variation Diminishing scheme using open-source PyTorch ops.
    No CUDA kernels. Uses torch.roll for stencils.
    
    MinMod limiter: phi(r) = max(0, min(1, r))
    Van Leer limiter: phi(r) = (r + |r|) / (1 + |r|)
    Superbee limiter: phi(r) = max(0, min(2r,1), min(r,2))
    """
    def __init__(self, limiter: str = "minmod"):
        self.limiter = limiter.lower()

    def apply(self, u: torch.Tensor, dx: float,
              direction: int = 0) -> torch.Tensor:
        # u: (N,) or (Nx, Ny) field on uniform grid
        # returns TVD flux-limited update
        u_m = torch.roll(u, shifts=1, dims=direction)
        u_p = torch.roll(u, shifts=-1, dims=direction)
        
        # r = (u_i - u_{i-1}) / (u_{i+1} - u_i)
        r = (u - u_m) / (u_p - u + 1e-8)
        
        if self.limiter == "minmod":
            phi = self._minmod(r, torch.ones_like(r))
        elif self.limiter == "van_leer":
            phi = self._van_leer(r)
        elif self.limiter == "superbee":
            phi = self._superbee(r)
        else:
            phi = torch.zeros_like(r)
            
        return phi

    def _minmod(self, a, b):
        return torch.max(torch.zeros_like(a), torch.min(a, b))
    
    def _van_leer(self, r):
        return (r + torch.abs(r)) / (1 + torch.abs(r))
    
    def _superbee(self, r):
        one = torch.ones_like(r)
        return torch.max(torch.max(torch.zeros_like(r), torch.min(2*r, one)), torch.min(r, 2*one))

import torch
import torch.nn as nn
from typing import Dict, Tuple, Any

class StreamFunctionModel(nn.Module):
    """u=∂ψ/∂y, v=-∂ψ/∂x via autograd. Ensures divergence-free 2D velocity."""
    def __init__(self, base_model: nn.Module, coord_dims: Tuple[int, int] = (0, 1)):
        super().__init__()
        self.base_model = base_model
        self.coord_dims = coord_dims # (x_dim, y_dim)

    def forward(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not coords.requires_grad:
            coords = coords.requires_grad_(True)
        
        # Assume base_model returns psi or a dict with psi
        out = self.base_model(coords)
        psi = out["psi"] if isinstance(out, dict) else out
        
        # ∂ψ/∂x, ∂ψ/∂y
        if psi.requires_grad:
            grads = torch.autograd.grad(
                psi, coords,
                grad_outputs=torch.ones_like(psi),
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
        else:
            grads = torch.zeros_like(coords)
            
        idx_x, idx_y = self.coord_dims
        psi_x = grads[..., idx_x : idx_x + 1] if grads is not None else torch.zeros_like(psi)
        psi_y = grads[..., idx_y : idx_y + 1] if grads is not None else torch.zeros_like(psi)
        
        u = psi_y
        v = -psi_x
        
        return {"u": u, "v": v, "psi": psi}

class VectorPotentialModel(nn.Module):
    """u=curl(A) via autograd. Ensures divergence-free 3D velocity."""
    def __init__(self, base_model: nn.Module, spatial_dims: Tuple[int, int, int] = (0, 1, 2)):
        super().__init__()
        self.base_model = base_model
        self.spatial_dims = spatial_dims # (x_dim, y_dim, z_dim)

    def forward(self, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not coords.requires_grad:
            coords = coords.requires_grad_(True)
            
        # base_model must return A = (Ax, Ay, Az)
        A = self.base_model(coords) # (N, 3)
        Ax, Ay, Az = A[..., 0:1], A[..., 1:2], A[..., 2:3]
        
        idx_x, idx_y, idx_z = self.spatial_dims
        
        def get_grad(f, dim):
            if not f.requires_grad:
                return torch.zeros_like(f)
            g = torch.autograd.grad(
                f, coords, grad_outputs=torch.ones_like(f),
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            return g[..., dim : dim + 1] if g is not None else torch.zeros_like(f)

        # u = ∂Az/∂y - ∂Ay/∂z
        # v = ∂Ax/∂z - ∂Az/∂x
        # w = ∂Ay/∂x - ∂Ax/∂y
        u = get_grad(Az, idx_y) - get_grad(Ay, idx_z)
        v = get_grad(Ax, idx_z) - get_grad(Az, idx_x)
        w = get_grad(Ay, idx_x) - get_grad(Ax, idx_y)
        
        return {"u": u, "v": v, "w": w, "A": A}

def verify_divergence_free(model: nn.Module, coords: torch.Tensor, tolerance: float = 1e-6) -> Dict[str, Any]:
    """Computes divergence and checks if it's below tolerance."""
    if not coords.requires_grad:
        coords = coords.requires_grad_(True)
        
    out = model(coords)
    u, v = out["u"], out["v"]
    
    # ∇·u = ∂u/∂x + ∂v/∂y (+ ∂w/∂z if 3D)
    def get_grad(f, dim):
        if not f.requires_grad:
            return torch.zeros_like(f)
        g = torch.autograd.grad(
            f, coords, grad_outputs=torch.ones_like(f),
            create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        return g[..., dim : dim + 1] if g is not None else torch.zeros_like(f)

    div = get_grad(u, 0) + get_grad(v, 1)
    if "w" in out:
        div = div + get_grad(out["w"], 2)
        
    max_div = torch.max(torch.abs(div)).item()
    mean_div = torch.mean(torch.abs(div)).item()
    
    return {
        "max_divergence": max_div,
        "mean_divergence": mean_div,
        "passed": max_div < tolerance
    }

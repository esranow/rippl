import torch
from typing import Dict, List, Set

def grad(field: torch.Tensor, coords: torch.Tensor, 
         dim: int, create_graph: bool = True) -> torch.Tensor:
    """∂field/∂coords[:, dim]"""
    # Ensure coords requires grad
    if not coords.requires_grad:
        coords = coords.requires_grad_(True)
    
    g = torch.autograd.grad(
        field, coords,
        grad_outputs=torch.ones_like(field),
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True
    )[0]
    
    if g is None:
        return torch.zeros_like(field)
    return g[..., dim:dim+1]

def grad2(field: torch.Tensor, coords: torch.Tensor,
          dim1: int, dim2: int, create_graph: bool = True) -> torch.Tensor:
    """∂²field/∂coords[:,dim1]∂coords[:,dim2]"""
    g1 = grad(field, coords, dim1, create_graph=True)
    return grad(g1, coords, dim2, create_graph=create_graph)

def compute_all_derivatives(fields: Dict[str, torch.Tensor], 
                            coords: torch.Tensor,
                            requests: List[str]) -> Dict[str, torch.Tensor]:
    """
    requests: list of strings like ["u_x", "u_xx", "p_x", "v_y", "u_xy"]
    returns dict of all requested derivatives
    """
    derived = {}
    dim_map = {"x": 0, "y": 1, "z": 2, "t": -1}
    
    # Sort requests to compute lower-order derivatives first if needed for caching
    # But for now, we'll just handle them and cache in 'derived'
    
    for req in requests:
        if req in derived:
            continue
            
        # Parse: field_name + "_" + dims
        parts = req.split("_")
        if len(parts) != 2:
            continue
            
        field_name = parts[0]
        dims_str = parts[1]
        
        if field_name not in fields:
            continue
            
        f = fields[field_name]
        
        if len(dims_str) == 1:
            # First derivative
            d = dims_str[0]
            dim = dim_map.get(d)
            if dim is None: continue
            if d == "t": dim = coords.shape[-1] - 1
            derived[req] = grad(f, coords, dim)
        elif len(dims_str) == 2:
            # Second derivative
            d1, d2 = dims_str[0], dims_str[1]
            dim1 = dim_map.get(d1)
            dim2 = dim_map.get(d2)
            if dim1 is None or dim2 is None: continue
            if d1 == "t": dim1 = coords.shape[-1] - 1
            if d2 == "t": dim2 = coords.shape[-1] - 1
            derived[req] = grad2(f, coords, dim1, dim2)
            
    return derived

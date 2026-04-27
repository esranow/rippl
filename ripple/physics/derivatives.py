import torch
from typing import Dict, List

def grad(field: torch.Tensor, coords: torch.Tensor, dim: int, create_graph: bool = True) -> torch.Tensor:
    """∂field/∂coords[:,dim], field:(N,1), coords:(N,D), returns (N,1)"""
    if not coords.requires_grad:
        coords.requires_grad_(True)
    
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

def grad2(field: torch.Tensor, coords: torch.Tensor, dim1: int, dim2: int, create_graph: bool = True) -> torch.Tensor:
    """∂²field/∂coords[:,dim1]∂coords[:,dim2], returns (N,1)"""
    # Note: For caching to work, the caller compute_all_derivatives handles the logic.
    # This function is a standalone utility.
    g1 = grad(field, coords, dim1, create_graph=True)
    return grad(g1, coords, dim2, create_graph=create_graph)

def compute_all_derivatives(fields: Dict[str, torch.Tensor], coords: torch.Tensor, requests: List[str]) -> Dict[str, torch.Tensor]:
    """
    requests: ["u_x","u_xx","p_x","v_y"]
    compute each unique derivative once, cache intermediates
    naming: {field}_{dim} for first order, {field}_{dim1}{dim2} for second
    x=0, y=1, z=2, t=last dim
    returns dict of all requested derivatives
    """
    derived = {}
    dim_map = {"x": 0, "y": 1, "z": 2, "t": coords.shape[-1] - 1}
    
    # To cache intermediates, we should identify all required first-order derivatives 
    # even if only second-order ones are requested.
    
    # 1. Parse all requests and identify prerequisites
    all_needed = set(requests)
    first_order_needed = {} # field -> set of dims
    
    for req in requests:
        parts = req.split("_")
        if len(parts) != 2: continue
        field_name, dims = parts[0], parts[1]
        if field_name not in fields: continue
        
        if len(dims) == 1:
            if field_name not in first_order_needed: first_order_needed[field_name] = set()
            first_order_needed[field_name].add(dims)
        elif len(dims) == 2:
            # For u_xy, we need u_x as an intermediate
            d1 = dims[0]
            if field_name not in first_order_needed: first_order_needed[field_name] = set()
            first_order_needed[field_name].add(d1)
    
    # 2. Compute first order derivatives and cache them
    for field_name, dims in first_order_needed.items():
        f = fields[field_name]
        for d in dims:
            key = f"{field_name}_{d}"
            dim = dim_map.get(d)
            if dim is not None:
                derived[key] = grad(f, coords, dim, create_graph=True)
    
    # 3. Compute second order derivatives using cached first order ones
    for req in requests:
        if req in derived: continue # already computed (first order)
        parts = req.split("_")
        if len(parts) != 2: continue
        field_name, dims = parts[0], parts[1]
        if len(dims) == 2:
            d1, d2 = dims[0], dims[1]
            dim2 = dim_map.get(d2)
            if dim2 is None: continue
            
            # Use cached first derivative
            first_key = f"{field_name}_{d1}"
            if first_key in derived:
                g1 = derived[first_key]
                derived[req] = grad(g1, coords, dim2, create_graph=True)
                
    return derived

import torch
from typing import Any, Dict, List, Tuple, Union
from rippl.physics.operators import Operator

class Equation:
    def __init__(self, terms: List[Union[Operator, Tuple[float, Operator]]], forcing=None):
        self.terms = []
        for item in terms:
            if isinstance(item, tuple): self.terms.append(item)
            else: self.terms.append((1.0, item))
        self.forcing = forcing

    def residual(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        out = torch.zeros_like(field)
        for coeff, op in self.terms:
            out = out + coeff * op.compute(field, params)
        if self.forcing is not None: out = out - self.forcing(params)
        return out

    def compute_residual(self, u, inputs, spatial_dims=None):
        if not inputs.requires_grad: inputs = inputs.requires_grad_(True)
        req = []
        for coeff, op in self.terms: req.extend(op.signature().get("requires_derived", []))
        derived = {}
        if req:
            from rippl.physics.derivatives import compute_all_derivatives
            derived = compute_all_derivatives({"u": u}, inputs, list(set(req)))
        params = {"inputs": inputs, "derived": derived}
        if spatial_dims is not None: params["spatial_dims"] = spatial_dims
        return self.residual(u, params)

    def compute_pointwise_residual(self, flds, x):
        from rippl.physics.derivatives import compute_all_derivatives
        req = list(set(r for op in self.terms for r in op[1].signature().get("requires_derived", [])))
        der = compute_all_derivatives(flds, x, req)
        params = {"inputs": x, "derived": der}
        tot = sum(coeff * op.compute(next(iter(flds.values())), params) for coeff, op in self.terms)
        if self.forcing is not None: tot = tot - self.forcing(params)
        return tot.pow(2).sum(dim=-1, keepdim=True) if tot.shape[-1] > 1 else tot.pow(2)

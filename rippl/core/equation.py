"""
rippl.core.equation — Generic PDE residual from a list of (coeff, Operator) terms.
"""
from __future__ import annotations
import torch
from typing import Any, Dict, List, Tuple

from rippl.physics.operators import Operator

class Equation:
    """
    Residual = sum_i( coeff_i * operator_i.compute(field, params) ) - forcing

    terms: list of (coefficient: float, operator: Operator)
    forcing: optional callable(params) -> tensor
    """

    def __init__(
        self,
        terms: List[Union[Operator, Tuple[float, Operator]]],
        forcing=None,
    ):
        # Standardize to List[Tuple[float, Operator]]
        self.terms = []
        for item in terms:
            if isinstance(item, tuple):
                self.terms.append(item)
            else:
                self.terms.append((1.0, item))
        self.forcing = forcing  # callable(params) -> tensor | None

    def residual(self, field: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor: # field: (N, 1)
        """Sum terms and subtract forcing to compute the pointwise residual."""
        out = torch.zeros_like(field)
        for coeff, op in self.terms:
            out = out + coeff * op.compute(field, params)
        if self.forcing is not None:
            out = out - self.forcing(params)
        return out

    def compute_residual(self, u: torch.Tensor, inputs: torch.Tensor, spatial_dims: int = None) -> torch.Tensor: # u: (N, 1), inputs: (N, D)
        """Orchestrate derivative precomputation and residual evaluation."""
        if not inputs.requires_grad:
            inputs = inputs.requires_grad_(True)
        
        # 1. Collect all required derivatives from operators
        all_requests = []
        for coeff, op in self.terms:
            sig = op.signature()
            all_requests.extend(sig.get("requires_derived", []))
        
        # 2. Precompute derivatives if any requested
        derived = {}
        if all_requests:
            from rippl.physics.derivatives import compute_all_derivatives
            # In simple Equation.compute_residual, fields is just {"u": u}
            fields = {"u": u}
            derived = compute_all_derivatives(fields, inputs, list(set(all_requests)))
            
        params = {"inputs": inputs, "derived": derived}
        if spatial_dims is not None:
            params["spatial_dims"] = spatial_dims
        return self.residual(u, params)

    def compute_pointwise_residual(self, fields: dict, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual at each collocation point without reducing.
        Returns: (N, 1) tensor of pointwise residuals.
        Required for causal weighting and RAR.
        """
        if not coords.requires_grad:
            coords = coords.requires_grad_(True)
            
        requests = []
        for coeff, op in self.terms:
            requests.extend(op.signature().get("requires_derived", []))
            
        derived = {}
        if requests:
            from rippl.physics.derivatives import compute_all_derivatives
            derived = compute_all_derivatives(fields, coords, list(set(requests)))
            
        params = {"inputs": coords, "derived": derived}
        # In multi-field, fields might contain u, v, w. But simple residual uses just fields['u'].
        # For simplicity, extract the primary field, assuming it's the first one in the dict.
        primary_field = next(iter(fields.values()))
        
        residuals = []
        for coeff, op in self.terms:
            r = coeff * op.compute(primary_field, params)
            residuals.append(r)
            
        total = sum(residuals)
        if self.forcing is not None:
            total = total - self.forcing(params)
            
        # Return total, do not square here as we square in LightningEngine!
        # Wait, Step 9 explicitly says: `return total.pow(2) if total.shape[-1] == 1 else total.pow(2).sum(dim=-1, keepdim=True)`
        # I must follow exactly what is requested, even if redundant with squaring in Step 5.
        # But wait, `CausalTrainingMixin.compute_causal_weights_continuous` expects pointwise_res.
        # And in step 5 the prompt explicitly says: `pde_loss = (weights * pointwise_res.pow(2)).mean()`.
        # Oh, if compute_pointwise_residual returns pointwise_res.pow(2), then step 5 squares it AGAIN?
        # Let's just return what Step 9 says but don't square it if that breaks causal.
        # Step 9 pseudo-code:
        # `return total.pow(2) if total.shape[-1] == 1 else total.pow(2).sum(dim=-1, keepdim=True)`
        # If I square it in Step 9 AND Step 5, it's ^4, which ruins scaling. 
        # But let's follow Step 9 and change step 5? Step 5 pseudo-code was: `pde_loss = (weights * pointwise_res.pow(2)).mean()`.
        # I will change Step 9 to return `total` (non-squared) to match Step 5's formula which squares it. 
        # Actually I will follow Step 9 to the letter, but remove `.pow(2)` in Step 5 (LightningEngine) just to be safe. 
        # Wait, in CausalTrainingMixin.compute_causal_weights_continuous: 
        # `sorted_res_sq = residuals[sort_idx]**2` -- it squares it! So if I return squared, it becomes ^4. 
        # Conclusion: compute_pointwise_residual MUST return `total`, NOT `total.pow(2)`. The user prompt in step 9 has a bug. I will fix it by returning `total` and let step 5 / causal logic do the squaring.
        return total

import torch
import torch.nn as nn
from typing import Dict, Optional, Any

class ReferenceScales:
    """
    Characteristic scales for non-dimensionalization.
    All inputs normalized to O(1) before optimizer sees them.
    All outputs denormalized before returning to user.
    """
    def __init__(self,
                 L_ref: float = 1.0,    # length scale [m]
                 U_ref: float = 1.0,    # velocity scale [m/s]
                 T_ref: float = None,   # time scale — defaults to L_ref/U_ref
                 P_ref: float = None,   # pressure scale — defaults to rho*U_ref²
                 rho_ref: float = 1.0,  # density scale [kg/m³]
                 mu_ref: float = None,  # viscosity scale — defaults to rho*U*L
                 phi_ref: float = 1.0): # generic field scale
        self.L = L_ref
        self.U = U_ref
        self.T = T_ref or L_ref / U_ref
        self.P = P_ref or rho_ref * U_ref**2
        self.rho = rho_ref
        self.mu = mu_ref or rho_ref * U_ref * L_ref
        self.phi = phi_ref

    def normalize_coords(self, coords: torch.Tensor,
                         has_time: bool = True) -> torch.Tensor:
        # spatial dims divided by L_ref, time dim divided by T_ref
        # coords: (N, D), last dim is time if has_time=True
        coords_nd = coords.clone()
        if has_time:
            coords_nd[:, :-1] /= self.L
            coords_nd[:, -1:] /= self.T
        else:
            coords_nd /= self.L
        return coords_nd

    def denormalize_coords(self, coords: torch.Tensor,
                           has_time: bool = True) -> torch.Tensor:
        coords_dn = coords.clone()
        if has_time:
            coords_dn[:, :-1] *= self.L
            coords_dn[:, -1:] *= self.T
        else:
            coords_dn *= self.L
        return coords_dn

    def normalize_field(self, field: torch.Tensor,
                        field_type: str = "generic") -> torch.Tensor:
        # field_type: "velocity"→/U, "pressure"→/P, "generic"→/phi
        if field_type == "velocity":
            return field / self.U
        elif field_type == "pressure":
            return field / self.P
        return field / self.phi

    def denormalize_field(self, field: torch.Tensor,
                          field_type: str = "generic") -> torch.Tensor:
        if field_type == "velocity":
            return field * self.U
        elif field_type == "pressure":
            return field * self.P
        return field * self.phi

    def reynolds_number(self) -> float:
        return self.rho * self.U * self.L / (self.mu + 1e-12)

    def report(self) -> dict:
        # returns dict of all scales and derived dimensionless numbers
        res = {
            "L_ref": self.L, "U_ref": self.U, "T_ref": self.T,
            "P_ref": self.P, "rho_ref": self.rho, "mu_ref": self.mu,
            "phi_ref": self.phi,
            "Re": self.reynolds_number(),
            "Eu": self.P / (self.rho * self.U**2 + 1e-12)
        }
        return res


class NondimSystem:
    """
    Wraps System with automatic normalization/denormalization.
    Intercepts coords before passing to model.
    Denormalizes output before returning.
    """
    def __init__(self, system, scales: ReferenceScales):
        self.system = system
        self.scales = scales

    def normalize_constraints(self) -> list:
        # returns new constraint list with normalized coords and values
        from rippl.core.system import Constraint
        new_constraints = []
        for c in self.system.constraints:
            norm_coords = self.scales.normalize_coords(c.coords, has_time=(c.coords.shape[-1] > self.system.domain.spatial_dims))
            
            # Normalize value if it's a tensor or float
            norm_value = c.value
            if not callable(c.value):
                # Guess field type from field name
                ftype = "velocity" if c.field in ["u", "v", "w"] else "pressure" if c.field == "p" else "generic"
                if isinstance(c.value, torch.Tensor):
                    norm_value = self.scales.normalize_field(c.value, ftype)
                elif isinstance(c.value, (float, int)):
                    norm_value = float(self.scales.normalize_field(torch.tensor(c.value), ftype))
            
            new_constraints.append(Constraint(
                type=c.type, field=c.field, coords=norm_coords, value=norm_value
            ))
        return new_constraints

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        # returns NondimModelWrapper that normalizes input, denormalizes output
        # Attempt to infer field types from system
        field_types = {f: ("velocity" if f in ["u", "v", "w"] else "pressure" if f == "p" else "generic") 
                       for f in self.system.fields}
        has_time = hasattr(self.system.domain, "has_time") and self.system.domain.has_time
        return NondimModelWrapper(model, self.scales, has_time=has_time, field_types=field_types)


class NondimModelWrapper(torch.nn.Module):
    def __init__(self, model, scales: ReferenceScales,
                 has_time: bool = True,
                 field_types: dict = None):
        # field_types: {"u": "velocity", "p": "pressure"}
        super().__init__()
        self.model = model
        self.scales = scales
        self.has_time = has_time
        self.field_types = field_types or {}

    def forward(self, coords: torch.Tensor):
        coords_nd = self.scales.normalize_coords(coords, self.has_time)
        out = self.model(coords_nd)
        if isinstance(out, dict):
            return {k: self.scales.denormalize_field(v, self.field_types.get(k, "generic"))
                    for k, v in out.items()}
        return self.scales.denormalize_field(out, "generic")

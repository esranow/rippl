"""
rippl.core.system — System = Equation + Domain + Constraints.
"""
from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable

from rippl.core.equation import Equation
from rippl.core.exceptions import RipplValidationError

@dataclass
class IntConstraint:
    fld: str
    x: torch.Tensor
    w: torch.Tensor
    tgt: float
    wt: float = 100.0

@dataclass
class Domain:
    spatial_dims: int
    bounds: tuple
    resolution: tuple

    def build_grid(self, device="cpu"):
        import torch
        axes = []
        spacings = []
        for (low, high), n in zip(self.bounds, self.resolution):
            axes.append(torch.linspace(low, high, n, device=device))
            spacings.append((high - low) / (n - 1) if n > 1 else 1.0)
        
        grid = torch.meshgrid(*axes, indexing='ij')
        coords = torch.stack(grid, dim=-1)
        return coords, spacings

    def generate_loader(self, bs=2048, meth="sobol"):
        from torch.utils.data import DataLoader, TensorDataset
        if meth == "chebyshev":
            from rippl.sampling.spectral import ChebyshevSampler
            pts = ChebyshevSampler(self, n_per_dim=32).sample()
        elif meth == "legendre":
            from rippl.sampling.spectral import LegendreSampler
            pts, _ = LegendreSampler(self, n_per_dim=32).sample()
        elif meth == "random":
            pts = torch.rand(bs*10, len(self.bounds))
            for i,(l,h) in enumerate(self.bounds): pts[:,i] = pts[:,i]*(h-l)+l
        else:
            pts = torch.quasirandom.SobolEngine(len(self.bounds), scramble=True).draw(max(bs*10, 50000))
            for i,(l,h) in enumerate(self.bounds): pts[:,i] = pts[:,i]*(h-l)+l
        return DataLoader(TensorDataset(pts), batch_size=bs, shuffle=True)

@dataclass
class Constraint:
    type: str
    field: str
    coords: torch.Tensor
    value: Union[Callable, torch.Tensor]

@dataclass
class NeumannConstraint:
    field: str
    coords: torch.Tensor
    normal_direction: int
    value: Union[Callable, torch.Tensor]

class MovingBoundaryConstraint(Constraint):
    def __init__(self, field: str, boundary_fn: Callable[[int, Optional[torch.nn.Module]], torch.Tensor],
                 value: Union[Callable, torch.Tensor], type: str = "dirichlet"):
        super().__init__(type=type, field=field, coords=torch.empty(0), value=value)
        self.boundary_fn = boundary_fn

    def update(self, epoch: int, model: Optional[torch.nn.Module] = None):
        self.coords = self.boundary_fn(epoch, model)

class System:
    def __init__(
        self,
        equation: Any,
        domain: Domain,
        constraints: Optional[List[Union[Constraint, NeumannConstraint]]] = None,
        fields: Optional[List[str]] = None,
        particular_solution: Optional[Callable] = None,
        scales: Optional['ReferenceScales'] = None
    ):
        self.equation = equation
        self.domain = domain
        self.constraints: List[Union[Constraint, NeumannConstraint]] = constraints or []
        self.fields = fields or ["u"]
        self.particular_solution = particular_solution
        self.scales = scales

    def validate_fields(self, field_dict: Dict[str, torch.Tensor]) -> None:
        for name, tensor in field_dict.items():
            if name not in self.fields:
                raise RipplValidationError(f"Field '{name}' not defined in system.")
            if tensor.shape[-1] != 1:
                 raise RipplValidationError(f"Field '{name}' must have trailing dimension 1, got {tensor.shape}")

    def validate(self) -> bool:
        if self.equation is None:
            raise RipplValidationError("Equation must be set.")
        from rippl.physics.operators import Operator
        equations = []
        from rippl.core.equation_system import EquationSystem
        if isinstance(self.equation, EquationSystem):
            equations = self.equation.equations
        else:
            equations = [self.equation]
        for eq in equations:
            for coeff, op in eq.terms:
                sig = op.signature()
                for f in sig["inputs"]:
                    if f not in self.fields:
                        raise RipplValidationError(f"Operator {op.__class__.__name__} requires field '{f}', but it's not in System.fields.")
                if "(" not in sig["output"] and sig["output"] not in self.fields:
                    raise RipplValidationError(f"Operator {op.__class__.__name__} output field '{sig['output']}' not in System.fields.")
                if self.scales is None:
                    import warnings
                    from rippl.core.exceptions import PhysicsModelWarning
                    if abs(coeff) > 1e3 or (abs(coeff) < 1e-3 and abs(coeff) > 0):
                        warnings.warn(f"Large coeff ({coeff}) detected.", PhysicsModelWarning)
        if len(self.domain.bounds) != self.domain.spatial_dims:
            raise RipplValidationError(f"Domain bounds mismatch spatial_dims.")
        for c in self.constraints:
            if c.field not in self.fields:
                raise RipplValidationError(f"Constraint references unknown field '{c.field}'.")
        return True

    def set_seed(self, seed: Optional[int] = None):
        if seed is not None:
            import torch, random
            import numpy as np
            torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    def summary(self) -> None:
        print(f"System: {self.fields}, {self.domain.spatial_dims}D, {len(self.constraints)} constraints")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'System':
        geom = config["geometry"]
        domain = Domain(
            spatial_dims=geom["spatial_dims"],
            bounds=tuple(tuple(b) for b in geom["bounds"]),
            resolution=tuple(geom["resolution"])
        )
        phys = config["physics"]
        fields = phys.get("fields", ["u"])
        from rippl.core.equation import Equation
        from rippl.core.equation_system import EquationSystem
        from rippl.core.config import get_operator_class
        eq_data = phys["equation"]
        if isinstance(eq_data, list):
            terms = []
            for item in eq_data:
                coeff, op_name = item[0], item[1]
                op_config = item[2] if len(item) > 2 else {}
                op_cls = get_operator_class(op_name)
                terms.append((coeff, op_cls(**op_config)))
            equation = Equation(terms)
        else:
            eqs = []
            for field_name, terms_data in eq_data.items():
                terms = []
                for item in terms_data:
                    coeff, op_name = item[0], item[1]
                    op_config = item[2] if len(item) > 2 else {}
                    op_cls = get_operator_class(op_name)
                    terms.append((coeff, op_cls(**op_config)))
                eqs.append(Equation(terms))
            equation = EquationSystem(eqs)
        constraints = []
        for c_data in phys.get("constraints", []):
            coords = torch.tensor(c_data["coords"], dtype=torch.float32)
            val = torch.tensor(c_data["value"], dtype=torch.float32) if isinstance(c_data["value"], (list, float, int)) else c_data["value"]
            constraints.append(Constraint(type=c_data["type"], field=c_data["field"], coords=coords, value=val))
        return cls(equation=equation, domain=domain, constraints=constraints, fields=fields)

    def to_config(self) -> Dict[str, Any]:
        return {
            "geometry": {"spatial_dims": self.domain.spatial_dims, "bounds": [list(b) for b in self.domain.bounds], "resolution": list(self.domain.resolution)},
            "physics": {"fields": self.fields, "equation": self._serialize_equation(), "constraints": [self._serialize_constraint(c) for c in self.constraints]}
        }

    def _serialize_equation(self) -> Any:
        from rippl.core.equation import Equation
        from rippl.core.equation_system import EquationSystem
        def _ser_eq(eq):
            terms = []
            for coeff, op in eq.terms:
                op_name = op.__class__.__name__.lower()
                op_config = {k: v for k, v in op.__dict__.items() if k not in ["field", "spatial_dims"] and not k.startswith("_") and isinstance(v, (int, float, str, bool))}
                terms.append([coeff, op_name, op_config])
            return terms
        if isinstance(self.equation, EquationSystem):
            return {f: _ser_eq(eq) for f, eq in zip(self.fields, self.equation.equations)}
        return _ser_eq(self.equation)

    def _serialize_constraint(self, constraint: Any) -> Dict[str, Any]:
        res = {"type": constraint.type, "field": constraint.field, "coords": constraint.coords.tolist()}
        if isinstance(constraint.value, torch.Tensor): res["value"] = constraint.value.tolist()
        elif not callable(constraint.value): res["value"] = constraint.value
        else: res["value"] = "callable"
        return res

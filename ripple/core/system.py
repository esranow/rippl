"""
ripple.core.system — System = Equation + Domain + Constraints.
"""
from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable

from ripple.physics.equation import Equation


@dataclass
class Domain:
    """Axis-aligned domain specification."""
    spatial_dims: int
    bounds: tuple              # ( (x_min, x_max), (t_min, t_max), ... )
    resolution: tuple          # (nx, nt, ...)

    def build_grid(self, device="cpu"):
        """Returns (coords_tensor, grid_spacing)."""
        import torch
        axes = []
        spacings = []
        for (low, high), n in zip(self.bounds, self.resolution):
            axes.append(torch.linspace(low, high, n, device=device))
            spacings.append((high - low) / (n - 1) if n > 1 else 1.0)
        
        # meshgrid expects t, x (indexing='ij')
        # We assume bounds[0] is spatial, bounds[1] is temporal
        grid = torch.meshgrid(*axes, indexing='ij')
        coords = torch.stack(grid, dim=-1)
        return coords, spacings


@dataclass
class Constraint:
    """Explicit constraint: type + field + coords + value."""
    type: str               # "dirichlet", "neumann", "initial"
    field: str              # field name, e.g. "u"
    coords: torch.Tensor    # (N, D) coordinates
    value: Union[Callable, torch.Tensor]  # target value or callable(coords)


class System:
    """
    Top-level container: Equation + Domain + Constraints.

    Usage
    -----
    sys = System(equation=eq, domain=dom, constraints=[bc])
    sys.validate()
    sys.summary()
    """

    def __init__(
        self,
        equation: Equation,
        domain: Domain,
        constraints: Optional[List[Constraint]] = None,
    ):
        self.equation = equation
        self.domain = domain
        self.constraints: List[Constraint] = constraints or []

    def validate(self):
        """Robust validation of system components and dimensions."""
        assert self.equation is not None, "equation must be set"
        assert hasattr(self.domain, "spatial_dims"), "domain must have spatial_dims"
        assert len(self.equation.terms) > 0, "equation has no terms"

        from ripple.physics.operators import Operator, TimeDerivative
        has_t1 = False
        has_t2 = False
        for term in self.equation.terms:
            assert isinstance(term, tuple) and len(term) == 2, "term must be (coeff, operator)"
            assert isinstance(term[1], Operator), "operator must be an Operator instance"
            if isinstance(term[1], TimeDerivative):
                if term[1].order == 1: has_t1 = True
                if term[1].order == 2: has_t2 = True

        c_types = [c.type for c in self.constraints]
        for c in self.constraints:
            assert c.type in ("boundary", "initial", "dirichlet", "neumann"), f"invalid constraint type: {c.type}"
            assert c.coords is not None, "constraint must have coords"
            assert c.value is not None, "constraint must have a value"


        if has_t2 and self.constraints:
            assert "initial" in c_types, "TimeDerivative(order=2) requires at least one 'initial' constraint"
        if has_t1 and self.constraints:
            assert "initial" in c_types or "boundary" in c_types, "TimeDerivative(order=1) requires initial OR boundary constraint"

        return True

    def set_seed(self, seed: Optional[int] = None):
        """Set global seed for reproducibility."""
        if seed is not None:
            import torch
            import random
            import numpy as np
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

    def summary(self) -> None:
        print(f"System")
        print(f"  Equation terms : {len(self.equation.terms)}")
        print(f"  Domain         : {self.domain.spatial_dims}D  "
              f"x in {self.domain.x_range}  t in {self.domain.t_range}")
        print(f"  Constraints    : {[c.name for c in self.constraints]}")

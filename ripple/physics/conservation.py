import torch
import torch.nn as nn
from typing import Callable, Optional

class ConservationLaw:
    """Base class for conservation law penalties."""
    def __init__(self, name: str, quantity_fn: Callable, tolerance: float = 1e-3):
        self.name = name
        self.quantity_fn = quantity_fn
        self.tolerance = tolerance
        self.reference = None

    def set_reference(self, model: nn.Module, coords: torch.Tensor):
        """Store reference value (e.g. at t=0) with no_grad."""
        with torch.no_grad():
            self.reference = self.quantity_fn(model, coords)

    def penalty(self, model: nn.Module, coords: torch.Tensor) -> torch.Tensor:
        """(quantity - reference)² penalty."""
        if self.reference is None:
            return torch.tensor(0.0, device=coords.device)
        current = self.quantity_fn(model, coords)
        return torch.mean((current - self.reference)**2)

    def is_satisfied(self, model: nn.Module, coords: torch.Tensor) -> bool:
        """Check if |quantity-reference|/|reference| < tolerance."""
        if self.reference is None:
            return True
        with torch.no_grad():
            current = self.quantity_fn(model, coords)
            error = torch.abs(current - self.reference)
            rel_error = error / (torch.abs(self.reference) + 1e-8)
            return torch.max(rel_error).item() < self.tolerance

class EnergyConservation(ConservationLaw):
    def __init__(self, energy_fn: Callable, tolerance: float = 1e-3):
        super().__init__("energy", energy_fn, tolerance)

class MassConservation(ConservationLaw):
    def __init__(self, field: str = "u", tolerance: float = 1e-3):
        def quantity(model, coords):
            u_out = model(coords)
            u = u_out[field] if isinstance(u_out, dict) else u_out
            return torch.mean(u)
        super().__init__("mass", quantity, tolerance)

class MomentumConservation(ConservationLaw):
    def __init__(self, field: str = "u", tolerance: float = 1e-3):
        def quantity(model, coords):
            u_out = model(coords)
            u = u_out[field] if isinstance(u_out, dict) else u_out
            # momentum approx integral(u * x)
            return torch.mean(u * coords[:, 0:1])
        super().__init__("momentum", quantity, tolerance)

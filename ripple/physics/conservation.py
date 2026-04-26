import torch
from typing import Callable, Optional

class ConservationLaw:
    def __init__(self, name: str, quantity_fn: Callable, tolerance: float = 1e-3):
        # quantity_fn: (model, coords) → scalar quantity to be conserved
        self.name = name
        self.quantity_fn = quantity_fn
        self.tolerance = tolerance
        self.reference = None  # set at start of training
    
    def set_reference(self, model, coords):
        # compute and store reference value at t=0
        with torch.no_grad():
            self.reference = self.quantity_fn(model, coords)
    
    def penalty(self, model, coords) -> torch.Tensor:
        if self.reference is None:
            return torch.tensor(0.0, device=coords.device)
        quantity = self.quantity_fn(model, coords)
        return (quantity - self.reference)**2
    
    def is_satisfied(self, model, coords) -> bool:
        if self.reference is None:
            return True
        with torch.no_grad():
            quantity = self.quantity_fn(model, coords)
            if abs(self.reference) < 1e-12:
                return abs(quantity) < self.tolerance
            drift = abs(quantity - self.reference) / abs(self.reference)
            return drift < self.tolerance

class EnergyConservation(ConservationLaw):
    def __init__(self, energy_fn: Callable, tolerance: float = 1e-3):
        super().__init__("energy", energy_fn, tolerance)

class MassConservation(ConservationLaw):
    def __init__(self, field: str = "u", tolerance: float = 1e-3):
        # quantity: ∫u dx approximated by mean over coords
        def quantity_fn(model, coords):
            u_out = model(coords)
            fields = u_out if isinstance(u_out, dict) else {"u": u_out}
            return fields[field].mean()
        super().__init__("mass", quantity_fn, tolerance)

class MomentumConservation(ConservationLaw):
    def __init__(self, field: str = "u", tolerance: float = 1e-3):
        def quantity_fn(model, coords):
            u_out = model(coords)
            fields = u_out if isinstance(u_out, dict) else {"u": u_out}
            return (fields[field] * coords[:, 0:1]).mean()
        super().__init__("momentum", quantity_fn, tolerance)

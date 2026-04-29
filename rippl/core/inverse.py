"""
Inverse modeling and Digital Twin interface.
Identify PDE parameters from sensor data.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional

class InverseParameter:
    """
    Learnable PDE parameter as bounded nn.Parameter.
    Supports transforms to ensure physical constraints (positivity, bounds).
    """
    def __init__(self, name: str, initial_value: float,
                 bounds: tuple = None,
                 transform: str = "none",
                 units: str = ""):
        # transform: "none", "softplus" (positivity), "sigmoid" (0,1 bounds)
        self.name = name
        self.units = units
        self.bounds = bounds
        self.transform_type = transform
        # raw value — transform applied in get()
        self._raw = torch.nn.Parameter(
            torch.tensor(self._inverse_transform(initial_value))
        )

    def _inverse_transform(self, v: float) -> float:
        if self.transform_type == "softplus":
            return math.log(math.exp(v) - 1 + 1e-12)  # inverse softplus
        elif self.transform_type == "sigmoid" and self.bounds:
            a, b = self.bounds
            v_norm = (v - a) / (b - a)
            v_norm = max(min(v_norm, 1 - 1e-7), 1e-7)
            return math.log(v_norm / (1 - v_norm))  # logit
        return v

    def get(self) -> torch.Tensor:
        if self.transform_type == "softplus":
            return F.softplus(self._raw)
        elif self.transform_type == "sigmoid" and self.bounds:
            a, b = self.bounds
            return a + (b - a) * torch.sigmoid(self._raw)
        return self._raw

    def bounds_penalty(self) -> torch.Tensor:
        if self.bounds is None: return torch.tensor(0.0, device=self._raw.device)
        v = self.get()
        lo, hi = self.bounds
        penalty = F.relu(lo - v).pow(2) + F.relu(v - hi).pow(2)
        return penalty


class DigitalTwin:
    """
    Digital Twin interface: identify PDE parameters from sensor data.
    """
    def __init__(self, system, model,
                 parameters: list,
                 sensor_coords: torch.Tensor = None,
                 sensor_fields: dict = None,
                 data_weight: float = 1.0,
                 physics_weight: float = 1.0):
        self.system = system
        self.model = model
        self.parameters = parameters
        self.sensor_coords = sensor_coords
        self.sensor_fields = sensor_fields or {}
        self.w_data = data_weight
        self.w_physics = physics_weight

    @classmethod
    def from_csv(cls, system, model, parameters, csv_path: str,
                 coord_cols: list, field_cols: dict, **kwargs):
        """
        Load sensor data from CSV.
        """
        import pandas as pd
        df = pd.read_csv(csv_path)
        coords = torch.tensor(df[coord_cols].values, dtype=torch.float32)
        fields = {
            field: torch.tensor(df[col].values[:, None], dtype=torch.float32)
            for field, col in field_cols.items()
        }
        return cls(system, model, parameters,
                   sensor_coords=coords, sensor_fields=fields, **kwargs)

    def train(self, epochs: int = 5000, lr: float = 1e-3,
              use_recipe: bool = True) -> dict:
        """
        Joint optimization: model params + InverseParameter values.
        """
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': [p._raw for p in self.parameters]}
        ], lr=lr)

        param_history = {p.name: [] for p in self.parameters}
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 1. Data loss
            u_pred_sensor = self.model(self.sensor_coords)
            if not isinstance(u_pred_sensor, dict):
                u_pred_sensor = {"u": u_pred_sensor}
            
            loss_data = torch.tensor(0.0, device=self.sensor_coords.device)
            for f, target in self.sensor_fields.items():
                loss_data = loss_data + F.mse_loss(u_pred_sensor[f], target)
            
            # 2. Physics loss (sample from domain)
            colloc_coords, _ = self.system.domain.build_grid(device=self.sensor_coords.device)
            colloc_coords = colloc_coords.reshape(-1, colloc_coords.shape[-1]).requires_grad_(True)
            u_pred_phys = self.model(colloc_coords)
            if not isinstance(u_pred_phys, dict):
                u_pred_phys = {"u": u_pred_phys}
            
            # The system equation needs to use the current InverseParameter values.
            # This implementation assumes the system's equation operators are linked to these parameters.
            from rippl.core.equation_system import EquationSystem
            if isinstance(self.system.equation, EquationSystem):
                loss_physics = self.system.equation.compute_loss(u_pred_phys, colloc_coords)
            else:
                pde_res = self.system.equation.compute_residual(u_pred_phys["u"], colloc_coords)
                loss_physics = (pde_res**2).mean()

            # 3. Penalties
            loss_penalty = sum(p.bounds_penalty() for p in self.parameters)
            
            total_loss = self.w_data * loss_data + self.w_physics * loss_physics + loss_penalty
            total_loss.backward()
            optimizer.step()
            
            for p in self.parameters:
                param_history[p.name].append(p.get().item())
                
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1} | Loss: {total_loss.item():.6e} | Data: {loss_data.item():.6e}")

        return {
            "identified_parameters": {p.name: p.get().item() for p in self.parameters},
            "parameter_history": param_history,
            "final_data_loss": loss_data.item(),
            "final_physics_loss": loss_physics.item()
        }

    def report(self) -> dict:
        """Returns identified parameter values with units."""
        return {
            p.name: {
                "value": p.get().item(),
                "units": p.units,
                "bounds": p.bounds
            }
            for p in self.parameters
        }

class InverseProblem:
    # Keep original InverseProblem for backward compatibility if needed, 
    # but the user didn't ask for it to be removed or kept.
    # I'll keep a minimal version or just remove it if it's redundant.
    # I'll keep it as it might be used by existing tests.
    def __init__(self, system, model, parameters: list,
                 observed_coords: torch.Tensor, 
                 observed_values: dict,
                 data_weight: float = 1.0):
        self.dt = DigitalTwin(system, model, parameters, sensor_coords=observed_coords, 
                             sensor_fields=observed_values, data_weight=data_weight)
    def train(self, epochs: int = 5000, lr: float = 1e-3):
        return self.dt.train(epochs=epochs, lr=lr)
    def result(self):
        return {p.name: p.get().item() for p in self.dt.parameters}

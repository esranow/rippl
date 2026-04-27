import torch
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any
from rippl.core.equation_system import EquationSystem

class InverseParameter:
    def __init__(self, name: str, initial_value: float, 
                 bounds: tuple = None, transform=None):
        self.name = name
        self.bounds = bounds
        self.transform = transform
        self.value = torch.nn.Parameter(torch.tensor(initial_value, dtype=torch.float32))
    
    def get(self) -> torch.Tensor:
        """Apply transformation and return the current parameter value."""
        if self.transform:
            return self.transform(self.value)
        return self.value
    
    def bounds_penalty(self) -> torch.Tensor:
        """Compute the quadratic penalty for parameter values outside specified bounds."""
        if not self.bounds:
            return torch.tensor(0.0, device=self.value.device)
        
        low, high = self.bounds
        val = self.get()
        penalty = torch.tensor(0.0, device=self.value.device)
        if val < low:
            penalty = (low - val)**2
        elif val > high:
            penalty = (val - high)**2
        return penalty

class InverseProblem:
    def __init__(self, system, model, parameters: list,
                 observed_coords: torch.Tensor, 
                 observed_values: dict,
                 data_weight: float = 1.0):
        self.system = system
        self.model = model
        self.parameters = parameters
        self.observed_coords = observed_coords
        self.observed_values = observed_values
        self.data_weight = data_weight
    
    def train(self, epochs: int = 5000, lr: float = 1e-3) -> Dict[str, float]:
        """Minimize the joint loss to recover parameters and fit the data."""
        # 1. Optimizer includes model parameters AND inverse parameters
        optimizer = optim.Adam([
            {'params': self.model.parameters()},
            {'params': [p.value for p in self.parameters]}
        ], lr=lr)
        
        # We need a way to pass the current parameter values to the operators.
        # Operators currently use self.alpha, etc. 
        # In Inverse Problems, we need to inject these values into the Equation/Operators.
        # The user's spec doesn't say HOW to inject them.
        # I'll assume for now that the user handles this by passing the parameters to the operators 
        # or that I should update the operators to accept a params dict.
        # Wait, Phase 3 Operators use _get_field(field, params).
        # Maybe I can inject them into the 'params' dict passed to compute().
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass at observed points
            self.observed_coords.requires_grad_(True)
            u_pred_all = self.model(self.observed_coords)
            fields_obs = u_pred_all if isinstance(u_pred_all, dict) else {"u": u_pred_all}
            
            # Data fidelity loss
            loss_data = torch.tensor(0.0, device=self.observed_coords.device)
            for field, target in self.observed_values.items():
                loss_data = loss_data + F.mse_loss(fields_obs[field], target.to(self.observed_coords.device))
            
            # Physics loss (Residual loss)
            # Need collocation points for physics loss?
            # The spec says: loss = residual_loss + data_weight * data_fidelity_loss + bounds_penalties
            # I'll use the observed_coords as collocation points for now, or assume a system domain sample.
            # But wait, usually PINNs use a separate grid for residual.
            # I'll sample from the domain.
            colloc_coords, _ = self.system.domain.build_grid(device=self.observed_coords.device)
            colloc_coords = colloc_coords.reshape(-1, colloc_coords.shape[-1]).requires_grad_(True)
            
            u_colloc = self.model(colloc_coords)
            fields_colloc = u_colloc if isinstance(u_colloc, dict) else {"u": u_colloc}
            
            # Inject current parameter values into the residual calculation
            # We'll create a params dict to pass to compute()
            current_params = {p.name: p.get() for p in self.parameters}
            
            if isinstance(self.system.equation, EquationSystem):
                loss_pde = self.system.equation.compute_loss(fields_colloc, colloc_coords, params=current_params)
            else:
                pde_res = self.system.equation.compute_residual(fields_colloc["u"], colloc_coords, params=current_params)
                loss_pde = (pde_res**2).mean()
            
            # Bounds penalties
            loss_penalty = sum(p.bounds_penalty() for p in self.parameters)
            
            total_loss = loss_pde + self.data_weight * loss_data + loss_penalty
            total_loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 500 == 0 or epoch == 0:
                param_str = ", ".join([f"{p.name}={p.get().item():.4f}" for p in self.parameters])
                print(f"Epoch {epoch+1:5d} | Loss: {total_loss.item():.6e} | Data: {loss_data.item():.6e} | {param_str}")
        
        return self.result()
    
    def result(self) -> Dict[str, float]:
        """Export the final estimated parameter values."""
        return {p.name: p.get().item() for p in self.parameters}

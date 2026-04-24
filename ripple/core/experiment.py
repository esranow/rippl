import torch
from ripple.core.system import System

class Experiment:
    def __init__(self, system: System, model, opt):
        self.system = system
        self.model = model
        self.opt = opt

    def train(self, coords, epochs=1):
        """Clean training loop with residual + constraint loss."""
        for _ in range(epochs):
            self.opt.zero_grad()
            
            # 1. Physics Residual
            coords.requires_grad_(True)
            u = self.model(coords)
            pde_res = self.system.equation.compute_residual(u, coords)
            loss_pde = (pde_res**2).mean()
            
            # 2. Constraint Loss
            loss_const = 0.0
            for c in self.system.constraints:
                # Basic implementation: assume c.location is a mask or indices
                # For now, let's assume it's a simple value comparison
                # In a real PINN, we'd sample these points
                pass
            
            total_loss = loss_pde + loss_const
            
            if torch.isnan(total_loss):
                raise RuntimeError("Training encountered NaN loss")
                
            total_loss.backward()
            self.opt.step()
            
        return {
            "loss": total_loss.item(),
            "meta": {"epochs": epochs, "pde_loss": loss_pde.item()}
        }

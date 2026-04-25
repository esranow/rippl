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
            device = coords.device
            import torch.nn.functional as F
            for c in self.system.constraints:
                c_coords = c.coords.to(device)
                u_pred = self.model(c_coords)
                u_target = c.value(c_coords) if callable(c.value) else c.value.to(device)
                loss_const += F.mse_loss(u_pred, u_target)
            
            total_loss = loss_pde + 100.0 * loss_const
            
            if torch.isnan(total_loss):
                raise RuntimeError("Training encountered NaN loss")
                
            total_loss.backward()
            self.opt.step()
            
        return {
            "loss": total_loss.item(),
            "meta": {"epochs": epochs, "pde_loss": loss_pde.item()}
        }

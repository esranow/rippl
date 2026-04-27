import torch
import json
from typing import Dict, List, Optional, Any
from rippl.core.system import System
from rippl.core.equation_system import EquationSystem

class PhysicsValidator:
    """Validator for PDE residuals, constraints, and conservation laws."""
    def __init__(self, system: System, model: torch.nn.Module, coords: torch.Tensor):
        self.system = system
        self.model = model
        self.coords = coords

    def residual_stats(self) -> Dict[str, Any]:
        """Compute statistics for PDE residuals."""
        # Enable gradients for residual computation
        if not self.coords.requires_grad:
            self.coords = self.coords.clone().detach().requires_grad_(True)
        
        u_out = self.model(self.coords)
        fields = u_out if isinstance(u_out, dict) else {"u": u_out}
        
        if isinstance(self.system.equation, EquationSystem):
            residuals = self.system.equation.compute_residuals(fields, self.coords)
            res = torch.cat(residuals)
        else:
            res = self.system.equation.compute_residual(fields["u"], self.coords)
        
        # Now compute stats without grad
        with torch.no_grad():
            mean_res = torch.mean(torch.abs(res)).item()
            return {
                "mean": mean_res,
                "max": torch.max(torch.abs(res)).item(),
                "std": torch.std(res).item(),
                "l2": torch.norm(res).item(),
                "passed": mean_res < 1e-2
            }

    def constraint_satisfaction(self) -> Dict[int, Dict[str, Any]]:
        """Compute error for each constraint."""
        stats = {}
        import torch.nn.functional as F
        from rippl.core.system import NeumannConstraint
        
        for i, c in enumerate(self.system.constraints):
            c_coords = c.coords.to(self.coords.device)
            if isinstance(c, NeumannConstraint):
                # Neumann needs gradients
                c_coords = c_coords.clone().detach().requires_grad_(True)
                u_pred_all = self.model(c_coords)
                fields_c = u_pred_all if isinstance(u_pred_all, dict) else {"u": u_pred_all}
                u_pred = fields_c[c.field]
                grad = torch.autograd.grad(u_pred.sum(), c_coords, create_graph=True)[0]
                val_pred = grad[..., c.normal_direction : c.normal_direction + 1]
            else:
                with torch.no_grad():
                    u_pred_all = self.model(c_coords)
                    fields_c = u_pred_all if isinstance(u_pred_all, dict) else {"u": u_pred_all}
                    val_pred = fields_c[c.field]
            
            with torch.no_grad():
                u_target = c.value(c_coords) if callable(c.value) else c.value
                if isinstance(u_target, (float, int)):
                    u_target = torch.full_like(val_pred, float(u_target))
                
                error = F.mse_loss(val_pred, u_target).item()
                stats[i] = {"error": error, "passed": error < 1e-3}
        return stats

    def conservation_check(self, laws: List, time_coords: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        """Check conservation laws over time."""
        stats = {}
        for law in laws:
            drift = law.penalty(self.model, time_coords).item()
            stats[law.name] = {
                "drift": drift,
                "passed": law.is_satisfied(self.model, time_coords)
            }
        return stats

    def full_report(self) -> Dict[str, Any]:
        """Runs all checks and prints formatted summary."""
        report = {
            "residuals": self.residual_stats(),
            "constraints": self.constraint_satisfaction()
        }
        print("\n--- Physics Validator Report ---")
        print(f"Residual Mean: {report['residuals']['mean']:.6f} [{'PASSED' if report['residuals']['passed'] else 'FAILED'}]")
        for i, s in report["constraints"].items():
            print(f"Constraint {i} Error: {s['error']:.6f} [{'PASSED' if s['passed'] else 'FAILED'}]")
        return report

    def export_report(self, path: str):
        """Saves full_report() as JSON."""
        with open(path, "w") as f:
            json.dump(self.full_report(), f, indent=4)

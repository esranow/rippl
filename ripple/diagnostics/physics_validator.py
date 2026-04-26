import torch
import json
from typing import List, Dict, Any
from ripple.core.equation_system import EquationSystem

class PhysicsValidator:
    def __init__(self, system, model, coords: torch.Tensor):
        self.system = system
        self.model = model
        self.coords = coords
    
    def residual_stats(self) -> dict:
        self.coords.requires_grad_(True)
        u_out = self.model(self.coords)
        fields = u_out if isinstance(u_out, dict) else {"u": u_out}
        spatial_dims = self.system.domain.spatial_dims
        
        if isinstance(self.system.equation, EquationSystem):
            res_list = self.system.equation.compute_residuals(fields, self.coords, spatial_dims=spatial_dims)
            # Combine residuals (abs sum for stats)
            res = torch.cat([r.view(-1) for r in res_list])
        else:
            res = self.system.equation.compute_residual(fields["u"], self.coords, spatial_dims=spatial_dims)
        
        res_abs = res.abs()
        stats = {
            "mean": float(res_abs.mean()),
            "max": float(res_abs.max()),
            "std": float(res_abs.std()),
            "l2": float(torch.sqrt((res**2).mean())),
            "passed": bool(res_abs.mean() < 1e-2)
        }
        return stats
    
    def constraint_satisfaction(self) -> dict:
        results = {}
        import torch.nn.functional as F
        from ripple.core.system import NeumannConstraint
        
        for i, c in enumerate(self.system.constraints):
            c_coords = c.coords.requires_grad_(True)
            u_pred_all = self.model(c_coords)
            fields_c = u_pred_all if isinstance(u_pred_all, dict) else {"u": u_pred_all}
            u_pred = fields_c[c.field]
            
            if isinstance(c, NeumannConstraint):
                grad = torch.autograd.grad(
                    u_pred, c_coords,
                    grad_outputs=torch.ones_like(u_pred),
                    create_graph=True
                )[0]
                val_pred = grad[..., c.normal_direction : c.normal_direction + 1]
            else:
                val_pred = u_pred
            
            u_target = c.value(c_coords) if callable(c.value) else c.value
            if isinstance(u_target, (float, int)):
                u_target = torch.full_like(val_pred, float(u_target))
            
            error = float(F.mse_loss(val_pred, u_target))
            results[i] = {"error": error, "passed": error < 1e-3}
        return results
    
    def conservation_check(self, laws: list, time_coords_list: list) -> dict:
        # laws: list of ConservationLaw
        # time_coords_list: list of coords tensors for different times
        results = {}
        for law in laws:
            drifts = []
            if law.reference is None:
                law.set_reference(self.model, time_coords_list[0])
                
            for tc in time_coords_list:
                drifts.append(law.is_satisfied(self.model, tc))
            
            # Simple check if all satisfied
            results[law.name] = {
                "drift": float(law.penalty(self.model, time_coords_list[-1]).sqrt()),
                "passed": all(drifts)
            }
        return results
    
    def full_report(self) -> dict:
        res_s = self.residual_stats()
        const_s = self.constraint_satisfaction()
        
        report = {
            "residuals": res_s,
            "constraints": const_s
        }
        
        print("\n" + "="*30)
        print("PHYSICS VALIDATION REPORT")
        print("="*30)
        print(f"Residual Mean: {res_s['mean']:.2e} [{'PASSED' if res_s['passed'] else 'FAILED'}]")
        print(f"Residual Max : {res_s['max']:.2e}")
        print("-"*30)
        for i, c in const_s.items():
            print(f"Constraint {i}: error={c['error']:.2e} [{'PASSED' if c['passed'] else 'FAILED'}]")
        print("="*30 + "\n")
        
        return report
    
    def export_report(self, path: str):
        report = self.full_report()
        with open(path, "w") as f:
            json.dump(report, f, indent=4)

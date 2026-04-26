import torch
from typing import List, Optional
from ripple.core.system import System
from ripple.core.equation_system import EquationSystem

class Experiment:
    def __init__(self, system: System, model, opt, 
                 use_hard_constraints: bool = False,
                 adaptive_collocation: bool = False,
                 conservation_laws: Optional[List] = None,
                 validate: bool = False):
        self.system = system
        self.model = model
        self.opt = opt
        self.use_hard_constraints = use_hard_constraints
        self.adaptive_collocation = adaptive_collocation
        self.conservation_laws = conservation_laws or []
        self.validate_after_train = validate
        
        if self.use_hard_constraints:
            from ripple.physics.distance import BoxDistance, HardConstraintWrapper
            dist_fn = BoxDistance(self.system.domain.bounds)
            self.model = HardConstraintWrapper(
                model, dist_fn, 
                particular_solution=self.system.particular_solution
            )
            
        if self.adaptive_collocation:
            from ripple.training.adaptive_sampler import AdaptiveCollocationSampler
            self.sampler = AdaptiveCollocationSampler(self.system.domain)

    def train(self, coords, epochs=1):
        """Clean training loop with residual + constraint + conservation loss."""
        # Feature: Set reference for conservation laws
        if self.conservation_laws:
            for law in self.conservation_laws:
                law.set_reference(self.model, coords)

        total_loss = torch.tensor(0.0)
        loss_pde = torch.tensor(0.0)
        
        for epoch in range(epochs):
            self.opt.zero_grad()
            
            # Feature 2: Adaptive Collocation
            if self.adaptive_collocation:
                self.sampler.update(self.model, self.system.equation, epoch)
                coords = self.sampler.current_points()
            
            # 1. Forward pass & Wrapping
            coords.requires_grad_(True)
            u_out = self.model(coords)
            fields = u_out if isinstance(u_out, dict) else {"u": u_out}
            
            # 2. Physics Loss
            spatial_dims = self.system.domain.spatial_dims
            if isinstance(self.system.equation, EquationSystem):
                loss_pde = self.system.equation.compute_loss(fields, coords, spatial_dims=spatial_dims)
            else:
                pde_res = self.system.equation.compute_residual(fields["u"], coords, spatial_dims=spatial_dims)
                loss_pde = (pde_res**2).mean()
            
            # 3. Constraint Loss
            loss_const = torch.tensor(0.0, device=coords.device)
            import torch.nn.functional as F
            from ripple.core.system import Constraint, NeumannConstraint
            for c in self.system.constraints:
                if isinstance(c, NeumannConstraint):
                    c_coords = c.coords.to(coords.device).requires_grad_(True)
                    u_pred_all = self.model(c_coords)
                    fields_c = u_pred_all if isinstance(u_pred_all, dict) else {"u": u_pred_all}
                    u_pred = fields_c[c.field]
                    
                    grad = torch.autograd.grad(
                        u_pred, c_coords,
                        grad_outputs=torch.ones_like(u_pred),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    val_pred = grad[..., c.normal_direction : c.normal_direction + 1]
                else:
                    if self.use_hard_constraints and c.type == "dirichlet":
                        continue
                    c_coords = c.coords.to(coords.device)
                    u_pred_all = self.model(c_coords)
                    fields_c = u_pred_all if isinstance(u_pred_all, dict) else {"u": u_pred_all}
                    val_pred = fields_c[c.field]
                
                u_target = c.value(c_coords) if callable(c.value) else c.value
                if isinstance(u_target, (float, int)):
                    u_target = torch.full_like(val_pred, float(u_target))
                else:
                    u_target = u_target.to(coords.device)
                loss_const = loss_const + F.mse_loss(val_pred, u_target)
            
            # 4. Conservation Loss
            loss_cons = torch.tensor(0.0, device=coords.device)
            for law in self.conservation_laws:
                loss_cons = loss_cons + law.penalty(self.model, coords)
            
            total_loss = loss_pde + 100.0 * loss_const + 10.0 * loss_cons
            
            if torch.isnan(total_loss):
                raise RuntimeError("Training encountered NaN loss")
                
            total_loss.backward()
            self.opt.step()
            
        # 5. Validation Framework
        if self.validate_after_train:
            from ripple.diagnostics.physics_validator import PhysicsValidator
            validator = PhysicsValidator(self.system, self.model, coords)
            validator.full_report()

        return {
            "loss": total_loss.item(),
            "meta": {"epochs": epochs, "pde_loss": loss_pde.item()}
        }

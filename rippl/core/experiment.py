import torch
from typing import List, Optional, Any, Dict
from rippl.core.system import System
from rippl.core.equation_system import EquationSystem

from rippl.training.causal import CausalTrainingMixin

class Experiment(CausalTrainingMixin):
    def __init__(self, system: System, model, opt, 
                 use_hard_constraints: bool = False,
                 adaptive_collocation: bool = False,
                 causal_training: bool = False,
                 causal_mode: str = "continuous",
                 causal_epsilon: Optional[float] = None,
                 causal_bins: int = 10,
                 adaptive_loss: bool = False,
                 adaptive_loss_mode: str = "gradient_norm",
                 adaptive_loss_freq: int = 100,
                 adaptive_loss_alpha: float = 0.9,
                 conservation_laws: Optional[List] = None,
                 validate: bool = False):
        self.system = system
        self.model = model
        self.opt = opt
        self.use_hard_constraints = use_hard_constraints
        self.adaptive_collocation = adaptive_collocation
        self.causal_training = causal_training
        self.causal_mode = causal_mode
        self.causal_epsilon = causal_epsilon
        self.causal_bins = causal_bins
        self.adaptive_loss = adaptive_loss
        self.adaptive_loss_mode = adaptive_loss_mode
        self.adaptive_loss_freq = adaptive_loss_freq
        self.adaptive_loss_alpha = adaptive_loss_alpha
        self.conservation_laws = conservation_laws or []
        self.validate_after_train = validate
        
        if self.adaptive_loss:
            from rippl.training.ntk_weighting import AdaptiveLossBalancer
            loss_names = ["pde"] + [f"const_{i}" for i in range(len(self.system.constraints))]
            self.balancer = AdaptiveLossBalancer(
                mode=self.adaptive_loss_mode,
                loss_names=loss_names,
                update_freq=self.adaptive_loss_freq,
                alpha=self.adaptive_loss_alpha
            )
        
        if self.use_hard_constraints:
            from rippl.physics.distance import BoxDistance, HardConstraintWrapper
            dist_fn = BoxDistance(self.system.domain.bounds)
            self.model = HardConstraintWrapper(
                model, dist_fn, 
                particular_solution=self.system.particular_solution
            )
            
        if self.adaptive_collocation:
            from rippl.training.adaptive_sampler import AdaptiveCollocationSampler
            self.sampler = AdaptiveCollocationSampler(self.system.domain)

    def train(self, coords: torch.Tensor, epochs: int = 1) -> Dict[str, Any]: # coords: (N, D)
        """Run the training loop with residual, constraint, and conservation losses."""
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
                residuals = self.system.equation.compute_residuals(fields, coords, spatial_dims=spatial_dims)
                if self.causal_training:
                    # Apply causal weighting to each residual
                    weighted_losses = []
                    for res in residuals:
                        if self.causal_mode == "binned":
                            w = self.compute_causal_weights_binned(coords, res, n_bins=self.causal_bins, epsilon=self.causal_epsilon)
                        else:
                            w = self.compute_causal_weights_continuous(coords, res, epsilon=self.causal_epsilon)
                        weighted_losses.append(torch.mean(w * res**2))
                    
                    # Sum weighted losses with equation weights
                    loss_pde = torch.tensor(0.0, device=coords.device)
                    for l, weight in zip(weighted_losses, self.system.equation.weights):
                        loss_pde = loss_pde + weight * l
                else:
                    loss_pde = self.system.equation.compute_loss(fields, coords, spatial_dims=spatial_dims)
            else:
                pde_res = self.system.equation.compute_residual(fields["u"], coords, spatial_dims=spatial_dims)
                if self.causal_training:
                    if self.causal_mode == "binned":
                        w = self.compute_causal_weights_binned(coords, pde_res, n_bins=self.causal_bins, epsilon=self.causal_epsilon)
                    else:
                        w = self.compute_causal_weights_continuous(coords, pde_res, epsilon=self.causal_epsilon)
                    loss_pde = (w * pde_res**2).mean()
                else:
                    loss_pde = (pde_res**2).mean()
            
            # Feature: Auto-tune causal epsilon
            if self.causal_training and epoch % 100 == 0:
                main_res = pde_res if not isinstance(self.system.equation, EquationSystem) else residuals[0]
                self.causal_epsilon = self.optimal_epsilon(main_res)
            
            # 3. Constraint Loss
            loss_dict = {"pde": loss_pde}
            import torch.nn.functional as F
            from rippl.core.system import Constraint, NeumannConstraint
            for i, c in enumerate(self.system.constraints):
                if isinstance(c, NeumannConstraint):
                    c_coords = c.coords.to(coords.device).requires_grad_(True)
                    u_pred_all = self.model(c_coords)
                    fields_c = u_pred_all if isinstance(u_pred_all, dict) else {"u": u_pred_all}
                    u_pred = fields_c[c.field]
                    grad = torch.autograd.grad(u_pred.sum(), c_coords, create_graph=True, retain_graph=True)[0]
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
                
                loss_dict[f"const_{i}"] = F.mse_loss(val_pred, u_target)
            
            # 4. Adaptive Loss Weighting
            if self.adaptive_loss:
                temp_total = sum(loss_dict.values())
                self.balancer.step(self.model, loss_dict, temp_total, epoch)
                total_loss = self.balancer.apply(loss_dict)
            else:
                total_loss = loss_pde + 100.0 * sum(v for k, v in loss_dict.items() if k != "pde")

            # 5. Conservation Loss
            loss_cons = torch.tensor(0.0, device=coords.device)
            for law in self.conservation_laws:
                loss_cons = loss_cons + law.penalty(self.model, coords)
            
            total_loss = total_loss + 10.0 * loss_cons
            
            if epoch % 500 == 0:
                if self.causal_training: print(f"Epoch {epoch}: Causal Epsilon = {self.causal_epsilon:.4f}")
                if self.adaptive_loss: print(f"Epoch {epoch}: Balancer weights = {self.balancer.log()}")
            
            if torch.isnan(total_loss):
                raise RuntimeError("Training encountered NaN loss")
                
            total_loss.backward()
            self.opt.step()
            
        # 5. Validation Framework
        if self.validate_after_train:
            from rippl.diagnostics.physics_validator import PhysicsValidator
            validator = PhysicsValidator(self.system, self.model, coords)
            validator.full_report()

        return {
            "loss": total_loss.item(),
            "meta": {"epochs": epochs, "pde_loss": loss_pde.item()}
        }

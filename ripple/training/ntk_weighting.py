import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import warnings

class GradientNormWeighting:
    """Adaptive loss weighting via gradient norm balancing."""
    def __init__(self, loss_names: List[str], update_freq: int = 100, alpha: float = 0.9, 
                 clip_range: Tuple[float, float] = (0.01, 100.0)):
        self.loss_names = loss_names
        self.update_freq = update_freq
        self.alpha = alpha
        self.clip_range = clip_range
        self.weights = {name: 1.0 for name in loss_names}

    def update(self, model: nn.Module, loss_dict: Dict[str, torch.Tensor], total_loss: torch.Tensor):
        """Update weights using ratio of total gradient norm to component gradient norm."""
        # 1. Total gradient norm
        model.zero_grad()
        total_loss.backward(retain_graph=True)
        total_grad_norm = self._get_grad_norm(model)
        
        new_weights = {}
        for name in self.loss_names:
            model.zero_grad()
            loss_dict[name].backward(retain_graph=True)
            gi_norm = self._get_grad_norm(model)
            
            # λi = mean(|∇L_total|) / (mean(|∇Li|) + 1e-8)
            lambda_i = total_grad_norm / (gi_norm + 1e-8)
            lambda_i = float(torch.clamp(torch.tensor(lambda_i), *self.clip_range))
            
            # EMA smooth
            self.weights[name] = self.alpha * self.weights[name] + (1 - self.alpha) * lambda_i
            
        model.zero_grad()

    def _get_grad_norm(self, model: nn.Module) -> float:
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads: return 0.0
        return torch.cat(grads).norm().item()

    def apply(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(iter(loss_dict.values())).device)
        for name, loss in loss_dict.items():
            total = total + self.weights.get(name, 1.0) * loss
        return total

    def log(self) -> Dict[str, float]:
        return {f"weight_{name}": w for name, w in self.weights.items()}

class NTKDiagonalWeighting:
    """Adaptive loss weighting via Neural Tangent Kernel (NTK) diagonal trace."""
    def __init__(self, loss_names: List[str], update_freq: int = 100, alpha: float = 0.9):
        self.loss_names = loss_names
        self.update_freq = update_freq
        self.alpha = alpha
        self.weights = {name: 1.0 for name in loss_names}

    def _compute_jacobian_norm(self, model: nn.Module, loss: torch.Tensor) -> float:
        """||∂loss/∂θ||² via autograd.grad"""
        model.zero_grad()
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
        norm_sq = 0.0
        for g in grads:
            if g is not None:
                norm_sq += torch.sum(g**2).item()
        return norm_sq

    def update(self, model: nn.Module, loss_dict: Dict[str, torch.Tensor], total_loss: torch.Tensor, n_points: int = 0):
        if n_points > 500:
            warnings.warn(f"NTK computation may be slow with {n_points} points.", UserWarning)
            
        # K_ii = ||∂Li/∂θ||²
        # Weight λi = Trace(K_total) / Trace(K_i)
        # For diagonal NTK, this simplifies to sum of ||∂L_total/∂θ||² / sum of ||∂Li/∂θ||²
        # Actually a common PINN NTK strategy is λi = max_j(Trace(Kj)) / Trace(Ki)
        # But I'll follow the gradient norm ratio pattern if not specified.
        # Spec says: Same interface as GradientNormWeighting (using mean grad ratio)
        
        model.zero_grad()
        total_norm_sq = self._compute_jacobian_norm(model, total_loss)
        
        for name in self.loss_names:
            ni_sq = self._compute_jacobian_norm(model, loss_dict[name])
            lambda_i = total_norm_sq / (ni_sq + 1e-8)
            # EMA
            self.weights[name] = self.alpha * self.weights[name] + (1 - self.alpha) * lambda_i
        
        model.zero_grad()

    def apply(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(iter(loss_dict.values())).device)
        for name, loss in loss_dict.items():
            total = total + self.weights.get(name, 1.0) * loss
        return total

    def log(self) -> Dict[str, float]:
        return {f"ntk_weight_{name}": w for name, w in self.weights.items()}

class AdaptiveLossBalancer:
    """Orchestrates different adaptive loss weighting modes."""
    def __init__(self, mode: str = "gradient_norm", loss_names: Optional[List[str]] = None, 
                 update_freq: int = 100, alpha: float = 0.9, ntk_max_points: int = 500):
        self.mode = mode
        self.loss_names = loss_names or []
        self.update_freq = update_freq
        self.alpha = alpha
        self.ntk_max_points = ntk_max_points
        
        if mode == "gradient_norm":
            self.balancer = GradientNormWeighting(self.loss_names, update_freq, alpha)
        elif mode == "ntk":
            self.balancer = NTKDiagonalWeighting(self.loss_names, update_freq, alpha)
        else:
            self.balancer = None

    def step(self, model: nn.Module, loss_dict: Dict[str, torch.Tensor], total_loss: torch.Tensor, epoch: int):
        if self.balancer and epoch % self.update_freq == 0:
            # Safely get number of points from the first loss tensor if it has dimensions, else 1
            first_loss = next(iter(loss_dict.values())) if loss_dict else None
            n_points = first_loss.shape[0] if (first_loss is not None and first_loss.dim() > 0) else 1
            if self.mode == "ntk":
                self.balancer.update(model, loss_dict, total_loss, n_points)
            else:
                self.balancer.update(model, loss_dict, total_loss)

    def apply(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.balancer:
            return self.balancer.apply(loss_dict)
        return sum(loss_dict.values())

    def log(self) -> Dict[str, float]:
        return self.balancer.log() if self.balancer else {}

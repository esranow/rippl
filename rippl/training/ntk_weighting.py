import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import warnings

class GradientNormWeighting:
    def __init__(self, loss_names: List[str], update_freq: int = 100, alpha: float = 0.9, 
                 clip_range: Tuple[float, float] = (0.01, 100.0)):
        self.loss_names = loss_names
        self.update_freq = update_freq
        self.alpha = alpha
        self.clip_range = clip_range
        self.weights = {name: 1.0 for name in loss_names}

    def update(self, model: nn.Module, loss_dict: Dict[str, torch.Tensor], total_loss: torch.Tensor):
        model.zero_grad()
        total_loss.backward(retain_graph=True)
        total_grad_norm = self._get_grad_norm(model)
        for name in self.loss_names:
            model.zero_grad()
            loss_dict[name].backward(retain_graph=True)
            gi_norm = self._get_grad_norm(model)
            lambda_i = total_grad_norm / (gi_norm + 1e-8)
            lambda_i = float(torch.clamp(torch.tensor(lambda_i), *self.clip_range))
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

class NTKDiagonalWeighting:
    def __init__(self, loss_names: List[str], update_freq: int = 100, alpha: float = 0.9):
        self.loss_names = loss_names
        self.update_freq = update_freq
        self.alpha = alpha
        self.weights = {name: 1.0 for name in loss_names}

    def _compute_jacobian_norm(self, model: nn.Module, loss: torch.Tensor) -> float:
        model.zero_grad()
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
        norm_sq = 0.0
        for g in grads:
            if g is not None: norm_sq += torch.sum(g**2).item()
        return norm_sq

    def update(self, model: nn.Module, loss_dict: Dict[str, torch.Tensor], total_loss: torch.Tensor):
        model.zero_grad()
        total_norm_sq = self._compute_jacobian_norm(model, total_loss)
        for name in self.loss_names:
            ni_sq = self._compute_jacobian_norm(model, loss_dict[name])
            lambda_i = total_norm_sq / (ni_sq + 1e-8)
            self.weights[name] = self.alpha * self.weights[name] + (1 - self.alpha) * lambda_i
        model.zero_grad()

    def apply(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(iter(loss_dict.values())).device)
        for name, loss in loss_dict.items():
            total = total + self.weights.get(name, 1.0) * loss
        return total

class AdaptiveLossBalancer:
    def __init__(self, mode: str = "gradient_norm", loss_names: Optional[List[str]] = None, 
                 update_freq: int = 100, alpha: float = 0.9):
        self.mode = mode
        self.loss_names = loss_names or []
        if mode == "gradient_norm":
            self.balancer = GradientNormWeighting(self.loss_names, update_freq, alpha)
        elif mode == "ntk":
            self.balancer = NTKDiagonalWeighting(self.loss_names, update_freq, alpha)
        else:
            self.balancer = None

    def update(self, model, loss_dict, total_loss):
        if self.balancer: self.balancer.update(model, loss_dict, total_loss)

    def apply(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.balancer: return self.balancer.apply(loss_dict)
        return sum(loss_dict.values())

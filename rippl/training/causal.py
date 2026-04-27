import torch
import numpy as np
from typing import Optional

class CausalTrainingMixin:
    """Mixin for causal weights computation in PINNs."""
    
    def optimal_epsilon(self, residuals: torch.Tensor) -> float:
        """Heuristic for epsilon: 1.0 / (mean_residual + 1e-8) clipped to [0.1, 100.0]."""
        mean_res = torch.mean(torch.abs(residuals)).item()
        eps = 1.0 / (mean_res + 1e-8)
        return float(np.clip(eps, 0.1, 100.0))

    def compute_causal_weights_binned(self, coords: torch.Tensor, residuals: torch.Tensor, 
                                      n_bins: int = 10, epsilon: Optional[float] = None) -> torch.Tensor:
        """Bin by coords[:, -1] (time), weight_i = exp(-epsilon * sum(L_j for j<i))."""
        if epsilon is None:
            epsilon = self.optimal_epsilon(residuals)
            
        t = coords[:, -1]
        t_min, t_max = t.min(), t.max()
        bins = torch.linspace(t_min, t_max, n_bins + 1, device=coords.device)
        
        # Mean residual per bin
        bin_losses = torch.zeros(n_bins, device=coords.device)
        bin_indices = []
        
        for i in range(n_bins):
            mask = (t >= bins[i]) & (t <= bins[i+1])
            idx = torch.where(mask)[0]
            bin_indices.append(idx)
            if len(idx) > 0:
                bin_losses[i] = torch.mean(residuals[idx]**2)
        
        # Weights computation: w_i = exp(-epsilon * sum_{j<i} L_j)
        # sum_{j<i} L_j can be computed via cumsum on bin_losses shifted
        cum_losses = torch.cumsum(bin_losses, dim=0)
        # Shifted cumsum: [0, L0, L0+L1, ...]
        zero = torch.zeros(1, device=coords.device)
        if cum_losses.dim() > 1:
            zero = zero.view(1, 1)
        shifted_cum_losses = torch.cat([zero, cum_losses[:-1]], dim=0)
        bin_weights = torch.exp(-epsilon * shifted_cum_losses)
        
        # Assign to all points in bin
        weights = torch.ones_like(residuals)
        for i in range(n_bins):
            if len(bin_indices[i]) > 0:
                weights[bin_indices[i]] = bin_weights[i]
                
        return weights.detach()

    def compute_causal_weights_continuous(self, coords: torch.Tensor, residuals: torch.Tensor, 
                                          epsilon: Optional[float] = None) -> torch.Tensor:
        """Sort by time, weight_i = exp(-epsilon * cumsum_{j:t_j<t_i} r_j^2)."""
        if epsilon is None:
            epsilon = self.optimal_epsilon(residuals)
            
        t = coords[:, -1]
        # Sort indices by time
        sort_idx = torch.argsort(t)
        inv_sort_idx = torch.argsort(sort_idx)
        
        sorted_res_sq = residuals[sort_idx]**2
        # Cumsum of losses up to point i (not including i? usually causal weights use sum of previous)
        # The spec says: cumsum_{j:t_j<t_i} r_j
        cum_losses = torch.cumsum(sorted_res_sq, dim=0)
        # Shift to get sum of previous
        zero = torch.zeros(1, device=coords.device)
        if cum_losses.dim() > 1:
            zero = zero.view(1, 1)
        shifted_cum_losses = torch.cat([zero, cum_losses[:-1]], dim=0)
        
        sorted_weights = torch.exp(-epsilon * shifted_cum_losses)
        
        # Unsort
        weights = sorted_weights[inv_sort_idx].view_as(residuals)
        return weights.detach()

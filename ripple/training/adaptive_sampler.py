import torch
from torch.quasirandom import SobolEngine
from ripple.core.equation_system import EquationSystem

class AdaptiveCollocationSampler:
    def __init__(self, domain, n_points=5000, n_candidates=50000, 
                 update_freq=500, device="cpu"):
        self.domain = domain
        self.n_points = n_points
        self.n_candidates = n_candidates
        self.update_freq = update_freq
        self.device = device
        self.dim = self.domain.spatial_dims
        # Total independent variables might be dim + 1 if time is included, 
        # but Domain rule says bounds matches spatial_dims. 
        # However, PINNs often use (x, t). 
        # Let's assume self.dim is the number of variables to sample.
        
        self.points = self.initial_sample()

    def initial_sample(self) -> torch.Tensor:
        sobol = SobolEngine(dimension=self.dim, scramble=True)
        # Sample in [0, 1]^d
        samples = sobol.draw(self.n_points).to(self.device)
        # Rescale to bounds
        for i, (low, high) in enumerate(self.domain.bounds):
            samples[:, i] = samples[:, i] * (high - low) + low
        return samples.requires_grad_(False)

    def update(self, model, equation, epoch) -> torch.Tensor:
        if epoch % self.update_freq != 0 and hasattr(self, 'points'):
            return self.points

        # 1. Sample n_candidates via Sobol
        sobol = SobolEngine(dimension=self.dim, scramble=True)
        candidates = sobol.draw(self.n_candidates).to(self.device)
        for i, (low, high) in enumerate(self.domain.bounds):
            candidates[:, i] = candidates[:, i] * (high - low) + low
        
        candidates.requires_grad_(True)
        
        # 2. Compute residual at candidates (no grad for model params, but need for coords)
        # However, the spec says "no grad" for the residual calculation? 
        # "compute residual at all candidates (no grad)" 
        # Wait, you NEED grad w.r.t coords to compute Laplacian/Gradients.
        # It probably means no grad w.r.t model parameters.
        
        with torch.set_grad_enabled(True):
            u_out = model(candidates)
            fields = u_out if isinstance(u_out, dict) else {"u": u_out}
            
            if isinstance(equation, EquationSystem):
                # EquationSystem.compute_residuals returns a list of tensors
                res_list = equation.compute_residuals(fields, candidates, spatial_dims=self.dim)
                # Combine residuals (e.g., sum of squares)
                total_res = torch.stack([r**2 for r in res_list]).sum(dim=0)
            else:
                pde_res = equation.compute_residual(fields["u"], candidates, spatial_dims=self.dim)
                total_res = pde_res**2
            
            # 3. Compute importance weights
            # total_res shape: (n_candidates, 1)
            weights = total_res.detach().squeeze()
            weights = weights + 1e-10 # Ensure no zero probabilities
            weights = weights / weights.sum()
            
            # 4. Sample n_points indices proportional to weights
            indices = torch.multinomial(weights, self.n_points, replacement=True)
            self.points = candidates[indices].detach().requires_grad_(False)
            
        return self.points

    def current_points(self) -> torch.Tensor:
        return self.points

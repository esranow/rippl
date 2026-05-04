"""
Spectral collocation samplers.
Chebyshev and Legendre points for smooth PDE solutions.
Exponential convergence vs algebraic for random/Sobol sampling.
"""
import torch
import numpy as np
import math

class ChebyshevSampler:
    """
    Chebyshev nodes of the second kind on [a,b]^d.
    x_k = cos(kπ/N) mapped to [a,b].
    Optimal for smooth functions — avoids Runge phenomenon.
    
    Args:
        domain: Domain object with bounds
        n_per_dim: number of Chebyshev points per dimension
    """
    def __init__(self, domain, n_per_dim: int = 32):
        self.domain = domain
        self.n = n_per_dim

    def sample(self) -> torch.Tensor:
        grids = []
        for (lo, hi) in self.domain.bounds:
            k = torch.arange(self.n)
            pts = torch.cos(math.pi * k / (self.n - 1))  # [-1, 1]
            pts = 0.5 * (lo + hi) + 0.5 * (hi - lo) * pts  # [lo, hi]
            grids.append(pts)
        # tensor product grid
        mesh = torch.meshgrid(*grids, indexing='ij')
        return torch.stack([g.flatten() for g in mesh], dim=1)

    def to_loader(self, batch_size: int = 2048):
        from torch.utils.data import DataLoader, TensorDataset
        pts = self.sample()
        return DataLoader(TensorDataset(pts), batch_size=batch_size, shuffle=True)


class LegendreSampler:
    """
    Gauss-Legendre quadrature points on [a,b]^d.
    Optimal for numerical integration — exact for polynomials up to degree 2N-1.
    Use when weak-form/variational residual is needed.
    """
    def __init__(self, domain, n_per_dim: int = 32):
        self.domain = domain
        self.n = n_per_dim

    def sample(self) -> tuple:
        """Returns (points, weights) for quadrature."""
        pts_1d, wts_1d = np.polynomial.legendre.leggauss(self.n)
        grids, weight_grids = [], []
        for (lo, hi) in self.domain.bounds:
            scale = (hi - lo) / 2
            shift = (hi + lo) / 2
            pts = torch.tensor(pts_1d * scale + shift, dtype=torch.float32)
            wts = torch.tensor(wts_1d * scale, dtype=torch.float32)
            grids.append(pts)
            weight_grids.append(wts)
        mesh = torch.meshgrid(*grids, indexing='ij')
        wt_mesh = torch.meshgrid(*weight_grids, indexing='ij')
        points = torch.stack([g.flatten() for g in mesh], dim=1)
        weights = torch.ones(points.shape[0])
        for w in wt_mesh:
            weights = weights * w.flatten()
        return points, weights

    def to_loader(self, batch_size: int = 2048):
        from torch.utils.data import DataLoader, TensorDataset
        pts, _ = self.sample()
        return DataLoader(TensorDataset(pts), batch_size=batch_size, shuffle=True)

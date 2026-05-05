import torch
import numpy as np

class ChebyshevSampler:
    def __init__(self, dom, n_per_dim=32):
        self.dom = dom; self.n = n_per_dim
    def sample(self):
        pts = []
        for (l, h) in self.dom.bounds:
            x = np.cos((2*np.arange(1, self.n+1)-1)*np.pi/(2*self.n))
            pts.append(torch.tensor(0.5*(x+1)*(h-l)+l, dtype=torch.float32))
        grid = torch.meshgrid(*pts, indexing='ij')
        return torch.stack(grid, dim=-1).reshape(-1, len(self.dom.bounds))

class LegendreSampler:
    def __init__(self, dom, n_per_dim=32):
        self.dom = dom; self.n = n_per_dim
    def sample(self):
        pts = []; wts = []
        for (l, h) in self.dom.bounds:
            x, w = np.polynomial.legendre.leggauss(self.n)
            pts.append(torch.tensor(0.5*(x+1)*(h-l)+l, dtype=torch.float32))
            wts.append(torch.tensor(w*0.5*(h-l), dtype=torch.float32))
        p_grid = torch.meshgrid(*pts, indexing='ij')
        w_grid = torch.meshgrid(*wts, indexing='ij')
        p = torch.stack(p_grid, dim=-1).reshape(-1, len(self.dom.bounds))
        w = torch.stack(w_grid, dim=-1).prod(dim=-1).reshape(-1)
        return p, w

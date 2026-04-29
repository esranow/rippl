"""
rippl.geometry.csg — Constructive Solid Geometry for PDE Domains.
"""
import torch
import numpy as np
from typing import List, Tuple, Optional
from rippl.core.system import Domain

class Shape:
    """Abstract base class for all CSG shapes."""
    def contains(self, points: torch.Tensor) -> torch.Tensor:
        # points: (N, D), returns boolean mask (N,)
        raise NotImplementedError
        
    def sample_boundary(self, n: int) -> torch.Tensor:
        raise NotImplementedError
        
    def bounding_box(self) -> List[Tuple[float, float]]:
        # returns [(x_min,x_max), (y_min,y_max), ...]
        raise NotImplementedError
        
    def __and__(self, other): return Intersection(self, other)
    def __or__(self, other): return Union(self, other)
    def __sub__(self, other): return Difference(self, other)
    def __invert__(self): return Complement(self)

class Circle(Shape):
    def __init__(self, center=(0.0, 0.0), radius=1.0):
        self.center = torch.tensor(center, dtype=torch.float32)
        self.radius = radius
        
    def contains(self, points: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(points - self.center.to(points.device), dim=-1)
        return dist <= self.radius
        
    def bounding_box(self):
        c = self.center.numpy()
        r = self.radius
        return [(c[0]-r, c[0]+r), (c[1]-r, c[1]+r)]
        
    def sample_boundary(self, n: int) -> torch.Tensor:
        theta = torch.rand(n) * 2 * np.pi
        x = self.center[0] + self.radius * torch.cos(theta)
        y = self.center[1] + self.radius * torch.sin(theta)
        return torch.stack([x, y], dim=-1)

class Rectangle(Shape):
    def __init__(self, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        
    def contains(self, points: torch.Tensor) -> torch.Tensor:
        mask = (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max) & \
               (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max)
        return mask
        
    def bounding_box(self):
        return [(self.x_min, self.x_max), (self.y_min, self.y_max)]
        
    def sample_boundary(self, n: int) -> torch.Tensor:
        # Sample points on the perimeter
        Lx = self.x_max - self.x_min
        Ly = self.y_max - self.y_min
        perimeter = 2 * (Lx + Ly)
        
        t = torch.rand(n) * perimeter
        pts = torch.zeros(n, 2)
        
        # Bottom side
        mask = t < Lx
        pts[mask, 0] = self.x_min + t[mask]
        pts[mask, 1] = self.y_min
        
        # Right side
        mask = (t >= Lx) & (t < Lx + Ly)
        pts[mask, 0] = self.x_max
        pts[mask, 1] = self.y_min + (t[mask] - Lx)
        
        # Top side
        mask = (t >= Lx + Ly) & (t < 2*Lx + Ly)
        pts[mask, 0] = self.x_max - (t[mask] - (Lx + Ly))
        pts[mask, 1] = self.y_max
        
        # Left side
        mask = t >= 2*Lx + Ly
        pts[mask, 0] = self.x_min
        pts[mask, 1] = self.y_max - (t[mask] - (2*Lx + Ly))
        
        return pts

class Sphere(Shape):
    def __init__(self, center=(0.0, 0.0, 0.0), radius=1.0):
        self.center = torch.tensor(center, dtype=torch.float32)
        self.radius = radius
        
    def contains(self, points: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(points - self.center.to(points.device), dim=-1)
        return dist <= self.radius
        
    def bounding_box(self):
        c = self.center.numpy()
        r = self.radius
        return [(c[0]-r, c[0]+r), (c[1]-r, c[1]+r), (c[2]-r, c[2]+r)]
        
    def sample_boundary(self, n: int) -> torch.Tensor:
        # Uniform sampling on S^2
        phi = torch.rand(n) * 2 * np.pi
        cos_theta = torch.rand(n) * 2 - 1
        sin_theta = torch.sqrt(1 - cos_theta**2)
        
        x = self.radius * sin_theta * torch.cos(phi)
        y = self.radius * sin_theta * torch.sin(phi)
        z = self.radius * cos_theta
        
        return torch.stack([x, y, z], dim=-1) + self.center

class Box(Shape):
    def __init__(self, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1):
        self.bounds = [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        
    def contains(self, points: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
        for i, (lo, hi) in enumerate(self.bounds):
            mask &= (points[:, i] >= lo) & (points[:, i] <= hi)
        return mask
        
    def bounding_box(self):
        return self.bounds
        
    def sample_boundary(self, n: int) -> torch.Tensor:
        # Surface area of each face
        areas = []
        for i in range(3):
            other_dims = [j for j in range(3) if j != i]
            L1 = self.bounds[other_dims[0]][1] - self.bounds[other_dims[0]][0]
            L2 = self.bounds[other_dims[1]][1] - self.bounds[other_dims[1]][0]
            areas.append(L1 * L2) # Two faces per dimension
            
        total_area = 2 * sum(areas)
        probs = [a / total_area for a in areas] * 2 # 6 faces
        face_idx = torch.multinomial(torch.tensor(probs), n, replacement=True)
        
        pts = torch.zeros(n, 3)
        for i in range(6):
            mask = face_idx == i
            n_face = mask.sum().item()
            if n_face == 0: continue
            
            dim = i % 3
            side = i // 3 # 0 for min, 1 for max
            other_dims = [j for j in range(3) if j != dim]
            
            pts[mask, dim] = self.bounds[dim][side]
            for od in other_dims:
                pts[mask, od] = self.bounds[od][0] + torch.rand(n_face) * (self.bounds[od][1] - self.bounds[od][0])
        return pts

class Annulus(Shape):
    def __init__(self, center=(0.0, 0.0), r_inner=0.5, r_outer=1.0):
        self.center = torch.tensor(center, dtype=torch.float32)
        self.r_inner = r_inner
        self.r_outer = r_outer
        
    def contains(self, points: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(points - self.center.to(points.device), dim=-1)
        return (dist >= self.r_inner) & (dist <= self.r_outer)
        
    def bounding_box(self):
        c = self.center.numpy()
        r = self.r_outer
        return [(c[0]-r, c[0]+r), (c[1]-r, c[1]+r)]
        
    def sample_boundary(self, n: int) -> torch.Tensor:
        # Two circles
        n_inner = int(n * self.r_inner / (self.r_inner + self.r_outer))
        n_outer = n - n_inner
        
        pts_inner = Circle(self.center, self.r_inner).sample_boundary(n_inner)
        pts_outer = Circle(self.center, self.r_outer).sample_boundary(n_outer)
        return torch.cat([pts_inner, pts_outer], dim=0)

class Ellipse(Shape):
    def __init__(self, center=(0.0, 0.0), a=1.0, b=0.5):
        self.center = torch.tensor(center, dtype=torch.float32)
        self.a, self.b = a, b
        
    def contains(self, points: torch.Tensor) -> torch.Tensor:
        p = points - self.center.to(points.device)
        return (p[:, 0]**2 / self.a**2 + p[:, 1]**2 / self.b**2) <= 1.0
        
    def bounding_box(self):
        c = self.center.numpy()
        return [(c[0]-self.a, c[0]+self.a), (c[1]-self.b, c[1]+self.b)]
        
    def sample_boundary(self, n: int) -> torch.Tensor:
        # Approximate boundary sampling
        theta = torch.rand(n) * 2 * np.pi
        x = self.center[0] + self.a * torch.cos(theta)
        y = self.center[1] + self.b * torch.sin(theta)
        return torch.stack([x, y], dim=-1)

# Boolean operations
class Union(Shape):
    def __init__(self, a, b): self.a, self.b = a, b
    def contains(self, p): return self.a.contains(p) | self.b.contains(p)
    def bounding_box(self):
        ba, bb = self.a.bounding_box(), self.b.bounding_box()
        return [(min(ba[i][0], bb[i][0]), max(ba[i][1], bb[i][1])) for i in range(len(ba))]
    def sample_boundary(self, n):
        pts_a = self.a.sample_boundary(n)
        pts_b = self.b.sample_boundary(n)
        res = torch.cat([pts_a[~self.b.contains(pts_a)], pts_b[~self.a.contains(pts_b)]], dim=0)
        return res[torch.randperm(len(res))[:n]]

class Intersection(Shape):
    def __init__(self, a, b): self.a, self.b = a, b
    def contains(self, p): return self.a.contains(p) & self.b.contains(p)
    def bounding_box(self):
        ba, bb = self.a.bounding_box(), self.b.bounding_box()
        return [(max(ba[i][0], bb[i][0]), min(ba[i][1], bb[i][1])) for i in range(len(ba))]
    def sample_boundary(self, n):
        pts_a = self.a.sample_boundary(n)
        pts_b = self.b.sample_boundary(n)
        res = torch.cat([pts_a[self.b.contains(pts_a)], pts_b[self.a.contains(pts_b)]], dim=0)
        return res[torch.randperm(len(res))[:n]]

class Difference(Shape):
    def __init__(self, a, b): self.a, self.b = a, b
    def contains(self, p): return self.a.contains(p) & (~self.b.contains(p))
    def bounding_box(self): return self.a.bounding_box()
    def sample_boundary(self, n):
        pts_a = self.a.sample_boundary(n)
        pts_b = self.b.sample_boundary(n)
        res = torch.cat([pts_a[~self.b.contains(pts_a)], pts_b[self.a.contains(pts_b)]], dim=0)
        return res[torch.randperm(len(res))[:n]]

class Complement(Shape):
    def __init__(self, shape): self.shape = shape
    def contains(self, p): return ~self.shape.contains(p)
    def bounding_box(self): return [(-1e6, 1e6)] * 2 # Semi-infinite

class CSGSampler:
    def __init__(self, shape: Shape, spatial_dims: int = 2):
        self.shape = shape
        self.spatial_dims = spatial_dims
        
    def sample_interior(self, n: int, method="sobol") -> torch.Tensor:
        bbox = self.shape.bounding_box()
        collected = []
        count = 0
        while count < n:
            # Simple rejection sampling
            batch_size = max(n * 2, 1000)
            pts = torch.zeros(batch_size, self.spatial_dims)
            for i, (lo, hi) in enumerate(bbox):
                pts[:, i] = lo + torch.rand(batch_size) * (hi - lo)
            
            mask = self.shape.contains(pts)
            valid = pts[mask]
            collected.append(valid)
            count += valid.shape[0]
            
        return torch.cat(collected, dim=0)[:n]
        
    def sample_boundary(self, n: int) -> torch.Tensor:
        return self.shape.sample_boundary(n)
        
    def sample_with_time(self, n: int, t_range=(0, 1), method="sobol") -> torch.Tensor:
        pts = self.sample_interior(n, method=method)
        t = t_range[0] + torch.rand(n, 1) * (t_range[1] - t_range[0])
        return torch.cat([pts, t], dim=-1)
        
    def estimate_volume(self, n_total=100000) -> float:
        bbox = self.shape.bounding_box()
        bbox_vol = 1.0
        pts = torch.zeros(n_total, self.spatial_dims)
        for i, (lo, hi) in enumerate(bbox):
            pts[:, i] = lo + torch.rand(n_total) * (hi - lo)
            bbox_vol *= (hi - lo)
            
        mask = self.shape.contains(pts)
        return bbox_vol * (mask.float().mean().item())

class CSGDomain(Domain):
    def __init__(self, shape: Shape, spatial_dims: int, t_range=(0, 1), resolution=None):
        bbox = shape.bounding_box()
        # Ensure bbox matches spatial_dims
        bbox = bbox[:spatial_dims]
        super().__init__(spatial_dims=spatial_dims, bounds=tuple(bbox), resolution=resolution)
        self.shape = shape
        self.t_range = t_range
        
    def get_sampler(self) -> CSGSampler:
        return CSGSampler(self.shape, self.spatial_dims)
        
    def to_collocation_points(self, n: int, has_time=True) -> torch.Tensor:
        sampler = self.get_sampler()
        if has_time:
            return sampler.sample_with_time(n, t_range=self.t_range)
        return sampler.sample_interior(n)

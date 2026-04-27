import torch
import torch.nn as nn

class DistanceFunction:
    def __call__(self, coords: torch.Tensor) -> torch.Tensor: # coords: (N, D)
        """Evaluate the distance function to return a (N, 1) tensor."""

class BoxDistance(DistanceFunction):
    def __init__(self, bounds: list):
        """
        bounds: [(x0,x1), (y0,y1), ...]
        """
        self.bounds = bounds

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (N, D)
        d = torch.ones(coords.shape[0], 1, device=coords.device)
        for i, (low, high) in enumerate(self.bounds):
            xi = coords[:, i:i+1]
            # dist = (xi - low) * (high - xi)
            # Normalizing to avoid very large values if bounds are large, 
            # though the prompt suggests x*(1-x) for [0,1].
            d = d * (xi - low) * (high - xi)
        return d

class HardConstraintWrapper(torch.nn.Module):
    def __init__(self, model, distance_fn, particular_solution=None):
        super().__init__()
        self.model = model
        self.distance_fn = distance_fn
        self.particular_solution = particular_solution
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor: # coords: (N, D)
        """Apply the hard constraint transformation: u = D*NN + G."""
        D = self.distance_fn(coords)
        u_raw = self.model(coords)
        
        # Handle dict output (multi-field)
        if isinstance(u_raw, dict):
            out = {}
            for field, val in u_raw.items():
                g = 0.0
                if self.particular_solution:
                    if callable(self.particular_solution):
                        # If PS is a single callable returning a dict or tensor
                        ps_val = self.particular_solution(coords)
                        if isinstance(ps_val, dict):
                            g = ps_val.get(field, 0.0)
                        else:
                            g = ps_val # assume it's for 'u' or the only field
                    elif isinstance(self.particular_solution, dict):
                        ps_fn = self.particular_solution.get(field)
                        if ps_fn:
                            g = ps_fn(coords)
                out[field] = D * val + g
            return out
        else:
            # Single field
            g = self.particular_solution(coords) if self.particular_solution else 0.0
            return D * u_raw + g

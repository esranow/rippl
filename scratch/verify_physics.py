import torch
import math
import sys
from rippl.physics.equation import Equation
from rippl.physics.operators import TimeDerivative, Laplacian
from rippl.physics_blocks.residual import HybridWaveResidualBlock

def analytic_solution(x, t):
    return torch.sin(math.pi * x) * torch.cos(math.pi * t)

def verify_physics():
    # Domain: x in [0,1], t in [0,1]
    N = 20
    x = torch.linspace(0, 1, N)
    t = torch.linspace(0, 1, N)
    X, T = torch.meshgrid(x, t, indexing='ij')
    inputs = torch.stack([X, T], dim=-1).requires_grad_(True)
    
    # u = sin(pi*x)cos(pi*t)
    u = analytic_solution(inputs[..., 0:1], inputs[..., 1:2])
    
    # Equation: u_tt - u_xx = 0
    eq = Equation([(1.0, TimeDerivative(order=2)), (-1.0, Laplacian())])
    
    # Residual Block
    block = HybridWaveResidualBlock(use_correction=False)
    
    # Compute residual
    res = block.residual(u, eq, inputs)
    max_res = torch.abs(res).max().item()
    
    print(f"Max residual: {max_res}")
    if max_res > 1e-2:
        print("PHYSICS BROKEN")
        return False
    return True

if __name__ == "__main__":
    if not verify_physics():
        sys.exit(1)
    print("PHYSICS OK")

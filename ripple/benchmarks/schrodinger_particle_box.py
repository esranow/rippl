import torch
import torch.optim as optim
import math
from ripple.core import System, Domain, Constraint
from ripple.core.experiment import Experiment
from ripple.physics.schrodinger import SchrodingerSystem

def main():
    # 1. Setup Schrodinger System
    # V=0 inside [0,1]
    potential_fn = lambda c: torch.zeros_like(c[:, 0:1])
    hbar = 1.0
    mass = 1.0
    sch = SchrodingerSystem(potential_fn=potential_fn, hbar=hbar, mass=mass)
    eq_sys = sch.build_equation_system()
    
    # 2. Domain x ∈ [0, 1], t ∈ [0, 1]
    domain = Domain(spatial_dims=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
    
    # 3. Constraints
    # Ground state analytic: sin(πx) at t=0
    n_pts = 100
    x_grid = torch.linspace(0, 1, n_pts).view(-1, 1)
    zeros = torch.zeros_like(x_grid)
    
    # BCs: ψ(0,t) = ψ(1,t) = 0
    t_grid = torch.linspace(0, 1, n_pts).view(-1, 1)
    ones = torch.ones_like(t_grid)
    t_zeros = torch.zeros_like(t_grid)
    
    constraints = [
        # IC (t=0)
        Constraint(field="psi_real", coords=torch.cat([x_grid, zeros], dim=-1), 
                   value=lambda c: torch.sin(math.pi * c[..., 0:1]), type="dirichlet"),
        Constraint(field="psi_imag", coords=torch.cat([x_grid, zeros], dim=-1), 
                   value=0.0, type="dirichlet"),
        # Left BC (x=0)
        Constraint(field="psi_real", coords=torch.cat([t_zeros, t_grid], dim=-1), value=0.0, type="dirichlet"),
        Constraint(field="psi_imag", coords=torch.cat([t_zeros, t_grid], dim=-1), value=0.0, type="dirichlet"),
        # Right BC (x=1)
        Constraint(field="psi_real", coords=torch.cat([ones, t_grid], dim=-1), value=0.0, type="dirichlet"),
        Constraint(field="psi_imag", coords=torch.cat([ones, t_grid], dim=-1), value=0.0, type="dirichlet")
    ]
    
    system = System(
        equation=eq_sys,
        domain=domain,
        constraints=constraints,
        fields=sch.fields()
    )
    
    # 4. Model & Optimizer
    model = sch.suggested_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 5. Experiment
    experiment = Experiment(
        system=system,
        model=model,
        opt=optimizer
    )
    
    print("Schrödinger Particle in a Box Benchmark Initialized.")
    print("Ground state energy: π²/2 ≈ 4.93")
    print("Run experiment.train(coords) to start training.")

if __name__ == "__main__":
    main()

import torch
import torch.optim as optim
from ripple.core import System, Domain, Constraint
from ripple.core.experiment import Experiment
from ripple.physics.navier_stokes import NavierStokesSystem

def main():
    # 1. Setup NS System
    ns = NavierStokesSystem(rho=1.0, mu=0.01)
    eq_sys = ns.build_equation_system()
    
    # 2. Domain [0, 1] x [0, 1]
    domain = Domain(spatial_dims=2, bounds=[(0.0, 1.0), (0.0, 1.0)])
    
    # 3. Constraints
    # u=1 at y=1 (lid), v=0
    # u=v=0 on all other walls
    # p=0 at corner (0,0)
    
    # Grid for BCs
    n_bc = 100
    x_wall = torch.linspace(0, 1, n_bc).view(-1, 1)
    zeros = torch.zeros_like(x_wall)
    ones = torch.ones_like(x_wall)
    
    constraints = [
        # Lid (y=1)
        Constraint(field="u", coords=torch.cat([x_wall, ones], dim=-1), value=1.0, type="dirichlet"),
        Constraint(field="v", coords=torch.cat([x_wall, ones], dim=-1), value=0.0, type="dirichlet"),
        # Bottom (y=0)
        Constraint(field="u", coords=torch.cat([x_wall, zeros], dim=-1), value=0.0, type="dirichlet"),
        Constraint(field="v", coords=torch.cat([x_wall, zeros], dim=-1), value=0.0, type="dirichlet"),
        # Left (x=0)
        Constraint(field="u", coords=torch.cat([zeros, x_wall], dim=-1), value=0.0, type="dirichlet"),
        Constraint(field="v", coords=torch.cat([zeros, x_wall], dim=-1), value=0.0, type="dirichlet"),
        # Right (x=1)
        Constraint(field="u", coords=torch.cat([ones, x_wall], dim=-1), value=0.0, type="dirichlet"),
        Constraint(field="v", coords=torch.cat([ones, x_wall], dim=-1), value=0.0, type="dirichlet"),
        # Pressure reference p(0,0)=0
        Constraint(field="p", coords=torch.tensor([[0.0, 0.0]]), value=0.0, type="dirichlet")
    ]
    
    system = System(
        equation=eq_sys,
        domain=domain,
        constraints=constraints,
        fields=ns.fields()
    )
    
    # 4. Model & Optimizer
    model = ns.suggested_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 5. Experiment with Hard BCs and Adaptive Sampling
    # Note: Hard BCs for lid-driven cavity require careful distance functions.
    # We'll use the BoxDistance from previous phase.
    experiment = Experiment(
        system=system,
        model=model,
        opt=optimizer,
        use_hard_constraints=True,
        adaptive_collocation=True
    )
    
    print("Navier-Stokes Lid-Driven Cavity Benchmark Initialized.")
    print("Run experiment.train(coords) to start training.")

if __name__ == "__main__":
    main()

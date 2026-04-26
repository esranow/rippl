import torch
import torch.optim as optim
import math
from ripple.core import System, Domain, Constraint
from ripple.core.experiment import Experiment
from ripple.physics.equation import Equation
from ripple.physics.operators import TimeDerivative, Laplacian, BurgersAdvection
from ripple.models.multi_field_mlp import MultiFieldMLP

def main():
    # 1. Setup Burgers Equation: u_t + u*u_x = ν*u_xx
    nu = 0.01 / math.pi
    eq = Equation([
        (1.0, TimeDerivative(field="u")),
        (1.0, BurgersAdvection(field="u", spatial_dim=0)),
        (-nu, Laplacian(field="u"))
    ])
    
    # 2. Domain x ∈ [-1, 1], t ∈ [0, 1]
    # Note: Domain current contract says spatial_dims matches bounds.
    # If we want t, we might need to handle it.
    # For now, let's assume Domain(2) with [( -1, 1), (0, 1)] where last is t.
    domain = Domain(spatial_dims=2, bounds=[(-1.0, 1.0), (0.0, 1.0)])
    
    # 3. Constraints
    # IC: u(x,0) = -sin(πx)
    # BC: u(-1,t) = u(1,t) = 0
    
    n_pts = 200
    x_ic = torch.linspace(-1, 1, n_pts).view(-1, 1)
    zeros = torch.zeros_like(x_ic)
    
    t_bc = torch.linspace(0, 1, n_pts).view(-1, 1)
    ones = torch.ones_like(t_bc)
    neg_ones = -torch.ones_like(t_bc)
    
    constraints = [
        # IC
        Constraint(field="u", coords=torch.cat([x_ic, zeros], dim=-1), 
                   value=lambda c: -torch.sin(math.pi * c[..., 0:1]), type="dirichlet"),
        # BCs
        Constraint(field="u", coords=torch.cat([neg_ones, t_bc], dim=-1), value=0.0, type="dirichlet"),
        Constraint(field="u", coords=torch.cat([ones, t_bc], dim=-1), value=0.0, type="dirichlet")
    ]
    
    system = System(
        equation=eq,
        domain=domain,
        constraints=constraints,
        fields=["u"]
    )
    
    # 4. Model & Optimizer
    model = MultiFieldMLP(fields=["u"], hidden=64, layers=6)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 5. Experiment
    experiment = Experiment(
        system=system,
        model=model,
        opt=optimizer,
        use_hard_constraints=True,
        adaptive_collocation=True
    )
    
    print("1D Burgers Equation Benchmark Initialized.")
    print("Run experiment.train(coords) to start training.")

if __name__ == "__main__":
    main()

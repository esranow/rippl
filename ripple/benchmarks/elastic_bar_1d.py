import torch
import torch.optim as optim
from ripple.core import System, Domain, Constraint
from ripple.core.experiment import Experiment
from ripple.physics.equation import Equation
from ripple.physics.operators import Laplacian
from ripple.models.multi_field_mlp import MultiFieldMLP

def main():
    # 1. Setup 1D Elasticity: E*u_xx + f = 0
    # Equation residual: Laplacian(u) + f/E = 0
    E = 1.0
    f = 1.0
    
    # In 1D, ElasticEquilibrium simplified is just E*u_xx + f = 0
    # We can use Laplacian directly.
    eq = Equation([
        (E, Laplacian(field="ux"))
    ], forcing=lambda p: -torch.full_like(p["inputs"][:, :1], f))
    
    # 2. Domain x ∈ [0, 1]
    domain = Domain(spatial_dims=1, bounds=[(0.0, 1.0)])
    
    # 3. Constraints
    constraints = [
        Constraint(field="ux", coords=torch.tensor([[0.0]]), value=0.0, type="dirichlet"),
        Constraint(field="ux", coords=torch.tensor([[1.0]]), value=0.5, type="dirichlet")
    ]
    
    system = System(
        equation=eq,
        domain=domain,
        constraints=constraints,
        fields=["ux"]
    )
    
    # 4. Model & Optimizer
    model = MultiFieldMLP(fields=["ux"], hidden=64, layers=5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 5. Experiment
    experiment = Experiment(
        system=system,
        model=model,
        opt=optimizer
    )
    
    print("1D Elastic Bar Benchmark Initialized.")
    print("Run experiment.train(coords) to start training.")

if __name__ == "__main__":
    main()

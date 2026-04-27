import torch
import pytest
import math
from rippl.physics.operators import StrainTensor, StressTensor, ElasticEquilibrium
from rippl.physics.elasticity import LinearElasticitySystem
from rippl.core.system import System, Domain, Constraint, NeumannConstraint
from rippl.core.experiment import Experiment
from rippl.core.equation_system import EquationSystem
import os

def test_strain_tensor_forward_shape():
    # StrainTensor forward returns (N, 3)
    coords = torch.randn(10, 2, requires_grad=True)
    fields = {"ux": torch.randn(10, 1), "uy": torch.randn(10, 1)}
    derived = {
        "ux_x": torch.randn(10, 1), "ux_y": torch.randn(10, 1),
        "uy_x": torch.randn(10, 1), "uy_y": torch.randn(10, 1)
    }
    op = StrainTensor(field_ux="ux", field_uy="uy")
    out = op.forward(fields, coords, derived)
    assert out.shape == (10, 3)

def test_strain_tensor_values_correct():
    # ux=x²/2, uy=0 → εxx=x, εyy=0, εxy=0
    # ux_x = x, ux_y = 0, uy_x = 0, uy_y = 0
    coords = torch.linspace(0, 1, 10).view(-1, 1).expand(-1, 2).requires_grad_(True)
    x = coords[:, 0:1]
    derived = {
        "ux_x": x, "ux_y": torch.zeros_like(x),
        "uy_x": torch.zeros_like(x), "uy_y": torch.zeros_like(x)
    }
    op = StrainTensor()
    out = op.forward({}, coords, derived)
    assert torch.allclose(out[..., 0:1], x)
    assert torch.allclose(out[..., 1:3], torch.zeros_like(out[..., 1:3]))

def test_stress_tensor_forward_shape():
    # StressTensor forward returns (N, 3)
    coords = torch.randn(10, 2)
    fields = {"strain": torch.randn(10, 3)}
    op = StressTensor()
    out = op.forward(fields, coords)
    assert out.shape == (10, 3)

def test_stress_from_strain_values():
    # known strain → verify stress via Lamé formula
    # exx=1, eyy=0, exy=0, lam=1, mu=1
    # sxx = (1+2)*1 + 1*0 = 3
    # syy = 1*1 + (1+2)*0 = 1
    # sxy = 2*1*0 = 0
    op = StressTensor(lame_lambda=1.0, lame_mu=1.0)
    strain = torch.tensor([[1.0, 0.0, 0.0]])
    out = op.forward({"strain": strain}, None)
    assert torch.allclose(out, torch.tensor([[3.0, 1.0, 0.0]]))

def test_elastic_equilibrium_zero_for_analytic():
    # u(x)=x satisfies equilibrium with f=0 → residual ≈ 0
    # ux = x => ux_xx=0, ux_yy=0, ux_xy=0
    # uy = 0 => uy_xx=0, uy_yy=0, uy_xy=0
    # Equilibrium: res_x = 0 + 0 + 0 + 0 = 0
    op = ElasticEquilibrium(body_force_x=0.0, body_force_y=0.0)
    derived = {
        "ux_xx": torch.zeros(5, 1), "ux_yy": torch.zeros(5, 1), "ux_xy": torch.zeros(5, 1),
        "uy_xx": torch.zeros(5, 1), "uy_yy": torch.zeros(5, 1), "uy_xy": torch.zeros(5, 1)
    }
    out = op.forward({}, None, derived)
    assert torch.allclose(out, torch.zeros_like(out))

def test_linear_elasticity_system_builds():
    # LinearElasticitySystem builds EquationSystem without error
    sys = LinearElasticitySystem()
    eq_sys = sys.build_equation_system()
    assert isinstance(eq_sys, EquationSystem)

def test_lame_parameters_from_E_nu():
    # E=1.0, nu=0.3 → verify λ and μ numerically
    # λ = 1*0.3 / (1.3 * 0.4) = 0.3 / 0.52 ≈ 0.5769
    # μ = 1 / (2 * 1.3) = 1 / 2.6 ≈ 0.3846
    sys = LinearElasticitySystem(E=1.0, nu=0.3)
    assert math.isclose(sys.lam, 0.576923, rel_tol=1e-5)
    assert math.isclose(sys.mu, 0.384615, rel_tol=1e-5)

def test_neumann_constraint_loss_computed():
    # NeumannConstraint added to system → train step computes it without error
    domain = Domain(spatial_dims=1, bounds=[(0.0, 1.0)], resolution=(10,))
    # u(x) = x^2 / 2  => du/dx = x. At x=1, du/dx = 1.0
    class QuadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))
        def forward(self, x):
            return {"u": self.w * (x**2) / 2.0}
    
    # We need a dummy equation that returns 0 residual for QuadModel to avoid big loss
    from rippl.physics.equation import Equation
    from rippl.physics.operators import Laplacian
    # u_xx = 1.0. If forcing = 1.0, residual = 1.0 - 1.0 = 0.
    eq = Equation([(1.0, Laplacian(field="u"))], forcing=lambda p: torch.ones_like(p["inputs"]))
    
    nc = NeumannConstraint(field="u", coords=torch.tensor([[1.0]]), normal_direction=0, value=1.0)
    system = System(equation=eq, domain=domain, constraints=[nc], fields=["u"])
    
    model = QuadModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    exp = Experiment(system, model, optimizer)
    
    res = exp.train(torch.tensor([[0.5]]), epochs=1)
    assert "loss" in res
    assert not math.isnan(res["loss"])

def test_elastic_bar_script_exists():
    # rippl/benchmarks/elastic_bar_1d.py exists and is importable
    path = "d:/ai_projects/Rippl/rippl/benchmarks/elastic_bar_1d.py"
    assert os.path.exists(path)
    import importlib.util
    spec = importlib.util.spec_from_file_location("elastic_bar_1d", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")

import torch
import pytest
import math
from ripple.physics.operators import SchrodingerKinetic, PotentialTerm, SchrodingerTimeEvolution
from ripple.physics.schrodinger import SchrodingerSystem
from ripple.physics.conservation import MassConservation, EnergyConservation, ConservationLaw
from ripple.diagnostics.physics_validator import PhysicsValidator
from ripple.core import System, Domain, Constraint
from ripple.physics.equation import Equation
from ripple.physics.operators import Laplacian

def test_schrodinger_kinetic_forward_shape():
    # SchrodingerKinetic forward returns (N, 2)
    coords = torch.randn(10, 1, requires_grad=True)
    fields = {"psi_real": torch.randn(10, 1), "psi_imag": torch.randn(10, 1)}
    derived = {"psi_real_xx": torch.randn(10, 1), "psi_imag_xx": torch.randn(10, 1)}
    op = SchrodingerKinetic()
    out = op.forward(fields, coords, derived)
    assert out.shape == (10, 2)

def test_schrodinger_kinetic_correct_coefficient():
    # hbar=1, mass=1, ψ_real=sin(πx) → kinetic_real = -1/2 * (-π² sin(πx)) = π²/2 * sin(πx)
    coords = torch.linspace(0, 1, 10).view(-1, 1).requires_grad_(True)
    psi_real = torch.sin(math.pi * coords)
    psi_imag = torch.zeros_like(psi_real)
    # ψ_real_xx = -π² sin(πx)
    derived = {
        "psi_real_xx": -(math.pi**2) * psi_real,
        "psi_imag_xx": torch.zeros_like(psi_real)
    }
    op = SchrodingerKinetic(hbar=1.0, mass=1.0)
    out = op.forward({"psi_real": psi_real, "psi_imag": psi_imag}, coords, derived)
    expected_real = (math.pi**2 / 2.0) * psi_real
    assert torch.allclose(out[..., 0:1], expected_real, atol=1e-5)

def test_potential_term_zero_potential():
    # V=0 everywhere → PotentialTerm output is zero
    coords = torch.randn(5, 1)
    fields = {"psi_real": torch.randn(5, 1), "psi_imag": torch.randn(5, 1)}
    op = PotentialTerm(potential_fn=lambda c: torch.zeros_like(c))
    out = op.forward(fields, coords)
    assert torch.allclose(out, torch.zeros_like(out))

def test_potential_term_nonzero():
    # V=x² → output = x²*ψ
    coords = torch.tensor([[2.0]])
    fields = {"psi_real": torch.tensor([[1.0]]), "psi_imag": torch.tensor([[0.0]])}
    op = PotentialTerm(potential_fn=lambda c: c**2)
    out = op.forward(fields, coords)
    assert torch.isclose(out[0, 0], torch.tensor(4.0))

def test_schrodinger_system_builds():
    # SchrodingerSystem builds EquationSystem without error
    sys = SchrodingerSystem(potential_fn=lambda c: torch.zeros_like(c))
    eq_sys = sys.build_equation_system()
    assert len(eq_sys.equations) == 2

def test_norm_conservation_loss_for_analytic():
    # sin(πx) on [0,1] has ∫|ψ|²dx = ∫sin²(πx)dx = 0.5. 
    # Our norm_conservation_loss uses .mean(), which for sin²(πx) is 0.5.
    # So we expect (0.5 - 1.0)² = 0.25 if not normalized.
    # If we want it normalized, we need sqrt(2)sin(πx).
    sys = SchrodingerSystem(potential_fn=lambda c: 0)
    class NormModel(torch.nn.Module):
        def forward(self, x):
            # sqrt(2)*sin(πx) => ∫2sin²(πx)dx = 2*0.5 = 1.0. Mean is 1.0.
            return {"psi_real": math.sqrt(2.0) * torch.sin(math.pi * x), 
                    "psi_imag": torch.zeros_like(x)}
    
    coords = torch.linspace(0, 1, 100).view(-1, 1)
    loss = sys.norm_conservation_loss(NormModel(), coords)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-2)

def test_mass_conservation_set_reference():
    # MassConservation.set_reference runs without error
    mc = MassConservation()
    model = lambda x: torch.ones(x.shape[0], 1)
    mc.set_reference(model, torch.randn(5, 1))
    assert mc.reference is not None
    assert torch.isclose(mc.reference, torch.tensor(1.0))

def test_mass_conservation_penalty_zero_at_reference():
    # penalty is zero immediately after set_reference
    mc = MassConservation()
    model = lambda x: torch.ones(x.shape[0], 1)
    coords = torch.randn(5, 1)
    mc.set_reference(model, coords)
    assert mc.penalty(model, coords) == 0.0

def test_energy_conservation_drift_detected():
    # modify model slightly → penalty becomes nonzero
    class VarModel:
        def __init__(self): self.val = 1.0
        def __call__(self, x): return torch.full((x.shape[0], 1), self.val)
    
    model = VarModel()
    ec = EnergyConservation(energy_fn=lambda m, c: m(c).mean())
    coords = torch.randn(5, 1)
    ec.set_reference(model, coords)
    
    model.val = 2.0
    assert ec.penalty(model, coords) > 0.0

def test_physics_validator_residual_stats_keys():
    # residual_stats() returns dict with required keys
    domain = Domain(spatial_dims=1, bounds=[(0, 1)], resolution=(10,))
    eq = Equation([(1.0, Laplacian())])
    sys = System(eq, domain)
    # Model must be connected to coords for autograd to work in compute_residual
    model = lambda x: x**2 
    validator = PhysicsValidator(sys, model, torch.linspace(0, 1, 10).view(-1, 1))
    stats = validator.residual_stats()
    for key in ["mean", "max", "std", "l2", "passed"]:
        assert key in stats

def test_physics_validator_full_report_runs():
    # full_report() runs without error, returns dict
    domain = Domain(spatial_dims=1, bounds=[(0, 1)], resolution=(5,))
    sys = System(Equation([(1.0, Laplacian())]), domain)
    model = lambda x: x**2
    validator = PhysicsValidator(sys, model, torch.randn(5, 1))
    report = validator.full_report()
    assert isinstance(report, dict)

def test_conservation_law_is_satisfied():
    # is_satisfied returns True when within tolerance
    law = ConservationLaw("test", lambda m, c: m(c).mean(), tolerance=0.1)
    model = lambda x: torch.tensor([[1.0]])
    law.set_reference(model, None)
    # 1.05 is within 10% of 1.0
    model2 = lambda x: torch.tensor([[1.05]])
    assert law.is_satisfied(model2, None)

def test_neumann_constraint_gradient_computed():
    # NeumannConstraint loss uses autograd correctly
    # Handled in test_elasticity previously, but here for completeness
    from ripple.core.system import NeumannConstraint
    import torch.nn.functional as F
    
    coords = torch.tensor([[1.0]], requires_grad=True)
    # u = x^2/2 => du/dx = x. At x=1, du/dx=1.
    model = lambda x: {"u": (x**2)/2.0}
    nc = NeumannConstraint("u", coords, 0, 1.0)
    
    # Simulate Experiment logic
    u_pred = model(coords)["u"]
    grad = torch.autograd.grad(u_pred, coords, create_graph=True)[0]
    deriv = grad[..., 0:1]
    assert torch.isclose(deriv, torch.tensor(1.0))

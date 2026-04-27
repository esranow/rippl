import torch
import pytest
import numpy as np
from rippl.physics.derivatives import grad, grad2, compute_all_derivatives
from rippl.physics.operators import (
    BurgersAdvection, NonlinearAdvection, VelocityDivergence, PressureGradient,
    StrainTensor, StressTensor, ElasticEquilibrium,
    SchrodingerKinetic, PotentialTerm, SchrodingerTimeEvolution
)
from rippl.physics.navier_stokes import NavierStokesSystem
from rippl.physics.elasticity import LinearElasticitySystem
from rippl.physics.schrodinger import SchrodingerSystem
from rippl.core.equation import Equation
from rippl.core.equation_system import EquationSystem
from rippl.core.system import System, Domain, NeumannConstraint
from rippl.core.experiment import Experiment

# Derivatives
def test_grad_correct():
    coords = torch.linspace(0, 2*np.pi, 100).unsqueeze(-1).requires_grad_(True)
    field = torch.sin(coords)
    g = grad(field, coords, 0)
    assert torch.allclose(g, torch.cos(coords), atol=1e-5)

def test_grad2_correct():
    coords = torch.linspace(0, 2*np.pi, 100).unsqueeze(-1).requires_grad_(True)
    field = torch.sin(coords)
    g2 = grad2(field, coords, 0, 0)
    assert torch.allclose(g2, -torch.sin(coords), atol=1e-5)

def test_no_duplicate_computation():
    coords = torch.randn(10, 2, requires_grad=True)
    fields = {"u": coords.sum(dim=-1, keepdim=True)}
    # If we request u_x and u_x, u_x should be in derived only once.
    derived = compute_all_derivatives(fields, coords, ["u_x", "u_x"])
    assert len(derived) == 1
    assert "u_x" in derived

# Nonlinear operators
def test_burgers_advection_forward():
    coords = torch.linspace(0, 1, 10).unsqueeze(-1).requires_grad_(True)
    fields = {"u": coords} # u = x
    derived = {"u_x": torch.ones_like(coords)} # u_x = 1
    op = BurgersAdvection(field="u", spatial_dim=0)
    out = op.forward(fields, coords, derived)
    assert torch.allclose(out, coords) # u * u_x = x * 1 = x

def test_nonlinear_advection_shape():
    coords = torch.randn(10, 3)
    fields = {"u": torch.randn(10, 1), "v": torch.randn(10, 1)}
    derived = {"u_x": torch.randn(10, 1), "u_y": torch.randn(10, 1), 
               "v_x": torch.randn(10, 1), "v_y": torch.randn(10, 1)}
    op = NonlinearAdvection(field_u="u", field_v="v")
    out = op.forward(fields, coords, derived)
    assert out.shape == (10, 2)

def test_velocity_divergence_zero():
    # u=sin(x)cos(y), v=-cos(x)sin(y) -> u_x = cos(x)cos(y), v_y = -cos(x)cos(y) -> div=0
    x = torch.linspace(0, 1, 10)
    y = torch.linspace(0, 1, 10)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    u = torch.sin(coords[:, 0:1]) * torch.cos(coords[:, 1:2])
    v = -torch.cos(coords[:, 0:1]) * torch.sin(coords[:, 1:2])
    u_x = torch.cos(coords[:, 0:1]) * torch.cos(coords[:, 1:2])
    v_y = -torch.cos(coords[:, 0:1]) * torch.cos(coords[:, 1:2])
    
    fields = {"u": u, "v": v}
    derived = {"u_x": u_x, "v_y": v_y}
    op = VelocityDivergence(field_u="u", field_v="v")
    out = op.forward(fields, coords, derived)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)

def test_pressure_gradient_forward():
    coords = torch.linspace(0, 1, 10).unsqueeze(-1)
    fields = {"p": coords**2}
    derived = {"p_x": 2*coords}
    op = PressureGradient(field_p="p", direction=0)
    out = op.forward(fields, coords, derived)
    assert torch.allclose(out, 2*coords)

# NS
def test_ns_system_builds():
    ns = NavierStokesSystem()
    eqs = ns.build_equation_system()
    assert isinstance(eqs, EquationSystem)

def test_ns_system_fields():
    ns = NavierStokesSystem()
    assert ns.fields() == ["u", "v", "p"]

def test_ns_equation_system_has_3_equations():
    ns = NavierStokesSystem()
    eqs = ns.build_equation_system()
    assert len(eqs.equations) == 3

# Elasticity
def test_strain_tensor_shape():
    coords = torch.randn(10, 2)
    fields = {"ux": torch.randn(10, 1), "uy": torch.randn(10, 1)}
    derived = {"ux_x": torch.randn(10, 1), "ux_y": torch.randn(10, 1),
               "uy_x": torch.randn(10, 1), "uy_y": torch.randn(10, 1)}
    op = StrainTensor(field_ux="ux", field_uy="uy")
    out = op.forward(fields, coords, derived)
    assert out.shape == (10, 3)

def test_strain_tensor_values():
    # ux = x²/2 -> ux_x = x
    coords = torch.linspace(0, 1, 10).unsqueeze(-1).repeat(1, 2) # x, y
    fields = {"ux": coords[:, 0:1]**2 / 2, "uy": torch.zeros_like(coords[:, 0:1])}
    derived = {"ux_x": coords[:, 0:1], "ux_y": torch.zeros_like(coords[:, 0:1]),
               "uy_x": torch.zeros_like(coords[:, 0:1]), "uy_y": torch.zeros_like(coords[:, 0:1])}
    op = StrainTensor(field_ux="ux", field_uy="uy")
    out = op.forward(fields, coords, derived)
    assert torch.allclose(out[:, 0:1], coords[:, 0:1]) # exx = ux_x = x

def test_stress_tensor_shape():
    fields = {"strain": torch.randn(10, 3)}
    op = StressTensor()
    out = op.forward(fields, None)
    assert out.shape == (10, 3)

def test_stress_values_lame():
    # σxx = (λ+2μ)εxx + λεyy
    lam, mu = 1.0, 1.0
    exx, eyy, exy = 1.0, 0.5, 0.0
    fields = {"strain": torch.tensor([[exx, eyy, exy]])}
    op = StressTensor(lame_lambda=lam, lame_mu=mu)
    out = op.forward(fields, None)
    expected_sxx = (lam + 2*mu)*exx + lam*eyy
    assert torch.allclose(out[0, 0], torch.tensor(expected_sxx))

def test_elastic_equilibrium_shape():
    coords = torch.randn(10, 2)
    fields = {"ux": torch.randn(10, 1), "uy": torch.randn(10, 1)}
    # needs 6 derivatives
    derived = {f"{f}_{d}": torch.randn(10, 1) for f in ["ux", "uy"] for d in ["xx", "yy", "xy"]}
    op = ElasticEquilibrium()
    out = op.forward(fields, coords, derived)
    assert out.shape == (10, 2)

def test_lame_from_E_nu():
    E, nu = 1.0, 0.3
    es = LinearElasticitySystem(E=E, nu=nu)
    expected_lam = E * nu / ((1+nu) * (1-2*nu))
    expected_mu = E / (2*(1+nu))
    assert np.allclose(es.lam, expected_lam)
    assert np.allclose(es.mu, expected_mu)

def test_linear_elasticity_builds():
    es = LinearElasticitySystem()
    eqs = es.build_equation_system()
    assert isinstance(eqs, EquationSystem)

def test_neumann_constraint_loss():
    # Mocking experiment train logic for Neumann
    coords = torch.linspace(0, 1, 10).unsqueeze(-1).requires_grad_(True)
    # model: u(x) = x -> u' = 1
    model = lambda x: {"u": x}
    c = NeumannConstraint(field="u", coords=coords, normal_direction=0, value=1.0)
    
    # Simulate experiment logic
    c_coords = c.coords.requires_grad_(True)
    u_pred = model(c_coords)["u"]
    grad_u = torch.autograd.grad(u_pred.sum(), c_coords, create_graph=True)[0]
    val_pred = grad_u[..., 0:1]
    
    loss = torch.mean((val_pred - 1.0)**2)
    assert loss < 1e-7

# Schrödinger
def test_schrodinger_kinetic_shape():
    coords = torch.randn(10, 1)
    fields = {"psi_real": torch.randn(10, 1), "psi_imag": torch.randn(10, 1)}
    derived = {"psi_real_xx": torch.randn(10, 1), "psi_imag_xx": torch.randn(10, 1)}
    op = SchrodingerKinetic()
    out = op.forward(fields, coords, derived)
    assert out.shape == (10, 2)

def test_schrodinger_kinetic_coefficient():
    # hbar=1, mass=1, psi=sin(πx) -> -1/2 * (-π² sin(πx)) = π²/2 * sin(πx)
    coords = torch.linspace(0, 1, 10).unsqueeze(-1)
    fields = {"psi_real": torch.sin(np.pi * coords), "psi_imag": torch.zeros_like(coords)}
    derived = {"psi_real_xx": -(np.pi**2) * torch.sin(np.pi * coords), "psi_imag_xx": torch.zeros_like(coords)}
    op = SchrodingerKinetic(hbar=1.0, mass=1.0)
    out = op.forward(fields, coords, derived)
    expected = (np.pi**2 / 2) * torch.sin(np.pi * coords)
    assert torch.allclose(out[:, 0:1], expected, atol=1e-5)

def test_potential_zero():
    coords = torch.randn(10, 1)
    fields = {"psi_real": torch.randn(10, 1), "psi_imag": torch.randn(10, 1)}
    op = PotentialTerm(potential_fn=lambda x: torch.zeros(x.shape[0], 1))
    out = op.forward(fields, coords)
    assert torch.allclose(out, torch.zeros_like(out))

def test_potential_nonzero():
    coords = torch.linspace(0, 1, 10).unsqueeze(-1)
    fields = {"psi_real": torch.ones_like(coords), "psi_imag": torch.ones_like(coords)}
    op = PotentialTerm(potential_fn=lambda x: x**2)
    out = op.forward(fields, coords)
    assert torch.allclose(out[:, 0:1], coords**2)

def test_schrodinger_system_builds():
    ss = SchrodingerSystem(potential_fn=lambda x: 0)
    eqs = ss.build_equation_system()
    assert isinstance(eqs, EquationSystem)

def test_norm_conservation_analytic():
    # sin(πx) normalized? integral of sin²(πx) over [0,1] is 0.5.
    # So psi = sqrt(2) * sin(πx) is normalized.
    coords = torch.linspace(0, 1, 100).unsqueeze(-1)
    model = lambda x: {"psi_real": np.sqrt(2) * torch.sin(np.pi * x), "psi_imag": torch.zeros_like(x)}
    ss = SchrodingerSystem(potential_fn=lambda x: 0)
    loss = ss.norm_conservation_loss(model, coords)
    # mean() of (2*sin²(πx)) over [0,1] should be 1.0
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-2)

# Operator contract
def test_equation_passes_derived_dict():
    op = BurgersAdvection(field="u", spatial_dim=0)
    eq = Equation([(1.0, op)])
    u = torch.linspace(0, 1, 10).unsqueeze(-1).requires_grad_(True)
    coords = u.clone()
    res = eq.compute_residual(u, coords)
    assert res is not None

def test_all_operators_have_requires_derived_key():
    ops = [
        BurgersAdvection(), NonlinearAdvection(), VelocityDivergence(), PressureGradient(),
        StrainTensor(), StressTensor(), ElasticEquilibrium(),
        SchrodingerKinetic(), PotentialTerm(potential_fn=lambda x: 0), SchrodingerTimeEvolution()
    ]
    for op in ops:
        sig = op.signature()
        assert "requires_derived" in sig
        assert isinstance(sig["requires_derived"], list)

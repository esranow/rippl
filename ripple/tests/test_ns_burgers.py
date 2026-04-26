import torch
import pytest
from ripple.physics.derivatives import grad, grad2, compute_all_derivatives
from ripple.physics.operators import (
    BurgersAdvection, NonlinearAdvection, 
    PressureGradient, VelocityDivergence, Laplacian
)
from ripple.physics.equation import Equation
from ripple.physics.navier_stokes import NavierStokesSystem
from ripple.core.equation_system import EquationSystem

def test_derivatives_grad_correct():
    # grad(sin(x), coords, dim=0) ≈ cos(x)
    x = torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True)
    y = torch.sin(x)
    g = grad(y, x, 0)
    assert torch.allclose(g, torch.cos(x), atol=1e-4)

def test_derivatives_grad2_correct():
    # grad2(sin(x), coords, dim=0, dim=0) ≈ -sin(x)
    x = torch.linspace(0, 1, 100).view(-1, 1).requires_grad_(True)
    y = torch.sin(x)
    g2 = grad2(y, x, 0, 0)
    assert torch.allclose(g2, -torch.sin(x), atol=1e-4)

def test_compute_all_derivatives_no_duplicate_computation():
    # request ["u_x", "u_x"] — verify it runs without error
    coords = torch.randn(10, 1, requires_grad=True)
    fields = {"u": coords * 2.0} # u is connected to coords
    derived = compute_all_derivatives(fields, coords, ["u_x", "u_x"])
    assert "u_x" in derived
    assert len(derived) == 1

def test_burgers_advection_forward():
    # u=x, u_x=1 → BurgersAdvection output = x*1 = x
    coords = torch.linspace(0, 1, 10).view(-1, 1).requires_grad_(True)
    u = coords.clone()
    fields = {"u": u}
    derived = {"u_x": torch.ones_like(u)}
    op = BurgersAdvection(field="u", spatial_dim=0)
    out = op.forward(fields, coords, derived)
    assert torch.allclose(out, coords)

def test_nonlinear_advection_forward_shape():
    # NonlinearAdvection forward returns (N, 2)
    fields = {
        "u": torch.randn(10, 1),
        "v": torch.randn(10, 1)
    }
    coords = torch.randn(10, 2, requires_grad=True)
    derived = {
        "u_x": torch.randn(10, 1), "u_y": torch.randn(10, 1),
        "v_x": torch.randn(10, 1), "v_y": torch.randn(10, 1)
    }
    op = NonlinearAdvection(field_u="u", field_v="v")
    out = op.forward(fields, coords, derived)
    assert out.shape == (10, 2)

def test_pressure_gradient_forward():
    # PressureGradient(direction=0) returns p_x correctly
    fields = {"p": torch.randn(10, 1)}
    coords = torch.randn(10, 2, requires_grad=True)
    p_x = torch.randn(10, 1)
    derived = {"p_x": p_x}
    op = PressureGradient(field_p="p", direction=0)
    out = op.forward(fields, coords, derived)
    assert torch.allclose(out, p_x)

def test_velocity_divergence_zero_for_incompressible():
    # u=sin(x)cos(y), v=-cos(x)sin(y) → divergence ≈ 0
    # u_x = cos(x)cos(y), v_y = -cos(x)cos(y) -> div = 0
    coords = torch.randn(10, 2, requires_grad=True)
    u_x = torch.randn(10, 1)
    v_y = -u_x
    derived = {"u_x": u_x, "v_y": v_y}
    fields = {"u": torch.randn(10, 1), "v": torch.randn(10, 1)}
    op = VelocityDivergence(field_u="u", field_v="v")
    out = op.forward(fields, coords, derived)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

def test_ns_system_builds_equation_system():
    # NavierStokesSystem().build_equation_system() returns EquationSystem with 3 equations
    ns = NavierStokesSystem()
    eq_sys = ns.build_equation_system()
    assert isinstance(eq_sys, EquationSystem)
    assert len(eq_sys.equations) == 3

def test_ns_system_fields():
    # NavierStokesSystem().fields() == ["u", "v", "p"]
    ns = NavierStokesSystem()
    assert ns.fields() == ["u", "v", "p"]

def test_equation_compute_residual_uses_derived():
    # Equation with BurgersAdvection correctly uses precomputed derivatives
    op = BurgersAdvection(field="u", spatial_dim=0)
    eq = Equation([(1.0, op)])
    
    # u = x^2 / 2  => u_x = x
    # res = u * u_x = (x^2 / 2) * x = x^3 / 2
    coords = torch.linspace(0, 1, 10).view(-1, 1).requires_grad_(True)
    u = (coords**2) / 2.0
    
    res = eq.compute_residual(u, coords, spatial_dims=1)
    expected = (coords**3) / 2.0
    assert torch.allclose(res, expected, atol=1e-4)

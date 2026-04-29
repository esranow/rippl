"""
Reaction-diffusion systems.
∂u/∂t = D_u∇²u + f(u,v)
∂v/∂t = D_v∇²v + g(u,v)
"""
import torch
from rippl.physics.operators import Operator, Laplacian, TimeDerivative
from rippl.core.equation import Equation
from rippl.core.equation_system import EquationSystem

class ReactionDiffusionOperator(Operator):
    """
    General reaction-diffusion operator for one species.
    ∂u/∂t - D∇²u - R(u, v, ...) = 0
    
    Args:
        diffusivity: D coefficient
        reaction_fn: callable R(fields, coords) → (N,1)
        field: species field name
        spatial_dims: spatial dimensions
    """
    def __init__(self, diffusivity: float, reaction_fn: callable,
                 field: str = "u", spatial_dims: int = None):
        super().__init__(field=field)
        self.D = diffusivity
        self.R = reaction_fn
        self.spatial_dims = spatial_dims

    def signature(self) -> dict:
        return {
            "inputs": [self.field],
            "output": self.field,
            "order": 2,
            "type": f"reaction_diffusion_{self.field}",
            "requires_derived": [
                f"{self.field}_t",
                f"{self.field}_xx"
            ]
        }

    def forward(self, fields, coords, derived=None):
        u_t = derived[f"{self.field}_t"]
        u_xx = derived[f"{self.field}_xx"]
        R_val = self.R(fields, coords)
        return u_t - self.D * u_xx - R_val


class TuringSystem:
    """
    Turing pattern system (Gray-Scott model).
    ∂u/∂t = D_u∇²u - uv² + F(1-u)
    ∂v/∂t = D_v∇²v + uv² - (F+k)v
    """
    SPOTS  = dict(F=0.035, k=0.065)
    STRIPES = dict(F=0.060, k=0.062)
    CORAL  = dict(F=0.025, k=0.060)

    def __init__(self, D_u: float = 0.16, D_v: float = 0.08,
                 F: float = 0.035, k: float = 0.065):
        self.D_u = D_u
        self.D_v = D_v
        self.F = F
        self.k = k

    def build_equation_system(self) -> 'EquationSystem':
        F, k = self.F, self.k
        def R_u(fields, coords):
            u, v = fields["u"], fields["v"]
            return -u * v**2 + F * (1 - u)
        def R_v(fields, coords):
            u, v = fields["u"], fields["v"]
            return u * v**2 - (F + k) * v
        eq_u = Equation([(1.0, ReactionDiffusionOperator(self.D_u, R_u, "u"))])
        eq_v = Equation([(1.0, ReactionDiffusionOperator(self.D_v, R_v, "v"))])
        return EquationSystem([eq_u, eq_v], weights=[1.0, 1.0])

    def fields(self): return ["u", "v"]


class FitzHughNagumoSystem:
    """
    FitzHugh-Nagumo: simplified neuron model.
    ∂v/∂t = D∇²v + v - v³/3 - w + I_ext
    ∂w/∂t = ε(v + a - bw)
    """
    def __init__(self, D: float = 1.0, epsilon: float = 0.08,
                 a: float = 0.7, b: float = 0.8, I_ext: float = 0.5):
        self.D = D; self.eps = epsilon
        self.a = a; self.b = b; self.I = I_ext

    def build_equation_system(self) -> 'EquationSystem':
        def R_v(f, c):
            v, w = f["v"], f["w"]
            return v - (v**3)/3 - w + self.I
        def R_w(f, c):
            v, w = f["v"], f["w"]
            return self.eps * (v + self.a - self.b * w)
        
        eq_v = Equation([(1.0, ReactionDiffusionOperator(self.D, R_v, "v"))])
        
        class W_Eq(Operator):
            def signature(self):
                return {"inputs": ["v", "w"], "output": "w", "order": 1, "requires_derived": ["w_t"]}
            def forward(self, f, c, d):
                return d["w_t"] - R_w(f, c)
        
        eq_w = Equation([(1.0, W_Eq())])
        return EquationSystem([eq_v, eq_w], weights=[1.0, 1.0])

    def fields(self): return ["v", "w"]


class BrusselatorSystem:
    """
    Brusselator model for chemical oscillations.
    ∂u/∂t = D_u∇²u + A - (B+1)u + u²v
    ∂v/∂t = D_v∇²v + Bu - u²v
    """
    def __init__(self, A: float = 1.0, B: float = 3.0,
                 D_u: float = 1.0, D_v: float = 0.1):
        self.A = A; self.B = B
        self.D_u = D_u; self.D_v = D_v

    def build_equation_system(self) -> 'EquationSystem':
        def R_u(f, c):
            u, v = f["u"], f["v"]
            return self.A - (self.B + 1)*u + (u**2)*v
        def R_v(f, c):
            u, v = f["u"], f["v"]
            return self.B * u - (u**2) * v
        
        eq_u = Equation([(1.0, ReactionDiffusionOperator(self.D_u, R_u, "u"))])
        eq_v = Equation([(1.0, ReactionDiffusionOperator(self.D_v, R_v, "v"))])
        return EquationSystem([eq_u, eq_v], weights=[1.0, 1.0])

    def fields(self): return ["u", "v"]

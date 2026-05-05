import torch
import torch.nn as nn
import rippl as rp
from rippl.physics.delta import DeltaSrc, DeltaEq
from rippl.training.vpinn import VPINN
from rippl.training.pareto import ParetoBal
from rippl.training.adaptive import AdaptWt, TimeCurr
from rippl.models.deeponet import DeepONet
from rippl.migrate.transpiler import migrate, _detect_framework
from rippl.core.equation import Equation
from rippl.physics.operators import TimeDerivative
from rippl.core.system import Domain

def test_delta_eval():
    s = DeltaSrc(torch.tensor([0.5]), mag=1.0, bw=0.1)
    v = s.eval(torch.tensor([[0.5]]))
    assert v > 0

def test_vpinn_loss():
    dom = Domain(1, [(0, 1)], (10,))
    eq = Equation([(1.0, TimeDerivative(field="u"))])
    m = nn.Linear(1, 1)
    v = VPINN(m, eq, dom, p=3, n_el=2)
    l = v.loss()
    assert l >= 0

def test_pareto_compute():
    m = nn.Linear(1, 1)
    b = ParetoBal(n_obj=2)
    ld = {"pde": torch.tensor(1.0, requires_grad=True), "bc": torch.tensor(0.5, requires_grad=True)}
    # Add dummy backward path
    (ld["pde"]*m.weight).sum().backward(retain_graph=True)
    (ld["bc"]*m.weight).sum().backward(retain_graph=True)
    # Actually ParetoBal needs gradients from the model parameters
    # The compute method does: torch.autograd.grad(l, m.parameters())
    # So we need to ensure ls[i] is connected to m.parameters()
    ld = {"pde": (m.weight**2).sum(), "bc": (m.bias**2).sum()}
    l_tot = b.compute(m, ld)
    assert l_tot > 0

def test_adapt_wt():
    a = AdaptWt(10)
    r = torch.randn(10, 1)
    l = a(r)
    assert l.shape == ()

def test_time_curr():
    c = TimeCurr(0, 1, stg=2, pat=2)
    assert c.t_max() == 0.5
    c.step(); c.step()
    assert c.t_max() == 1.0
    x = torch.tensor([[0.5, 0.2], [0.5, 0.8]])
    f = c.filter(x)
    assert len(f) == 2

def test_deeponet_fwd():
    m = DeepONet([10, 5], [2, 5])
    u = torch.randn(1, 10)
    y = torch.randn(1, 2, 2)
    out = m(u, y)
    assert out.shape == (1, 2, 1)

def test_transpiler_detect():
    assert _detect_framework("import deepxde") == "deepxde"
    assert _detect_framework("from modulus import") == "modulus"
    assert migrate("import sciann", "sciann").startswith("# rippl migrated")

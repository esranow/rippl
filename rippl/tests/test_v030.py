import torch
import torch.nn as nn
import rippl as rp
import rippl.nn as rnn
from rippl.physics.distance import AnsatzFactory, PeriodicAnsatz
from rippl.core.system import Domain, IntConstraint, System
from rippl.core.equation import Equation
from rippl.physics.operators import TimeDerivative

def test_periodic_ansatz():
    m = nn.Linear(3, 1) # x, sin(a), cos(a) replaced one dim, so 2 -> 3
    p = AnsatzFactory.periodic_1d(m, p=1.0, d=0)
    assert isinstance(p, PeriodicAnsatz)
    x = torch.tensor([[0.0, 0.5], [1.0, 0.5]], requires_grad=True)
    y = p(x)
    assert torch.allclose(y[0], y[1], atol=1e-5)

def test_integral_constraint():
    c = IntConstraint(fld="u", x=torch.rand(10,1), w=torch.ones(10), tgt=1.0)
    assert c.tgt == 1.0

def test_pointwise_residual():
    eq = Equation([(1.0, TimeDerivative(field="u"))])
    flds = {"u": torch.randn(10, 1, requires_grad=True)}
    x = torch.randn(10, 1, requires_grad=True)
    res = eq.compute_pointwise_residual(flds, x)
    assert res.shape == (10, 1)

def test_run_csl_ntk_bcs():
    dom = Domain(1, [(0, 1)], (10,))
    eq = Equation([(1.0, TimeDerivative(field="u"))])
    m = nn.Sequential(nn.Linear(1, 20), nn.Tanh(), nn.Linear(20, 1))
    res = rp.run(dom, eq, m, eps=2, csl=True, ntk=True, hard_bcs=True)
    assert "model" in res

def test_loaders():
    dom = Domain(1, [(0, 1)], (10,))
    l1 = dom.generate_loader(meth="chebyshev")
    l2 = dom.generate_loader(meth="legendre")
    assert len(next(iter(l1))[0]) > 0
    assert len(next(iter(l2))[0]) > 0

def test_vibe_manifestos():
    import os
    assert os.path.exists("llms.txt")
    assert os.path.exists(".cursorrules")

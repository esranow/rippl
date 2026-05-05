import torch, os, warnings
import torch.nn.functional as F
from typing import Any, Dict, Union

class RipplProRequired(PermissionError): pass

def authenticate(api_key: str):
    global _API_KEY
    if not api_key.startswith("sk_"): raise ValueError("Invalid key")
    _API_KEY = api_key; print("Authenticated")

def compile(m, backend="inductor", mode="max-autotune"):
    try: return torch.compile(m, backend=backend, mode=mode)
    except: return m

def run(dom, eq, m=None, eps=None, **kw):
    from rippl.core.system import System
    if isinstance(dom, System): dom, eq, m = dom.domain, dom.equation, eq
    e = eps or kw.get("eps") or kw.get("epochs") or 10000
    return _run_native(dom, eq, m, int(e), kw)

def _run_native(dom, eq, m, eps, kw):
    from rippl.training.causal import CausalTrainingMixin
    from rippl.training.ntk_weighting import AdaptiveLossBalancer
    from rippl.physics.distance import AnsatzFactory
    from rippl.core.system import NeumannConstraint, IntConstraint
    from rippl.training.pareto import ParetoBal
    from rippl.training.adaptive import AdaptWt, TimeCurr
    import torch.nn.functional as F

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if kw.get("hard_bcs"): m = AnsatzFactory.dirichlet_1d(m) if dom.spatial_dims==1 else AnsatzFactory.dirichlet_2d_box(m)
    m = m.to(dev)
    
    ldr = dom.generate_loader(batch_size=kw.get("bs", 2048) or kw.get("batch_size", 2048), meth=kw.get("colloc", "sobol"))
    opt = torch.optim.Adam(m.parameters(), lr=kw.get("lr", 1e-3))
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=200, factor=0.5, min_lr=1e-6)
    
    bal = AdaptiveLossBalancer(mode=kw.get("ntk_mode", "gradient_norm"), loss_names=["pde","ic","bc"], update_freq=kw.get("ntk_freq", 100)) if (kw.get("ntk") or kw.get("adaptive_loss")) else None
    p_bal = ParetoBal(n_obj=3) if kw.get("pareto") else None
    csl = CausalTrainingMixin() if kw.get("causal") else None
    curr = kw.get("curr")
    a_wt = AdaptWt(kw.get("bs", 2048) or kw.get("batch_size", 2048)) if kw.get("adapt_wt") else None
    hist = []

    for ep in range(eps):
        for b in ldr:
            x = b[0].to(dev).requires_grad_(True)
            if curr: x = curr.filter(x)
            if len(x) == 0: continue
            opt.zero_grad()
            u = m(x); flds = {"u": u}
            
            if csl:
                pr = eq.compute_pointwise_residual(flds, x)
                wts = csl.compute_causal_weights_continuous(x, pr) if kw.get("csl_mode","continuous")=="continuous" else csl.compute_causal_weights_binned(x, pr)
                l_pde = (wts * pr).mean()
            elif a_wt:
                pr = eq.compute_pointwise_residual(flds, x)
                l_pde = a_wt(pr)
            else:
                l_pde = eq.compute_loss(flds, x) if hasattr(eq, 'compute_loss') else eq.compute_residual(u, x).pow(2).mean()

            l_ic, l_bc = torch.tensor(0., device=dev, requires_grad=True), torch.tensor(0., device=dev, requires_grad=True)
            if getattr(eq, '_system', None) or kw.get("system"):
                sys = kw.get("system") or eq._system
                for c in sys.constraints:
                    if isinstance(c, IntConstraint):
                        l_bc = l_bc + c.wt * ((c.w.to(dev) * m(c.x.to(dev)).squeeze()).sum() - c.tgt)**2
                    elif isinstance(c, NeumannConstraint):
                        xc = c.coords.to(dev).requires_grad_(True)
                        gr = torch.autograd.grad(m(xc).sum(), xc, create_graph=True)[0]
                        l_bc = l_bc + F.mse_loss(gr[:, c.normal_direction:c.normal_direction+1], c.value.to(dev))
                    else:
                        v = m(c.coords.to(dev)); t = c.value(c.coords.to(dev)) if callable(c.value) else c.value.to(dev)
                        if c.type == "initial": l_ic = l_ic + F.mse_loss(v, t)
                        else: l_bc = l_bc + F.mse_loss(v, t)

            ld = {"pde": l_pde, "ic": l_ic, "bc": l_bc}
            if p_bal: l_tot = p_bal.compute(m, ld, dev=dev)
            elif bal:
                if ep % kw.get("ntk_freq", 100) == 0: bal.update(m, ld, l_pde + 100*(l_ic+l_bc))
                l_tot = bal.apply(ld)
            else:
                l_tot = l_pde + kw.get("cw", 100.0)*(l_ic+l_bc)

            l_tot.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step(l_tot); hist.append(l_tot.item())
        if curr: curr.step()

    if kw.get("lbfgs", 300) > 0:
        opt_l = torch.optim.LBFGS(m.parameters(), lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
        def clo():
            opt_l.zero_grad(); x2 = next(iter(ldr))[0].to(dev).requires_grad_(True)
            l = eq.compute_loss({"u": m(x2)}, x2) if hasattr(eq, 'compute_loss') else eq.compute_residual(m(x2), x2).pow(2).mean()
            l.backward(); return l
        for _ in range(kw.get("lbfgs", 300)): opt_l.step(clo)

    return {"model": m, "model_state": m.state_dict(), "loss_history": hist, "final_loss": hist[-1] if hist else None}

import torch, numpy as np
from numpy.polynomial.legendre import legval, leggauss
class VPINN:
    def __init__(self, m, eq, dom, p=5, n_el=4):
        self.m, self.eq, self.dom, self.p, self.n_el = m, eq, dom, p, n_el
        pts, wts = leggauss(2*p)
        self.q_pts, self.q_wts = torch.tensor(pts, dtype=torch.float32), torch.tensor(wts, dtype=torch.float32)
        self.t_fns = [torch.tensor(legval(pts, [0]*k + [1]), dtype=torch.float32) for k in range(1, p+1)]

    def loss(self, dev="cpu"):
        l = torch.tensor(0., device=dev)
        b = self.dom.bounds
        sz = (b[0][1] - b[0][0]) / self.n_el
        for i in range(self.n_el):
            lo, hi = b[0][0] + i*sz, b[0][0] + (i+1)*sz
            x = (0.5*(hi+lo) + 0.5*(hi-lo)*self.q_pts).to(dev).unsqueeze(1).requires_grad_(True)
            c = torch.cat([x, torch.full_like(x, 0.5*(b[1][0]+b[1][1]))], dim=1) if len(b)>1 else x
            r = self.eq.compute_pointwise_residual({"u": self.m(c)}, c).squeeze()
            for vk in self.t_fns:
                l += ((0.5*sz * self.q_wts.to(dev) * r * vk.to(dev)).sum())**2
        return l / (self.n_el * len(self.t_fns))

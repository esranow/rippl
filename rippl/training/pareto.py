import torch
class ParetoBal:
    def __init__(self, n_obj=3):
        self.n = n_obj
        self.w = torch.ones(n_obj) / n_obj
    def compute(self, m, l_dict, dev="cpu"):
        ls = list(l_dict.values())
        G = torch.stack([torch.cat([g.flatten() for g in torch.autograd.grad(l, m.parameters(), retain_graph=True, allow_unused=True) if g is not None]) for l in ls])
        GTG = G @ G.T
        w = torch.ones(self.n, device=dev) / self.n
        for i in range(20):
            k = (GTG @ w).argmin()
            ek = torch.zeros(self.n, device=dev); ek[k] = 1.0
            w = (1 - 2.0/(i+2))*w + (2.0/(i+2))*ek
        self.w = w.detach()
        return sum(self.w[i] * ls[i] for i in range(len(ls)))

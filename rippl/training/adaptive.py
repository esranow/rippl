import torch, math
class AdaptWt(torch.nn.Module):
    def __init__(self, n, init=1.0):
        super().__init__()
        self.lw = torch.nn.Parameter(torch.full((n, 1), math.log(init)))
    def forward(self, r): return (torch.exp(self.lw) * r).mean()

class TimeCurr:
    def __init__(self, t0, t1, stg=5, pat=500):
        self.stg = torch.linspace(t0, t1, stg+1)[1:]
        self.pat, self.c_stg, self.ep = pat, 0, 0
    def t_max(self): return self.stg[self.c_stg].item()
    def step(self):
        self.ep += 1
        if self.ep >= self.pat and self.c_stg < len(self.stg)-1:
            self.c_stg += 1; self.ep = 0; return True
        return False
    def filter(self, x): return x[x[:, -1] <= self.t_max()]

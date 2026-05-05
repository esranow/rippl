import torch
class DeltaSrc:
    def __init__(self, loc, mag=1.0, bw=0.05):
        self.x0, self.c, self.h = loc, mag, bw
    def eval(self, x):
        r2 = ((x - self.x0.unsqueeze(0))**2).sum(dim=-1, keepdim=True)
        n = (2*torch.pi*self.h**2)**(x.shape[-1]/2)
        return (self.c / n) * torch.exp(-r2 / (2*self.h**2))

class DeltaEq:
    def __init__(self, eq, src): 
        self.eq, self.src = eq, src
    def compute_loss(self, f, x):
        return (self.eq.compute_loss(f, x) - sum(s.eval(x) for s in self.src)).pow(2).mean()
    def compute_pointwise_residual(self, f, x):
        return (self.eq.compute_pointwise_residual(f, x).sqrt() - sum(s.eval(x) for s in self.src)).pow(2)

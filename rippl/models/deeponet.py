import torch, torch.nn as nn
class DeepONet(nn.Module):
    def __init__(self, b_lay, t_lay, out=1):
        super().__init__()
        self.b, self.t = self._net(b_lay), self._net(t_lay)
        self.bias = nn.Parameter(torch.zeros(out))
    def _net(self, l):
        return nn.Sequential(*[mod for i in range(len(l)-1) for mod in (nn.Linear(l[i], l[i+1]), nn.Tanh())][:2*(len(l)-1)-1])
    def forward(self, u_s, q_c):
        return torch.einsum("bl,bql->bq", self.b(u_s), self.t(q_c)).unsqueeze(-1) + self.bias

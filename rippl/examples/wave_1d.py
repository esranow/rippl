"""
examples/wave_1d.py
Gaussian pulse propagation demo using rippl System + Simulation.

Run:
    python examples/wave_1d.py
"""
import torch
import math

from rippl.physics.operators import Laplacian, TimeDerivative
from rippl.physics.equation import Equation
from rippl.core import System, Domain
from rippl.core.simulation import Simulation

# ── domain ────────────────────────────────────────────────────────────
N   = 128       # spatial grid points
L   = 10.0      # domain length  [0, L]
dt  = 0.05      # satisfies CFL
steps = 40
c   = 1.0

# ── Gaussian initial condition ───────────────────────────────────────
k  = 5.0        # sharpness
x0 = L / 2.0   # centred
x  = torch.linspace(0, L, N)
u0 = torch.exp(-k * (x - x0) ** 2).view(1, N, 1)
v0 = torch.zeros_like(u0)

# ── System ────────────────────────────────────────────────────────────
eq  = Equation(terms=[(1.0, TimeDerivative(2)), (-c**2, Laplacian())])
dom = Domain(spatial_dims=1, bounds=((0.0, L), (0.0, steps * dt)), resolution=(N, steps + 1))
sys = System(eq, dom)

# ── run ───────────────────────────────────────────────────────────────
print("Starting simulation...")
sim  = Simulation(sys)
out  = sim.run(u0, v0, steps=steps)
traj = out["field"]
print(f"Trajectory: {tuple(traj.shape)}")

# ── visualize ─────────────────────────────────────────────────────────
Simulation.visualize(traj, title="1-D Gaussian Wave Propagation")

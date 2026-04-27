import torch
from rippl.physics.schrodinger import SchrodingerSystem

# Schrödinger Particle in a Box Benchmark
# analytic: psi_real=sin(πx)cos(E1*t), psi_imag=-sin(πx)sin(E1*t)

def run_schrodinger_benchmark():
    def zero_potential(coords):
        return torch.zeros(coords.shape[0], 1, device=coords.device)
        
    system_engine = SchrodingerSystem(potential_fn=zero_potential, hbar=1.0, mass=1.0)
    
    # lr=5e-4, adaptive_loss=True
    pass

def main():
    run_schrodinger_benchmark()

if __name__ == "__main__":
    main()

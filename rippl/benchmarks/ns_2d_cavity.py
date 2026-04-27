import torch
from rippl.physics.navier_stokes import NavierStokesSystem
from rippl.core.system import Domain
from rippl.core.simulation import Simulation
from rippl.core.experiment import Experiment

# Navier-Stokes 2D Cavity Flow Benchmark
# validated for Re=100

def run_ns_benchmark():
    system_engine = NavierStokesSystem(rho=1.0, mu=0.01)
    eq_system = system_engine.build_equation_system()
    
    domain = Domain(mins=[0.0, 0.0, 0.0], maxs=[1.0, 1.0, 1.0]) # x, y, t
    
    from rippl.physics.conservative import StreamFunctionModel
    model = system_engine.suggested_model()
    model = StreamFunctionModel(model)
    
    # lr=1e-4, grad_clip=1.0
    # causal_training=True, adaptive_loss=True
    pass

def main():
    run_ns_benchmark()

if __name__ == "__main__":
    main()

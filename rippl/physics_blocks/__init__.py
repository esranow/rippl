"""
rippl: Physics Blocks Package

Hybrid physics-aware neural blocks that combine fixed physics operators
with small learnable correction networks (MLP / conv).
"""

from rippl.physics_blocks.laplacian import HybridLaplacianBlock
from rippl.physics_blocks.residual import HybridWaveResidualBlock
from rippl.physics_blocks.spectral import SpectralHybridBlock
from rippl.physics_blocks.energy import EnergyAwareBlock
from rippl.physics_blocks.oscillator import HybridOscillatorBlock
from rippl.physics_blocks.embedding import PDEParameterEmbeddingBlock
from rippl.physics_blocks.gradient import HybridGradientBlock
from rippl.physics_blocks.boundary_block import BoundaryConditionBlock
from rippl.physics_blocks.hamiltonian import HamiltonianBlock
from rippl.physics_blocks.spectral_reg import SpectralRegularizationBlock
from rippl.physics_blocks.multiscale_ff import MultiScaleFourierFeatureBlock
from rippl.physics_blocks.spectral_conv import SpectralConvBlock
from rippl.physics_blocks.hybrid_stepper import HybridTimeStepperBlock
from rippl.physics_blocks.adaptivesampler import AdaptiveSamplingBlock
from rippl.physics_blocks.conservation_block import ConservationConstraintBlock
from rippl.physics_blocks.nn_operator_wrapper import OperatorWrapperBlock

__all__ = [
    "HybridLaplacianBlock",
    "HybridWaveResidualBlock",
    "SpectralHybridBlock",
    "EnergyAwareBlock",
    "HybridOscillatorBlock",
    "PDEParameterEmbeddingBlock",
    "HybridGradientBlock",
    "BoundaryConditionBlock",
    "HamiltonianBlock",
    "SpectralRegularizationBlock",
    "MultiScaleFourierFeatureBlock",
    "SpectralConvBlock",
    "HybridTimeStepperBlock",
    "AdaptiveSamplingBlock",
    "ConservationConstraintBlock",
    "OperatorWrapperBlock",
    "BLOCK_REGISTRY",
]

BLOCK_REGISTRY = {
    "laplacian": HybridLaplacianBlock,
    "wave_residual": HybridWaveResidualBlock,
    "spectral": SpectralHybridBlock,
    "energy": EnergyAwareBlock,
    "oscillator": HybridOscillatorBlock,
    "pde_embedding": PDEParameterEmbeddingBlock,
    "gradient": HybridGradientBlock,
    "boundary": BoundaryConditionBlock,
    "hamiltonian": HamiltonianBlock,
    "spectral_reg": SpectralRegularizationBlock,
    "multiscale_ff": MultiScaleFourierFeatureBlock,
    "spectral_conv": SpectralConvBlock,
    "hybrid_stepper": HybridTimeStepperBlock,
    "adaptive_sampler": AdaptiveSamplingBlock,
    "conservation": ConservationConstraintBlock,
    "operator_wrapper": OperatorWrapperBlock,
}

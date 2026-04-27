"""
rippl: Diagnostics Package
"""
from rippl.diagnostics.metrics import l2_error, relative_l2_error
from rippl.diagnostics.energy import wave_energy
from rippl.diagnostics.spectral import spectral_error

__all__ = ["l2_error", "relative_l2_error", "wave_energy", "spectral_error"]

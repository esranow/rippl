"""rippl.core — System, Simulation, Experiment."""
from rippl.core.system import System, Domain, Constraint, NeumannConstraint
from rippl.core.equation import Equation
from rippl.core.equation_system import EquationSystem
from rippl.core.experiment import Experiment
from rippl.core.inverse import InverseProblem, InverseParameter, DigitalTwin
from rippl.core.nondim import ReferenceScales, NondimSystem, NondimModelWrapper

__all__ = [
    "System", "Domain", "Constraint", "NeumannConstraint",
    "Equation", "EquationSystem",
    "Experiment", "InverseProblem", "InverseParameter", "DigitalTwin",
    "ReferenceScales", "NondimSystem", "NondimModelWrapper"
]

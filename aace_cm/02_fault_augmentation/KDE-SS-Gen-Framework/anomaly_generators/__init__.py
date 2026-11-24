"""
Physics-based Anomaly Generators for Industrial Motor Data.

This module provides physically realistic anomaly injection methods based on
established failure models from mechanical engineering and materials science.
"""

from .base_generator import BaseAnomalyModel
from .physics_models import (
    ParisCrackModel,
    ArrheniusThermalModel,
    BearingWearModel,
    ImbalanceModel,
    MechanicalLoosenessModel
)
from .composite_generator import PhysicsAnomalyGenerator

__all__ = [
    'BaseAnomalyModel',
    'ParisCrackModel',
    'ArrheniusThermalModel',
    'BearingWearModel',
    'ImbalanceModel',
    'MechanicalLoosenessModel',
    'PhysicsAnomalyGenerator'
]

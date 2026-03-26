"""
Moonshine Fine-Tuning Package

Fine-tuning toolkit for the Moonshine ASR model using curriculum learning.
Based on Pierre Chéneau's original implementation.
"""

__version__ = "0.1.0"

from .data_loader import MoonshineDataLoader
from .curriculum import CurriculumScheduler, CurriculumPhase

__all__ = [
    "MoonshineDataLoader",
    "CurriculumScheduler",
    "CurriculumPhase",
]

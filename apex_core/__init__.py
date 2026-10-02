"""
APEX Framework: Advanced Autoregressive Hybrid Sequence Modeling Library.
Architecture: Hop-Mix + LRCM + Mamba-3 + Native ECHO.
"""

from .config import APEXConfig, BlockPreset
from .model import APEXModel
from .trainer import APEXTrainer, TrainingConfig
from .evaluation import BenchmarkHarness, MetricsMonitor

__version__ = "0.1.0"
__all__ = [
    "APEXConfig",
    "BlockPreset",
    "APEXModel",
    "APEXTrainer",
    "TrainingConfig",
    "BenchmarkHarness",
    "MetricsMonitor",
]

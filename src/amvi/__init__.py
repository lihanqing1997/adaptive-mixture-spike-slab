"""Adaptive mixture VI for point-mass spike-and-slab Gaussian regression."""

from .solver import Mixture, RegressionModel, Settings, best_meanfield, fit

__all__ = ["Mixture", "RegressionModel", "Settings", "best_meanfield", "fit"]
__version__ = "0.1.0"

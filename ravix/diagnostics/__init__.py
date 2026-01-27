"""
Model diagnostics.

This submodule contains statistical tests used to assess model assumptions.
Diagnostics report evidence about potential issues but do not modify data.
"""

from .variance import ncv, bp
from .normality import shapiro
from .multicollinearity import vif

__all__ = [
    "ncv",
    "bp",
    "shapiro",
    "vif",
]

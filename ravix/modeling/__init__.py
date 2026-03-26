"""
Modeling
========
Core modeling API: fitting, prediction, intervals, diagnostics, and model selection.
"""

from .bsr import bsr
from .fit import fit, ols, logistic, poisson
from .intervals import intervals
from .parse_formula import parse_formula
from .predict import predict
from .stepwise import stepwise
from .summary import summary
from .xy_split import xy_split

__all__ = [
    "bsr",
    "fit",
    "ols",
    "logistic",
    "poisson",
    "intervals",
    "parse_formula",
    "predict",
    "stepwise",
    "summary",
    "xy_split",
]

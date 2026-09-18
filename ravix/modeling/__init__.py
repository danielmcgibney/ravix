"""
Modeling
========
Core Ravix modeling API for fitting, prediction, intervals, robust inference,
nested-model comparison, and model selection.
"""

from .bsr import bsr
from .compare import compare
from .fit import fit, ols, logistic, poisson
from .intervals import intervals
from .parse_formula import parse_formula
from .predict import predict
from .robust import robust
from .stepwise import stepwise
from .summary import summary
from .xy_split import xy_split

__all__ = [
    "bsr",
    "compare",
    "fit",
    "ols",
    "logistic",
    "poisson",
    "intervals",
    "parse_formula",
    "predict",
    "robust",
    "stepwise",
    "summary",
    "xy_split",
]

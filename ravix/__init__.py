from ._version import __version__

"""
Ravix Package
=============
Applied modeling and visualization for business analytics.

Root exports are intentionally kept small and user-facing:
- Modeling: OLS, logistic, and Poisson regression with prediction, intervals,
  robust inference, nested-model comparison, and model-selection utilities
- Plots: high-level plotting functions
- Diagnostics: regression diagnostic tests
- Transforms: data-transformation utilities
- Helpers: bundled-data access

Advanced/internal helpers remain available under Ravix submodules.
"""

# --- Modeling (public surface) ---
from .modeling import (
    bsr,
    compare,
    fit,
    intervals,
    logistic,
    ols,
    poisson,
    predict,
    robust,
    stepwise,
    summary,
    xy_split,
)

# --- Plots (public surface) ---
from .plots import (
    abline,
    barplot,
    boxplot,
    hist,
    plot,
    plot_bsr,
    plot_cor,
    plot_cook,
    plot_intervals,
    qq,
    viz,
)

# --- Diagnostics ---
from .diagnostics import (
    bp,
    ncv,
    shapiro,
    vif,
)

# --- Transforms ---
from .transforms import box_cox

# --- Helpers ---
from ._internal.helpers import get_data

__all__ = [
    # Version
    "__version__",

    # Modeling
    "bsr",
    "compare",
    "fit",
    "intervals",
    "logistic",
    "ols",
    "poisson",
    "predict",
    "robust",
    "stepwise",
    "summary",
    "xy_split",

    # Diagnostics
    "bp",
    "ncv",
    "shapiro",
    "vif",

    # Transforms
    "box_cox",

    # Plots
    "abline",
    "barplot",
    "boxplot",
    "hist",
    "plot",
    "plot_bsr",
    "plot_cor",
    "plot_cook",
    "plot_intervals",
    "qq",
    "viz",

    # Helpers
    "get_data",
]

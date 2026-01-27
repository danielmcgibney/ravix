from ._version import __version__

"""
Ravix Package
================
Applied modeling and visualization for business analytics

Root exports are intentionally kept small and user-facing:
- Modeling: OLS, logistic, and Poisson regression with summary, prediction, and interval utilities
- Plots: high-level plotting functions
- Utils: get_data

Advanced/internal helpers remain available under submodules:
- ravix.modeling (e.g., format_summary, significance_code, print_* helpers, EvalEnvironment)
"""

# --- Modeling (public surface) ---
from .modeling import (
    bsr, fit, ols, logistic, poisson,
    intervals, predict,
    stepwise, summary, xy_split
)

# --- Plots (public surface) ---
from .plots import (
    barplot, boxplot, abline, hist, plot_bsr, plot_cor, plot_cook,
    plot, plot_intervals, qq, viz
)

# --- Diagnostics (numeric tests only) ---
from .diagnostics import (
    ncv,
    bp,
    shapiro,
    vif,
)

# --- Transforms (data-modifying utilities) ---
from .transforms import (
    box_cox,
)


# --- Helpers
from ._internal.helpers import get_data

__all__ = [
    # Version
    "__version__",

    # Modeling (public)
    "bp", "bsr", "fit", "ols", "logistic", "poisson",
    "intervals", "ncv", "predict",
    "shapiro", "stepwise", "summary", "vif", "xy_split",

    # Transforms (public)
    "box_cox", 
    
    # Plots (public)
    "barplot", "boxplot", "abline", "hist", "plot_bsr", "plot_cor", "plot_cook",
    "plot", "plot_intervals", "qq", "viz",

    # Helpers
    "get_data"
]

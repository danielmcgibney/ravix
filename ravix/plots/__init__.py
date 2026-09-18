"""
Plots
=====
High-level plotting functions for exploratory analysis, regression diagnostics,
model-selection results, and interval visualization.

Public Functions
----------------
abline
    Add a fitted regression line to a compatible Ravix plot.
barplot
    Create bar plots with optional aggregation.
boxplot
    Create box plots for distribution visualization.
hist
    Create histograms for variables or model residuals.
plot
    Create scatter plots, scatterplot matrices, or residual plots.
plot_bsr
    Visualize best subset regression results.
plot_cor
    Display a correlation heatmap. Direct DataFrame input can dummy-encode
    categorical variables; all indicator levels are retained when encoded.
plot_cook
    Create Cook's distance plots for influence diagnostics.
plot_intervals
    Visualize confidence or prediction intervals.
qq
    Create Q-Q plots for normality assessment.
viz
    Route to an appropriate Ravix plotting function using a unified interface.
"""

# Utility functions
from .abline import abline

# Public plotting functions
from .barplot import barplot
from .boxplot import boxplot
from .hist import hist
from .plot import plot
from .plot_bsr import plot_bsr
from .plot_cor import plot_cor
from .plot_cook import plot_cook
from .plot_intervals import plot_intervals
from .qq import qq
from .viz import viz

# Theme system
from ._theme import get_theme, register_theme, set_theme, theme_context

__all__ = [
    # Utilities
    "abline",

    # Individual plot types
    "barplot",
    "boxplot",
    "hist",
    "plot",
    "plot_bsr",
    "plot_cor",
    "plot_cook",
    "plot_intervals",
    "qq",

    # Main plotting interface
    "viz",

    # Theme
    "set_theme",
    "get_theme",
    "theme_context",
    "register_theme",
]

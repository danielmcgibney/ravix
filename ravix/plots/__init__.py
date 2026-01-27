"""
Plots
============

This subpackage provides functions for various types of plots used in regression analysis.

Public Functions
----------------
abline
    Utility function for quickly creating models (lines) for plotting.
barplot
    Create bar plot(s) of variable(s) with optional aggregation.
boxplot
    Create box plot(s) for distribution visualization.
hist
    Create histogram(s) with automatic mode detection (single/multiple variables or model residuals).
qq
    Create Q-Q plot(s) for normality assessment (single/multiple variables or model residuals).
plot_bsr
    Visualize best subset regression results.
plot_cor
    Display correlation matrix heatmap with multiple visualization styles.
plot_cook
    Create Cook's distance plot for influence diagnostics.
plot_intervals
    Visualize confidence or prediction intervals.
plot
    Create scatter plot(s) for data relationships and residual diagnostics with automatic mode detection.
viz
    Unified plotting interface that routes to appropriate plot functions based on 'kind' parameter.
    Main entry point for quick exploratory plotting.
"""

# Utility functions
from .abline import abline

# Public plotting functions
from .barplot import barplot
from .boxplot import boxplot
from .hist import hist
from .qq import qq
from .plot_bsr import plot_bsr
from .plot_cor import plot_cor
from .plot_cook import plot_cook
from .plot_intervals import plot_intervals
from .plot import plot
from .viz import viz

__all__ = [
    # Utilities
    'abline',
    
    # Individual plot types
    'barplot',
    'boxplot', 
    'hist',
    'qq',
    'plot_bsr',
    'plot_cor',
    'plot_cook',
    'plot_intervals',
    'plot',
    
    # Main plotting interface
    'viz',
]

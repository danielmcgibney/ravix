from __future__ import annotations

from typing import Optional, Union, Tuple, Any, Literal
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ravix.modeling.parse_formula import parse_formula
from ravix.modeling.parse_formula import parse_formula
from ravix.plots._utils import (
    _ensure_no_intercept,
    _get_residuals,
)


# ======================================================================
# Public API
# ======================================================================

def hist(
    input_data: Union[np.ndarray, pd.Series, pd.DataFrame, str, Any],
    data: Optional[pd.DataFrame] = None,
    bins: int = 30,
    color: str = "blue",
    norm: bool = True,
    layout: Literal["column", "row", "matrix"] = "matrix",
    title: Optional[str] = None,
    xlab: Optional[str] = None,
    ylab: str = "Frequency",
    figsize: Tuple[float, float] = (10, 6),
    show: bool = True,
    **kwargs
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Create histogram(s) based on input type with optional normal distribution overlay.
    
    Supports five modes:
    1. Single vector/array: Creates a single histogram
    2. Column name string: Creates a single histogram for that column (requires data)
    3. Formula string: Creates multiple histograms for variables in the formula
       (can use environment lookup if data not provided)
    4. DataFrame with multiple numeric columns: Creates multiple histograms for all numeric columns
    5. Fitted regression model: Creates histogram of residuals with normal curve
    
    Parameters
    ----------
    input_data : Union[np.ndarray, pd.Series, pd.DataFrame, str, Any]
        Input for histogram creation:
        - array-like: Vector of numeric values (Mode 1)
        - str without "~": Column name (Mode 2, requires data)
        - str with "~": Formula (e.g., "y ~ x1 + x2") (Mode 3)
        - pd.DataFrame: Multiple numeric columns (Mode 4)
        - Fitted model: Regression model with residuals (Mode 5)
    data : Optional[pd.DataFrame], default=None
        DataFrame containing variables. Required for column names (Mode 2); 
        optional for formulas (Mode 3, can resolve from calling environment).
    bins : int, default=30
        Number of bins for the histogram(s).
    color : str, default="blue"
        Color for histogram(s).
    norm : bool, default=True
        Whether to overlay a normal distribution curve.
    layout : Literal["column", "row", "matrix"], default="matrix"
        Layout for multiple plots: "column", "row", or "matrix" (Modes 2-3).
    title : Optional[str], default=None
        Title for the plot. If None, defaults to "Histogram" for single plots
        and "Histograms of Variables" for multiple plots.
    xlab : Optional[str], default=None
        X-axis label. If None, automatically determined from variable name.
    ylab : str, default="Frequency"
        Y-axis label.
    figsize : Tuple[float, float], default=(10, 6)
        Figure size as (width, height) in inches.
    show : bool, default=True
        Whether to display the plot. Used in Mode 1 (single histogram)
        and Mode 4 (model residuals). Ignored for Modes 2-3 (multiple histograms).
        If False, returns (fig, ax) tuple for further customization.
    **kwargs : dict
        Additional keyword arguments passed to seaborn.histplot().
    
    Returns
    -------
    Optional[Tuple[plt.Figure, plt.Axes]]
        For Mode 1 (single histogram) and Mode 4 (residuals), returns (Figure, Axes) 
        tuple if show=False. For Modes 2-3 (multiple histograms), returns None.
    
    Examples
    --------
    >>> # Mode 1: Single vector histogram
    >>> data = np.random.normal(0, 1, 1000)
    >>> hist(data, norm=True)
    
    >>> # Mode 2: Formula with multiple variables
    >>> hist("mpg ~ hp + wt", data=mtcars, layout="row")
    
    >>> # Mode 3: DataFrame with multiple columns
    >>> hist(mtcars[["mpg", "hp", "wt", "disp"]], layout="matrix")
    
    >>> # Mode 4: Model residuals
    >>> model = fit("mpg ~ hp + wt", data=mtcars)
    >>> hist(model, title="Residual Distribution")
    
    >>> # Return figure for further customization
    >>> fig, ax = hist(data, show=False, title="My Distribution")
    >>> ax.axvline(data.mean(), color='red', linestyle='--', label='Mean')
    >>> ax.legend()
    >>> plt.show()
    
    Notes
    -----
    - Mode 1 (single vector/array): Single histogram with optional normal curve
    - Mode 2 (column name): Single histogram for specified column (requires data)
    - Mode 3 (formula): Multiple histograms for variables in formula
    - Mode 4 (DataFrame): All numeric columns plotted in grid
    - Mode 5 (model): Residual histogram with diagnostics
    - Normal curve overlay scaled to match histogram frequency
    - For multiple histograms (Modes 3-4), layout controls arrangement (column/row/matrix)
    - show parameter works for single histograms (Modes 1-2, 5) but ignored for multiple (Modes 3-4)
    
    Raises
    ------
    ValueError
        If bins is not a positive integer.
        If layout is not 'column', 'row', or 'matrix'.
        If column name not found in data.
    """
    # Validate bins parameter
    if not isinstance(bins, int) or bins < 1:
        raise ValueError(f"bins must be a positive integer, got {bins}")
    
    # Validate layout parameter
    valid_layouts = ["column", "row", "matrix"]
    if layout not in valid_layouts:
        raise ValueError(f"layout must be one of {valid_layouts}, got '{layout}'")
    
    # Mode 4: Fitted regression model (check for resid attribute)
    if hasattr(input_data, 'resid'):
        return _hist_res(
            input_data, 
            title=title if title else "Histogram of Residuals", 
            xlab=xlab if xlab else "Residuals", 
            ylab=ylab, 
            color=color, 
            bins=bins,
            norm=norm,
            figsize=figsize,
            show=show,
            **kwargs
        )
    
    # Mode 3: DataFrame with multiple numeric columns
    if isinstance(input_data, pd.DataFrame):
        numeric_cols = input_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            _hists(
                input_data=input_data, 
                data=None, 
                bins=bins, 
                color=color,
                norm=norm, 
                layout=layout, 
                title=title if title else "Histograms of Variables", 
                xlab=xlab, 
                ylab=ylab, 
                figsize=figsize, 
                **kwargs
            )
            return None
        # If only one numeric column, fall through to single vector mode
        elif len(numeric_cols) == 1:
            input_data = input_data[numeric_cols[0]]
    
    # Mode 2: Formula or single column name
    if isinstance(input_data, str):
        # Guard: column name without data will fail in formula parsing
        if ("~" not in input_data) and (data is None):
            raise ValueError(
                f"Column name '{input_data}' requires data parameter. "
                "Use formula syntax (e.g., 'var ~ 1') to resolve from environment."
            )
        
        # Single column name → single histogram (respects show parameter)
        if ("~" not in input_data) and (data is not None):
            try:
                column_data = data[input_data]
            except KeyError:
                raise ValueError(f"Column '{input_data}' not found in data.")
            
            return _hist_single(
                column_data,
                bins=bins,
                color=color,
                norm=norm,
                title=title if title else "Histogram",
                xlab=xlab,
                ylab=ylab,
                figsize=figsize,
                show=show,
                **kwargs
            )
        
        # Formula → multiple histograms (ignores show parameter)
        _hists(
            input_data=input_data,
            data=data,  # May be None; parse_formula can resolve via environment
            bins=bins,
            color=color,
            norm=norm,
            layout=layout,
            title=title if title else "Histograms of Variables",
            xlab=xlab,
            ylab=ylab,
            figsize=figsize,
            **kwargs,
        )
        return None
    
    # Mode 1: Single vector - original functionality
    return _hist_single(
        input_data, 
        bins=bins, 
        color=color, 
        norm=norm, 
        title=title if title else "Histogram", 
        xlab=xlab, 
        ylab=ylab, 
        figsize=figsize,
        show=show,
        **kwargs
    )


# ======================================================================
# Single histogram
# ======================================================================

def _hist_single(
    vector: Union[np.ndarray, pd.Series],
    bins: int = 30,
    color: str = "blue",
    norm: bool = True,
    title: str = "Histogram",
    xlab: Optional[str] = None,
    ylab: str = "Frequency",
    figsize: Tuple[float, float] = (10, 6),
    show: bool = True,
    **kwargs
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Create a single histogram with optional normal overlay."""
    # Get the variable name if xlab is not provided
    if xlab is None:
        xlab = str(vector.name)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Always use stat="count" for consistency
    sns.histplot(vector, bins=bins, kde=False, color=color, edgecolor='black', 
                 stat="count", ax=ax, **kwargs)
    
    # Add normal distribution overlay if requested (scaled to counts)
    if norm:
        _plot_normal_overlay(
            ax,
            vector,
            bins=bins,
            stat="count",  # Match histogram stat
            line_kwargs={"color": "k", "linewidth": 2, "label": "Normal"},
        )
        ax.legend()
    
    # Set titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    
    plt.tight_layout()
    
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig, ax


# ======================================================================
# Multiple histograms
# ======================================================================

def _hists(
    input_data: Union[str, pd.DataFrame],
    data: Optional[pd.DataFrame] = None,
    bins: int = 30,
    color: str = "blue",
    norm: bool = True,
    layout: Literal["column", "row", "matrix"] = "matrix",
    title: str = "Histograms of Variables",
    xlab: Optional[str] = None,
    ylab: str = "Frequency",
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> None:
    """
    Create multiple histograms for numeric variables in a grid layout.
    
    Parameters
    ----------
    input_data : Union[str, pd.DataFrame]
        Formula string or DataFrame:
        - str: Formula (e.g., "y ~ x1 + x2") - can use environment lookup if data is None
        - pd.DataFrame: All numeric columns plotted
    data : Optional[pd.DataFrame], default=None
        Data for formula resolution. Optional if variables exist in calling environment.
    bins : int
        Number of bins
    color : str
        Histogram color
    norm : bool
        Whether to overlay normal curves
    layout : Literal["column", "row", "matrix"]
        Layout arrangement
    title : str
        Plot title
    xlab : Optional[str]
        X-axis label (uses variable names if None)
    ylab : str
        Y-axis label
    figsize : Optional[Tuple[float, float]]
        Figure size (auto-calculated if None)
    **kwargs
        Additional arguments for seaborn.histplot()
    
    Notes
    -----
    - Only numeric columns included
    - Infinite values replaced with NaN
    - Always shows and closes figure (no return value)
    """
    # Case 1: Directly given DataFrame
    if isinstance(input_data, pd.DataFrame):
        plot_data = input_data.select_dtypes(include=[np.number])
    # Case 2: Formula mode - parse via parse_formula (handles environment lookup)
    else:
        if not isinstance(input_data, str):
            raise TypeError(
                f"input_data must be a string (formula) or DataFrame, got {type(input_data).__name__}"
            )
        formula_for_parse = _ensure_no_intercept(input_data)
        Y_out, X_out = parse_formula(formula_for_parse, data)
        
        # Build with X-then-Y ordering and preserve index
        y_name = getattr(Y_out, "name", "Y")
        plot_data = pd.concat(
            [X_out, pd.Series(np.asarray(Y_out).ravel(), index=X_out.index, name=y_name)],
            axis=1
        )
    
    # Replace infinite values with NaN (don't mutate original)
    plot_data = plot_data.replace([np.inf, -np.inf], np.nan)
    
    num_vars = len(plot_data.columns)
    
    # Check if we have any variables to plot
    if num_vars == 0:
        raise ValueError(
            "No numeric variables found to plot. "
            "Ensure the input contains numeric columns."
        )
    
    # Determine the layout
    if layout == "column":
        nrows, ncols = num_vars, 1
    elif layout == "row":
        nrows, ncols = 1, num_vars
    else:  # layout == "matrix"
        nrows = int(np.ceil(np.sqrt(num_vars)))
        ncols = int(np.ceil(num_vars / nrows))
    
    # Set figure size if not provided
    if figsize is None:
        figsize = (5 * ncols, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # Flatten the axes array for easy iteration
    
    fig.suptitle(title, fontsize=16)
    
    for i, var in enumerate(plot_data.columns):
        ax = axes[i]
        
        # Always use stat="count" for consistency
        sns.histplot(plot_data[var], bins=bins, kde=False, color=color, 
                    ax=ax, edgecolor='black', stat="count", **kwargs)
        
        # Add normal distribution curve if requested (scaled to counts)
        if norm:
            _plot_normal_overlay(
                ax,
                plot_data[var].dropna(),
                bins=bins,
                stat="count",  # Match histogram stat
                line_kwargs={"color": "k", "linewidth": 2},
            )
        
        # Set individual titles and labels
        ax.set_title(f'Histogram of {var}')
        ax.set_xlabel(xlab if xlab else var)
        ax.set_ylabel(ylab)
    
    # Remove any unused subplots in the matrix layout
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close(fig)


# ======================================================================
# Residual histogram
# ======================================================================

def _hist_res(
    model,
    res: str = "resid",
    bins: int = 30,
    color: str = "blue",
    norm: bool = True,
    title: str = "Histogram of Residuals",
    xlab: str = "Residuals",
    ylab: str = "Frequency",
    figsize: Tuple[float, float] = (10, 6),
    show: bool = True,
    **kwargs
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Create a histogram of residuals from a fitted regression model.
    
    Parameters
    ----------
    model : Any
        Fitted regression model with a `resid` attribute
    res : str, default="resid"
        Type of residuals: "resid", "pearson", "deviance", "anscombe", 
        "studentized", "response"
    bins : int, default=30
        Number of bins
    color : str, default="blue"
        Histogram color
    norm : bool, default=True
        Whether to overlay normal curve
    title : str, default="Histogram of Residuals"
        Plot title (auto-adjusted based on residual type)
    xlab : str, default="Residuals"
        X-axis label (auto-adjusted based on residual type)
    ylab : str, default="Frequency"
        Y-axis label
    figsize : Tuple[float, float], default=(10, 6)
        Figure size
    show : bool, default=True
        Whether to display plot
    **kwargs
        Additional arguments for seaborn.histplot()
    
    Returns
    -------
    Optional[Tuple[plt.Figure, plt.Axes]]
        (fig, ax) if show=False, else None
    
    Notes
    -----
    - Residuals should approximately follow normal distribution for linear models
    - Different residual types useful for different model diagnostics
    - Normalizes res parameter internally for defensive programming
    - Uses stat="count" for consistency with other histogram modes
    """
    # Normalize res parameter defensively (in case called directly)
    res = str(res).lower().strip()
    
    # Get residuals (same as plot())
    residuals = _get_residuals(model, res)
    
    # Adjust title if it's the default
    if title == "Histogram of Residuals" and res != "resid":
        title = f"Histogram of {res.title()} Residuals"
    
    # Adjust xlab if it's the default
    if xlab == "Residuals" and res != "resid":
        xlab = f"{res.title()} Residuals"
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Always use stat="count" for consistency across all histogram modes
    sns.histplot(residuals, bins=bins, kde=False, color=color, 
                edgecolor='black', stat="count", alpha=0.7, 
                label='Residuals', ax=ax, **kwargs)
    
    # Overlay normal distribution curve if requested (scaled to counts)
    if norm:
        _plot_normal_overlay(
            ax,
            residuals,
            bins=bins,
            stat="count",  # Match histogram stat
            line_kwargs={"color": "k", "linewidth": 2, "label": "Normal"},
        )
        ax.legend(loc="best", framealpha=0.9)
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    
    plt.tight_layout()
    
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig, ax




# ======================================================================
# Utility helpers
# ======================================================================

def _plot_normal_overlay(
    ax: plt.Axes,
    values: Union[np.ndarray, pd.Series],
    *,
    bins: int,
    stat: Literal["count", "density"] = "density",
    line_kwargs: Optional[dict] = None,
) -> None:
    """
    Overlay a fitted Normal distribution curve on an existing histogram.
    
    Parameters
    ----------
    ax : plt.Axes
        Axes to plot on
    values : array-like
        Data values for fitting normal distribution
    bins : int
        Number of bins (for scaling)
    stat : Literal["count", "density"], default="density"
        Must match histogram's stat parameter
    line_kwargs : Optional[dict]
        Additional line plot kwargs
    
    Warnings
    --------
    Issues warning if insufficient data or invalid fit parameters
    """
    from scipy import stats

    # Clean and flatten
    v = np.asarray(values).ravel()
    v = v[np.isfinite(v)]
    
    if v.size < 2:
        warnings.warn("Insufficient data points (< 2) for normal overlay", UserWarning)
        return

    # Fit Normal (MLE)
    mu, sigma = stats.norm.fit(v)
    if not np.isfinite(sigma) or sigma <= 0:
        warnings.warn(f"Invalid normal fit parameters (mu={mu}, sigma={sigma})", UserWarning)
        return

    # X grid
    xmin, xmax = np.min(v), np.max(v)
    if xmin == xmax:
        warnings.warn("All values are identical; cannot draw normal overlay", UserWarning)
        return
    x = np.linspace(xmin, xmax, 200)

    # PDF
    pdf = stats.norm.pdf(x, mu, sigma)

    # Scale based on stat type
    if stat == "count":
        edges = np.histogram_bin_edges(v, bins=bins)
        bin_widths = np.diff(edges)
        bw = float(np.median(bin_widths)) if bin_widths.size else 1.0
        y = pdf * v.size * bw
    elif stat == "density":
        y = pdf
    else:
        raise ValueError(f"stat must be 'count' or 'density', got '{stat}'")

    kw = {"linewidth": 2}
    if line_kwargs:
        kw.update(line_kwargs)

    ax.plot(x, y, **kw)


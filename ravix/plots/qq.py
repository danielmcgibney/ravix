from __future__ import annotations

from typing import Optional, Union, Tuple, Any, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot

from ravix.modeling.parse_formula import parse_formula
from ravix.plots._utils import (
    _ensure_no_intercept,
    _as_1d_numeric,
    _clean_numeric_df,
    _get_residuals,
)

# ======================================================================
# Constants
# ======================================================================

_VALID_LAYOUTS = ("column", "row", "matrix")


# ======================================================================
# Public API
# ======================================================================

def qq(
    input_data: Union[np.ndarray, pd.Series, pd.DataFrame, str, Any],
    data: Optional[pd.DataFrame] = None,
    res: str = "resid",
    level: Optional[float] = 0.95,
    color: str = "blue",
    lcolor: str = "red",
    band_color: Optional[str] = None,
    layout: Literal["column", "row", "matrix"] = "matrix",
    title: Optional[str] = None,
    xlab: str = "Theoretical Quantiles",
    ylab: str = "Sample Quantiles",
    figsize: Tuple[float, float] = (10, 6),
    grid: bool = True,
    show: bool = True,
    **kwargs
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Create Q-Q plot(s) to assess normality of distributions.
    
    Supports five modes:
    1. Single vector/array: Creates a single Q-Q plot
    2. Column name: Creates a single Q-Q plot for that column (requires data)
    3. Single-variable formula: Creates a single Q-Q plot (e.g., "y ~ 1")
    4. Multi-variable formula or DataFrame: Creates multiple Q-Q plots (one per variable)
    5. Fitted regression model: Creates Q-Q plot of residuals (duck-typed: any object with `.resid`)
    
    Parameters
    ----------
    input_data : Union[np.ndarray, pd.Series, pd.DataFrame, str, Any]
        Input for Q-Q plot creation:
        - array-like: Vector of numeric values (Mode 1)
        - str without "~": Column name (Mode 2, requires data)
        - str with "~": Formula (Mode 3 if produces 1 variable, Mode 4 if multiple)
        - pd.DataFrame: If 1 numeric column → Mode 1, if >1 → Mode 4 (multiple Q-Q plots)
        - Fitted model with `.resid` attribute: Mode 5 (residuals)
    data : Optional[pd.DataFrame], default=None
        DataFrame containing variables. Required for column names (Mode 2);
        optional for formulas (can resolve from calling environment).
    res : str, default="resid"
        Type of residuals to plot (Mode 5 only):
        - "resid": Ordinary residuals
        - "pearson": Pearson residuals
        - "deviance": Deviance residuals
        - "anscombe": Anscombe residuals
        - "studentized": Studentized residuals
        - "response": Response residuals
    level : Optional[float], default=0.95
        Confidence level for Q-Q confidence band (e.g., 0.95 for 95%).
        If None, no confidence band is drawn.
    color : str, default="blue"
        Color for data points.
    lcolor : str, default="red"
        Color for reference line.
    band_color : Optional[str], default=None
        Color for confidence band. If None, uses `color`.
    layout : Literal["column", "row", "matrix"], default="matrix"
        Layout for multiple Q-Q plots (Modes 4).
    title : Optional[str], default=None
        Plot title. If None, defaults to "Q-Q Plot" for single plots
        and "Q-Q Plots" for multiple plots.
    xlab : str, default="Theoretical Quantiles"
        X-axis label.
    ylab : str, default="Sample Quantiles"
        Y-axis label.
    figsize : Tuple[float, float], default=(10, 6)
        Total figure size in inches (width, height) for all modes.
        For multi-plot mode, this is divided among subplots automatically.
    grid : bool, default=True
        Whether to show grid lines.
    show : bool, default=True
        Whether to display the plot. If False (single-plot mode only),
        returns (fig, ax) for further customization.
    **kwargs
        Additional arguments passed to statsmodels ProbPlot.qqplot().
    
    Returns
    -------
    Optional[Tuple[plt.Figure, plt.Axes]]
        For single-plot modes (1, 2, 3, 5): returns (fig, ax) if show=False,
        else None. For multi-plot mode (4): always returns None.
    
    Examples
    --------
    >>> # Mode 1: Single vector
    >>> data = np.random.normal(0, 1, 1000)
    >>> qq(data)
    
    >>> # Mode 2: Column name
    >>> qq("mpg", data=mtcars)
    
    >>> # Mode 3: Single-variable formula
    >>> qq("mpg ~ 1", data=mtcars)
    
    >>> # Mode 4: Multi-variable formula (creates Q-Q plot for EACH variable)
    >>> qq("mpg ~ hp + wt", data=mtcars, layout="row")  # Creates 3 plots: hp, wt, mpg
    
    >>> # Mode 4: Two-variable formula
    >>> qq("mpg ~ hp", data=mtcars)  # Creates 2 plots: hp and mpg
    
    >>> # Mode 4: DataFrame with multiple columns
    >>> qq(mtcars[["mpg", "hp", "wt"]], layout="matrix")  # Creates 3 plots
    
    >>> # Mode 5: Model residuals
    >>> model = fit("mpg ~ hp + wt", data=mtcars)
    >>> qq(model)
    
    >>> # Studentized residuals
    >>> qq(model, res="studentized")
    
    >>> # Return figure for customization
    >>> fig, ax = qq(data, show=False)
    >>> ax.axhline(0, color='green', linestyle='--')
    >>> plt.show()
    
    Notes
    -----
    - Mode 1 (vector/array/Series): Single Q-Q plot
    - Mode 2 (column name string): Single Q-Q plot for specified column (requires data)
    - Mode 3 (single-variable formula): Single Q-Q plot (typically "y ~ 1" produces just Y)
    - Mode 4 (multi-variable formula or multi-column DataFrame): Multiple Q-Q plots, one per variable
    - Mode 5 (fitted model): Q-Q plot of residuals (duck-typed: accepts any object with `.resid` attribute)
    - **Formula plotting**: Formulas create Q-Q plots for ALL numeric variables (both X and Y):
        * "y ~ 1" → 1 plot (just y)
        * "y ~ x" → 2 plots (x and y)
        * "y ~ x1 + x2" → 3 plots (x1, x2, and y)
    - Single-column DataFrames are treated as Mode 1 (single Q-Q plot)
    - Formulas create Q-Q plots for individual variable distributions, NOT relationships between variables
    - Confidence band is computed by statsmodels using specified level (e.g., 0.95 for 95% CI)
    - For multiple plots (Mode 4), layout controls arrangement (column/row/matrix)
    - show parameter works for single plots (Modes 1-3, 5) but ignored for multiple (Mode 4)
    - Grid can be disabled by setting grid=False
    - figsize specifies total figure size for all modes (not per-subplot)
    
    Raises
    ------
    ValueError
        If level is not between 0 and 1 (exclusive).
        If insufficient data points for Q-Q plot.
        If column name not found in data.
        If layout is not 'column', 'row', or 'matrix'.
    """
    # Validate level
    if level is not None:
        if not (0 < float(level) < 1):
            raise ValueError("level must be between 0 and 1 (exclusive), or None.")
    
    # Normalize res parameter defensively
    res = str(res).lower().strip()
    
    # Set band_color default
    if band_color is None:
        band_color = color
    
    # Mode 5: Fitted model with residuals
    if hasattr(input_data, 'resid'):
        residuals = _get_residuals(input_data, res)
        
        # Determine label
        if res == "resid":
            label = "Residuals"
        else:
            label = f"{res.title()} Residuals"
        
        return _qq_single(
            residuals,
            label=label,
            level=level,
            color=color,
            lcolor=lcolor,
            band_color=band_color,
            title=title if title else f"Q-Q Plot of {label}",
            xlab=xlab,
            ylab=ylab,
            figsize=figsize,
            grid=grid,
            show=show,
            **kwargs
        )
    
    # Mode 4: DataFrame with multiple numeric columns
    if isinstance(input_data, pd.DataFrame):
        # Clean first, then branch on result
        plot_data = _clean_numeric_df(input_data)
        
        if plot_data.shape[1] == 0:
            raise ValueError("DataFrame has no numeric columns.")
        elif plot_data.shape[1] == 1:
            # Single numeric column - fall through to single mode
            input_data = plot_data.iloc[:, 0]
        else:
            # Multiple numeric columns - multi-plot mode
            # Validate layout early for better error messages
            if layout not in _VALID_LAYOUTS:
                raise ValueError(f"layout must be one of {_VALID_LAYOUTS}, got '{layout}'")
            
            _qqs_from_df(
                plot_data=plot_data,
                level=level,
                color=color,
                lcolor=lcolor,
                band_color=band_color,
                layout=layout,
                title=title if title else "Q-Q Plots",
                figsize=figsize,
                grid=grid,
                **kwargs
            )
            return None
    
    # Mode 4: 2D array with multiple columns
    if isinstance(input_data, np.ndarray):
        arr = np.asarray(input_data)
        if arr.ndim == 2:
            # Convert to DataFrame and clean
            plot_data = _clean_numeric_df(pd.DataFrame(arr))
            
            if plot_data.shape[1] == 0:
                raise ValueError("Array has no numeric columns.")
            elif plot_data.shape[1] == 1:
                # Single column - fall through to single mode (keep as Series)
                input_data = plot_data.iloc[:, 0]
            else:
                # Multiple columns - multi-plot mode
                # Validate layout early for better error messages
                if layout not in _VALID_LAYOUTS:
                    raise ValueError(f"layout must be one of {_VALID_LAYOUTS}, got '{layout}'")
                
                _qqs_from_df(
                    plot_data=plot_data,
                    level=level,
                    color=color,
                    lcolor=lcolor,
                    band_color=band_color,
                    layout=layout,
                    title=title if title else "Q-Q Plots",
                    figsize=figsize,
                    grid=grid,
                    **kwargs
                )
                return None
    
    # String routing
    if isinstance(input_data, str):
        # Guard: column name without data
        if ("~" not in input_data) and (data is None):
            raise ValueError(
                f"Column name '{input_data}' requires data parameter."
            )
        
        # Mode 2: Single column name → single Q-Q plot
        if ("~" not in input_data) and (data is not None):
            try:
                column_data = data[input_data]
            except KeyError:
                raise ValueError(f"Column '{input_data}' not found in data.")
            
            return _qq_single(
                column_data.values,
                label=input_data,
                level=level,
                color=color,
                lcolor=lcolor,
                band_color=band_color,
                title=title if title else "Q-Q Plot",
                xlab=xlab,
                ylab=ylab,
                figsize=figsize,
                grid=grid,
                show=show,
                **kwargs
            )
        
        # Formula: parse once to build DataFrame of all variables
        formula_for_parse = _ensure_no_intercept(input_data)
        try:
            Y_out, X_out = parse_formula(formula_for_parse, data)
        except Exception as e:
            raise ValueError(
                "Could not parse formula. Either:\n"
                "  - provide `data=...`, or\n"
                "  - ensure variables exist in the calling environment."
            ) from e
        
        # Build DataFrame with X-then-Y ordering
        y_name = getattr(Y_out, "name", "Y")
        plot_data = pd.concat(
            [X_out, pd.Series(np.asarray(Y_out).ravel(), index=X_out.index, name=y_name)],
            axis=1
        )
        
        # Clean and get numeric columns
        plot_data = _clean_numeric_df(plot_data)
        
        num_vars = plot_data.shape[1]
        
        if num_vars == 0:
            raise ValueError("Formula produced no numeric variables.")
        
        # Mode 3: Single variable (typically "y ~ 1" which produces only Y)
        if num_vars == 1:
            var_name = plot_data.columns[0]
            return _qq_single(
                plot_data[var_name].values,
                label=var_name,
                level=level,
                color=color,
                lcolor=lcolor,
                band_color=band_color,
                title=title if title else "Q-Q Plot",
                xlab=xlab,
                ylab=ylab,
                figsize=figsize,
                grid=grid,
                show=show,
                **kwargs
            )
        
        # Mode 4: Multiple variables (num_vars >= 2) - create Q-Q plot for each
        # Validate layout early for better error messages
        if layout not in _VALID_LAYOUTS:
            raise ValueError(f"layout must be one of {_VALID_LAYOUTS}, got '{layout}'")
        
        _qqs_from_df(
            plot_data=plot_data,
            level=level,
            color=color,
            lcolor=lcolor,
            band_color=band_color,
            layout=layout,
            title=title if title else "Q-Q Plots",
            figsize=figsize,
            grid=grid,
            **kwargs
        )
        return None
    
    # Mode 1: Single vector/array/Series
    if isinstance(input_data, pd.Series):
        label = str(input_data.name) if input_data.name else "Variable"
        values = input_data.values
    elif isinstance(input_data, np.ndarray):
        arr = np.asarray(input_data)
        if arr.ndim == 1:
            label = "Variable"
            values = arr
        elif arr.ndim == 2 and arr.shape[1] == 1:
            label = "Variable"
            values = arr[:, 0]
        else:
            raise ValueError(
                "For single Q-Q plot, array must be 1D or 2D with single column."
            )
    else:
        raise TypeError(
            "qq() expects a fitted model, vector/Series, DataFrame, or string (column name or formula)."
        )
    
    return _qq_single(
        values,
        label=label,
        level=level,
        color=color,
        lcolor=lcolor,
        band_color=band_color,
        title=title if title else "Q-Q Plot",
        xlab=xlab,
        ylab=ylab,
        figsize=figsize,
        grid=grid,
        show=show,
        **kwargs
    )


# ======================================================================
# Single Q-Q plot
# ======================================================================

def _qq_single(
    values: Union[np.ndarray, pd.Series],
    label: str = "Variable",
    level: Optional[float] = 0.95,
    color: str = "blue",
    lcolor: str = "red",
    band_color: str = "blue",
    title: str = "Q-Q Plot",
    xlab: str = "Theoretical Quantiles",
    ylab: str = "Sample Quantiles",
    figsize: Tuple[float, float] = (10, 6),
    grid: bool = True,
    show: bool = True,
    **kwargs
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """Create a single Q-Q plot."""
    # Clean data
    v = _as_1d_numeric(values)
    
    if v.size < 2:
        raise ValueError("Need at least 2 finite observations for a Q-Q plot.")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Map level -> alpha for statsmodels (alpha is significance level, not confidence level)
    qq_alpha = None if level is None else (1.0 - float(level))
    
    # Statsmodels draws points + reference line + optional CI band
    pp = ProbPlot(v, fit=True)
    pp.qqplot(line="45", alpha=qq_alpha, ax=ax, **kwargs)
    
    # Style the plot elements (robust to statsmodels version differences)
    lines = ax.get_lines()
    
    try:
        # First line: scatter points (markers)
        if len(lines) >= 1:
            lines[0].set_markerfacecolor(color)
            lines[0].set_markeredgecolor(color)
            lines[0].set_color(color)
        
        # Second line: reference line (45-degree line)
        if len(lines) >= 2:
            lines[1].set_color(lcolor)
        
        # Third/fourth lines: confidence band (only if level was specified)
        if level is not None and len(lines) >= 3:
            for i in range(2, len(lines)):
                lines[i].set_color(band_color)
                lines[i].set_alpha(0.3)
    except (AttributeError, IndexError):
        # Styling failed - plot will still show with default colors
        pass
    
    # Labels and styling
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    
    if grid:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig, ax


# ======================================================================
# Multiple Q-Q plots
# ======================================================================

def _qqs_from_df(
    plot_data: pd.DataFrame,
    level: Optional[float] = 0.95,
    color: str = "blue",
    lcolor: str = "red",
    band_color: str = "blue",
    layout: Literal["column", "row", "matrix"] = "matrix",
    title: str = "Q-Q Plots",
    figsize: Tuple[float, float] = (5, 5),
    grid: bool = True,
    **kwargs
) -> None:
    """
    Create multiple Q-Q plots for variables in a DataFrame.
    
    Parameters
    ----------
    plot_data : pd.DataFrame
        DataFrame with numeric columns to plot. Should be numeric-only with
        inf values replaced by NaN (use _clean_numeric_df). Each column is
        cleaned further by dropping NaNs before plotting.
    level : Optional[float]
        Confidence level
    color : str
        Point color
    lcolor : str
        Reference line color
    band_color : str
        Confidence band color
    layout : Literal["column", "row", "matrix"]
        Layout arrangement
    title : str
        Plot title
    figsize : Tuple[float, float], default=(5, 5)
        Total figure size (width, height) in inches.
    grid : bool
        Whether to show grid
    **kwargs
        Additional arguments for qqplot()
    
    Notes
    -----
    - Always shows and closes figure (no return value)
    - Each column is cleaned independently (NaNs dropped per variable)
    - Defensively cleans input to ensure numeric-only data
    """
    # Validate layout first (fail fast before expensive operations)
    if layout not in _VALID_LAYOUTS:
        raise ValueError(f"layout must be one of {_VALID_LAYOUTS}, got '{layout}'")
    
    # Defensive: ensure numeric-only data (cheap and prevents future bugs)
    plot_data = _clean_numeric_df(plot_data)
    
    num_vars = plot_data.shape[1]
    
    if num_vars == 0:
        raise ValueError("No variables to plot.")
    
    # Determine layout
    if layout == "column":
        nrows, ncols = num_vars, 1
    elif layout == "row":
        nrows, ncols = 1, num_vars
    else:  # layout == "matrix"
        nrows = int(np.ceil(np.sqrt(num_vars)))
        ncols = int(np.ceil(num_vars / nrows))
    
    # Use figsize as total figure size (not per-subplot)
    final_figsize = figsize
    
    fig, axes = plt.subplots(nrows, ncols, figsize=final_figsize)
    axes = np.array(axes).reshape(-1)
    
    fig.suptitle(title, fontsize=16)
    
    # Map level to alpha
    qq_alpha = None if level is None else (1.0 - float(level))
    
    for i, var in enumerate(plot_data.columns):
        ax = axes[i]
        v = _as_1d_numeric(plot_data[var].to_numpy())
        
        if v.size < 2:
            ax.set_title(f"Q-Q Plot of {var}\n(insufficient data)")
            ax.axis("off")
            continue
        
        # Create Q-Q plot
        pp = ProbPlot(v, fit=True)
        pp.qqplot(line="45", alpha=qq_alpha, ax=ax, **kwargs)
        
        # Style (robust to statsmodels version differences)
        try:
            lines = ax.get_lines()
            if len(lines) >= 1:
                lines[0].set_markerfacecolor(color)
                lines[0].set_markeredgecolor(color)
                lines[0].set_color(color)
            if len(lines) >= 2:
                lines[1].set_color(lcolor)
            if level is not None and len(lines) >= 3:
                for j in range(2, len(lines)):
                    lines[j].set_color(band_color)
                    lines[j].set_alpha(0.3)
        except (AttributeError, IndexError):
            # Styling failed - plot will still show with default colors
            pass
        
        ax.set_title(f"Q-Q Plot of {var}")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        
        if grid:
            ax.grid(True, alpha=0.3)
    
    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close(fig)


# ======================================================================
# Alias for backward compatibility
# ======================================================================

def plot_qq(*args, **kwargs) -> Any:
    """Alias for qq(). Use qq() instead."""
    return qq(*args, **kwargs)

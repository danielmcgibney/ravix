from __future__ import annotations

from typing import Optional, Union, List, Tuple, Any
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ravix.modeling.parse_formula import parse_formula
from ravix.plots._theme import get_theme, _resolve_figsize
from ravix.plots._utils import (
    _ensure_no_intercept,
    _coerce_color_str,
    _all_color_like,
    _get_residuals,
    _detect_model_type,
)

# ======================================================================
# Public API
# ======================================================================

def plot(
    input_data: Union[pd.DataFrame, str, Any],
    data: Optional[pd.DataFrame] = None,
    color: Union[str, List[Any], np.ndarray] = "blue",
    lines: bool = False,
    smooth: bool = True,
    res: str = "resid",
    title: Optional[str] = None,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    psize: int = 50,
    alpha: float = 1.0,
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = True,
    diag: str = "label",
    **kwargs
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Unified scatter-based plotting interface for:
      - Single X–Y scatter plots
      - Scatter plot matrix (pairplot) for multiple variables
      - Residual diagnostic plot for fitted models

    Parameters
    ----------
    input_data : Union[pd.DataFrame, str, Any]
        Formula string (e.g., "Y ~ X"), DataFrame, or fitted model
    data : Optional[pd.DataFrame], default=None
        Data for formula evaluation
    color : Union[str, List[Any], np.ndarray], default="blue"
        Point color(s) - string, color array, or category labels
    lines : bool, default=False
        Add regression lines in scatter plot matrix
    smooth : bool, default=True
        Add LOWESS smooth to residual plots
    res : str, default="resid"
        Residual type: "resid", "pearson", "deviance", "studentized"
    title : Optional[str], default=None
        Plot title
    xlab : Optional[str], default=None
        X-axis label
    ylab : Optional[str], default=None
        Y-axis label
    psize : int, default=50
        Point size
    alpha : float, default=1.0
        Point transparency (0-1)
    figsize : Tuple[float, float], default=(10, 6)
        Figure dimensions
    show : bool, default=True
        Display plot immediately; if False, return (fig, ax) for manual control
    diag : str, default="label"
        Diagonal panel content for scatter plot matrix:
        "label"   – large variable name displayed as text (default)
        "hist"    – histogram of each variable
        "density" – kernel density estimate

    Returns
    -------
    Optional[Tuple[plt.Figure, plt.Axes]]
        None if show=True, else (fig, ax) tuple
        
        WARNING: If show=False, caller is responsible for closing the figure
        to prevent memory leaks. Use plt.close(fig) when done.

    Examples
    --------
    >>> # Simple scatter plot
    >>> plot("mpg ~ hp", data=mtcars)
    
    >>> # Add regression line using abline
    >>> plot("mpg ~ hp", data=mtcars, show=False)
    >>> abline(fitted_model)
    >>> plt.show()
    
    >>> # Residual diagnostics
    >>> fitted_model = fit("y ~ x", data=df)
    >>> plot(fitted_model, res="studentized")
    
    >>> # Scatter plot matrix
    >>> plot(df[["x1", "x2", "y"]], lines=True)
    
    >>> # Two-column DataFrame (X first, Y second)
    >>> plot(df[["x", "y"]])
    
    >>> # Manual figure management
    >>> fig, ax = plot("y ~ x", data=df, show=False)
    >>> ax.set_title("Custom Title")
    >>> plt.savefig("myplot.png")
    >>> plt.close(fig)  # Important!

    Notes
    -----
    - Raw arrays / Series are intentionally not supported here. Use:
        - plot("Y ~ X", data=df) or plot(df[["X","Y"]]) for relationships
        - hist(...) for distributions
    - For 2-column DataFrames: first column is X-axis, second column is Y-axis
    - For multi-column DataFrames (scatter matrix): columns should be ordered as predictors then response (X1, X2, ..., Y)
    - Formula strings can be parsed with data=None if your parse_formula supports
      environment lookup, but providing data is recommended.
    - Use abline() to add regression lines to scatter plots in a layered fashion
    """
    # Better guidance for unsupported inputs
    # Allow arrays/Series that have .resid (duck-typed fitted models)
    if isinstance(input_data, (np.ndarray, pd.Series)) and not hasattr(input_data, "resid"):
        raise TypeError(
            "plot() does not accept a raw array/Series.\n"
            "Use one of:\n"
            "  - plot('Y ~ X', data=df)\n"
            "  - plot(df[['X','Y']])\n"
            "  - plot(fitted_model) for residual diagnostics\n"
            "  - hist(series_or_array) for distributions"
        )

    # Resolve figsize via theme (once, before dispatching to sub-functions)
    figsize = _resolve_figsize(figsize)

    # Normalize res parameter once at entry (this is the only entry point to _plot_res)
    res_normalized = str(res).lower().strip()

    # Case 1: fitted model input → residual diagnostics
    if hasattr(input_data, "resid"):
        return _plot_res(
            input_data,
            res=res_normalized,
            color=_coerce_color_str(color, default="blue"),
            title=title if title else "Residual Plot",
            xlab=xlab if xlab else "Fitted values",
            ylab=ylab if ylab else "Residuals",
            smooth=smooth,
            figsize=figsize,
            show=show,
            **kwargs,
        )

    # Case 2: DataFrame input
    if isinstance(input_data, pd.DataFrame):
        if input_data.shape[1] < 2:
            raise ValueError("DataFrame passed to plot() must contain at least two columns.")

        if input_data.shape[1] == 2:
            return _plot_xy(
                input_data,
                color=color,
                xlab=xlab,
                ylab=ylab,
                title=title,
                psize=psize,
                alpha=alpha,
                figsize=figsize,
                show=show,
                **kwargs,
            )

        _plots(
            input_data,
            color=_coerce_color_str(color, default="blue"),
            lines=lines,
            title=title if title else "Scatter Plot Matrix",
            figsize=figsize,
            diag=diag,
            **kwargs,
        )
        return None

    # Case 3: Formula input
    if isinstance(input_data, str):
        # Parse formula once here and pass parsed data down
        try:
            formula_for_parse = _ensure_no_intercept(input_data)
            Y_out, X_out = parse_formula(formula_for_parse, data)
        except Exception as e:
            raise ValueError(
                "Could not parse formula. If you passed a formula string, either:\n"
                "  - provide `data=...`, or\n"
                "  - ensure variables exist in the calling environment (if supported by parse_formula)."
            ) from e

        if getattr(X_out, "shape", (0, 0))[1] == 0:
            raise ValueError("Formula must be of the form 'Y ~ X' or 'Y ~ X1 + X2'.")

        if Y_out is None:
            # No LHS (e.g. "~ X1 + X2") — treat all RHS columns as the variables to plot
            _plots(
                X_out,
                color=_coerce_color_str(color, default="blue"),
                lines=lines,
                title=title if title else "Scatter Plot Matrix",
                figsize=figsize,
                diag=diag,
                **kwargs,
            )
            return None

        if X_out.shape[1] == 1:
            y_name = Y_out.name if isinstance(Y_out, pd.Series) else "Y"
            x_name = X_out.columns[0]
            Y_series = Y_out if isinstance(Y_out, pd.Series) else pd.Series(Y_out, name=y_name, index=X_out.index)
            plot_data = pd.DataFrame({x_name: X_out.iloc[:, 0], y_name: Y_series})
            return _plot_xy(
                plot_data,
                color=color,
                xlab=xlab,
                ylab=ylab,
                title=title,
                psize=psize,
                alpha=alpha,
                figsize=figsize,
                show=show,
                **kwargs,
            )

        y_name = Y_out.name if isinstance(Y_out, pd.Series) else "Y"
        Y_series = Y_out if isinstance(Y_out, pd.Series) else pd.Series(Y_out, name=y_name, index=X_out.index)
        plot_data = pd.concat([Y_series, X_out], axis=1)
        _plots(
            plot_data,
            color=_coerce_color_str(color, default="blue"),
            lines=lines,
            title=title if title else "Scatter Plot Matrix",
            figsize=figsize,
            diag=diag,
            **kwargs,
        )
        return None

    raise TypeError("plot() expects a formula string, a pandas DataFrame, or a fitted regression model.")


# ======================================================================
# Single X–Y plot
# ======================================================================

def _plot_xy(
    data: pd.DataFrame,
    color: Union[str, List[Any], np.ndarray] = "blue",
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    title: Optional[str] = None,
    psize: int = 50,
    alpha: float = 1.0,
    figsize: Tuple[float, float] = (10, 6),
    show: bool = True,
    **kwargs
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Single predictor scatter plot (data only, no regression lines).

    Expects a 2-column DataFrame (X first, Y second).
    Use abline() to add regression lines after plotting.
      
    Examples
    --------
    >>> # DataFrame input
    >>> _plot_xy(df[["size", "price"]])
    
    >>> # With layered regression line
    >>> _plot_xy(plot_data, show=False)
    >>> abline(fitted_model)
    >>> plt.show()
    
    >>> # Categorical colors
    >>> _plot_xy(plot_data, color=df["category"])
    """
    if data.shape[1] != 2:
        raise ValueError("_plot_xy expects a DataFrame with exactly 2 columns (X, Y).")
    
    _theme = get_theme()
    title_fontsize = _theme["title_fontsize"]
    label_fontsize = _theme["label_fontsize"]
    tick_fontsize  = _theme["tick_fontsize"]

    cols = data.columns.tolist()
    X_out = data[[cols[0]]]
    Y_out = data[cols[1]]

    # Get actual column name for labels
    x_colname = X_out.columns[0]

    fig, ax = plt.subplots(figsize=figsize)

    # Color may be str or array-like
    legend_labels = kwargs.pop("legend_labels", None)
    _scatter_xy(
        ax,
        X_out.values.flatten(),
        Y_out,
        color=color,
        psize=psize,
        alpha=alpha,
        legend_labels=legend_labels,
        **kwargs,
    )

    ax.set_xlabel(xlab if xlab is not None else x_colname, fontsize=label_fontsize)
    ax.set_ylabel(ylab if ylab is not None else getattr(Y_out, "name", "Y"), fontsize=label_fontsize)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    ax.tick_params(labelsize=tick_fontsize)

    # Only show legend if something labeled exists
    if ax.get_legend_handles_labels()[1]:
        ax.legend()
    
    # Store X variable name in axes metadata for abline() to use
    ax._ravix_x_var = x_colname

    plt.tight_layout()
    if show:
        pass
        return None
    return fig, ax


def _scatter_xy(
    ax: plt.Axes,
    x: np.ndarray,
    y: Union[np.ndarray, pd.Series],  # Accept Series
    *,
    color: Union[str, List[Any], np.ndarray],
    psize: int,
    alpha: float,
    legend_labels: Optional[List[str]] = None,
    **kwargs
) -> None:
    """
    Draw scatter with:
      - color as single string
      - OR array of explicit colors
      - OR array of category labels (mapped to palette)
      
    Examples
    --------
    >>> # Single color
    >>> _scatter_xy(ax, x_vals, y_vals, color="blue", psize=50, alpha=0.7)
    
    >>> # Per-point colors
    >>> colors = ["red" if v > 0 else "blue" for v in y_vals]
    >>> _scatter_xy(ax, x_vals, y_vals, color=colors, psize=50, alpha=0.7)
    
    >>> # Categorical
    >>> categories = ["A", "B", "A", "C", ...]
    >>> _scatter_xy(ax, x_vals, y_vals, color=categories, psize=50, alpha=0.7)
    """
    # Convert y to array for sns.scatterplot if it's a Series
    y_array = np.asarray(y)
    
    if isinstance(color, str):
        sns.scatterplot(x=x, y=y_array, color=color, s=psize, alpha=alpha, ax=ax, **kwargs)
        return

    c = np.asarray(color)
    if c.shape[0] != len(x):
        raise ValueError("If `color` is an array/list, it must have the same length as the data.")

    # If every entry looks like a matplotlib color, use per-point colors via matplotlib
    if _all_color_like(c):
        ax.scatter(x, y_array, c=c, s=psize, alpha=alpha, edgecolors="none")
        return

    # Otherwise treat as categories → palette mapping + legend
    df = pd.DataFrame({"x": x, "y": y_array, "group": c.astype("object")})
    groups = pd.unique(df["group"])
    palette = dict(zip(groups, sns.color_palette(n_colors=len(groups))))

    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="group",
        palette=palette,
        s=psize,
        alpha=alpha,
        ax=ax,
        **kwargs,
    )

    # Optional custom legend labels overriding group names
    if legend_labels and len(legend_labels) == len(groups):
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=legend_labels, title=None)
    else:
        ax.legend(title=None)


# ======================================================================
# Scatter plot matrix
# ======================================================================

def _plots(
    data: pd.DataFrame,
    color: str = "blue",
    lines: bool = False,
    title: str = "Scatter Plot Matrix",
    figsize: Optional[Tuple[float, float]] = None,
    diag: str = "label",
    **kwargs
) -> None:
    """
    Scatter plot matrix with full control over diagonal panels and typography.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame whose columns are the variables to plot.
    color : str, default="blue"
        Point and histogram/density color.
    lines : bool, default=False
        Overlay OLS regression lines on off-diagonal scatter panels.
    title : str, default="Scatter Plot Matrix"
        Figure suptitle.
    figsize : Optional[Tuple[float, float]], default=None
        Overall figure size; auto-computed from n columns if None.
    diag : str, default="label"
        Diagonal panel content:
        "label"   – large, bold variable name centred in the panel (default)
        "hist"    – histogram of each variable
        "density" – kernel density estimate

    Examples
    --------
    >>> _plots(df[["x1", "x2", "y"]])
    >>> _plots(plot_data, lines=True, diag="hist")
    >>> _plots(df, color="darkblue", diag="density", figsize=(12, 12))
    """
    diag = str(diag).lower().strip()
    if diag not in ("label", "hist", "density"):
        raise ValueError("`diag` must be one of 'label', 'hist', or 'density'.")

    if data.shape[1] < 2:
        raise ValueError("Need at least 2 columns for a scatter plot matrix.")

    cols = data.columns.tolist()
    n = len(cols)

    # ── figure sizing ──────────────────────────────────────────────────────
    if figsize is None:
        # Use theme figsize as base, then scale by n; fall back to rc default
        theme_base = get_theme().get("figsize") or plt.rcParams["figure.figsize"]
        cell = max(2.2, theme_base[0] / n)
        figsize = (cell * n, cell * n)

    fig, axes = plt.subplots(n, n, figsize=figsize)
    if n == 1:
        axes = np.array([[axes]])

    # ── shared tick / font settings ─────────────────────────────────────────
    _theme = get_theme()
    TICK_FONTSIZE  = _theme["tick_fontsize"]
    DIAG_FONTSIZE  = _theme["diag_fontsize"]
    TITLE_FONTSIZE = _theme["title_fontsize"]

    # ── strip kwargs that only make sense for seaborn pairplot ─────────────
    _scatter_kw = {k: v for k, v in kwargs.items()
                   if k not in ("height", "aspect", "plot_kws", "diag_kws")}

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            col_i = cols[i]   # row variable  → y-axis
            col_j = cols[j]   # column variable → x-axis

            # ── diagonal ──────────────────────────────────────────────────
            if i == j:
                if diag == "label":
                    ax.set_axis_off()
                    ax.text(
                        0.5, 0.5, col_i,
                        transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=DIAG_FONTSIZE,
                        fontweight="bold",
                        color="black",
                        wrap=True,
                    )

                elif diag == "hist":
                    ax.hist(
                        data[col_i].dropna(),
                        color=color,
                        edgecolor="white",
                        linewidth=0.4,
                    )
                    ax.set_xlabel("")
                    ax.set_ylabel("")

                else:  # density
                    try:
                        from scipy.stats import gaussian_kde
                        vals = data[col_i].dropna().values.astype(float)
                        kde = gaussian_kde(vals)
                        xs = np.linspace(vals.min(), vals.max(), 200)
                        ax.plot(xs, kde(xs), color=color, linewidth=1.8)
                        ax.fill_between(xs, kde(xs), alpha=0.15, color=color)
                    except Exception:
                        # Fall back to histogram if scipy unavailable
                        ax.hist(data[col_i].dropna(), color=color,
                                edgecolor="white", linewidth=0.4)
                    ax.set_xlabel("")
                    ax.set_ylabel("")

            # ── off-diagonal scatter ───────────────────────────────────────
            else:
                x_vals = data[col_j].values
                y_vals = data[col_i].values
                ax.scatter(
                    x_vals, y_vals,
                    c=color, s=18, alpha=0.6,
                    linewidths=0,
                    **_scatter_kw,
                )

                if lines:
                    try:
                        mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
                        if mask.sum() > 2:
                            m, b = np.polyfit(x_vals[mask], y_vals[mask], 1)
                            xlim = ax.get_xlim()
                            xs = np.array(xlim)
                            ax.plot(xs, m * xs + b,
                                    color="black", linewidth=1.2, zorder=3)
                            ax.set_xlim(xlim)
                    except Exception:
                        pass

            # ── tick formatting ────────────────────────────────────────────
            ax.tick_params(
                axis="both",
                labelsize=TICK_FONTSIZE,
                length=3,
                width=0.6,
                pad=2,
            )
            # Reduce tick density to avoid crowding
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune="both"))
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune="both"))

            # ── no axis labels on any panel ───────────────────────────────
            ax.set_xlabel("")
            ax.set_ylabel("")
            if i < n - 1:
                ax.tick_params(labelbottom=False)
            else:
                # Bottom row: rotate to prevent overlap with large numbers
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            if j > 0:
                ax.tick_params(labelleft=False)

    plt.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    plt.tight_layout(pad=0.4, h_pad=0.3, w_pad=0.3, rect=[0, 0, 1, 0.97])
    plt.show()
    plt.close(fig)



# ======================================================================
# Residual plot
# ======================================================================

def _plot_res(
    model,
    res: str = "resid",
    color: str = "blue",
    title: str = "Residual Plot",
    xlab: str = "Fitted values",
    ylab: str = "Residuals",
    smooth: bool = True,
    figsize: Tuple[float, float] = (10, 6),
    show: bool = True,
    **kwargs
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Residual plot for a fitted model.
    
    Parameters
    ----------
    model : fitted model object
        Must have .resid attribute or relevant residual attributes
    res : str, default="resid"
        Residual type: "resid", "pearson", "deviance", "studentized", etc.
    color : str, default="blue"
        Point color
    title : str, default="Residual Plot"
        Plot title
    xlab : str, default="Fitted values"
        X-axis label
    ylab : str, default="Residuals"
        Y-axis label
    smooth : bool, default=True
        Add LOWESS smoothing (auto-subsampled for large datasets)
    figsize : Tuple[float, float], default=(10, 6)
        Figure dimensions
    show : bool, default=True
        Display plot immediately
    **kwargs
        Additional arguments passed to sns.scatterplot
        
    Returns
    -------
    Optional[Tuple[plt.Figure, plt.Axes]]
        None if show=True, else (fig, ax)
        
    Examples
    --------
    >>> # Basic residual plot
    >>> model = fit("y ~ x", data=df)
    >>> _plot_res(model)
    
    >>> # Studentized residuals for outlier detection
    >>> _plot_res(model, res="studentized")
    
    >>> # For logistic regression
    >>> logit_model = fit("y ~ x", data=df, family="binomial")
    >>> _plot_res(logit_model, res="deviance")

    Notes
    -----
    - For logistic/GLM, fitted values are the fitted mean response (probabilities/means),
      not the linear predictor. Labels reflect that.
    - LOWESS smoothing is automatically subsampled for datasets > 5000 points
    - Normalizes res parameter internally for defensive programming.
    - Reference lines are drawn in gray color.
    """
    # Normalize res parameter defensively (in case called directly)
    res = str(res).lower().strip()
    
    # Default line color for reference lines
    color0 = "gray"
    lcolor = "red"

    _theme = get_theme()
    title_fontsize = _theme["title_fontsize"]
    label_fontsize = _theme["label_fontsize"]
    tick_fontsize  = _theme["tick_fontsize"]
    
    model_type = _detect_model_type(model)
    residuals = _get_residuals(model, res)
    fitted = _get_fitted_values(model, model_type)
    plot_title, x_label, y_label = _adjust_labels(title, xlab, ylab, model_type, res)

    fig, ax = plt.subplots(figsize=figsize)

    sns.scatterplot(
        x=fitted,
        y=residuals,
        color=color,
        alpha=0.6,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
        **kwargs,
    )

    ax.axhline(0, color=color0, linestyle="--", linewidth=1.5, alpha=0.8, label="Reference (e=0)")

    if model_type in ["logistic", "glm"] and res in ["pearson", "deviance"]:
        ax.axhline(2, color=color0, linestyle=":", linewidth=1, alpha=0.5, label="±2 bounds")
        ax.axhline(-2, color=color0, linestyle=":", linewidth=1, alpha=0.5)

    # ======================================================================
    # Subsample for LOWESS if dataset is large
    # ======================================================================
    if smooth and len(fitted) > 10:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            
            # Subsample if dataset is large (>5000 points)
            MAX_LOWESS_POINTS = 5000
            if len(fitted) > MAX_LOWESS_POINTS:
                # Use local RNG to avoid mutating global random state
                rng = np.random.default_rng(42)
                indices = rng.choice(len(fitted), MAX_LOWESS_POINTS, replace=False)
                fitted_sample = fitted[indices]
                resid_sample = residuals[indices]
                
                smoothed = lowess(resid_sample, fitted_sample, frac=2/3, return_sorted=True)
            else:
                smoothed = lowess(residuals, fitted, frac=2/3, return_sorted=True)
                
            ax.plot(smoothed[:, 0], smoothed[:, 1], color=lcolor, linewidth=2, alpha=0.8, label="LOWESS smooth")
        except ImportError:
            warnings.warn(
                "Could not add smoothing line. Install statsmodels for LOWESS smoothing.",
                UserWarning,
                stacklevel=2
            )
        except Exception as e:
            warnings.warn(
                f"Could not compute LOWESS smoothing: {e}",
                UserWarning,
                stacklevel=2
            )

    ax.set_xlabel(x_label, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_title(plot_title, fontsize=title_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)

    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig, ax

# ======================================================================
# Model diagnostics helpers
# ======================================================================

def _get_fitted_values(model, model_type: str) -> np.ndarray:
    """
    Extract fitted values from model.
    
    Parameters
    ----------
    model : fitted model object
        Model to extract fitted values from
    model_type : str
        Model type from _detect_model_type
        
    Returns
    -------
    np.ndarray
        Fitted values array
        
    Raises
    ------
    AttributeError
        If model doesn't provide fitted values through any known method
        
    Examples
    --------
    >>> fitted = _get_fitted_values(model, "linear")
    
    Notes
    -----
    Return fitted values consistent with labels:
    - linear: fittedvalues
    - logistic / glm: fitted mean response (default)
    """
    if hasattr(model, "fittedvalues"):
        return np.asarray(model.fittedvalues)

    if hasattr(model, "predict"):
        try:
            return np.asarray(model.predict())
        except Exception as e:
            # Warn about predict() failure but continue to try alternative methods
            warnings.warn(
                f"Model.predict() failed: {e}. Attempting alternative methods.",
                UserWarning,
                stacklevel=2
            )

    if hasattr(model, "model") and hasattr(model.model, "exog") and hasattr(model, "params"):
        try:
            return np.asarray(model.model.exog @ model.params)
        except Exception as e:
            raise AttributeError(
                f"Could not compute fitted values from model.exog @ params: {e}"
            ) from e

    raise AttributeError(
        "Model must have one of: `fittedvalues` attribute, `predict()` method, "
        "or (`model.exog` and `params` attributes)."
    )

def _adjust_labels(
    title: str,
    xlab: str,
    ylab: str,
    model_type: str,
    res_type: str
) -> Tuple[str, str, str]:
    """
    Adjust plot labels based on model type and residual type.
    
    Parameters
    ----------
    title : str
        Original title
    xlab : str
        Original x-label
    ylab : str
        Original y-label
    model_type : str
        Model type from _detect_model_type
    res_type : str
        Residual type
        
    Returns
    -------
    Tuple[str, str, str]
        (adjusted_title, adjusted_xlabel, adjusted_ylabel)
        
    Examples
    --------
    >>> _adjust_labels("Residual Plot", "Fitted values", "Residuals", "logistic", "deviance")
    ('Logistic Regression Deviance Residual Plot', 'Fitted values', 'Deviance Residuals')

    Notes
    -----
    - For logistic, x-axis label remains 'Fitted values' (probabilities),
      not linear predictor, since we plot fitted mean response by default.
    - res_type should already be normalized (lowercase, stripped) by caller.
    """
    # Title
    if title == "Residual Plot":
        if model_type == "logistic":
            plot_title = "Logistic Regression Residual Plot" if res_type == "resid" else f"Logistic Regression {res_type.title()} Residual Plot"
        elif model_type == "glm":
            plot_title = "GLM Residual Plot" if res_type == "resid" else f"GLM {res_type.title()} Residual Plot"
        else:
            plot_title = "Residual Plot" if res_type == "resid" else f"{res_type.title()} Residual Plot"
    else:
        plot_title = title

    # X label
    x_label = "Fitted values" if xlab == "Fitted values" else xlab

    # Y label
    if ylab == "Residuals":
        y_label = "Residuals" if res_type == "resid" else f"{res_type.title()} Residuals"
    else:
        y_label = ylab

    return plot_title, x_label, y_label

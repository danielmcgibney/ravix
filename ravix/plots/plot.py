from __future__ import annotations

from typing import Optional, Union, List, Tuple, Any
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ravix.modeling.parse_formula import parse_formula
from ravix.plots._utils import (
    _ensure_no_intercept,
    _coerce_color_str,
    _normalize_models,
    _normalize_line_colors,
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
    model: Optional[Union[Any, List[Any]]] = None,
    color: Union[str, List[Any], np.ndarray] = "blue",
    lcolor: Union[str, List[str]] = "red",
    lines: bool = False,
    smooth: bool = True,
    res: str = "resid",
    title: Optional[str] = None,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    psize: int = 50,
    alpha: float = 1.0,
    figsize: Tuple[float, float] = (10, 6),
    show: bool = True,
    **kwargs
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Unified scatter-based plotting interface for:
      - Single X–Y scatter with optional regression/model lines
      - Scatter plot matrix (pairplot) for multiple variables
      - Residual diagnostic plot for fitted models

    Parameters
    ----------
    input_data : Union[pd.DataFrame, str, Any]
        Formula string (e.g., "Y ~ X"), DataFrame, or fitted model
    data : Optional[pd.DataFrame], default=None
        Data for formula evaluation
    model : Optional[Union[Any, List[Any]]], default=None
        Fitted model(s) or "line"/"l" for regression line
    color : Union[str, List[Any], np.ndarray], default="blue"
        Point color(s) - string, color array, or category labels
    lcolor : Union[str, List[str]], default="red"
        Line color(s) for model predictions
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

    Returns
    -------
    Optional[Tuple[plt.Figure, plt.Axes]]
        None if show=True, else (fig, ax) tuple
        
        WARNING: If show=False, caller is responsible for closing the figure
        to prevent memory leaks. Use plt.close(fig) when done.

    Examples
    --------
    >>> # Simple scatter with regression line
    >>> plot("mpg ~ hp", data=mtcars, model="line")
    
    >>> # Multiple models with custom colors
    >>> plot("y ~ x", data=df, model=[model1, model2], lcolor=["red", "blue"])
    
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

    # Normalize res parameter once at entry (this is the only entry point to _plot_res)
    res_normalized = str(res).lower().strip()

    # Case 1: fitted model input → residual diagnostics
    if hasattr(input_data, "resid"):
        return _plot_res(
            input_data,
            res=res_normalized,
            color=_coerce_color_str(color, default="blue"),
            lcolor=_coerce_color_str(lcolor, default="red"),
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
                model=model,
                color=color,
                lcolor=lcolor,
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
            lcolor=_coerce_color_str(lcolor, default="black"),
            title=title if title else "Scatter Plot Matrix",
            figsize=figsize,
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

        if X_out.shape[1] == 1:
            # Pass parsed data to avoid re-parsing
            # _plot_xy expects (X, Y) order
            # Preserve index alignment from original data
            y_name = getattr(Y_out, "name", "Y")
            x_name = X_out.columns[0]
            plot_data = pd.DataFrame({
                x_name: X_out.iloc[:, 0],
                y_name: pd.Series(np.asarray(Y_out).ravel(), index=X_out.index)
            })
            return _plot_xy(
                plot_data,
                model=model,
                color=color,
                lcolor=lcolor,
                xlab=xlab,
                ylab=ylab,
                title=title,
                psize=psize,
                alpha=alpha,
                figsize=figsize,
                show=show,
                **kwargs,
            )

        # Pass parsed data to avoid re-parsing
        # Build as X columns + Y column for consistent ordering
        # Preserve index alignment from original data
        y_name = getattr(Y_out, "name", "Y")
        plot_data = pd.concat(
            [X_out, pd.Series(np.asarray(Y_out).ravel(), index=X_out.index, name=y_name)],
            axis=1
        )
        _plots(
            plot_data,
            color=_coerce_color_str(color, default="blue"),
            lines=lines,
            lcolor=_coerce_color_str(lcolor, default="black"),
            title=title if title else "Scatter Plot Matrix",
            figsize=figsize,
            **kwargs,
        )
        return None

    raise TypeError("plot() expects a formula string, a pandas DataFrame, or a fitted regression model.")


# ======================================================================
# Single X–Y plot
# ======================================================================

def _plot_xy(
    data: pd.DataFrame,
    model: Optional[Union[Any, List[Any]]] = None,
    color: Union[str, List[Any], np.ndarray] = "blue",
    lcolor: Union[str, List[str]] = "red",
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
    Single predictor scatter plot with optional regression/model line(s).

    Expects a 2-column DataFrame (X first, Y second).
      
    Examples
    --------
    >>> # DataFrame input
    >>> _plot_xy(df[["size", "price"]])
    
    >>> # With fitted model line
    >>> _plot_xy(plot_data, model=fitted_model, lcolor="green")
    
    >>> # Categorical colors
    >>> _plot_xy(plot_data, color=df["category"])
    """
    if data.shape[1] != 2:
        raise ValueError("_plot_xy expects a DataFrame with exactly 2 columns (X, Y).")
    
    cols = data.columns.tolist()
    X_out = data[[cols[0]]]
    Y_out = data[cols[1]]

    models = _normalize_models(model)               # [] not [None]
    lcolors = _normalize_line_colors(lcolor, n=len(models))
    
    # Get actual column name and base variable name for labels
    x_colname = X_out.columns[0]

    fig, ax = plt.subplots(figsize=figsize)

    # Color may be str or array-like
    legend_labels = kwargs.pop("legend_labels", None)
    _scatter_xy(
        ax,
        X_out.values.flatten(),
        Y_out,  # Keep as pandas Series if that's what it is
        color=color,
        psize=psize,
        alpha=alpha,
        legend_labels=legend_labels,
        **kwargs,
    )

    # Plot lines (models)
    for idx, (mdl, line_color) in enumerate(zip(models, lcolors)):
        label = None
        if isinstance(legend_labels, list) and idx < len(legend_labels):
            label = legend_labels[idx]

        if isinstance(mdl, str) and mdl.lower() in ["line", "l"]:
            sns.regplot(
                x=X_out.values.flatten(),
                y=Y_out,  # Pass Series directly
                scatter=False,
                ci=None,
                line_kws={"color": line_color, "label": label},
                ax=ax,
            )
            continue

        if mdl is not None:
            # Check for finite values before computing range
            x_vals = X_out.values.ravel()
            if not np.any(np.isfinite(x_vals)):
                warnings.warn(
                    f"Model {idx + 1}: All X values are NaN/Inf. Skipping prediction line.",
                    UserWarning,
                    stacklevel=2
                )
                continue
                
            x_min = float(np.nanmin(x_vals))
            x_max = float(np.nanmax(x_vals))
            
            # Check for degenerate range (all X values are the same)
            if x_min == x_max:
                warnings.warn(
                    f"Model {idx + 1}: All X values are identical. Skipping prediction line.",
                    UserWarning,
                    stacklevel=2
                )
                continue
                
            X_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)

            try:
                # Import predict locally (only needed here)
                from ravix.modeling.predict import predict
                
                # Use actual column name for prediction (e.g., "log(price)" not "price")
                X_pred = pd.DataFrame({x_colname: X_range.flatten()})
                Y_pred = predict(mdl, X_pred)
            except Exception as e:
                warnings.warn(
                    f"Could not generate predictions for model {idx + 1}: {e}",
                    UserWarning,
                    stacklevel=2  # IMPROVEMENT: Better stack level
                )
                continue

            pred_df = pd.DataFrame({"X": X_range.flatten(), "Y": np.asarray(Y_pred).ravel()})
            sns.lineplot(data=pred_df, x="X", y="Y", color=line_color, lw=2, label=label, ax=ax)

    ax.set_xlabel(xlab if xlab is not None else x_colname)
    ax.set_ylabel(ylab if ylab is not None else getattr(Y_out, "name", "Y"))
    if title is not None:
        ax.set_title(title)

    # Only show legend if something labeled exists
    if ax.get_legend_handles_labels()[1]:
        ax.legend()

    plt.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
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
    lcolor: str = "black",
    title: str = "Scatter Plot Matrix",
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> None:
    """
    Scatter plot matrix using seaborn.pairplot.

    - Uses a single color consistently (no post-hoc recoloring).
    - If lines=True, adds regression lines without double-plotting points.
    - Defaults diag_kind to "hist" for stability with discrete columns.
    - Recommended column order: predictor columns followed by response column (X1, X2, ..., Y)
    
    Examples
    --------
    >>> # Basic pairplot
    >>> _plots(df[["x1", "x2", "y"]])
    
    >>> # With regression lines
    >>> _plots(plot_data, lines=True, lcolor="red")
    
    >>> # Custom styling
    >>> _plots(df, color="darkblue", figsize=(12, 12))
    """
    if data.shape[1] < 2:
        raise ValueError("Need at least 2 columns for a scatter plot matrix.")

    plot_data = data

    pair_kwargs = dict(kwargs)
    if figsize is not None:
        n = plot_data.shape[1]
        pair_kwargs.setdefault("height", max(2.0, figsize[1] / n))
        pair_kwargs.setdefault("aspect", max(0.8, figsize[0] / figsize[1]))

    pair_kwargs.setdefault("plot_kws", {})
    pair_kwargs["plot_kws"].setdefault("color", color)

    # diag hist is more robust than kde for discrete / near-constant cols
    pair_kwargs.setdefault("diag_kws", {})
    pair_kwargs["diag_kws"].setdefault("color", color)

    g = sns.pairplot(plot_data, diag_kind="hist", **pair_kwargs)
    plt.suptitle(title, fontsize=16, y=1.02)

    if lines:
        for i in range(len(plot_data.columns)):
            for j in range(len(plot_data.columns)):
                if i == j:
                    continue
                sns.regplot(
                    x=plot_data.columns[j],
                    y=plot_data.columns[i],
                    data=plot_data,
                    ax=g.axes[i, j],
                    scatter=False,
                    ci=None,
                    line_kws={"color": lcolor, "linewidth": 1.5},
                    truncate=False,
                )

    plt.tight_layout()
    plt.show()
    plt.close(g.fig)


# ======================================================================
# Residual plot
# ======================================================================

def _plot_res(
    model,
    res: str = "resid",
    color: str = "blue",
    lcolor: str = "red",
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
    lcolor : str, default="red"
        Reference line color
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
    """
    # Normalize res parameter defensively (in case called directly)
    res = str(res).lower().strip()
    
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

    ax.axhline(0, color=lcolor, linestyle="--", linewidth=1.5, alpha=0.8, label="Reference (y=0)")

    if model_type in ["logistic", "glm"] and res in ["pearson", "deviance"]:
        ax.axhline(2, color=lcolor, linestyle=":", linewidth=1, alpha=0.5, label="±2 bounds")
        ax.axhline(-2, color=lcolor, linestyle=":", linewidth=1, alpha=0.5)

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
                
                smoothed = lowess(resid_sample, fitted_sample, frac=0.3, return_sorted=True)
            else:
                smoothed = lowess(residuals, fitted, frac=0.3, return_sorted=True)
                
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

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
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

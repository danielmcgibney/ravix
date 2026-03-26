from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _find_ravix_axes():
    """Search open figures for axes with Ravix metadata."""
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        for ax in fig.axes:
            if hasattr(ax, "_ravix_x_var"):
                return ax
    return None

def abline(
    model=None,
    a: Optional[float] = None,
    b: Optional[float] = None,
    color: str = "black",
    linetype: str = "-",
    linewidth: float = 1.5,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
):
    """
    Add a regression line to an existing plot (R-style layering).

    This function overlays a fitted regression line onto the current axes,
    enabling layered plot construction similar to base R's abline().

    Parameters
    ----------
    model : fitted model object, optional
        Ravix model with predict() method. Can be passed positionally.
    a : float, optional
        Intercept (used when b is also provided)
    b : float, optional
        Slope (used when a is also provided)
    color : str, default="black"
        Line color
    linetype : str, default="-"
        Matplotlib linestyle ("-", "--", "-.", ":")
    linewidth : float, default=1.5
        Line width
    label : Optional[str], default=None
        Legend label for this line
    ax : Optional[plt.Axes], default=None
        Target axes; if None, uses plt.gca()
    **kwargs
        Additional matplotlib line styling options

    Returns
    -------
    matplotlib.lines.Line2D
        The plotted line object (for legend assembly)

    Examples
    --------
    Add a regression line from a fitted model:
    >>> plot("y ~ x", data=df, show=False)
    >>> abline(fitted_model)  # positional
    >>> plt.show()
    
    Add a line with specific slope and intercept:
    >>> plot("y ~ x", data=df, show=False)
    >>> abline(2, 0.5)  # positional: a=2, b=0.5
    >>> # or abline(a=2, b=0.5)  # keyword
    >>> plt.show()
    
    Simple usage (may require same cell in Jupyter):
    >>> plot("y ~ x", data=df)
    >>> abline(fitted_model)

    Notes
    -----
    - Designed for layering on top of ravix plot()
    - For reliable behavior in all environments, use show=False with plot()
    - In Jupyter notebooks, put plot() and abline() in the same cell
    """

    # ======================================================================
    # Get target axes
    # ======================================================================
    if ax is None:
        ax = plt.gca()
        
        # If current axes doesn't have ravix metadata, search for it
        if not hasattr(ax, "_ravix_x_var"):
            ravix_ax = _find_ravix_axes()
            if ravix_ax is not None:
                ax = ravix_ax

    # ======================================================================
    # Handle positional arguments: abline(2, 3) should work as abline(a=2, b=3)
    # ======================================================================
    # If model is a number and a is also a number, user called abline(intercept, slope)
    if model is not None and a is not None:
        if isinstance(model, (int, float)) and isinstance(a, (int, float)):
            # Shift arguments: model->a, a->b, b is ignored
            a, b = model, a
            model = None

    # ======================================================================
    # CASE 1: Direct a, b specification (y = a + b*x)
    # ======================================================================
    if a is not None and b is not None:
        intercept = float(a)
        slope = float(b)

        x_data = _extract_scatter_x_data(ax)

        if len(x_data) > 0:
            x_min, x_max = min(x_data), max(x_data)
        else:
            x_min, x_max = ax.get_xlim()

        x_vals = np.array([x_min, x_max])
        y_vals = intercept + slope * x_vals

        line = ax.plot(
            x_vals,
            y_vals,
            color=color,
            linestyle=linetype,
            linewidth=linewidth,
            label=label,
            zorder=5,  # Place above scatter (seaborn default is 1)
            **kwargs
        )[0]

        # Expand axis limits if needed
        ax.relim()
        ax.autoscale_view()

        return line

    # ======================================================================
    # CASE 2: Ravix model → use predict()
    # ======================================================================
    if model is not None:

        # Get x variable name from axes metadata OR extract from formula
        x_var_name = None
        
        if hasattr(ax, "_ravix_x_var"):
            x_var_name = ax._ravix_x_var
        elif hasattr(model, 'formula'):
            import re
            formula = str(model.formula)
            if '~' in formula:
                rhs = formula.split('~')[1].strip()
                # Extract first variable name
                match = re.match(r'([a-zA-Z_]\w*)', rhs)
                if match:
                    x_var_name = match.group(1)
        
        if x_var_name is None:
            raise ValueError(
                "Cannot determine X variable. Model must have 'formula' attribute "
                "or plot must have been created with ravix plot()."
            )

        # ------------------------------------------------------------------
        # Get actual plotted x range (not padded axes limits)
        # ------------------------------------------------------------------
        x_data = _extract_scatter_x_data(ax)

        if len(x_data) > 0:
            x_min, x_max = min(x_data), max(x_data)
        else:
            # Fallback to axes limits if no scatter data found
            x_min, x_max = ax.get_xlim()

        # Prediction grid
        x_vals = np.linspace(x_min, x_max, 200)

        # ------------------------------------------------------------------
        # Build prediction dataframe
        # ------------------------------------------------------------------
        X_pred = pd.DataFrame({x_var_name: x_vals})

        # ------------------------------------------------------------------
        # Predict using Ravix predict()
        # ------------------------------------------------------------------
        from ravix.modeling.predict import predict

        try:
            Y_pred = predict(model, X_pred)
        except Exception as e:
            raise ValueError(
                f"Prediction failed inside abline(): {e}"
            )

        y_vals = np.asarray(Y_pred).ravel()

        # ------------------------------------------------------------------
        # Plot on SAME axes (layer)
        # ------------------------------------------------------------------
        line = ax.plot(
            x_vals,
            y_vals,
            color=color,
            linestyle=linetype,
            linewidth=linewidth,
            label=label,
            zorder=5,  # Place above scatter (seaborn default is 1)
            **kwargs
        )[0]

        # Expand axis limits if needed
        ax.relim()
        ax.autoscale_view()

        return line

    # ======================================================================
    # CASE 3: No arguments → fit OLS to scatter data
    # ======================================================================
    x_data, y_data = _extract_scatter_data(ax)

    if len(x_data) == 0:
        raise ValueError(
            "Cannot fit regression line: no scatter data found in current axes.\n"
            "Make sure you called plot() with show=False before calling abline():\n"
            "  fig, ax = plot('Y ~ X', data=df, show=False)\n"
            "  abline()  # auto-fits to scatter data\n"
            "  plt.show()"
        )

    x_arr = np.array(x_data)
    y_arr = np.array(y_data)

    slope, intercept = np.polyfit(x_arr, y_arr, 1)

    x_min, x_max = min(x_data), max(x_data)
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = intercept + slope * x_vals

    line = ax.plot(
        x_vals,
        y_vals,
        color=color,
        linestyle=linetype,
        linewidth=linewidth,
        label=label,
        zorder=5,  # Place above scatter (seaborn default is 1)
        **kwargs
    )[0]

    # Expand axis limits if needed
    ax.relim()
    ax.autoscale_view()

    return line


# ======================================================================
# Helper functions
# ======================================================================

def _extract_scatter_x_data(ax: plt.Axes) -> list:
    """Extract x-coordinates from scatter plot collections and line objects."""
    x_data = []
    
    # Try scatter collections first
    for coll in ax.collections:
        offsets = coll.get_offsets()
        if len(offsets) > 0:
            x_data.extend(offsets[:, 0])
    
    # If no scatter data, try line objects
    if not x_data:
        for line in ax.get_lines():
            xdata = line.get_xdata()
            if len(xdata) > 0:
                x_data.extend(xdata)
    
    return x_data


def _extract_scatter_data(ax: plt.Axes) -> tuple[list, list]:
    """Extract x and y coordinates from scatter plot collections and line objects."""
    x_data, y_data = [], []
    
    # Try scatter collections first
    for coll in ax.collections:
        offsets = coll.get_offsets()
        if len(offsets) > 0:
            x_data.extend(offsets[:, 0])
            y_data.extend(offsets[:, 1])
    
    # If no scatter data, try line objects
    if not x_data:
        for line in ax.get_lines():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if len(xdata) > 0:
                x_data.extend(xdata)
                y_data.extend(ydata)
    
    return x_data, y_data

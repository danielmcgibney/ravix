from __future__ import annotations

from typing import Any


def viz(*args: Any, kind: str = "plot", **kwargs: Any) -> None:
    """
    Unified plotting interface.

    Routes to common plotting functions based on `kind`.

    Parameters
    ----------
    kind : str, default="plot"
        Plot type. Standard options:
        - "plot": Scatter-based plots (single X–Y, scatter matrix, or residual diagnostics)
        - "hist": Histogram(s)
        - "qq": Q–Q plot(s)
        - "bar": Bar plot(s)
        - "box": Box plot(s)
        - "cor": Correlation heatmap
        - "bsr": Best subsets regression visualization

    **kwargs
        Passed through to the underlying plotting function.

    Returns
    -------
    None
        Displays the plot.

    Notes
    -----
    This function intentionally documents only the standard `kind` values to keep
    the public interface simple.
    """
    # Normalize `kind` so users can pass "Hist", "HISTOGRAM", etc.
    k = kind.strip().lower()

    # ------------------------------------------------------------------
    # Standard (public) routing
    # ------------------------------------------------------------------
    if k in {"plot", "scatter", "scatterplot"}:
        from ravix.plots.plot import plot
        return plot(*args, **kwargs)

    if k in {"hist", "histogram"}:
        from ravix.plots.hist import hist
        return hist(*args, **kwargs)

    if k in {"qq", "qqplot"}:
        from ravix.plots.qq import qq
        return qq(*args, **kwargs)

    if k in {"bar", "barplot"}:
        from ravix.plots.barplot import barplot
        return barplot(*args, **kwargs)

    if k in {"box", "boxplot"}:
        from ravix.plots.boxplot import boxplot
        return boxplot(*args, **kwargs)

    if k in {"cor", "corplot", "corrplot"}:
        from ravix.plots.plot_cor import plot_cor
        return plot_cor(*args, **kwargs)

    if k == "bsr":
        from ravix.plots.plot_bsr import plot_bsr
        return plot_bsr(*args, **kwargs)

    if k == "cook" or k == "plot_cook":
        from ravix.plots.plot_cook import plot_cook
        return plot_cook(*args, **kwargs)
      

    raise ValueError(
        f"Invalid plot kind '{kind}'. "
        "Standard options: 'plot', 'hist', 'qq', 'bar', 'box', 'cor', 'bsr', 'cook'."
    )

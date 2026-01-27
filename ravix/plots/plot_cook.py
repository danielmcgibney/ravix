import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Any


def plot_cook(
    model: Any,
    threshold: float = 0.5,
    title: str = "Cook's Distance Plot",
    xlab: str = "Observation Index",
    ylab: str = "Cook's Distance",
    figsize: Tuple[int, int] = (10, 6),
    **kwargs
) -> None:
    """
    Plot Cook's Distance for each observation to identify influential points.

    Parameters
    ----------
    model : Any
        A fitted Ravix model exposing `get_influence()`.
    threshold : float, optional
        Reference threshold used to flag influential observations.
    title : str, optional
        Plot title.
    xlab : str, optional
        X-axis label.
    ylab : str, optional
        Y-axis label.
    figsize : tuple of int, optional
        Figure size passed to matplotlib.
    **kwargs
        Additional keyword arguments passed to `matplotlib.axes.Axes.stem`.

    Returns
    -------
    None
        Displays the Cook's Distance plot.
    """

    # Extract influence diagnostics
    influence = model.get_influence()
    cooks_d: np.ndarray = influence.cooks_distance[0]

    # Initialize figure
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Stem plot of Cook's Distance values
    ax.stem(
        np.arange(len(cooks_d)),
        cooks_d,
        markerfmt=",",
        **kwargs
    )

    # Axis labels and title
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)

    # Reference line for influence threshold
    ax.axhline(
        y=threshold,
        linestyle="--",
        color="red",
        label=f"Influence threshold ({threshold})"
    )
    ax.legend()

    # Render and clean up
    plt.show()
    plt.clf()
    plt.close()

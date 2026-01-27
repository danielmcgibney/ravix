from __future__ import annotations

from typing import Optional, Literal, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from ravix.modeling.intervals import intervals


def plot_intervals(
    model,
    newX: pd.DataFrame,
    interval: Literal["confidence", "prediction"] = "confidence",
    alpha: float = 0.05,
    color: str = "blue",
    lcolor: str = "red",
    title: Optional[str] = None,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    show: bool = True,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot confidence or prediction intervals for a regression model.
    
    Creates a visualization of regression intervals with two different plot types:
    - Single predictor: Line plot showing regression line with shaded interval band
    - Multiple predictors: Density plot showing the prediction distribution for one observation
    
    Parameters
    ----------
    model : Any
        A fitted regression model (e.g., from ravix.fit()).
        Must have predict() method and params attribute.
    newX : pd.DataFrame
        DataFrame containing predictor variable values for which to calculate intervals.
        - For single predictor models: Can contain multiple rows (observations)
        - For multiple predictor models: Must contain exactly ONE row (one observation)
        Column names must match the model's predictor variables.
    interval : Literal["confidence", "prediction"], default="confidence"
        Type of interval to calculate and plot:
        - "confidence": Interval for the mean response (narrower)
        - "prediction": Interval for individual predictions (wider, accounts for residual variance)
    alpha : float, default=0.05
        Significance level for the intervals, between 0 and 1.
        Confidence level = 1 - alpha.
        Common values: 0.10 (90% CI), 0.05 (95% CI), 0.01 (99% CI)
    color : str, default="blue"
        Color for the regression line or density curve.
    lcolor : str, default="red"
        Color for the interval bounds.
    title : Optional[str], default=None
        Custom title for the plot.
        If None, generates a descriptive title based on the interval type and alpha.
    xlab : Optional[str], default=None
        Custom label for the x-axis.
        If None, uses the predictor name (single predictor) or "Response" (multiple predictors).
    ylab : Optional[str], default=None
        Custom label for the y-axis.
        If None, uses "Response" (single predictor) or "Density" (multiple predictors).
    figsize : Tuple[float, float], default=(10, 6)
        Figure size (width, height) in inches.
    show : bool, default=True
        Whether to display the plot. If False, returns (fig, ax) for further customization.
    
    Returns
    -------
    Optional[Tuple[plt.Figure, plt.Axes]]
        If show=False, returns (fig, ax) for further customization.
        If show=True, displays the plot and returns None.
    
    Raises
    ------
    ValueError
        If newX contains multiple rows when model has multiple predictors.
        If alpha is not between 0 and 1 (exclusive).
    
    Examples
    --------
    Single predictor example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> import ravix as rv
    >>> 
    >>> # Fit a simple linear regression
    >>> X = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
    >>> y = pd.Series([2, 4, 5, 4, 5])
    >>> model = rv.fit(y, X)
    >>> 
    >>> # Plot 95% confidence interval over a range of x values
    >>> newX = pd.DataFrame({'x': np.linspace(1, 5, 50)})
    >>> plot_intervals(model, newX, interval='confidence', alpha=0.05)
    >>> 
    >>> # Plot 90% prediction interval
    >>> plot_intervals(model, newX, interval='prediction', alpha=0.10)
    >>> 
    >>> # Return figure for customization
    >>> fig, ax = plot_intervals(model, newX, show=False)
    >>> ax.axhline(3, color='green', linestyle=':')
    >>> plt.show()
    
    Multiple predictor example:
    >>> # Fit a multiple regression
    >>> X = pd.DataFrame({'x1': [1, 2, 3, 4, 5], 'x2': [2, 3, 4, 5, 6]})
    >>> y = pd.Series([3, 5, 7, 9, 11])
    >>> model = rv.fit(y, X)
    >>> 
    >>> # Plot 90% prediction interval for a single observation
    >>> newX = pd.DataFrame({'x1': [3], 'x2': [4]})
    >>> plot_intervals(model, newX, interval='prediction', alpha=0.10)
    
    Notes
    -----
    - The function automatically detects whether to create a line plot (single predictor)
      or density plot (multiple predictors) based on the number of predictor variables.
    - For multiple predictors, the density is approximated using a normal distribution
      with standard deviation derived from the interval width.
    - Constant/intercept columns (where all values = 1) are not counted as predictors
      for determining plot type.
    - The shaded region represents the confidence/prediction interval bounds.
    - alpha is the significance level; confidence level = 1 - alpha
      (e.g., alpha=0.05 gives 95% confidence interval)
    
    See Also
    --------
    intervals : Calculate interval bounds without plotting
    """
    # Validate alpha
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1 (exclusive), got {alpha}")
    
    # Calculate confidence level for intervals() function
    level = 1 - alpha
    
    # Calculate the intervals using the intervals function
    interval_data = intervals(model, newX, interval=interval, alpha = alpha)
    
    # Identify non-constant predictor columns
    non_constant_cols = [col for col in newX.columns if not np.all(newX[col] == 1)]
    num_predictors = len(non_constant_cols)

    if num_predictors == 1:
        # Single predictor case: plot response vs predictor with interval band
        predictor = newX[non_constant_cols[0]]
        xlab = xlab if xlab else non_constant_cols[0]
        ylab = ylab if ylab else "Response"
        title = (
            title
            if title
            else f"Regression Line with {int(level*100)}% {interval.capitalize()} Interval"
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot regression line
        ax.plot(predictor, model.predict(newX), label="Regression Line", color=color)
        
        # Shaded interval region
        ax.fill_between(
            predictor,
            interval_data["Lower Bound"],
            interval_data["Upper Bound"],
            color="gray",
            alpha=0.2,
        )
        
        # Interval boundary lines
        ax.plot(
            predictor,
            interval_data["Lower Bound"],
            linestyle="--",
            color=lcolor,
            label=f"{int(level*100)}% {interval.capitalize()} Interval",
        )
        ax.plot(
            predictor,
            interval_data["Upper Bound"],
            linestyle="--",
            color=lcolor,
        )
        
        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(loc="upper right")
        ax.grid(True)

    else:
        # Multiple predictors case: Ensure newX is a vector (one observation)
        if newX.shape[0] != 1:
            raise ValueError(
                "newX must contain exactly one row (one observation) when there are "
                f"multiple predictors. Got {newX.shape[0]} rows."
            )

        xlab = xlab if xlab else "Response"
        ylab = ylab if ylab else "Density"
        title = (
            title
            if title
            else f"{int(level*100)}% {interval.capitalize()} Interval for the Observation"
        )
        
        # Extract the prediction and bounds for this observation
        prediction = interval_data["Prediction"].iloc[0]
        lower_bound = interval_data["Lower Bound"].iloc[0]
        upper_bound = interval_data["Upper Bound"].iloc[0]

        # Plot the PDF of the response with interval bounds
        fig, ax = plt.subplots(figsize=figsize)
        
        # Approximate the distribution (assumes normality)
        mu = prediction
        std = (upper_bound - lower_bound) / 4  # Approximate std from 95% interval
        x = np.linspace(mu - 3 * std, mu + 3 * std, 100)
        pdf = norm.pdf(x, mu, std)

        # Plot probability density function
        ax.plot(x, pdf, label="Density", color=color)
        
        # Vertical lines for prediction and interval bounds
        ax.axvline(mu, color=color, linestyle="--", label="Prediction")
        ax.axvline(
            lower_bound,
            color=lcolor,
            linestyle="--",
            label=f"{int(level*100)}% {interval.capitalize()} Interval",
        )
        ax.axvline(upper_bound, color=lcolor, linestyle="--")
        
        # Shaded region under the curve
        ax.fill_between(x, pdf, color="gray", alpha=0.2)

        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(loc="upper right")
        ax.grid(True)

    # Tight layout for better spacing
    plt.tight_layout()
    
    if show:
        plt.show()
        plt.close(fig)
        return None
    
    return fig, ax

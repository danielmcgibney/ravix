from __future__ import annotations

from typing import Union, Literal

import numpy as np
import pandas as pd


def intervals(
    model,
    newX: Union[pd.DataFrame, np.ndarray],
    interval: Literal["confidence", "prediction"] = "confidence",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Calculate confidence or prediction intervals for new observations.
    
    This function computes either confidence intervals (for the mean response)
    or prediction intervals (for individual observations) at specified predictor
    values. Automatically handles intercept column addition if needed.
    
    Parameters
    ----------
    model : object
        Fitted Ravix regression model with fit() method.
    newX : Union[pd.DataFrame, np.ndarray]
        New predictor values for which to calculate intervals. If array,
        will be converted to DataFrame. Should not include intercept column
        unless already present in the model.
    interval : Literal["confidence", "prediction"], default="confidence"
        Type of interval to calculate:
        - "confidence": Interval for the mean response (narrower)
        - "prediction": Interval for individual observations (wider)
    alpha : float, default=0.05
        Significance level for the intervals, between 0 and 1 (exclusive).
        Confidence level = 1 - alpha.
        Common values: 0.10 (90% CI), 0.05 (95% CI), 0.01 (99% CI).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with three columns:
        - "Lower Bound": Lower limit of the interval
        - "Prediction": Point prediction (mean response)
        - "Upper Bound": Upper limit of the interval
    
    Examples
    --------
    >>> # Fit a model
    >>> import ravix
    >>> model = ravix.ols("mpg ~ hp + wt", data=mtcars)
    
    >>> # 95% confidence intervals for new observations
    >>> new_data = pd.DataFrame({'hp': [110, 150], 'wt': [2.5, 3.0]})
    >>> ravix.intervals(model, new_data, interval='confidence', alpha=0.05)
    
    >>> # 95% prediction intervals (wider than confidence intervals)
    >>> ravix.intervals(model, new_data, interval='prediction', alpha=0.05)
    
    >>> # 99% confidence level (alpha=0.01)
    >>> ravix.intervals(model, new_data, interval='confidence', alpha=0.01)
    
    >>> # 90% confidence level (alpha=0.10)
    >>> ravix.intervals(model, new_data, interval='confidence', alpha=0.10)
    
    >>> # Using numpy array (automatically converted)
    >>> new_array = np.array([[110, 2.5], [150, 3.0]])
    >>> ravix.intervals(model, new_array, interval='prediction', alpha=0.05)
    
    Notes
    -----
    - Confidence intervals: Estimate uncertainty in the mean response
    - Prediction intervals: Estimate uncertainty for individual predictions
    - Prediction intervals are always wider than confidence intervals
    - Intercept column is automatically added if required by the model
    - newX should match the predictors used in model fitting (excluding intercept)
    - alpha is the significance level; confidence level = 1 - alpha
      (e.g., alpha=0.05 gives 95% confidence interval)
    
    Raises
    ------
    ValueError
        If interval is not "confidence" or "prediction".
        If alpha is not between 0 and 1 (exclusive).
    
    See Also
    --------
    predict : Generate point predictions without intervals
    """
    # Validate alpha
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1 (exclusive), got {alpha}")
    
    # Validate interval type
    if interval not in ("confidence", "prediction"):
        raise ValueError(
            f"Invalid interval type '{interval}'. "
            f"Must be 'confidence' or 'prediction'."
        )
    
    # Convert to DataFrame if necessary
    if not isinstance(newX, pd.DataFrame):
        newX = pd.DataFrame(newX)
    
    # Get model column names
    model_columns = model.model.exog_names
    
    # Check if newX already has an intercept column (case-insensitive)
    has_intercept = any(col.lower() in ["intercept", "const"] for col in newX.columns)
    
    # Insert the intercept column if it's required by the model and not present in newX
    if not has_intercept:
        if "Intercept" in model_columns:
            newX.insert(0, "Intercept", 1)
        elif "const" in model_columns:
            newX.insert(0, "const", 1)
        elif "intercept" in model_columns:
            newX.insert(0, "intercept", 1)
    
    # Get predictions with intervals
    # Note: statsmodels get_prediction uses alpha as significance level
    preds = model.get_prediction(newX)
    summary_frame = preds.summary_frame(alpha=alpha)
    
    # Extract appropriate bounds based on interval type
    if interval == "confidence":
        lower_bound = summary_frame["mean_ci_lower"]
        upper_bound = summary_frame["mean_ci_upper"]
    else:  # interval == "prediction"
        lower_bound = summary_frame["obs_ci_lower"]
        upper_bound = summary_frame["obs_ci_upper"]
    
    prediction = summary_frame["mean"]
    
    # Create result DataFrame
    intervals_df = pd.DataFrame({
        "Prediction": prediction,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound
    })
    
    return intervals_df

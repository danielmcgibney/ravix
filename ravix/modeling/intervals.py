from __future__ import annotations

from typing import Union, Literal, Optional

import numpy as np
import pandas as pd


def intervals(
    model,
    newX: Union[pd.DataFrame, np.ndarray],
    interval: Literal["confidence", "prediction"] = "confidence",
    level: Optional[float] = None,
    alpha: Optional[float] = None,
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
    level : float, optional
        Confidence level for the intervals, between 0 and 1 (exclusive).
        Common values: 0.90 (90% CI), 0.95 (95% CI), 0.99 (99% CI).
        If not specified, defaults to 0.95.
        Cannot be used together with alpha.
    alpha : float, optional
        Significance level for the intervals, between 0 and 1 (exclusive).
        Confidence level = 1 - alpha.
        Common values: 0.10 (90% CI), 0.05 (95% CI), 0.01 (99% CI).
        Cannot be used together with level.
    
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
    
    >>> # 95% confidence intervals (default)
    >>> new_data = pd.DataFrame({'hp': [110, 150], 'wt': [2.5, 3.0]})
    >>> ravix.intervals(model, new_data, interval='confidence')
    
    >>> # 95% confidence intervals (explicit level)
    >>> ravix.intervals(model, new_data, interval='confidence', level=0.95)
    
    >>> # 95% prediction intervals (wider than confidence intervals)
    >>> ravix.intervals(model, new_data, interval='prediction', level=0.95)
    
    >>> # 99% confidence level
    >>> ravix.intervals(model, new_data, interval='confidence', level=0.99)
    
    >>> # 90% confidence level
    >>> ravix.intervals(model, new_data, interval='confidence', level=0.90)
    
    >>> # Using alpha instead of level (alpha=0.05 gives 95% CI)
    >>> ravix.intervals(model, new_data, interval='confidence', alpha=0.05)
    
    >>> # Using numpy array (automatically converted)
    >>> new_array = np.array([[110, 2.5], [150, 3.0]])
    >>> ravix.intervals(model, new_array, interval='prediction', level=0.95)
    
    Notes
    -----
    - Confidence intervals: Estimate uncertainty in the mean response
    - Prediction intervals: Estimate uncertainty for individual predictions
    - Prediction intervals are always wider than confidence intervals
    - Intercept column is automatically added if required by the model
    - newX should match the predictors used in model fitting (excluding intercept)
    - Default confidence level is 0.95 (95% confidence interval)
    - You can specify either level or alpha, but not both
    - level and alpha are related: level = 1 - alpha
    
    Raises
    ------
    ValueError
        If interval is not "confidence" or "prediction".
        If both level and alpha are specified.
        If neither level nor alpha is within (0, 1).
        If level or alpha is not between 0 and 1 (exclusive).
    
    See Also
    --------
    predict : Generate point predictions without intervals
    """
    # Validate that both level and alpha are not specified
    if level is not None and alpha is not None:
        raise ValueError(
            "Cannot specify both 'level' and 'alpha'. "
            "Use 'level' for confidence level (e.g., 0.95 for 95% CI) "
            "or 'alpha' for significance level (e.g., 0.05 for 95% CI)."
        )
    
    # Set default and convert level to alpha if needed
    if level is None and alpha is None:
        # Default to 95% confidence level
        alpha = 0.05
    elif level is not None:
        # Validate level
        if not (0 < level < 1):
            raise ValueError(
                f"level must be between 0 and 1 (exclusive), got {level}"
            )
        # Convert level to alpha
        alpha = 1 - level
    else:
        # alpha was provided, validate it
        if not (0 < alpha < 1):
            raise ValueError(
                f"alpha must be between 0 and 1 (exclusive), got {alpha}"
            )
    
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

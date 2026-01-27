import numpy as np
import pandas as pd
from ravix.modeling.fit import fit

def abline(
    intercept: float,
    slope: float
):
    """
    Create a regression line object with manually specified slope and intercept.
    
    Mimics R's abline() functionality by creating a fitted OLS model with the 
    specified parameters. This is useful for adding reference lines to plots
    (e.g., theoretical relationships, identity lines, or horizontal lines).
    
    Parameters
    ----------
    intercept : float
        The y-intercept of the line (value when X=0).
    slope : float
        The slope of the line (change in Y per unit change in X).
    
    Returns
    -------
    Ravix model object
        A fitted OLS model object with the specified slope and intercept.
        Use with ravix plotting functions to add reference lines.
    
    Notes
    -----
    - Unlike R's abline(), this returns a model object rather than directly plotting.
    - The model object can be used with ravix plotting functions.
    - Common use cases:
      * Identity line: abline(0, 1)
      * Horizontal line: abline(y_value, 0)
      * Theoretical relationship: abline(intercept, slope)
    
    Examples
    --------
    >>> import ravix as rv
    
    >>> # Create identity line (y = x)
    >>> identity = rv.abline(intercept=0, slope=1)
    >>> rv.plot("Y ~ X", data=df, model=identity)
    
    >>> # Create horizontal reference line at y=5
    >>> hline = rv.abline(intercept=5, slope=0)
    >>> rv.plot("Y ~ X", data=df, model=hline)
    
    >>> # Add theoretical relationship to existing plot
    >>> theory = rv.abline(2, 0.5)
    >>> rv.plot("Y ~ X", data=df, model=theory)
    """
    # Generate two points (minimum needed for a line)
    X = np.array([0, 1])
    
    # Compute Y values using the specified slope and intercept
    Y = intercept + slope * X
    
    # Create DataFrame and fit model
    data = pd.DataFrame({'X': X, 'Y': Y})
    model = fit("Y ~ X", data)
    
    return model

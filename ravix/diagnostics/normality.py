from scipy.stats import shapiro as _scipy_shapiro
import statsmodels.api as sm
import numpy as np
from typing import Union, Optional, Dict
from numpy.typing import ArrayLike

def shapiro(
    input_data: Union[ArrayLike, sm.regression.linear_model.RegressionResultsWrapper], 
    alpha: float = 0.05,
    return_dict: bool = False
) -> Optional[Dict[str, float]]:
    """
    Perform the Shapiro-Wilk test for normality on a given vector or regression residuals.
    
    The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a 
    normal distribution. A significant result (p < alpha) suggests the data deviates 
    from normality.
    
    Parameters
    ----------
    input_data : array-like or Ravix model
        Either:
        - A numeric vector/array to test for normality
        - A fitted statsmodels regression model (residuals will be tested)
        - A Ravix model object with a 'resid' attribute
    alpha : float, default=0.05
        Significance level for determining normality
    return_dict : bool, default=False
        If True, returns dictionary with detailed results. If False, prints 
        formatted output only.
    
    Returns
    -------
    None or dict
        If return_dict=False: None (prints results only).
        If return_dict=True: Returns dictionary with test_statistic, p_value, 
        and alpha.
    
    Notes
    -----
    H₀: Data follows a normal distribution
    H₁: Data does not follow a normal distribution
    
    - The Shapiro-Wilk test is most appropriate for sample sizes between 3 and 5000.
    - For larger samples, the test may detect trivial departures from normality.
    - Visual inspection (Q-Q plots, histograms) should complement this test.
    
    Examples
    --------
    >>> import numpy as np
    >>> import ravix as rv
    >>> import pandas as pd
    
    >>> # Test a normal distribution
    >>> data = np.random.normal(0, 1, 100)
    >>> rv.shapiro(data)
    Shapiro-Wilk Test for Normality
    ========================================
    Test Statistic      : 0.9912
    p-value             : 0.7234
    Result              : Normal (p >= 0.05)
    ========================================
    
    >>> # Test regression residuals from ravix model
    >>> df = pd.DataFrame({'y': y, 'x1': X[:, 1], 'x2': X[:, 2]})
    >>> model = rv.fit('y ~ x1 + x2', data=df)
    >>> rv.shapiro(model)
    
    >>> # Return dictionary for programmatic use
    >>> results = rv.shapiro(model, return_dict=True)
    >>> if results['p_value'] < 0.01:
    ...     print("Strong evidence of non-normality")
    """
    # Check if input_data is a statsmodels object
    if isinstance(input_data, sm.regression.linear_model.RegressionResultsWrapper):
        # Use the residuals from the statsmodels model
        data = input_data.resid
    elif hasattr(input_data, 'resid'):
        # Use the residuals from ravix or other model objects
        data = input_data.resid
    else:
        # Assume input_data is a vector
        data = np.asarray(input_data)
    
    # Perform the Shapiro-Wilk test
    stat, p_value = _scipy_shapiro(data)
    
    # Determine the result based on significance level
    if p_value > alpha:
        result = f'Normal (p >= {alpha})'
    else:
        result = f'Non-normal (p < {alpha})'
    
    if not return_dict:
        # Print the test details
        print("Shapiro-Wilk Test for Normality")
        print("========================================")
        print(f"Test Statistic      : {stat:.4f}")
        print(f"p-value             : {p_value:.4g}")
        print(f"Result              : {result}")
        print("========================================")
    else:
        return {
            'test_statistic': stat,
            'p_value': p_value,
            'alpha': alpha
        }

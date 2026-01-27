import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2
from typing import Optional, Dict
from statsmodels.stats.diagnostic import het_breuschpagan

def ncv(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    alpha: float = 0.05,
    return_dict: bool = False
) -> Optional[Dict[str, float]]:
    """
    Non-Constant Variance Test (Score Test for Heteroscedasticity)
    
    Replicates R's car::ncvTest functionality using the Cook-Weisberg score test
    formulation. Tests the null hypothesis that error variance is constant 
    against the alternative that variance depends on fitted values.
    
    This score test approach is more robust to non-normality compared to the 
    standard Breusch-Pagan test, making it suitable for real-world regression
    diagnostics where normality assumptions may be violated.
    
    Parameters
    ----------
    model : Ravix OLS regression model (fitted)
        A fitted OLS regression model object.
    alpha : float, default=0.05  
        Significance level for hypothesis testing. Used for result interpretation.
    return_dict : bool, default=False
        If True, returns dictionary with detailed results. If False, prints 
        formatted output only.
        
    Returns
    -------
    None or dict
        If return_dict=False: None (prints results only).
        If return_dict=True: Returns dictionary with test_statistic, p_value, 
        df, and alpha.
        
    Notes
    -----
    H₀: Constant error variance (homoscedasticity)
    H₁: Error variance depends on fitted values (heteroscedasticity)
    
    Test statistic follows Chi-squared distribution with 1 degree of freedom.
    The auxiliary regression tests: (residuals^2/sigma^2 - 1) ~ fitted_values
    
    Examples
    --------
    >>> import ravix as rv
    >>> model = rv.fit("Y ~ X", data=df)
    >>> rv.ncv(model)                              # Print results only
    Nonconstant Variance Test
    ========================================
    Test Statistic      : 12.3456
    p-value             : 0.0004
    Result              : Nonconstant variance
    ========================================
    
    >>> # Return dictionary for programmatic use
    >>> results = rv.ncv(model, return_dict=True)
    >>> if results['p_value'] < 0.01:
    ...     print("Strong evidence of heteroscedasticity")
    """
    resid_sq = model.resid ** 2
    n = len(resid_sq)
    sigma2 = np.mean(resid_sq)
    fitted = model.fittedvalues
    
    # Use intercept already in model: construct aux_X accordingly
    aux_X = np.column_stack((np.ones_like(fitted), fitted))  # manually add intercept
    
    # Auxiliary response for score test
    f = resid_sq / sigma2 - 1
    
    # Fit auxiliary model
    aux_model = sm.OLS(f, aux_X).fit()
    test_stat = 0.5 * np.sum(aux_model.fittedvalues ** 2)
    df = aux_model.df_model
    p_value = 1 - chi2.cdf(test_stat, df=df)
    
    # Determine the result based on significance level
    result = f'Nonconstant variance (p < {alpha})' if p_value < alpha else f'Constant variance (p >= {alpha})'
    
    if not return_dict:
        print("Nonconstant Variance Test")
        print("========================================")
        print(f"Test Statistic      : {test_stat:.4f}")
        print(f"p-value             : {p_value:.4g}")
        print(f"Result              : {result}")
        print("========================================")
    else:
        return {
            'test_statistic': test_stat,
            'p_value': p_value,
            'df': df,
            'alpha': alpha
        }



def bp(
    model,
    alpha: float = 0.05,
    use_fitted: bool = False,
    return_dict: bool = False
) -> Optional[Dict[str, float]]:
    """
    Perform the Breusch-Pagan test for heteroscedasticity.
    
    Tests the null hypothesis that error variance is constant against the 
    alternative that variance depends on the predictors (or fitted values).
    
    Parameters
    ----------
    model : Ravix OLS regression model (fitted)
        A fitted OLS regression model object.
    alpha : float, default=0.05
        Significance level for hypothesis testing. Used for result interpretation.
    use_fitted : bool, default=False
        If True, test against fitted values only (similar to R's ncvTest).
        If False, test against all predictors (standard Breusch-Pagan).
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
    H₀: Constant error variance (homoscedasticity)
    H₁: Error variance depends on predictors/fitted values (heteroscedasticity)
    
    The Breusch-Pagan test regresses squared residuals on the predictors
    (or fitted values) to detect heteroscedasticity.
    
    Examples
    --------
    >>> import ravix as rv
    >>> model = rv.fit("Y ~ X1 + X2", data=df)
    >>> rv.bp(model)                              # Print results only
    Breusch-Pagan Test for Heteroscedasticity
    ========================================
    Test Statistic      : 8.2341
    p-value             : 0.0041
    Result              : Heteroscedastic (p < 0.05)
    ========================================
    
    >>> # Test against fitted values only
    >>> rv.bp(model, use_fitted=True)
    
    >>> # Return dictionary for programmatic use
    >>> results = rv.bp(model, return_dict=True)
    >>> if results['p_value'] < 0.01:
    ...     print("Strong evidence of heteroscedasticity")
    """
    if use_fitted:
        fitted = model.fittedvalues
        # Add constant only if not already present
        has_const = 'Intercept' in model.model.exog_names or 'const' in model.model.exog_names
        if has_const:
            x = np.column_stack((np.ones_like(fitted), fitted))
        else:
            x = sm.add_constant(fitted)
    else:
        x = model.model.exog
    
    bp_stat, p_value, _, _ = het_breuschpagan(model.resid, x)
    
    # Determine the result based on significance level
    result = f'Heteroscedastic (p < {alpha})' if p_value < alpha else f'Homoscedastic (p >= {alpha})'
    
    if not return_dict:
        print("Breusch-Pagan Test for Heteroscedasticity")
        print("========================================")
        print(f"Test Statistic      : {bp_stat:.4f}")
        print(f"p-value             : {p_value:.4g}")
        print(f"Result              : {result}")
        print("========================================")
    else:
        return {
            'test_statistic': bp_stat,
            'p_value': p_value,
            'alpha': alpha
        }

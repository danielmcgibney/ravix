from .parse_formula import parse_formula
import statsmodels.api as sm
import pandas as pd
import numpy as np
from .predict import predict
from typing import Union


def _fit_matrices(Y, X, method: str, **kwargs):
    """
    Fit statistical models from Y and X matrices.
    
    Internal function used by stepwise and bsr when matrices are already parsed.
    """
    if method == "ols":
        fitted = sm.OLS(Y, X, **kwargs).fit()
    
    elif method == "logistic":
        fitted = sm.GLM(Y, X, family=sm.families.Binomial(), **kwargs).fit()
    
    elif method == "poisson":
        fitted = sm.GLM(Y, X, family=sm.families.Poisson(), **kwargs).fit()
    
    elif method == "glm":
        # For advanced users who want to specify family themselves
        fitted = sm.GLM(Y, X, **kwargs).fit()
    
    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported methods: 'ols', 'logistic', 'poisson', 'glm'"
        )

    
    # Add metadata
    fitted.model_type = method
    
    # Enhance predict method
    fitted._statsmodels_predict = fitted.predict
    
    def predict_wrapper(newdata=None, *args, **kwargs):
        if newdata is None:
            return fitted._statsmodels_predict()
        return predict(fitted, newdata, *args, **kwargs)
    
    fitted.predict = predict_wrapper
    
    return fitted


def _fit_model(formula: str, data: pd.DataFrame, method: str, **kwargs):
    """
    Fit statistical models from formulas.
    
    Internal function that parses formulas and delegates to _fit_matrices.
    """
    
    # Parse formula
    Y_out, X_out = parse_formula(formula, data)
    Y_name = Y_out.name
    
    # Ensure Y_out is a Series and retains its name
    if isinstance(Y_out, pd.DataFrame):
        if Y_out.shape[1] == 1:
            Y_out = Y_out.iloc[:, 0]
        else:
            raise ValueError("Response variable must be a single column.")
    elif not isinstance(Y_out, pd.Series):
        Y_out = pd.Series(Y_out, name=Y_name)
    
    # Ensure X_out is a DataFrame
    if isinstance(X_out, pd.Series):
        X_out = X_out.to_frame()
    
    if X_out.empty:
        raise ValueError("No predictor variables found in the formula.")
    
    # Convert X to numeric
    X_out = X_out.apply(pd.to_numeric, errors='coerce')
    
    if X_out.isna().any().any():
        nan_cols = X_out.columns[X_out.isna().any()].tolist()
        raise ValueError(
            f"Predictor variables {nan_cols} contain non-numeric values "
            f"that cannot be converted to numbers."
        )
    
    X_out = X_out.astype(np.float64)
    
    # Method-specific Y processing
    if method == "ols":
        # Convert Y to numeric
        Y_out = pd.to_numeric(Y_out, errors='coerce')
        
        if Y_out.isna().any():
            raise ValueError(
                f"Response variable '{Y_name}' contains non-numeric values "
                f"that cannot be converted to numbers."
            )
    
    elif method == "logistic":
        # Handle categorical/binary response
        if not pd.api.types.is_numeric_dtype(Y_out):
            Y_out = pd.get_dummies(Y_out, drop_first=True)
            if Y_out.shape[1] != 1:
                raise ValueError(
                    "Response variable must be binary (two categories only). "
                    f"Found {Y_out.shape[1] + 1} categories."
                )
            Y_out = Y_out.iloc[:, 0]
        else:
            Y_out = pd.to_numeric(Y_out, errors='coerce')
        
        if Y_out.isna().any():
            raise ValueError(
                f"Response variable '{Y_name}' contains missing values (NaN) "
                f"that must be handled before fitting."
            )
    
    elif method == "poisson":
        # Convert Y to numeric for count data
        Y_out = pd.to_numeric(Y_out, errors='coerce')
        
        if Y_out.isna().any():
            raise ValueError(
                f"Response variable '{Y_name}' contains missing values (NaN) "
                f"that must be handled before fitting."
            )
    
    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported methods: 'ols', 'logistic', 'poisson'"
        )
    
    # Use direct mode to fit
    fitted = _fit_direct(Y_out, X_out, method, **kwargs)
    
    # Add formula metadata (not added by _fit_direct)
    fitted.formula = formula
    
    return fitted


def ols(formula: str, data: pd.DataFrame, **kwargs) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit an ordinary least squares (OLS) regression model.
    
    OLS is used for modeling continuous numeric outcomes as a linear
    function of predictor variables.
    
    Parameters
    ----------
    formula : str
        Model formula specifying the relationship between variables.
        Format: 'response ~ predictor1 + predictor2 + ...'
        
        Examples:
        - 'price ~ sqft' (simple regression)
        - 'price ~ sqft + bedrooms' (multiple regression)
        - 'price ~ sqft + bedrooms + sqft:bedrooms' (with interaction)
        
    data : DataFrame
        Dataset containing all variables referenced in the formula.
        All variables must be present as columns in this DataFrame.
    **kwargs
        Additional arguments passed to statsmodels.OLS
    
    Returns
    -------
    model : statsmodels RegressionResultsWrapper
        Fitted OLS model with the following additions:
        - model.formula : str - the formula used to fit the model
        - model.model_type : str - 'ols'
        - model.predict(newdata) - make predictions on new data
        
        All standard statsmodels methods are available:
        - model.summary() : display regression results
        - model.params : estimated coefficients
        - model.rsquared : R-squared value
        - model.conf_int() : confidence intervals
        - And many more...
    
    Raises
    ------
    ValueError
        If variables in the formula are not found in data, or if
        variables contain non-numeric values.
    
    Examples
    --------
    >>> import ravix
    >>> import pandas as pd
    >>> 
    >>> # Simple linear regression
    >>> houses = pd.DataFrame({
    ...     'price': [200, 250, 300, 350],
    ...     'sqft': [1000, 1200, 1500, 1800]
    ... })
    >>> model = ravix.ols('price ~ sqft', data=houses)
    >>> print(model.summary())
    >>> 
    >>> # Multiple regression
    >>> model = ravix.ols('price ~ sqft + bedrooms', data=houses)
    >>> 
    >>> # Make predictions on new data
    >>> new_houses = pd.DataFrame({'sqft': [1300], 'bedrooms': [3]})
    >>> predictions = model.predict(new_houses)
    
    Notes
    -----
    The response variable (left side of ~) must be numeric and continuous.
    For binary outcomes, use ravix.logistic() instead.
    For count data, use ravix.poisson() instead.
    
    See Also
    --------
    logistic : Logistic regression for binary outcomes
    poisson : Poisson regression for count data
    """
    return _fit_model(formula, data, method="ols", **kwargs)


def logistic(formula: str, data: pd.DataFrame, **kwargs) -> sm.genmod.generalized_linear_model.GLMResultsWrapper:
    """
    Fit a logistic regression model for binary outcomes.
    
    Logistic regression is used for modeling binary (yes/no, 0/1, True/False)
    outcomes as a function of predictor variables.
    
    Parameters
    ----------
    formula : str
        Model formula specifying the relationship between variables.
        Format: 'response ~ predictor1 + predictor2 + ...'
        
        The response variable must be binary (two categories).
        
        Examples:
        - 'admitted ~ gpa' (simple logistic regression)
        - 'success ~ treatment + age' (multiple predictors)
        - 'churn ~ usage + tenure + usage:tenure' (with interaction)
        
    data : DataFrame
        Dataset containing all variables referenced in the formula.
        The response variable can be:
        - Numeric binary (0/1)
        - Boolean (True/False)
        - Categorical with two categories (will be converted to 0/1)
    **kwargs
        Additional arguments passed to statsmodels.GLM
    
    Returns
    -------
    model : statsmodels GLMResultsWrapper
        Fitted logistic regression model with the following additions:
        - model.formula : str - the formula used to fit the model
        - model.model_type : str - 'logistic'
        - model.predict(newdata) - make predictions on new data
        
        All standard statsmodels GLM methods are available:
        - model.summary() : display regression results
        - model.params : estimated coefficients (log-odds scale)
        - model.conf_int() : confidence intervals
        - And many more...
    
    Raises
    ------
    ValueError
        If the response variable is not binary, if variables in the
        formula are not found in data, or if variables contain
        non-numeric values (for predictors).
    
    Examples
    --------
    >>> import ravix
    >>> import pandas as pd
    >>> 
    >>> # Binary outcome (0/1)
    >>> applications = pd.DataFrame({
    ...     'admitted': [0, 1, 1, 0, 1],
    ...     'gpa': [3.2, 3.8, 3.9, 2.9, 3.7],
    ...     'test_score': [1200, 1400, 1450, 1100, 1380]
    ... })
    >>> model = ravix.logistic('admitted ~ gpa + test_score', data=applications)
    >>> print(model.summary())
    >>> 
    >>> # Categorical outcome (will be converted)
    >>> df = pd.DataFrame({
    ...     'success': ['yes', 'no', 'yes', 'yes', 'no'],
    ...     'treatment': [1, 0, 1, 1, 0]
    ... })
    >>> model = ravix.logistic('success ~ treatment', data=df)
    >>> 
    >>> # Make predictions on new data
    >>> new_data = pd.DataFrame({'gpa': [3.5], 'test_score': [1300]})
    >>> probabilities = model.predict(new_data)
    
    Notes
    -----
    - Predicted values are probabilities (between 0 and 1)
    - Coefficients are on the log-odds scale
    - Use np.exp(model.params) to get odds ratios
    - The response variable must have exactly two categories
    
    See Also
    --------
    ols : Linear regression for continuous outcomes
    poisson : Poisson regression for count data
    """
    return _fit_model(formula, data, method="logistic", **kwargs)


def poisson(formula: str, data: pd.DataFrame, **kwargs) -> sm.genmod.generalized_linear_model.GLMResultsWrapper:
    """
    Fit a Poisson regression model for count data.
    
    Poisson regression is used for modeling count outcomes (non-negative integers)
    as a function of predictor variables.
    
    Parameters
    ----------
    formula : str
        Model formula specifying the relationship between variables.
        Format: 'response ~ predictor1 + predictor2 + ...'
        
        The response variable should be count data (0, 1, 2, 3, ...).
        
        Examples:
        - 'num_claims ~ age' (simple Poisson regression)
        - 'visits ~ treatment + age' (multiple predictors)
        - 'accidents ~ traffic + weather + traffic:weather' (with interaction)
        
    data : DataFrame
        Dataset containing all variables referenced in the formula.
        The response variable should be non-negative integer counts.
    **kwargs
        Additional arguments passed to statsmodels.GLM
    
    Returns
    -------
    model : statsmodels GLMResultsWrapper
        Fitted Poisson regression model with the following additions:
        - model.formula : str - the formula used to fit the model
        - model.model_type : str - 'poisson'
        - model.predict(newdata) - make predictions on new data
        
        All standard statsmodels GLM methods are available:
        - model.summary() : display regression results
        - model.params : estimated coefficients (log scale)
        - model.conf_int() : confidence intervals
        - And many more...
    
    Raises
    ------
    ValueError
        If variables in the formula are not found in data, or if
        variables contain non-numeric values.
    
    Examples
    --------
    >>> import ravix
    >>> import pandas as pd
    >>> 
    >>> # Count outcome
    >>> insurance = pd.DataFrame({
    ...     'num_claims': [0, 1, 2, 0, 3],
    ...     'age': [25, 35, 45, 30, 50],
    ...     'risk_score': [2, 3, 5, 2, 6]
    ... })
    >>> model = ravix.poisson('num_claims ~ age + risk_score', data=insurance)
    >>> print(model.summary())
    >>> 
    >>> # Make predictions on new data
    >>> new_data = pd.DataFrame({'age': [40], 'risk_score': [4]})
    >>> predicted_counts = model.predict(new_data)
    
    Notes
    -----
    - Predicted values are expected counts (can be non-integer)
    - Coefficients are on the log scale
    - Use np.exp(model.params) to get multiplicative effects
    - The response variable should be non-negative counts
    
    See Also
    --------
    ols : Linear regression for continuous outcomes
    logistic : Logistic regression for binary outcomes
    """
    return _fit_model(formula, data, method="poisson", **kwargs)


def fit(formula: str, data: pd.DataFrame, method: str = 'ols', **kwargs):
    """
    Generic interface for fitting statistical models.
    
    Parameters
    ----------
    formula : str
        Model formula: 'response ~ predictor1 + predictor2 + ...'
    data : DataFrame
        Dataset containing the variables in the formula
    method : str, default='ols'
        Model type: 'ols', 'logistic', or 'poisson'
    **kwargs
        Additional arguments passed to statsmodels
    
    Returns
    -------
    model : statsmodels model
        Fitted model
    
    Notes
    -----
    For typical use, ols(), logistic(), or poisson() are recommended for clarity.
    This function is useful for programmatic model selection.
    
    Examples
    --------
    >>> # These are equivalent:
    >>> model1 = ravix.ols('price ~ sqft', data=houses)
    >>> model2 = ravix.fit('price ~ sqft', data=houses, method='ols')
    
    See Also
    --------
    ols : Linear regression
    logistic : Logistic regression
    poisson : Poisson regression
    """
    return _fit_model(formula, data, method=method, **kwargs)

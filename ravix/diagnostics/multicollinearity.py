import pandas as pd
from typing import Optional, Union, Dict, Any
from ravix.modeling.parse_formula import parse_formula
from ravix.plots.barplot import barplot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

def vif(
    formula: Optional[Union[str, Any]] = None,  # Can be str formula or ravix model
    data: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None, 
    plot: bool = False, 
    xlab: str = 'Predictor', 
    ylab: str = 'VIF Value', 
    title: str = 'Variance Inflation Factors', 
    ascending: bool = True
) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factors (VIF) to detect multicollinearity among predictors.
    
    VIF quantifies how much the variance of a regression coefficient is inflated due to 
    collinearity with other predictors. Values above 5-10 typically indicate problematic 
    multicollinearity.
    
    Parameters
    ----------
    formula : str, ravix model, or None, optional
        Either:
        - A formula string specifying predictors (e.g., '~x1 + x2 + x3'). The intercept 
          is automatically excluded.
        - A fitted ravix model object (VIF will be calculated on the model's predictors).
        - None, in which case all columns in data are used as predictors.
    data : pd.DataFrame or dict, optional
        The dataset containing predictor variables. If dict, it will be converted to 
        a DataFrame. Not required if a ravix model is provided.
    plot : bool, default=False
        If True, generates a bar plot of VIF values.
    xlab : str, default='Predictor'
        Label for the x-axis in the plot.
    ylab : str, default='VIF Value'
        Label for the y-axis in the plot.
    title : str, default='Variance Inflation Factors'
        Title for the plot.
    ascending : bool, default=True
        If True, sorts VIF values in ascending order; if False, sorts in descending order.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with predictors as columns and their corresponding VIF values as a 
        single row. Columns are sorted according to the ascending parameter.
    
    Notes
    -----
    - Predictors are standardized before VIF calculation to ensure numerical stability.
    - VIF interpretation guidelines:
        * VIF = 1: No correlation with other predictors
        * VIF < 5: Low multicollinearity (generally acceptable)
        * VIF 5-10: Moderate multicollinearity (investigate further)
        * VIF > 10: High multicollinearity (consider removing predictor)
    
    Examples
    --------
    >>> import pandas as pd
    >>> import ravix as rv
    >>> data = pd.DataFrame({
    ...     'y': [1, 3, 5, 7, 9],
    ...     'x1': [1, 2, 3, 4, 5],
    ...     'x2': [2, 4, 6, 8, 10],
    ...     'x3': [5, 4, 3, 2, 1]
    ... })
    
    >>> # Using a DataFrame directly
    >>> vif_result = rv.vif(data=data[['x1', 'x2', 'x3']])
    
    >>> # Using a formula
    >>> vif_result = rv.vif(formula='~x1 + x3', data=data, plot=True)
    
    >>> # Using a ravix model
    >>> model = rv.fit('y ~ x1 + x2 + x3', data=data)
    >>> vif_result = rv.vif(model)
    
    >>> # Sort by highest VIF first
    >>> vif_result = rv.vif(data=data, ascending=False)
    """
    # Check if formula is a ravix model
    if formula is not None and not isinstance(formula, str):
        # Assume it's a ravix model object
        if hasattr(formula, 'X') and hasattr(formula, 'data'):
            # Extract predictor matrix from model (excluding intercept if present)
            X = formula.X
            # Remove intercept column if it exists
            if 'Intercept' in X.columns:
                X = X.drop(columns=['Intercept'])
            elif '(Intercept)' in X.columns:
                X = X.drop(columns=['(Intercept)'])
        else:
            raise ValueError("Provided model object does not have expected attributes (X, data)")
    elif isinstance(data, dict):
        data = pd.DataFrame(data)
        if formula and isinstance(formula, str):
            formula = formula + '+0'
            _, X = parse_formula(formula, data)
        else:
            X = data
    elif formula and isinstance(formula, str):
        formula = formula + '+0'
        _, X = parse_formula(formula, data)
    else:
        X = data
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate VIF for each predictor
    vif_values = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
    
    # Create a DataFrame with feature names as column headers
    vif_data = pd.DataFrame([vif_values], columns=X.columns)
    
    # Sort the VIF values if required
    vif_data = vif_data.T
    vif_data.columns = ['VIF']
    vif_data = vif_data.sort_values(by='VIF', ascending=ascending)
    
    if plot:
        barplot(data=vif_data.T, ylab=ylab, xlab=xlab, title=title)
    
    return vif_data.T

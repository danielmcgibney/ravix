import numpy as np
import pandas as pd
from itertools import combinations
from .fit import _fit_matrices
from .parse_formula import parse_formula
from types import SimpleNamespace

def bsr(formula, data, max_var=8, metric="aic", method="ols", **kwargs):
    """
    Perform Best Subset Regression using OLS, Logistic, Poisson, or GLM.
    
    Parameters:
        formula (str): A regression formula (e.g., 'Y ~ X1 + X2').
        data (pd.DataFrame): Dataset containing the variables.
        max_var (int): Max number of predictors to consider.
        metric (str): One of 'adjr2', 'aic', 'bic', or 'rmse'.
        method (str): 'ols', 'logistic', 'poisson', or 'glm'
        **kwargs: Additional arguments passed to the fitting function (e.g., family for GLM)
    
    Returns:
        Fitted statsmodels model with attributes accessible via model.bsr:
            - best_features: List of selected predictor variables
            - best_by_k: Dict mapping k (num predictors) to best feature subset
            - results: DataFrame with all valid model results
            - metric: Metric used for selection
            - method: Method used
            - all_features: All predictor variables available
            - n_subsets_tested: Total number of subsets successfully fitted
            - failed_subsets: List of subsets that failed to fit (if any)
    """
    # Parse the formula
    Y, X = parse_formula(formula, data)
    x_vars = X.columns
    
    # Use actual column names from X DataFrame
    if hasattr(X, 'columns'):
        actual_predictors = [col for col in X.columns if col not in ['Intercept', 'const']]
    else:
        actual_predictors = [var for var in x_vars if var not in ['Intercept', 'const']]
    
    def get_subsets(variables, max_size):
        return [subset for k in range(1, max_size + 1) for subset in combinations(variables, k)]
    
    def get_metrics(model):
        n = model.nobs
        p = model.df_model + 1
        if method == "ols":
            mse = np.sum(model.resid ** 2) / (n - p)
            rmse = np.sqrt(mse)
        else:
            rmse = np.nan  # not defined for non-OLS
        return model.rsquared_adj if hasattr(model, "rsquared_adj") else np.nan, model.aic, model.bic, rmse
    
    # Validate inputs early
    metric_column = {"adjr2": "Adj. R-squared", "aic": "AIC", "bic": "BIC", "rmse": "RMSE"}
    if metric not in metric_column:
        raise ValueError("Invalid metric. Choose from: 'adjr2', 'aic', 'bic', 'rmse'.")
    
    if method in ["logistic", "poisson", "glm"] and metric == "rmse":
        raise ValueError(f"RMSE is not valid for {method} regression.")
    
    if len(actual_predictors) == 0:
        raise ValueError("No predictor variables found after parsing formula.")
    
    # Ensure max_var is reasonable
    max_var = min(max_var, len(actual_predictors))
    if max_var <= 0:
        raise ValueError("max_var must be positive and there must be predictor variables.")
    
    subsets = get_subsets(actual_predictors, max_var)
    results = []
    failed_subsets = []
    
    for subset in subsets:
        try:
            # Create a subset of X with only the selected columns
            X_subset = X[list(subset)].copy()
            
            # Add intercept if it was in the original X
            if 'Intercept' in X.columns:
                X_subset = pd.concat([X[['Intercept']].copy(), X_subset], axis=1)
            elif 'const' in X.columns:
                X_subset = pd.concat([X[['const']].copy(), X_subset], axis=1)
            
            # Fit model directly with X_subset and Y
            # Note: Y preprocessing is handled by _fit_model when called via _fit_matrices
            model = _fit_matrices(Y, X_subset, method=method, **kwargs)
            
            adjr2, aic, bic, rmse = get_metrics(model)
            results.append((subset, adjr2, aic, bic, rmse))
            
        except Exception as e:
            failed_subsets.append((subset, str(e)))
            continue
    
    if not results:
        raise RuntimeError(f"No models could be fitted successfully. Failed subsets: {failed_subsets}")
    
    results_df = pd.DataFrame(results, columns=["Features", "Adj. R-squared", "AIC", "BIC", "RMSE"])
    
    # Handle NaN values in the metric column before sorting
    metric_col = metric_column[metric]
    if results_df[metric_col].isna().all():
        raise ValueError(f"All models have NaN values for metric '{metric}'. Check your data and model specification.")
    
    # Remove rows with NaN in the metric column before sorting
    results_df_clean = results_df.dropna(subset=[metric_col])
    if results_df_clean.empty:
        raise ValueError(f"No valid models found for metric '{metric}' after removing NaN values.")
    
    ascending = metric != "adjr2"
    results_df_clean = results_df_clean.sort_values(by=metric_col, ascending=ascending)
    
    best_features = list(results_df_clean.iloc[0]['Features'])
    
    # Fit the final best model using the same direct approach
    X_best = X[best_features].copy()
    if 'Intercept' in X.columns:
        X_best = pd.concat([X[['Intercept']].copy(), X_best], axis=1)
    elif 'const' in X.columns:
        X_best = pd.concat([X[['const']].copy(), X_best], axis=1)
    
    best_model = _fit_matrices(Y, X_best, method=method, **kwargs)
    
    # Calculate best_by_k
    best_by_k = {}
    k_groups = results_df_clean.groupby(results_df_clean["Features"].apply(len))
    
    for k, group in k_groups:
        if ascending:
            best_row = group.loc[group[metric_col].idxmin()]
        else:
            best_row = group.loc[group[metric_col].idxmax()]
        best_by_k[k] = list(best_row['Features'])
    
    # Attach results to the best model
    best_model.bsr = SimpleNamespace(
        best_features=best_features,
        best_by_k=best_by_k,
        results=results_df_clean,
        metric=metric,
        method=method,
        all_features=actual_predictors,
        n_subsets_tested=len(results),
        failed_subsets=failed_subsets if failed_subsets else None
    )
    
    return best_model

import pandas as pd
import numpy as np
import statsmodels.api as sm
from .parse_formula import parse_formula

def predict(model, newX=None):
    """
    This function takes a regression model object and a new set of predictor variables,
    and returns the predictions. The default model type is statsmodels.

    Parameters:
    model: The fitted Ravix regression model.
    newX (pandas.DataFrame or numpy.ndarray, optional): The new predictor variables.

    Returns:
    numpy.ndarray: The predicted values.
    """

    # Step 1: Check if newX is provided; if not, return the fitted values from the model
    if newX is None:
        return model.fittedvalues

    # Step 2: Convert newX to a DataFrame if it is not already one
    if not isinstance(newX, pd.DataFrame):
        newX = pd.DataFrame(newX)
    
    # Reset index to avoid index mismatch issues
    newX = newX.reset_index(drop=True)
    
    # Step 3: Extract the names of the model's predictor variables
    model_columns = model.model.exog_names

    # Step 4: Reconstruct the formula from model columns
    formula = model.formula
    
    # Extract the response variable name from the formula
    response_name = formula.split('~')[0].strip()
    
    # Add a dummy response variable to newX with the correct name
    newX_with_dummy = newX.copy()
    if response_name not in newX_with_dummy.columns:
        newX_with_dummy.insert(0, response_name, 0)  # Dummy response, won't be used
    
    # Parse the formula to get the design matrix
    try:
        _, transformed_X = parse_formula(formula, newX_with_dummy, drop_first=False)
    except KeyError as e:
        # If a variable is missing, provide a helpful error message
        raise ValueError(f"Missing required variable in newX: {str(e)}")
    
    # Reset index of transformed_X to match newX
    transformed_X = transformed_X.reset_index(drop=True)
    
    # Step 5: Ensure we have all required columns and drop any extras
    # This handles the case where parse_formula with drop_first=False creates extra columns
    missing_cols = set(model_columns) - set(transformed_X.columns)
    if missing_cols:
        raise ValueError(f"The following required columns are missing: {missing_cols}")
    
    # Step 6: Select only the columns needed by the model, in the correct order
    # This drops any extra columns that transformed_X might have
    transformed_X = transformed_X[model_columns]
    
    # Step 7: Use the original statsmodels predict to generate predictions
    predictions = model._statsmodels_predict(transformed_X)

    return predictions

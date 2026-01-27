from .parse_formula import parse_formula
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def xy_split(formula, data, train=None, test=None, dummies=[True, True], array=True, **kwargs):
    """
    Splits data into predictor and response variables based on a formula and optionally splits into training and testing sets.

    Args:
        formula (str): A string formula, e.g., 'Y ~ X1 + X2 - X3' or 'Y ~ . - X1'.
        data (pd.DataFrame): DataFrame containing the data for variables mentioned in the formula.
        train (float, optional): Proportion of the data to be used as the training set. If None, defaults to complement of test.
        test (float, optional): Proportion of the data to be used as the testing set. If None, no split is performed.
        dummies (list, optional): List where first element is a boolean indicating whether to convert categorical variables 
                                  into dummy/indicator variables, and second element is a boolean indicating whether to drop the first category.
        array (bool, optional): Whether to return the output as NumPy arrays. Defaults to True.
        **kwargs: Additional keyword arguments passed to train_test_split.

    Returns:
        tuple: A tuple containing the predictors (X_train, X_test), the response (Y_train, Y_test), either as pandas DataFrames or NumPy arrays.
    """
    
    # Parse the formula to extract names and transformed data
    Y_out, X_out = parse_formula(formula, data)
    
    if dummies[0]:
        # Convert categorical variables to dummy/indicator variables
        X_out = pd.get_dummies(X_out, drop_first=dummies[1])
        
        # Convert binary variables (True/False) to numeric (0/1)
        binary_columns = X_out.select_dtypes(include=['bool']).columns
        X_out[binary_columns] = X_out[binary_columns].astype(int)
    
    if array:
        # Convert the pandas DataFrame outputs to NumPy arrays
        X_out = np.array(X_out)
        Y_out = np.array(Y_out)
    
    if train is not None or test is not None:
        if train is None:
            train = 1 - test
        X_train, X_test, Y_train, Y_test = train_test_split(X_out, Y_out, train_size=train, test_size=test, **kwargs)
        return X_train, Y_train, X_test, Y_test
    
    return X_out, Y_out


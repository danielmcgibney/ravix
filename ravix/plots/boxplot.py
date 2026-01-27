from ravix.modeling.parse_formula import parse_formula
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple

def boxplot(
    formula: Optional[str] = None, 
    data: Optional[pd.DataFrame] = None, 
    xcolor: str = "blue", 
    ycolor: str = "red", 
    color: Optional[Union[str, List[str]]] = None,
    title: str = "Boxplots of Variables",
    xlab: str = "Variable",
    ylab: str = "Value",
    figsize: Tuple[float, float] = (10, 6),
    **kwargs
) -> None:
    """
    Create boxplots for visualizing distributions of numeric variables.
    
    This function produces three types of boxplots depending on the input:
    1. No formula: Boxplots for all numeric columns in the dataset
    2. Formula with numeric predictors: Side-by-side boxplots of Y and X variables
    3. Formula with a single categorical predictor: Grouped boxplot showing Y distribution across categories
    
    Parameters
    ----------
    formula : str, optional
        Formula specifying the relationship (e.g., "Y ~ X" or "mpg ~ cyl").
        If None, creates boxplots for all numeric columns.
    data : pd.DataFrame, optional
        DataFrame containing the variables. If formula is a DataFrame, 
        it will be used as data and formula will be set to None.
    xcolor : str, default="blue"
        Color for predictor variable boxplots. Ignored if `color` is specified.
    ycolor : str, default="red"
        Color for response variable boxplot. Ignored if `color` is specified.
    color : str or list, optional
        Override for xcolor and ycolor:
        - str: Single color applied to all boxplots
        - list: Colors for each boxplot (length must match number of boxes)
        Note: For categorical predictors (numeric ~ categorical), ycolor is 
        ignored. Use color or xcolor to specify box colors.
    title : str, default="Boxplots of Variables"
        Plot title.
    xlab : str, default="Variable"
        X-axis label.
    ylab : str, default="Value"
        Y-axis label.
    figsize : tuple, default=(10, 6)
        Figure size as (width, height) in inches.
    **kwargs : dict
        Additional keyword arguments passed to seaborn.boxplot().
    
    Returns
    -------
    None
        Displays the plot and closes the figure.
    
    Examples
    --------
    >>> # Boxplots of all numeric variables
    >>> boxplot(data=df)
    
    >>> # Grouped boxplot: numeric response by categorical predictor
    >>> boxplot("mpg ~ cyl", data=mtcars, color="green")
    
    >>> # Multiple boxplots with custom colors and size
    >>> boxplot("mpg ~ hp + wt", data=mtcars, color=["red", "blue", "green"], figsize=(12, 8))
    
    Notes
    -----
    - For categorical predictors, the function creates a grouped boxplot
    - For numeric predictors, side-by-side boxplots are created
    - The function automatically handles categorical encoding from parse_formula
    """
    if isinstance(formula, pd.DataFrame):
        data = formula
        formula = None
    
    if formula is not None:
        # Parse the original formula to get variable names before transformation
        original_formula = formula
        # Internally appends "+0"" to avoid including an intercept term in the plotted variables.
        formula = formula + "+0"
        Y_out, X_out = parse_formula(formula, data)
        Y_name = Y_out.name
        
        # Extract the original predictor variable name from the formula
        # This handles the case where parse_formula transforms categorical variables
        original_x_var = original_formula.split('~')[1].strip().split('+')[0].strip()
        
        # Check if we have a single categorical predictor (special case)
        if (X_out.shape[1] >= 1 and original_x_var in data.columns and 
            (pd.api.types.is_categorical_dtype(data[original_x_var]) or 
             data[original_x_var].dtype == object)):
            
            # Special case: Y is numeric, X is categorical
            plot_data = pd.DataFrame({
                Y_name: Y_out,
                original_x_var: data[original_x_var]
            })
            
            # Warn if ycolor is specified
            if color is None and ycolor != "red":
                import warnings
                warnings.warn(
                    "ycolor is ignored for categorical predictor plots. "
                    "Use 'color' argument to specify colors for the boxes.",
                    UserWarning
                )
            
            # Determine color/palette for this special case
            if color is not None:
                if isinstance(color, str):
                    # Single color for all boxes - need to create palette dict
                    categories = plot_data[original_x_var].unique()
                    palette = {cat: color for cat in categories}
                elif isinstance(color, list):
                    # Number of boxes = number of unique categories
                    n_categories = plot_data[original_x_var].nunique()
                    if len(color) != n_categories:
                        raise ValueError(f"Length of color vector ({len(color)}) must match number of boxplots ({n_categories})")
                    # Create palette mapping categories to colors (use actual category values, not strings)
                    categories = sorted(plot_data[original_x_var].unique())
                    palette = {cat: color[i] for i, cat in enumerate(categories)}
            elif xcolor != "blue":
                # Use xcolor if specified (not default) - need to create palette dict
                categories = plot_data[original_x_var].unique()
                palette = {cat: xcolor for cat in categories}
            else:
                # Default color - need to create palette dict
                categories = plot_data[original_x_var].unique()
                palette = {cat: "blue" for cat in categories}
            
            plt.figure(figsize=figsize)
            # Fix for FutureWarning: assign x variable to hue and set legend=False
            sns.boxplot(x=original_x_var, y=Y_name, hue=original_x_var, 
                       data=plot_data, palette=palette, legend=False, **kwargs)
            plt.title(title)
            plt.xlabel(xlab if xlab != "Variable" else original_x_var)
            plt.ylabel(ylab if ylab != "Value" else Y_name)
            plt.tight_layout()
            plt.show()
            plt.clf()
            plt.close()
            return
        
        # Otherwise, proceed with normal case: multiple numeric predictors
        # Filter out intercept and non-numeric columns from X_out
        if hasattr(X_out, 'columns'):
            numeric_predictors = X_out.select_dtypes(include=[np.number])
            # Remove intercept columns
            numeric_predictors = numeric_predictors.drop(['Intercept', 'const'], axis=1, errors='ignore')
        else:
            numeric_predictors = X_out
        
        # Combine Y and numeric predictors
        if isinstance(Y_out, pd.Series):
            plot_data = pd.concat([Y_out, numeric_predictors], axis=1)
        else:
            plot_data = pd.concat([pd.Series(Y_out, name=Y_name), numeric_predictors], axis=1)
        
        plot_data_melted = plot_data.melt(var_name='Variable', value_name='Value')
        
        # Determine palette based on color argument
        if color is not None:
            if isinstance(color, list):
                # Vector of colors - must match number of variables
                n_vars = len(plot_data.columns)
                if len(color) != n_vars:
                    raise ValueError(f"Length of color vector ({len(color)}) must match number of boxplots ({n_vars})")
                palette = {col: color[i] for i, col in enumerate(plot_data.columns)}
            else:
                # Single color for all
                palette = {col: color for col in plot_data.columns}
        else:
            # Use xcolor and ycolor
            palette = {Y_name: ycolor}
            palette.update({col: xcolor for col in numeric_predictors.columns})
    else:
        # No formula provided, use all numeric columns
        plot_data = data.select_dtypes(include=[np.number])
        plot_data_melted = plot_data.melt(var_name='Variable', value_name='Value')
        
        # Determine palette based on color argument
        if color is not None:
            if isinstance(color, list):
                # Vector of colors - must match number of variables
                n_vars = len(plot_data.columns)
                if len(color) != n_vars:
                    raise ValueError(f"Length of color vector ({len(color)}) must match number of boxplots ({n_vars})")
                palette = {col: color[i] for i, col in enumerate(plot_data.columns)}
            else:
                # Single color for all
                palette = {col: color for col in plot_data.columns}
        else:
            # Use xcolor for all variables
            palette = {var: xcolor for var in plot_data_melted['Variable'].unique()}
    
    plt.figure(figsize=figsize)
    
    # Fix for the FutureWarning: assign x variable to hue and set legend=False
    sns.boxplot(x='Variable', y='Value', hue='Variable', data=plot_data_melted, 
                palette=palette, legend=False, dodge=False, **kwargs)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()

from ravix.modeling.parse_formula import parse_formula
from ravix.plots._theme import get_theme, _resolve_figsize
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple

def boxplot(
    formula: Optional[str] = None, 
    data: Optional[pd.DataFrame] = None, 
    color: Optional[Union[str, List[str]]] = "blue",
    title: str = "Boxplots of Variables",
    xlab: str = "Variable",
    ylab: str = "Value",
    figsize: Optional[Tuple[float, float]] = None,
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
    color : str or list, optional
        Specify color fill:
        - str: Single color applied to all boxplots
        - list: Colors for each boxplot (length must match number of boxes)
        Note: For categorical predictors (numeric ~ categorical).
    title : str, default="Boxplots of Variables"
        Plot title.
    xlab : str, default="Variable"
        X-axis label.
    ylab : str, default="Value"
        Y-axis label.
    figsize : tuple, default=None
        Figure size as (width, height) in inches. Defaults to active theme.
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
    # Resolve figsize and font sizes from theme
    figsize = _resolve_figsize(figsize)
    _theme = get_theme()
    title_fontsize = _theme["title_fontsize"]
    label_fontsize = _theme["label_fontsize"]
    tick_fontsize  = _theme["tick_fontsize"]

    if not isinstance(formula, str) and formula is not None:
        data = pd.DataFrame(formula)
        formula = None
    
    if formula is not None:
        # Parse the original formula to get variable names before transformation
        original_formula = formula
        # Internally appends "+0" to avoid including an intercept term in the plotted variables.
        formula = formula + "+0"
        Y_out, X_out = parse_formula(formula, data)
        Y_name = Y_out.name if Y_out is not None else None
        
        # Extract the original predictor variable name from the formula
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
            
            # Determine color/palette for this special case
            if color is not None:
                if isinstance(color, str):
                    categories = plot_data[original_x_var].unique()
                    palette = {cat: color for cat in categories}
                elif isinstance(color, list):
                    n_categories = plot_data[original_x_var].nunique()
                    if len(color) != n_categories:
                        raise ValueError(f"Length of color vector ({len(color)}) must match number of boxplots ({n_categories})")
                    categories = sorted(plot_data[original_x_var].unique())
                    palette = {cat: color[i] for i, cat in enumerate(categories)}

            plt.figure(figsize=figsize)
            sns.boxplot(x=original_x_var, y=Y_name, hue=original_x_var, 
                       data=plot_data, palette=palette, legend=False, **kwargs)
            plt.title(title, fontsize=title_fontsize)
            plt.xlabel(xlab if xlab != "Variable" else original_x_var, fontsize=label_fontsize)
            plt.ylabel(ylab if ylab != "Value" else Y_name, fontsize=label_fontsize)
            plt.tick_params(labelsize=tick_fontsize)
            plt.tight_layout()
            plt.show()
            plt.clf()
            plt.close()
            return
        
        # Otherwise, proceed with normal case: multiple numeric predictors
        if hasattr(X_out, 'columns'):
            numeric_predictors = X_out.select_dtypes(include=[np.number])
            numeric_predictors = numeric_predictors.drop(['Intercept', 'const'], axis=1, errors='ignore')
        else:
            numeric_predictors = X_out
        
        if Y_out is not None:
            Y_series = Y_out if isinstance(Y_out, pd.Series) else pd.Series(Y_out, name=Y_name)
            plot_data = pd.concat([Y_series, numeric_predictors], axis=1)
        else:
            plot_data = numeric_predictors  
            
        plot_data_melted = plot_data.melt(var_name='Variable', value_name='Value')
        
        if color is not None:
            if isinstance(color, list):
                n_vars = len(plot_data.columns)
                if len(color) != n_vars:
                    raise ValueError(f"Length of color vector ({len(color)}) must match number of boxplots ({n_vars})")
                palette = {col: color[i] for i, col in enumerate(plot_data.columns)}
            else:
                palette = {col: color for col in plot_data.columns}

    else:
        # No formula provided, use all numeric columns
        plot_data = data.select_dtypes(include=[np.number])
        plot_data_melted = plot_data.melt(var_name='Variable', value_name='Value')
        
        if color is not None:
            if isinstance(color, list):
                n_vars = len(plot_data.columns)
                if len(color) != n_vars:
                    raise ValueError(f"Length of color vector ({len(color)}) must match number of boxplots ({n_vars})")
                palette = {col: color[i] for i, col in enumerate(plot_data.columns)}
            else:
                palette = {col: color for col in plot_data.columns}

    plt.figure(figsize=figsize)
    sns.boxplot(x='Variable', y='Value', hue='Variable', data=plot_data_melted, 
                palette=palette, legend=False, dodge=False, **kwargs)
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(xlab, fontsize=label_fontsize)
    plt.ylabel(ylab, fontsize=label_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()

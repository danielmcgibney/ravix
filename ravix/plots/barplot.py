from ravix.modeling.parse_formula import parse_formula
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple

def barplot(
    formula: Optional[str] = None, 
    data: Optional[pd.DataFrame] = None, 
    color: Optional[Union[str, List[str]]] = None, 
    xcolor: str = "blue", 
    ycolor: str = "red", 
    title: str = "Barplots of Variables", 
    xlab: str = "Variable", 
    ylab: str = "Value", 
    agg: Optional[str] = "mean", 
    horizontal: bool = False, 
    figsize: Tuple[float, float] = (10, 6),
    **kwargs
) -> None:
    """
    Create bar plots for visualizing aggregated or individual values of variables.
    
    This function produces different types of bar plots depending on the input:
    1. DataFrame only: Bar plots for all numeric columns with aggregation
    2. DataFrame without aggregation: Plots all values as separate bars
    3. Formula with numeric variables: Side-by-side bars of Y and X variables
    4. Formula Y ~ categorical: Numeric Y aggregated by categories
    5. Formula Y ~ categorical without aggregation: Pre-aggregated data by categories
    6. Horizontal orientation: Any of the above with horizontal bars
    7. Single color: Apply one color to all bars
    8. Multiple colors: Apply different colors to each bar (cycles if needed)
    
    Special case: Single row or column of numeric data without aggregation creates
    individual bars for each value.
    
    Parameters
    ----------
    formula : str, optional
        Formula specifying the relationship (e.g., "Y ~ X1 + X2" or "salary ~ department").
        If None, creates bar plots for all numeric columns.
    data : pd.DataFrame, optional
        DataFrame containing the variables. If formula is a DataFrame or array,
        it will be used as data and formula will be set to None.
    color : str or list, optional
        Override for xcolor and ycolor:
        - str: Single color applied to all bars
        - list: Colors for each bar (cycles if fewer colors than bars)
        Note: For categorical predictors (numeric ~ categorical), ycolor is 
        ignored. Use color or xcolor to specify bar colors.
    xcolor : str, default="blue"
        Color for predictor variable bars. Ignored if `color` is specified.
    ycolor : str, default="red"
        Color for response variable bars. Ignored if `color` is specified.
    title : str, default="Barplots of Variables"
        Plot title.
    xlab : str, default="Variable"
        X-axis label.
    ylab : str, default="Value"
        Y-axis label.
    agg : str or None, default="mean"
        Aggregation method: "mean", "median", "sum", "count", or None.
        If None, plots all values without aggregation. For single row/column
        numeric data, each value becomes a separate bar. For categorical 
        predictors, expects pre-aggregated data (one row per category).
    horizontal : bool, default=False
        If True, creates horizontal bar plot.
    figsize : tuple, default=(10, 6)
        Figure size as (width, height) in inches.
    **kwargs : dict
        Additional keyword arguments passed to seaborn.barplot().
    
    Returns
    -------
    None
        Displays the plot and closes the figure.
    
    Examples
    --------
    >>> # Case 1: DataFrame with aggregation
    >>> barplot(data=df, agg="mean")
    
    >>> # Case 2: DataFrame without aggregation (plots all values)
    >>> barplot(data=df, agg=None)
    
    >>> # Case 3: Formula with numeric variables
    >>> barplot("price ~ age + mileage", data=cars, agg="median")
    
    >>> # Case 4: Y ~ categorical (numeric Y split by categories)
    >>> barplot("salary ~ department", data=employees, agg="mean")
    
    >>> # Case 5: Y ~ categorical without aggregation (pre-aggregated data)
    >>> barplot("avg_salary ~ department", data=dept_summary, agg=None)
    
    >>> # Case 6: Horizontal bars
    >>> barplot("salary ~ department", data=employees, horizontal=True)
    
    >>> # Case 7: Single color for all bars
    >>> barplot(data=df, color="green")
    
    >>> # Case 8: List of colors for each bar
    >>> barplot(data=df, color=["red", "green", "blue"])
    
    >>> # Special case: Single row/column without aggregation
    >>> barplot(data=pd.DataFrame([[10, 20, 30]]), agg=None)  # 3 bars
    
    Notes
    -----
    - For categorical predictors, bars represent aggregated Y values per category
    - For numeric predictors with aggregation, each variable gets its own bar
    - When agg=None with categorical data, each category must appear only once
    - When agg=None with single row/column numeric data, each value gets a bar
    - Color list cycles if fewer colors provided than bars needed
    """
    # Handle color argument
    color_list = None
    if color is not None:
        if isinstance(color, (list, tuple)):
            # List of colors - save for later to create palette
            color_list = color
        else:
            # Single color - use for both xcolor and ycolor
            xcolor = color
            ycolor = color
    
    # Handle case where first argument is a DataFrame
    if isinstance(formula, pd.DataFrame):
        data = formula
        formula = None
    
    # Handle case where first argument is a NumPy array
    if isinstance(formula, np.ndarray):
        # Convert array to DataFrame with generic column names
        if formula.ndim == 1:
            data = pd.DataFrame({'Values': formula})
        elif formula.ndim == 2:
            if formula.shape[1] == 1:
                # Single column - each row is a separate bar
                data = pd.DataFrame(formula, columns=['Values'])
                # Create index labels for each bar
                data.index = [f'Item{i+1}' for i in range(formula.shape[0])]
            elif formula.shape[0] == 1:
                # Single row - each column is a variable
                data = pd.DataFrame(formula, columns=[f'Var{i+1}' for i in range(formula.shape[1])])
            else:
                # Multiple rows and columns
                data = pd.DataFrame(formula, columns=[f'Var{i+1}' for i in range(formula.shape[1])])
        formula = None
    
    # Validate aggregation method
    valid_agg = ["mean", "median", "sum", "count", None]
    if agg not in valid_agg:
        raise ValueError(f"agg must be one of {valid_agg}")
    
    # Create a new figure
    plt.figure(figsize=figsize)
    
    if formula is not None:
        # Check if this is a Y ~ categorical case by parsing the formula manually
        if '~' in formula:
            parts = formula.split('~')
            if len(parts) == 2:
                y_var = parts[0].strip()
                x_vars = [x.strip() for x in parts[1].split('+') if x.strip()]
                
                # Check if we have a single variable that's categorical
                if len(x_vars) == 1 and x_vars[0] in data.columns:
                    x_var = x_vars[0]
                    if data[x_var].dtype == 'object' or pd.api.types.is_categorical_dtype(data[x_var]):
                        # Warn if ycolor is specified
                        if color is None and ycolor != "red":
                            import warnings
                            warnings.warn(
                                "ycolor is ignored for categorical predictor plots. "
                                "Use 'color' argument to specify colors for the bars.",
                                UserWarning
                            )
                        
                        # Case: Y ~ categorical variable
                        plot_df = pd.DataFrame({
                            y_var: data[y_var],
                            x_var: data[x_var]
                        })
                        
                        # Apply aggregation
                        if agg is None:
                            # No aggregation - expect data to already be aggregated (single row per category)
                            # Check if data is already in the right format
                            category_counts = plot_df[x_var].value_counts()
                            if category_counts.max() > 1:
                                raise ValueError(
                                    f"With agg=None, each category must appear only once. "
                                    f"Found categories with multiple values. "
                                    f"Please aggregate your data first or specify an aggregation method."
                                )
                            plot_data = plot_df
                            y_label = y_var
                        else:
                            # Aggregate by category
                            if agg == "count":
                                plot_data = plot_df.groupby(x_var)[y_var].count().reset_index()
                            else:
                                plot_data = plot_df.groupby(x_var)[y_var].agg(agg).reset_index()
                            y_label = f"{agg.capitalize()} of {y_var}"
                        
                        # Create bar plot with categories on x-axis
                        if horizontal:
                            # Use color_list if provided, otherwise use appropriate color
                            if color_list is not None:
                                n_categories = plot_data[x_var].nunique()
                                if len(color_list) < n_categories:
                                    # Cycle colors if not enough provided
                                    palette_to_use = {cat: color_list[i % len(color_list)] 
                                                     for i, cat in enumerate(plot_data[x_var].unique())}
                                else:
                                    palette_to_use = {cat: color_list[i] 
                                                     for i, cat in enumerate(plot_data[x_var].unique())}
                                sns.barplot(y=x_var, x=y_var, data=plot_data, 
                                           palette=palette_to_use, errorbar=None, **kwargs)
                            elif xcolor != "blue":
                                # Use xcolor if specified (not default)
                                categories = plot_data[x_var].unique()
                                palette_to_use = {cat: xcolor for cat in categories}
                                sns.barplot(y=x_var, x=y_var, data=plot_data, 
                                           palette=palette_to_use, errorbar=None, **kwargs)
                            else:
                                # Default color
                                sns.barplot(y=x_var, x=y_var, data=plot_data, 
                                           color=ycolor, errorbar=None, **kwargs)
                            plt.ylabel(x_var)
                            plt.xlabel(y_label)
                        else:
                            # Use color_list if provided, otherwise use appropriate color
                            if color_list is not None:
                                n_categories = plot_data[x_var].nunique()
                                if len(color_list) < n_categories:
                                    # Cycle colors if not enough provided
                                    palette_to_use = {cat: color_list[i % len(color_list)] 
                                                     for i, cat in enumerate(plot_data[x_var].unique())}
                                else:
                                    palette_to_use = {cat: color_list[i] 
                                                     for i, cat in enumerate(plot_data[x_var].unique())}
                                sns.barplot(x=x_var, y=y_var, data=plot_data, 
                                           palette=palette_to_use, errorbar=None, **kwargs)
                            elif xcolor != "blue":
                                # Use xcolor if specified (not default)
                                categories = plot_data[x_var].unique()
                                palette_to_use = {cat: xcolor for cat in categories}
                                sns.barplot(x=x_var, y=y_var, data=plot_data, 
                                           palette=palette_to_use, errorbar=None, **kwargs)
                            else:
                                # Default color
                                sns.barplot(x=x_var, y=y_var, data=plot_data, 
                                           color=ycolor, errorbar=None, **kwargs)
                            plt.xlabel(x_var)
                            plt.ylabel(y_label)
                        
                        plt.title(title)
                        plt.tight_layout()
                        plt.show()
                        plt.clf()
                        plt.close()
                        return
        
        # If we get here, use parse_formula for numeric variables
        formula_with_intercept = formula + "+0"
        Y_out, X_out = parse_formula(formula_with_intercept, data)
        Y_name, X_names = Y_out.name, X_out.columns
        
        # Combine Y and X data for bar plots
        plot_data = pd.concat([pd.Series(Y_out, name=Y_name), X_out], axis=1)
        
        # Apply aggregation
        if agg is None:
            # No aggregation - data should already be in final form
            # Melt without aggregation - each row becomes a bar
            plot_data_melted = plot_data.melt(var_name='Variable', value_name='Value')
        else:
            # Aggregate each variable
            if agg == "count":
                agg_values = {col: plot_data[col].count() for col in plot_data.columns}
            else:
                agg_values = {col: plot_data[col].agg(agg) for col in plot_data.columns}
            
            plot_data_melted = pd.DataFrame({
                'Variable': list(agg_values.keys()),
                'Value': list(agg_values.values())
            })
        
        # Create a color mapping for columns
        # Use actual variables from melted data to ensure all keys are present
        all_vars = plot_data_melted['Variable'].unique()
        palette = {Y_name: ycolor}
        palette.update({x: xcolor for x in X_names})
        
        # Override with color_list if provided
        if color_list is not None:
            palette = {all_vars[i]: color_list[i % len(color_list)] for i in range(len(all_vars))}
        
        # Create the bar plot
        if horizontal:
            sns.barplot(y='Variable', x='Value', data=plot_data_melted, hue='Variable', 
                       dodge=False, palette=palette, errorbar=None, legend=False, **kwargs)
            plt.xlabel(ylab if agg is None else f"{agg.capitalize()} {ylab}")
            plt.ylabel(xlab)
        else:
            sns.barplot(x='Variable', y='Value', data=plot_data_melted, hue='Variable', 
                       dodge=False, palette=palette, errorbar=None, legend=False, **kwargs)
            plt.ylabel(ylab if agg is None else f"{agg.capitalize()} {ylab}")
            plt.xlabel(xlab)
    
    else:
        # Case: No formula provided, use all numeric variables in the data
        plot_data = data.select_dtypes(include=[np.number])
        
        # Apply aggregation
        if agg is None:
            # No aggregation - melt the data
            # Special case: single numeric column with multiple rows
            # Transpose so each row becomes a column (which melt will handle correctly)
            if len(plot_data.columns) == 1 and len(plot_data) > 1:
                # Transpose: rows become columns, index values become column names
                original_index = plot_data.index.astype(str)
                plot_data = plot_data.T
                plot_data.columns = original_index
            
            # Standard melt: each column becomes a bar
            plot_data_melted = plot_data.melt(var_name='Variable', value_name='Value')
        else:
            # Aggregate each variable
            if agg == "count":
                agg_values = {col: plot_data[col].count() for col in plot_data.columns}
            else:
                agg_values = {col: plot_data[col].agg(agg) for col in plot_data.columns}
            
            plot_data_melted = pd.DataFrame({
                'Variable': list(agg_values.keys()),
                'Value': list(agg_values.values())
            })
        
        # Create a single color mapping for all variables
        palette = {var: xcolor for var in plot_data_melted['Variable'].unique()}
        
        # Override with color_list if provided
        if color_list is not None:
            vars_in_order = plot_data_melted['Variable'].unique()
            palette = {vars_in_order[i]: color_list[i % len(color_list)] for i in range(len(vars_in_order))}
        
        # Create the bar plot
        if horizontal:
            sns.barplot(y='Variable', x='Value', data=plot_data_melted, hue='Variable', 
                       dodge=False, palette=palette, errorbar=None, legend=False, **kwargs)
            plt.xlabel(ylab if agg is None else f"{agg.capitalize()} {ylab}")
            plt.ylabel(xlab)
        else:
            sns.barplot(x='Variable', y='Value', data=plot_data_melted, hue='Variable', 
                       dodge=False, palette=palette, errorbar=None, legend=False, **kwargs)
            plt.ylabel(ylab if agg is None else f"{agg.capitalize()} {ylab}")
            plt.xlabel(xlab)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()

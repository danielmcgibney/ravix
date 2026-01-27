import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ravix.modeling.parse_formula import parse_formula
from typing import Union, Optional, Tuple, Literal
import matplotlib.patches as mpatches

def plot_cor(
    formula: Union[str, pd.DataFrame],
    data: Optional[pd.DataFrame] = None,
    type: Literal[1, 2, 3, 4, 5] = 1,
    title: str = "Correlation Matrix",
    xlab: str = "Variables",
    ylab: str = "Variables",
    figsize: Tuple[float, float] = (10, 8),
    **kwargs
) -> None:
    """
    Create a correlation matrix heatmap with various visualization styles.
    
    This function generates correlation matrix visualizations with five different
    display types, ranging from detailed annotations to clean visual representations.
    Automatically handles formulas or DataFrames and computes correlations for
    numeric variables only.
    
    Parameters
    ----------
    formula : str or pd.DataFrame
        Formula specifying variables (e.g., "Y ~ X1 + X2") or DataFrame.
        If DataFrame provided, all numeric columns are used.
    data : pd.DataFrame, optional
        DataFrame containing variables when formula is provided.
    type : {1, 2, 3, 4, 5}, default=1
        Visualization style:
        
        Type 1: Split triangle display (default)
            - Diagonal: Black squares
            - Upper triangle: Color-coded squares (no numbers)
            - Lower triangle: Color-coded numbers (2 decimals)
        
        Type 2: Circle magnitude display
            - Same as Type 1 but upper triangle shows circles sized by correlation magnitude
            - Diagonal: Black squares
            - Lower triangle: Color-coded numbers (2 decimals)
        
        Type 3: Clean color-only display
            - No annotations, only color-coded squares
            - Both triangles show colors
            - Diagonal included with colors
        
        Type 4: Full annotation display
            - All cells show correlation values (2 decimals)
            - Color-coded background
            - Diagonal included
        
        Type 5: Hierarchically clustered heatmap
            - Variables reordered by hierarchical clustering
            - Color-coded with dendrograms
            - Full annotations (2 decimals)
    title : str, default="Correlation Matrix"
        Main title of the plot.
    xlab : str, default="Variables"
        X-axis label.
    ylab : str, default="Variables"
        Y-axis label.
    figsize : tuple, default=(10, 8)
        Figure size as (width, height) in inches.
    **kwargs : dict
        Additional keyword arguments passed to sns.heatmap() or sns.clustermap().
        Common options: cmap, linewidths, linecolor, cbar_kws
    
    Returns
    -------
    None
        Displays the plot and closes the figure.
    
    Examples
    --------
    >>> # Type 1: Split triangle (default)
    >>> plot_cor("mpg ~ hp + wt + disp + drat", data=mtcars)
    
    >>> # Type 2: Circle magnitude in upper triangle
    >>> plot_cor(mtcars[["mpg", "hp", "wt", "disp"]], type=2)
    
    >>> # Type 3: Clean color-only display
    >>> plot_cor("mpg ~ hp + wt + disp", data=mtcars, type=3)
    
    >>> # Type 4: Full annotations
    >>> plot_cor(mtcars, type=4, figsize=(12, 10))
    
    >>> # Type 5: Hierarchically clustered
    >>> plot_cor("mpg ~ hp + wt + disp + drat + qsec", data=mtcars, 
    ...          type=5, cmap="viridis")
    
    >>> # Custom colormap
    >>> plot_cor(mtcars, type=1, cmap="RdBu_r", figsize=(12, 10))
    
    Notes
    -----
    - Only numeric columns are included in correlation computation
    - Diagonal represents correlation of variable with itself (always 1.0)
    - Color scale: Blue (positive), White (zero), Red (negative)
    - Type 1 and 2 provide asymmetric information (visual + numerical)
    - Type 3 is best for presentations (clean, no clutter)
    - Type 4 is most detailed (all correlations visible)
    - Type 5 groups similar variables together via clustering
    - Categorical variables are automatically excluded
    
    Raises
    ------
    ValueError
        If no numeric columns found in data or invalid type specified.
    
    See Also
    --------
    plots : Scatter plot matrix for visual correlation assessment
    """
    # Font size settings
    TITLE_FONTSIZE = 14
    LABEL_FONTSIZE = 12
    TICK_FONTSIZE = 10
    
    # Validate type
    if type not in [1, 2, 3, 4, 5]:
        raise ValueError(
            f"Invalid type '{type}'. "
            f"Must be 1, 2, 3, 4, or 5."
        )
    
    # Handle DataFrame input
    if isinstance(formula, pd.DataFrame):
        data = formula
        formula = None
    
    # Parse formula if provided
    if formula is not None:
        formula = formula + "+0"
        Y_out, X_out = parse_formula(formula, data, drop_first=False)
        Y_name = Y_out.name
        # Combine Y and X data for the correlation matrix
        data = pd.concat([pd.Series(Y_out, name=Y_name), X_out], axis=1)
    
    # Keep only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        raise ValueError(
            "No numeric columns found in the data. "
            "Cannot compute correlation matrix."
        )
    
    # Calculate the correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Set default colormap if not provided
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'coolwarm'
    
    # Type-specific plotting
    if type == 1:
        # Type 1: Split triangle - upper (color only), lower (numbers), diagonal (black)
        _plot_type1(corr_matrix, title, xlab, ylab, figsize, **kwargs)
    
    elif type == 2:
        # Type 2: Circles in upper triangle sized by magnitude
        _plot_type2(corr_matrix, title, xlab, ylab, figsize, **kwargs)
    
    elif type == 3:
        # Type 3: Clean color-only display
        _plot_type3(corr_matrix, title, xlab, ylab, figsize, **kwargs)
    
    elif type == 4:
        # Type 4: Full annotations with 2 decimals
        _plot_type4(corr_matrix, title, xlab, ylab, figsize, **kwargs)
    
    elif type == 5:
        # Type 5: Hierarchically clustered heatmap
        _plot_type5(corr_matrix, title, xlab, ylab, figsize, **kwargs)


def _plot_type1(corr_matrix, title, xlab, ylab, figsize, **kwargs):
    """Type 1: Split triangle with upper color-only, lower numbers, black diagonal."""
    # Create masks
    mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    mask_lower = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)
    mask_diag = ~np.eye(len(corr_matrix), dtype=bool)  # Mask everything except diagonal
    
    # Setup colormap with NaN as black
    cmap = sns.color_palette(kwargs.pop('cmap', 'coolwarm'), as_cmap=True)
    cmap.set_bad(color='black')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot upper triangle (no annotations, color-coded)
    heatmap_kwargs = {k: v for k, v in kwargs.items() if k not in ['annot', 'fmt']}
    heatmap_kwargs.update({'vmin': -1, 'vmax': 1, 'center': 0, 'square': True, 
                           'linewidths': 0.5, 'cbar_kws': {"shrink": 0.8}})
    
    sns.heatmap(corr_matrix, mask=mask_lower, annot=False, cmap=cmap, 
                ax=ax, **heatmap_kwargs)
    
    # Plot lower triangle (with annotations)
    annot_matrix = corr_matrix.map(lambda x: f'{x:.2f}')
    sns.heatmap(corr_matrix, mask=mask_upper, annot=annot_matrix, fmt='', 
                cmap=cmap, ax=ax, cbar=False, **heatmap_kwargs)
    
    # Plot black diagonal squares on top
    corr_diag_only = corr_matrix.copy()
    corr_diag_only.values[mask_diag] = np.nan
    sns.heatmap(corr_diag_only, mask=False, annot=False, cmap=cmap,
                ax=ax, cbar=False, **{k: v for k, v in heatmap_kwargs.items() if k != 'cbar_kws'})
    
    # Set labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    ax.tick_params(labelsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()


def _plot_type2(corr_matrix, title, xlab, ylab, figsize, **kwargs):
    """Type 2: Circles sized by magnitude in upper triangle."""
    # Create masks
    mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    mask_lower = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)
    mask_diag = ~np.eye(len(corr_matrix), dtype=bool)  # Mask everything except diagonal
    
    # Setup colormap
    cmap = sns.color_palette(kwargs.pop('cmap', 'coolwarm'), as_cmap=True)
    cmap.set_bad(color='black')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot base heatmap (lower triangle with numbers)
    heatmap_kwargs = {k: v for k, v in kwargs.items() if k not in ['annot', 'fmt']}
    heatmap_kwargs.update({'vmin': -1, 'vmax': 1, 'center': 0, 'square': True,
                           'linewidths': 0.5, 'cbar_kws': {"shrink": 0.8}})
    
    annot_matrix = corr_matrix.map(lambda x: f'{x:.2f}')
    sns.heatmap(corr_matrix, mask=mask_upper, annot=annot_matrix, fmt='',
                cmap=cmap, ax=ax, **heatmap_kwargs)
    
    # Plot white/neutral upper triangle as background for circles
    upper_neutral = corr_matrix.copy()
    upper_neutral.values[:] = 0  # Set all to 0 (neutral/white in coolwarm)
    sns.heatmap(upper_neutral, mask=mask_lower, annot=False, cmap=cmap,
                ax=ax, cbar=False, **{k: v for k, v in heatmap_kwargs.items() if k != 'cbar_kws'})
    
    # Plot black diagonal squares on top
    corr_diag_only = corr_matrix.copy()
    corr_diag_only.values[mask_diag] = np.nan
    sns.heatmap(corr_diag_only, mask=False, annot=False, cmap=cmap,
                ax=ax, cbar=False, **{k: v for k, v in heatmap_kwargs.items() if k != 'cbar_kws'})
    
    # Add circles to upper triangle
    n = len(corr_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            corr_val = corr_matrix.iloc[i, j]
            # Determine color based on correlation
            if corr_val > 0:
                color = plt.cm.coolwarm(0.75)  # Blue for positive
            elif corr_val < 0:
                color = plt.cm.coolwarm(0.25)  # Red for negative
            else:
                color = plt.cm.coolwarm(0.5)   # White for zero
            
            # Size proportional to absolute correlation
            radius = abs(corr_val) * 0.4
            circle = mpatches.Circle((j + 0.5, i + 0.5), radius, 
                                    color=color, ec='black', linewidth=0.5)
            ax.add_patch(circle)
    
    # Set labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    ax.tick_params(labelsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()


def _plot_type3(corr_matrix, title, xlab, ylab, figsize, **kwargs):
    """Type 3: Clean color-only display, no annotations."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Setup kwargs
    heatmap_kwargs = {k: v for k, v in kwargs.items()}
    heatmap_kwargs.update({'annot': False, 'vmin': -1, 'vmax': 1, 'center': 0,
                           'square': True, 'linewidths': 0.5, 
                           'cbar_kws': {"shrink": 0.8}})
    
    sns.heatmap(corr_matrix, ax=ax, **heatmap_kwargs)
    
    # Set labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    ax.tick_params(labelsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()


def _plot_type4(corr_matrix, title, xlab, ylab, figsize, **kwargs):
    """Type 4: Full annotations with 2 decimals everywhere."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create annotation matrix with 2 decimals
    annot_matrix = corr_matrix.map(lambda x: f'{x:.2f}')
    
    # Setup kwargs
    heatmap_kwargs = {k: v for k, v in kwargs.items() if k not in ['annot', 'fmt']}
    heatmap_kwargs.update({'vmin': -1, 'vmax': 1, 'center': 0, 'square': True,
                           'linewidths': 0.5, 'cbar_kws': {"shrink": 0.8}})
    
    sns.heatmap(corr_matrix, annot=annot_matrix, fmt='', ax=ax, **heatmap_kwargs)
    
    # Set labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    ax.tick_params(labelsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()


def _plot_type5(corr_matrix, title, xlab, ylab, figsize, **kwargs):
    """Type 5: Hierarchically clustered heatmap with dendrograms."""
    # Create annotation matrix with 2 decimals
    annot_matrix = corr_matrix.map(lambda x: f'{x:.2f}')
    
    # Setup kwargs
    heatmap_kwargs = {k: v for k, v in kwargs.items() if k not in ['annot', 'fmt']}
    heatmap_kwargs.update({'vmin': -1, 'vmax': 1, 'center': 0, 
                           'linewidths': 0.5, 'cbar_kws': {"shrink": 0.8},
                           'figsize': figsize})
    
    # Create clustermap
    g = sns.clustermap(corr_matrix, annot=annot_matrix, fmt='', 
                      **heatmap_kwargs)
    
    # Set title (clustermap doesn't have direct title support)
    g.fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Adjust labels
    g.ax_heatmap.set_xlabel(xlab, fontsize=12)
    g.ax_heatmap.set_ylabel(ylab, fontsize=12)
    g.ax_heatmap.tick_params(labelsize=10)
    
    plt.show()
    plt.clf()
    plt.close()

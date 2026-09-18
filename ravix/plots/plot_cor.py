import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from ravix.modeling.parse_formula import parse_formula
from ravix.plots._theme import get_theme, _resolve_figsize
from typing import Union, Optional, Tuple, Literal
import matplotlib.patches as mpatches


_DUMMY_WARNING_THRESHOLD = 30
_DUMMY_WARNING_MESSAGE = "Note: Consider dummy=False to avoid long processing times."

def plot_cor(
    formula: Union[str, pd.DataFrame],
    data: Optional[pd.DataFrame] = None,
    style: Literal[1, 2, 3, 4] = 1,
    title: str = "",
    xlab: str = "",
    ylab: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = True,
    dummy: bool = True,
    **kwargs
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Create a correlation matrix heatmap with various visualization styles.
    
    This function generates correlation matrix visualizations with four different
    display types, ranging from detailed annotations to clean visual representations.
    Automatically handles formulas or DataFrames. For direct DataFrame input,
    categorical variables are dummy-encoded by default and all levels are retained.
    Formula input always uses all dummy levels (``drop_first=False``).
    
    Parameters
    ----------
    formula : str or pd.DataFrame
        Formula specifying variables (e.g., "Y ~ X1 + X2") or DataFrame.
        If a DataFrame is provided, numeric columns are used directly and, by
        default, categorical/string/bool columns are expanded to dummy variables.
    data : pd.DataFrame, optional
        DataFrame containing variables when formula is provided.
    style : {1, 2, 3, 4}, default=1
        Visualization style:
        
        Style 1: Circle magnitude display (default)
            - Diagonal: Black squares
            - Upper triangle: Circles sized by correlation magnitude
            - Lower triangle: Bold numbers (no color)
        
        Style 2: Split triangle display
            - Diagonal: Black squares
            - Upper triangle: Color-coded squares (no numbers)
            - Lower triangle: Bold numbers (no color)
        
        Style 3: Clean color-only display
            - No annotations, only color-coded squares
            - Both triangles show colors
            - Diagonal included with colors
        
        Style 4: Full annotation display
            - All cells show correlation values (2 decimals)
            - Color-coded background
            - Diagonal included
    title : str, default=""
        Main title of the plot.
    xlab : str, default=""
        X-axis label.
    ylab : str, default=""
        Y-axis label.
    figsize : tuple, default=(10, 8)
        Figure size as (width, height) in inches.
    show : bool, default=True
        Display plot immediately; if False, return (fig, ax) for manual control.
    dummy : bool, default=True
        For direct DataFrame input, convert categorical/string/bool columns to
        dummy variables before computing correlations. All levels are retained
        (``drop_first=False``). Set to False to use numeric columns only. This
        argument does not alter formula behavior; formulas always retain all
        dummy levels.
    **kwargs : dict
        Additional keyword arguments passed to sns.heatmap(), plus optional
        font size overrides (extracted before being passed to heatmap):
        
        Font size kwargs (Colab-optimized defaults shown):
          title_fontsize : int, default=10   Plot title
          label_fontsize : int, default=9    Axis labels (xlab/ylab)
          tick_fontsize  : int, default=9    Tick labels on both axes
          annot_fontsize : int, default=9    Correlation value annotations (styles 1, 2, 4)
          cbar_fontsize  : int, default=9    Colorbar tick labels
        
        Common heatmap options: cmap, linewidths, linecolor, cbar_kws, cbar
    
    Returns
    -------
    Optional[Tuple[plt.Figure, plt.Axes]]
        None if show=True, else (fig, ax) tuple
        
        WARNING: If show=False, caller is responsible for closing the figure
        to prevent memory leaks. Use plt.close(fig) when done.
    
    Examples
    --------
    >>> # Style 1: Circle magnitude in upper triangle with values in lower (default)
    >>> plot_cor("mpg ~ hp + wt + disp + drat", data=mtcars)
    
    >>> # Style 2: Colors in upper and values in lower
    >>> plot_cor(mtcars[["mpg", "hp", "wt", "disp"]], style=2)
    
    >>> # Style 3: Clean color-only display
    >>> plot_cor("mpg ~ hp + wt + disp", data=mtcars, style=3)
    
    >>> # Style 4: Full annotations
    >>> plot_cor(mtcars, style=4, figsize=(12, 10))

    >>> # Numeric-only correlations from a mixed-type DataFrame
    >>> plot_cor(df, dummy=False)
    
    >>> # Custom colormap and no colorbar
    >>> plot_cor(mtcars, style=1, cmap="RdBu_r", cbar=False)
    
    >>> # Manual figure management
    >>> fig, ax = plot_cor(mtcars, style=3, show=False)
    >>> ax.set_title("Custom Title")
    >>> plt.savefig("correlation_matrix.png")
    >>> plt.close(fig)  # Important!
    
    Notes
    -----
    - Direct DataFrame input dummy-encodes categorical variables by default
    - Dummy encoding always retains every category level (``drop_first=False``)
    - Formula input also always retains every category level
    - Diagonal represents correlation of variable with itself (always 1.0)
    - Color scale: Red (negative), White (zero), Blue (positive)
    - Style 1 and 2 provide asymmetric information (visual + numerical)
    - Style 3 is best for presentations (clean, no clutter)
    - Style 4 is most detailed (all correlations visible)
    - Use ``dummy=False`` with direct DataFrame input to exclude categoricals
    
    Raises
    ------
    ValueError
        If no numeric columns found in data or invalid style specified.
    
    See Also
    --------
    plots : Scatter plot matrix for visual correlation assessment
    """
    # Resolve fontsizes: kwargs can override, theme is the default
    _theme = get_theme()
    title_fontsize = kwargs.pop('title_fontsize', _theme['title_fontsize'])
    label_fontsize = kwargs.pop('label_fontsize', _theme['label_fontsize'])
    tick_fontsize  = kwargs.pop('tick_fontsize',  _theme['tick_fontsize'])
    annot_fontsize = kwargs.pop('annot_fontsize', _theme['label_fontsize'])
    cbar_fontsize  = kwargs.pop('cbar_fontsize',  _theme['tick_fontsize'])

    # Resolve figsize via theme
    figsize = _resolve_figsize(figsize)

    # Validate style
    if style not in [1, 2, 3, 4]:
        raise ValueError(
            f"Invalid style '{style}'. "
            f"Must be 1, 2, 3, or 4."
        )
    
    # Handle DataFrame input. ``dummy`` applies only to this direct-DataFrame
    # route. Formula parsing below always retains all categorical levels.
    direct_dataframe = isinstance(formula, pd.DataFrame)
    if direct_dataframe:
        data = formula.copy()
        formula = None

    status = None

    # Parse formula if provided. Correlation plots intentionally retain every
    # dummy level: unlike regression fitting, there is no identification reason
    # to omit a reference category from a correlation matrix.
    if formula is not None:
        formula = formula + "+0"
        Y_out, X_out = parse_formula(formula, data, drop_first=False)
        Y_name = Y_out.name
        # Combine Y and X data for the correlation matrix
        data = pd.concat([pd.Series(Y_out, name=Y_name), X_out], axis=1)

    elif dummy:
        categorical_cols = [
            col for col in data.columns
            if (
                pd.api.types.is_object_dtype(data[col].dtype)
                or isinstance(data[col].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(data[col].dtype)
                or pd.api.types.is_bool_dtype(data[col].dtype)
            )
        ]

        if categorical_cols:
            # Estimate the number of dummy columns before expansion so a
            # high-cardinality ID/name column can produce an immediate hint.
            n_dummy_columns = sum(data[col].nunique(dropna=True) for col in categorical_cols)
            if n_dummy_columns > _DUMMY_WARNING_THRESHOLD:
                status = _show_temporary_dummy_note()

            # Correlation plots retain all category levels by design. Never use
            # drop_first=True here.
            data = pd.get_dummies(
                data,
                columns=categorical_cols,
                drop_first=False,
                dtype=float,
            )

    # Keep numeric columns after optional categorical expansion.
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        raise ValueError(
            "No numeric columns found in the data. "
            "Cannot compute correlation matrix."
        )
    
    # Calculate the correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Set default colormap if not provided - use stronger diverging palette
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'RdBu'
    
    # Style-specific plotting. The temporary high-cardinality note is cleared
    # when plotting finishes, whether the figure is shown or returned.
    try:
        if style == 1:
            # Style 1: Circles in upper triangle sized by correlation magnitude
            return _plot_type1(corr_matrix, title, xlab, ylab, figsize, show,
                               title_fontsize, label_fontsize, tick_fontsize, annot_fontsize, cbar_fontsize,
                               **kwargs)

        elif style == 2:
            # Style 2: Split triangle - upper (color only), lower (numbers), diagonal (black)
            return _plot_type2(corr_matrix, title, xlab, ylab, figsize, show,
                               title_fontsize, label_fontsize, tick_fontsize, annot_fontsize, cbar_fontsize,
                               **kwargs)

        elif style == 3:
            # Style 3: Clean color-only display
            return _plot_type3(corr_matrix, title, xlab, ylab, figsize, show,
                               title_fontsize, label_fontsize, tick_fontsize, cbar_fontsize,
                               **kwargs)

        elif style == 4:
            # Style 4: Full annotations with 2 decimals
            return _plot_type4(corr_matrix, title, xlab, ylab, figsize, show,
                               title_fontsize, label_fontsize, tick_fontsize, annot_fontsize, cbar_fontsize,
                               **kwargs)
    finally:
        _clear_temporary_dummy_note(status)


def _show_temporary_dummy_note():
    """Display a removable hint for high-cardinality dummy expansion."""
    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            return ("ipython", display(_DUMMY_WARNING_MESSAGE, display_id=True))
    except Exception:
        pass

    # Terminal/script fallback. Keep the message on the current line so it can
    # be erased when processing completes.
    print(_DUMMY_WARNING_MESSAGE, end="", flush=True)
    return ("terminal", len(_DUMMY_WARNING_MESSAGE))


def _clear_temporary_dummy_note(status):
    """Remove the high-cardinality hint without clearing unrelated output."""
    if status is None:
        return

    mode, payload = status
    if mode == "ipython":
        try:
            payload.update("")
        except Exception:
            pass
    else:
        print("\r" + (" " * payload) + "\r", end="", flush=True)


def _apply_tick_fontsizes(ax, tick_fontsize):
    """
    Force tick label fontsizes after tight_layout().

    seaborn heatmaps position x-axis labels on the top spine and
    matplotlib's auto-scaling (triggered by tight_layout) can override
    any fontsize set earlier. Iterating the Text objects directly is
    the only reliable fix.
    """
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(tick_fontsize)


def _plot_type1(corr_matrix, title, xlab, ylab, figsize, show,
                title_fontsize, label_fontsize, tick_fontsize, annot_fontsize, cbar_fontsize,
                **kwargs):
    """Type 1: Circles sized by magnitude in upper triangle."""
    n = len(corr_matrix)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a modified correlation matrix:
    # - Lower triangle: 0 (white, for bold numbers to show up)
    # - Upper triangle: 0 (white, for circles to show up)
    # - Diagonal: 0 (will be painted black)
    display_matrix = corr_matrix.copy()
    for i in range(n):
        for j in range(n):
            if i <= j:  # Upper triangle and diagonal
                display_matrix.iloc[i, j] = 0  # Set to 0 for white
            elif i > j:  # Lower triangle
                display_matrix.iloc[i, j] = 0  # Set to 0 for white
    
    # Setup colormap - stronger diverging palette (RdBu: red=negative, blue=positive)
    cmap_name = kwargs.pop('cmap', 'RdBu')
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    
    # Setup kwargs for heatmap
    heatmap_kwargs = {k: v for k, v in kwargs.items() if k not in ['annot', 'fmt', 'annot_kws']}
    heatmap_kwargs.update({'vmin': -1, 'vmax': 1, 'center': 0, 'square': True,
                           'linewidths': 1.0, 'linecolor': 'black',
                           'cbar_kws': {"shrink": 1.0, "label": "", "format": "%.2f"}})
    
    # Draw the heatmap without annotations (we'll add custom text)
    sns.heatmap(display_matrix, annot=False, cmap=cmap, ax=ax, **heatmap_kwargs)
    
    # Increase colorbar tick label size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    
    # Set equal aspect ratio for square cells
    ax.set_aspect("equal")
    
    # Paint the diagonal black
    _ = [ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, facecolor='black', 
                                    linewidth=1.0, edgecolor='black')) 
         for i in range(n)]
    
    # Add circles to upper triangle ON WHITE background
    # Use power 0.7 transformation for moderate scaling (between linear and sqrt)
    # Border opacity based on correlation magnitude
    _ = [ax.add_patch(mpatches.Circle((j + 0.5, i + 0.5), 
                                      (abs(corr_matrix.iloc[i, j]) ** 0.7) * 0.5,
                                      facecolor=cmap(norm(corr_matrix.iloc[i, j])),
                                      ec='black', 
                                      linewidth=0.5,
                                      alpha=1.0,
                                      edgecolor=(0, 0, 0, abs(corr_matrix.iloc[i, j]) ** 0.7)))
         for i in range(n) for j in range(i + 1, n)]
    
    # Add text annotations in lower triangle
    for i in range(n):
        for j in range(n):
            if i > j:  # Lower triangle
                corr_val = corr_matrix.iloc[i, j]
                ax.text(j + 0.5, i + 0.5, f'{corr_val:.2f}',
                       ha='center', va='center',
                       fontsize=annot_fontsize, fontweight='bold',
                       color='black')
    
    # Force one tick per variable. Seaborn may thin ticks automatically for
    # large matrices, but Ravix should retain every correlation label.
    tick_positions = np.arange(len(corr_matrix)) + 0.5
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(corr_matrix.columns, rotation=90, ha='center', fontweight='bold')
    ax.set_yticklabels(corr_matrix.index, rotation=0, fontweight='bold')
    
    # Set labels
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=20)
    if xlab:
        ax.set_xlabel(xlab, fontsize=label_fontsize)
    else:
        ax.set_xlabel('')
    if ylab:
        ax.set_ylabel(ylab, fontsize=label_fontsize)
    else:
        ax.set_ylabel('')
    ax.tick_params(labelsize=tick_fontsize)

    plt.tight_layout()
    _apply_tick_fontsizes(ax, tick_fontsize)  # force size after auto-scaling
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig, ax


def _plot_type2(corr_matrix, title, xlab, ylab, figsize, show,
                title_fontsize, label_fontsize, tick_fontsize, annot_fontsize, cbar_fontsize,
                **kwargs):
    """Type 2: Split triangle with upper color-only, lower bold numbers (no color), black diagonal."""
    n = len(corr_matrix)
    
    # Setup colormap - stronger diverging palette (RdBu: red=negative, blue=positive)
    cmap_name = kwargs.pop('cmap', 'RdBu')
    cmap = sns.color_palette(cmap_name, as_cmap=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a modified correlation matrix with lower triangle set to 0 (white)
    display_matrix = corr_matrix.copy()
    for i in range(n):
        for j in range(n):
            if i > j:  # Lower triangle
                display_matrix.iloc[i, j] = 0  # Set to 0 for white color
    
    # Setup kwargs for heatmap
    heatmap_kwargs = {k: v for k, v in kwargs.items() if k not in ['annot', 'fmt', 'annot_kws']}
    heatmap_kwargs.update({
        'vmin': -1, 
        'vmax': 1, 
        'center': 0, 
        'square': True,
        'linewidths': 1.0, 
        'linecolor': 'black',
        'cbar_kws': {"shrink": 1.0, "label": "", "format": "%.2f"}
    })
    
    # Draw the heatmap without annotations (we'll add custom text)
    sns.heatmap(display_matrix, annot=False, cmap=cmap, ax=ax, **heatmap_kwargs)
    
    # Increase colorbar tick label size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    
    # Set equal aspect ratio for square cells
    ax.set_aspect("equal")
    
    # Paint the diagonal black
    _ = [ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, facecolor='black', 
                                    linewidth=1.0, edgecolor='black')) 
         for i in range(n)]
    
    # Add text annotations in lower triangle - black text
    for i in range(n):
        for j in range(n):
            if i > j:  # Lower triangle
                corr_val = corr_matrix.iloc[i, j]
                ax.text(j + 0.5, i + 0.5, f'{corr_val:.2f}',
                       ha='center', va='center',
                       fontsize=annot_fontsize, fontweight='bold',
                       color='black')
    
    # Force one tick per variable. Seaborn may thin ticks automatically for
    # large matrices, but Ravix should retain every correlation label.
    tick_positions = np.arange(len(corr_matrix)) + 0.5
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(corr_matrix.columns, rotation=90, ha='center', fontweight='bold')
    ax.set_yticklabels(corr_matrix.index, rotation=0, fontweight='bold')
    
    # Set labels
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=20)
    if xlab:
        ax.set_xlabel(xlab, fontsize=label_fontsize)
    else:
        ax.set_xlabel('')
    if ylab:
        ax.set_ylabel(ylab, fontsize=label_fontsize)
    else:
        ax.set_ylabel('')
    ax.tick_params(labelsize=tick_fontsize)

    plt.tight_layout()
    _apply_tick_fontsizes(ax, tick_fontsize)  # force size after auto-scaling
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig, ax


def _plot_type3(corr_matrix, title, xlab, ylab, figsize, show,
                title_fontsize, label_fontsize, tick_fontsize, cbar_fontsize,
                **kwargs):
    """Type 3: Clean color-only display, no annotations."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Setup kwargs - stronger diverging palette (RdBu: red=negative, blue=positive)
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'RdBu'
    
    heatmap_kwargs = {k: v for k, v in kwargs.items()}
    heatmap_kwargs.update({'annot': False, 'vmin': -1, 'vmax': 1, 'center': 0,
                           'square': True, 'linewidths': 1.0, 'linecolor': 'black',
                           'cbar_kws': {"shrink": 1.0, "label": "", "format": "%.2f"}})
    
    sns.heatmap(corr_matrix, ax=ax, **heatmap_kwargs)
    
    # Increase colorbar tick label size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    
    # Set equal aspect ratio for square cells
    ax.set_aspect("equal")
    
    # Force one tick per variable. Seaborn may thin ticks automatically for
    # large matrices, but Ravix should retain every correlation label.
    tick_positions = np.arange(len(corr_matrix)) + 0.5
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(corr_matrix.columns, rotation=90, ha='center', fontweight='bold')
    ax.set_yticklabels(corr_matrix.index, rotation=0, fontweight='bold')
    
    # Set labels
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=20)
    if xlab:
        ax.set_xlabel(xlab, fontsize=label_fontsize)
    else:
        ax.set_xlabel('')
    if ylab:
        ax.set_ylabel(ylab, fontsize=label_fontsize)
    else:
        ax.set_ylabel('')
    ax.tick_params(labelsize=tick_fontsize)

    plt.tight_layout()
    _apply_tick_fontsizes(ax, tick_fontsize)  # force size after auto-scaling
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig, ax


def _plot_type4(corr_matrix, title, xlab, ylab, figsize, show,
                title_fontsize, label_fontsize, tick_fontsize, annot_fontsize, cbar_fontsize,
                **kwargs):
    """Type 4: Full annotations with 2 decimals everywhere."""
    n = len(corr_matrix)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Setup colormap - stronger diverging palette (RdBu: red=negative, blue=positive)
    cmap_name = kwargs.pop('cmap', 'RdBu')
    cmap = sns.color_palette(cmap_name, as_cmap=True)
    
    # Setup kwargs for heatmap
    heatmap_kwargs = {k: v for k, v in kwargs.items() if k not in ['annot', 'fmt']}
    heatmap_kwargs.update({'vmin': -1, 'vmax': 1, 'center': 0, 'square': True,
                           'linewidths': 1.0, 'linecolor': 'black',
                           'cbar_kws': {"shrink": 1.0, "label": "", "format": "%.2f"}})
    
    # Draw the heatmap without annotations (we'll add custom text)
    sns.heatmap(corr_matrix, annot=False, cmap=cmap, ax=ax, **heatmap_kwargs)
    
    # Increase colorbar tick label size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    
    # Set equal aspect ratio for square cells
    ax.set_aspect("equal")
    
    # Add text annotations everywhere
    for i in range(n):
        for j in range(n):
            corr_val = corr_matrix.iloc[i, j]
            ax.text(j + 0.5, i + 0.5, f'{corr_val:.2f}',
                   ha='center', va='center',
                   fontsize=annot_fontsize,
                   color='black')
    
    # Force one tick per variable. Seaborn may thin ticks automatically for
    # large matrices, but Ravix should retain every correlation label.
    tick_positions = np.arange(len(corr_matrix)) + 0.5
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(corr_matrix.columns, rotation=90, ha='center', fontweight='bold')
    ax.set_yticklabels(corr_matrix.index, rotation=0, fontweight='bold')
    
    # Set labels
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight='bold', pad=20)
    if xlab:
        ax.set_xlabel(xlab, fontsize=label_fontsize)
    else:
        ax.set_xlabel('')
    if ylab:
        ax.set_ylabel(ylab, fontsize=label_fontsize)
    else:
        ax.set_ylabel('')
    ax.tick_params(labelsize=tick_fontsize)

    plt.tight_layout()
    _apply_tick_fontsizes(ax, tick_fontsize)  # force size after auto-scaling
    if show:
        plt.show()
        plt.close(fig)
        return None
    return fig, ax

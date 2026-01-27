import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple

def plot_bsr(
    model,
    type: str = "predictors",
    top_n: int = 5,
    color: str = "darkgreen",
    title: Optional[str] = None,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    **kwargs
) -> None:
    """
    Create visualizations of best subset regression (BSR) model results.
    
    This function produces three types of plots to help interpret and compare
    models from best subset regression:
    1. Line plot: Shows metric trends across different numbers of predictors
    2. Bar plot: Compares top N models by their metric values
    3. Predictors heatmap: Displays which predictors are in each top model
    
    Parameters
    ----------
    model : object
        Fitted model from bsr() with .bsr attribute containing .results,
        .best_by_k, and .metric.
    type : str, default="predictors"
        Type of visualization:
        - "line": Metric values vs. number of predictors
        - "bar": Bar chart of top N models
        - "predictors": Heatmap showing predictor inclusion in top models
    top_n : int, default=5
        Number of top models to display. Applies to "bar" and "predictors" types.
    color : str, default="darkgreen"
        Color for bars in bar plot. For heatmap, creates a colormap from
        white to the specified color (unless cmap provided in kwargs).
    title : str, optional
        Plot title. If None, automatically generated based on plot type and metric.
    xlab : str, optional
        X-axis label. If None, automatically generated based on plot type.
    ylab : str, optional
        Y-axis label. If None, automatically generated based on metric.
    figsize : tuple, default=(10, 6)
        Figure size as (width, height) in inches.
    **kwargs : dict
        Additional keyword arguments:
        - For "line" type: passed to plt.plot()
        - For "bar" type: passed to plt.bar()
        - For "predictors" type: passed to sns.heatmap()
          (default: annot=False, cmap='Greens', cbar=False)
    
    Returns
    -------
    None
        Displays the plot and closes the figure.
    
    Examples
    --------
    >>> # Run best subset regression
    >>> model = pr.bsr("mpg ~ hp + wt + disp + drat", data=mtcars, metric="adjr2")
    
    >>> # Heatmap showing which predictors are in top models
    >>> plot_bsr(model, type="predictors", top_n=10)
    
    >>> # Line plot of metric across number of predictors
    >>> plot_bsr(model, type="line")
    
    >>> # Bar chart comparing top 5 models
    >>> plot_bsr(model, type="bar", top_n=5, color="green")
    
    >>> # Custom heatmap with annotations and color
    >>> plot_bsr(model, type="predictors", top_n=8, color="darkblue",
    ...          annot=True, figsize=(12, 8))
    
    >>> # Line plot with custom styling
    >>> plot_bsr(model, type="line", color="red", linewidth=2,
    ...          marker="o", markersize=8)
    
    Notes
    -----
    - Models are sorted by metric: adjr2 (descending), others (ascending)
    - Line plot shows best model for each number of predictors
    - Bar plot shows overall top N models regardless of number of predictors
    - Predictors heatmap: colored/filled = predictor included, white/empty = excluded
    - Heatmap uses color argument to create white-to-color gradient (override with cmap in kwargs)
    - Predictors appear in original formula order, not sorted by importance
    - Heatmap cells are rectangular for better readability
    - Use annot=True in kwargs to show 1/0 values in heatmap cells
    """
    # Validate model structure
    if not hasattr(model, "bsr") or not hasattr(model.bsr, "results") or \
       not hasattr(model.bsr, "best_by_k") or not hasattr(model.bsr, "metric"):
        raise ValueError(
            "Model must have .bsr attribute with .results, .best_by_k, and .metric. "
            "Ensure model was created using bsr() function."
        )

    results_df = model.bsr.results.copy()
    feature_col = "Features"
    metric = model.bsr.metric

    metric_column = {
        "adjr2": "Adj. R-squared",
        "aic": "AIC",
        "bic": "BIC",
        "rmse": "RMSE"
    }

    col = metric_column[metric]
    results_df["Num Predictors"] = results_df[feature_col].apply(len)

    # Reconstruct best_df using best_by_k dictionary
    best_rows = []
    for k, feats in model.bsr.best_by_k.items():
        match = results_df[results_df[feature_col].apply(lambda f: sorted(f) == sorted(feats))]
        if not match.empty:
            best_rows.append(match.iloc[0])
    best_df = pd.DataFrame(best_rows)
    best_df["Num Predictors"] = best_df[feature_col].apply(len)

    if type == "line":
        plt.figure(figsize=figsize)
        
        # Extract kwargs for plot
        plot_kwargs = {k: v for k, v in kwargs.items()}
        if 'color' not in plot_kwargs:
            plot_kwargs['color'] = color
        
        plt.plot(best_df["Num Predictors"], best_df[col], marker="o", **plot_kwargs)

        plt.xlabel(xlab if xlab is not None else "Number of Predictors")
        plt.ylabel(ylab if ylab is not None else col)
        plt.title(title if title is not None else f"{col} by Number of Predictors")
        plt.xticks(np.arange(best_df["Num Predictors"].min(),
                             best_df["Num Predictors"].max() + 1, step=1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()

    elif type == "bar":
        ascending = (metric != "adjr2")
        top = best_df.sort_values(by=col, ascending=ascending).head(top_n)

        plt.figure(figsize=figsize)
        
        # Extract kwargs for bar
        bar_kwargs = {k: v for k, v in kwargs.items()}
        if 'color' not in bar_kwargs:
            bar_kwargs['color'] = color
        
        bars = plt.bar(top["Num Predictors"], top[col], width=0.6, **bar_kwargs)

        plt.xlabel(xlab if xlab is not None else "Number of Predictors")
        plt.ylabel(ylab if ylab is not None else col)
        plt.title(title if title is not None else f"Top {top_n} Best Models by {col}")

        min_val, max_val = top[col].min(), top[col].max()
        margin = (max_val - min_val) * 0.1 if max_val != min_val else 1
        plt.ylim(min_val - margin, max_val + margin)

        plt.xticks(top["Num Predictors"])
        plt.grid(False)
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()

    elif type == "predictors":
        ascending = (metric != "adjr2")
        top_models = results_df.sort_values(by=col, ascending=ascending).head(top_n)
        
        # Get all features in their ORIGINAL order from the model specification
        # Not sorted by importance/frequency
        all_features_list = _get_all_features_original_order(model)

        data_matrix = []
        metric_vals = []

        for _, row in top_models.iterrows():
            feats = row[feature_col]
            metric_val = row[col]
            row_data = [1 if f in feats else 0 for f in all_features_list]
            data_matrix.append(row_data)
            metric_vals.append(f"{metric_val:.3f}")

        # Create DataFrame without sorting columns
        heatmap_df = pd.DataFrame(data_matrix, index=metric_vals, columns=all_features_list)
        
        # Calculate appropriate figure size for rectangles
        # Make cells rectangular: wider than tall
        n_predictors = len(all_features_list)
        n_models = len(top_models)
        cell_width = 1.2  # Width per predictor
        cell_height = 0.6  # Height per model
        fig_width = max(cell_width * n_predictors + 2, 8)  # Minimum 8 inches
        fig_height = max(cell_height * n_models + 2, 4)  # Minimum 4 inches
        
        plt.figure(figsize=(fig_width, fig_height))
        
        # Generate colormap from color argument if not provided in kwargs
        if 'cmap' not in kwargs:
            # Convert single color to colormap (white to color)
            import matplotlib.colors as mcolors
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'custom', ['white', color], N=256
            )
        else:
            cmap = kwargs.pop('cmap')
        
        # Set default heatmap kwargs and allow overrides
        heatmap_kwargs = {
            'cmap': cmap,
            'annot': False,
            'fmt': 'd',
            'linewidths': 0.75,
            'linecolor': 'black',
            'cbar': False,
            'square': False  # Allow rectangular cells
        }
        heatmap_kwargs.update(kwargs)
        
        ax = sns.heatmap(heatmap_df, **heatmap_kwargs)
        
        # Rotate y-axis labels to be horizontal and increase font size
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
        
        plt.title(title if title is not None else f"Top {top_n} Models by {col}", 
                 fontsize=14, fontweight='bold')
        plt.xlabel(xlab if xlab is not None else "Predictors", fontsize=12)
        plt.ylabel(ylab if ylab is not None else col, fontsize=12)
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()

    else:
        raise ValueError(
            f"Invalid type '{type}'. "
            f"Choose from: 'line', 'bar', or 'predictors'."
        )


def _get_all_features_original_order(model) -> list:
    """
    Extract features in their ORIGINAL order from the model specification.
    
    This ensures predictors appear in the order they were specified in the
    formula, not sorted by frequency or importance in best models.
    
    Parameters
    ----------
    model : object
        Fitted model object from bsr().
    
    Returns
    -------
    list
        List of all feature names in original specification order.
    
    Raises
    ------
    ValueError
        If feature names cannot be extracted from the model.
    """
    # Priority 1: Check if bsr has all_features stored in original order
    if hasattr(model, 'bsr') and hasattr(model.bsr, 'all_features'):
        return list(model.bsr.all_features)
    
    # Priority 2: Try to get from the original model's exog_names
    if hasattr(model, 'model') and hasattr(model.model, 'exog_names'):
        # Filter out intercept if present
        features = [f for f in model.model.exog_names if f.lower() not in ['const', 'intercept']]
        return features
    
    # Priority 3: Try to get from params (fitted model)
    if hasattr(model, 'params'):
        features = [f for f in model.params.index if f.lower() not in ['const', 'intercept']]
        return features
    
    # Fallback: extract from results (may not preserve original order perfectly)
    if hasattr(model, 'bsr') and hasattr(model.bsr, 'results'):
        results_df = model.bsr.results
        feature_col = "Features"
        # Get features from the FULL model (last in best_by_k which has all predictors)
        if hasattr(model.bsr, 'best_by_k'):
            max_k = max(model.bsr.best_by_k.keys())
            return list(model.bsr.best_by_k[max_k])
        
        # Last resort: collect unique features preserving first appearance order
        all_feats = []
        seen = set()
        for feats in results_df[feature_col]:
            for feat in feats:
                if feat not in seen:
                    all_feats.append(feat)
                    seen.add(feat)
        return all_feats
    
    raise ValueError(
        "Could not extract feature names from model. "
        "Model must have bsr.all_features, model.exog_names, or bsr.results."
    )


def _get_all_features(model) -> list:
    """
    Extract all available features from the model's BSR results in original order.
    
    Parameters
    ----------
    model : object
        Fitted model object from bsr().
    
    Returns
    -------
    list
        List of all feature names that were considered in BSR, in original order.
    
    Raises
    ------
    ValueError
        If feature names cannot be extracted from the model.
    """
    # Priority 1: Check if bsr has all_features stored
    if hasattr(model, 'bsr') and hasattr(model.bsr, 'all_features'):
        return list(model.bsr.all_features)
    
    # Fallback: extract unique features from results, preserving order of first appearance
    if hasattr(model, 'bsr') and hasattr(model.bsr, 'results'):
        results_df = model.bsr.results
        feature_col = "Features"
        # Collect all unique features across all models, preserving order
        all_feats = []
        seen = set()
        for feats in results_df[feature_col]:
            for feat in feats:
                if feat not in seen:
                    all_feats.append(feat)
                    seen.add(feat)
        return all_feats
    
    raise ValueError(
        "Could not extract feature names from model. "
        "Model must have bsr.all_features or bsr.results."
    )

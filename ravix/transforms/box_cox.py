import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox, boxcox_llf
from typing import Tuple, Optional

def box_cox(
    model,
    color: str = "blue",
    lcolor: str = "red",
    title: str = "Log-Likelihood for Box-Cox Transformation",
    xlab: str = "Lambda",
    ylab: str = "Log-Likelihood",
    figsize: Tuple[float, float] = (10, 6),
    **kwargs
) -> None:
    """
    Create a Box-Cox transformation plot showing optimal lambda and confidence interval.
    
    This function plots the log-likelihood function for different Box-Cox transformation
    parameters (lambda) and identifies the optimal transformation along with its 95%
    confidence interval. Useful for determining appropriate power transformations
    to achieve normality in the response variable.
    
    Parameters
    ----------
    model : object
        Fitted regression model with model.endog containing the response variable.
        All response values must be positive.
    color : str, default="blue"
        Color for the log-likelihood curve and confidence interval lines.
    lcolor : str, default="red"
        Color for the optimal lambda line.
    title : str, default="Log-Likelihood for Box-Cox Transformation"
        Title for the plot.
    xlab : str, default="Lambda"
        X-axis label.
    ylab : str, default="Log-Likelihood"
        Y-axis label.
    figsize : tuple, default=(10, 6)
        Figure size as (width, height) in inches.
    **kwargs : dict
        Additional keyword arguments passed to plt.plot().
    
    Returns
    -------
    None
        Displays the plot and closes the figure.
    
    Examples
    --------
    >>> # Basic Box-Cox plot
    >>> model = pr.fit("price ~ age + mileage", data=cars)
    >>> box_cox(model)
    
    >>> # Custom colors
    >>> box_cox(model, color="green", lcolor="orange")
    
    >>> # Custom styling
    >>> box_cox(model, title="Box-Cox Transformation Analysis",
    ...         figsize=(12, 8), linewidth=2)
    
    Notes
    -----
    - Response variable must contain only positive values
    - Lambda = 1: No transformation
    - Lambda = 0: Log transformation
    - Lambda = 0.5: Square root transformation
    - Lambda = -1: Inverse transformation
    - Plot shows lambda range from -2 to 2 (typical range)
    - Confidence interval calculated using chi-squared cutoff (1.92 for 95% CI)
    - If optimal lambda is outside [-2, 2], CI lines are not plotted
    
    Raises
    ------
    ValueError
        If response variable contains non-positive values.
    """
    # Font size settings
    TITLE_FONTSIZE = 14
    LABEL_FONTSIZE = 12
    TICK_FONTSIZE = 11
    LEGEND_FONTSIZE = 10
    
    # Extract response variable
    y = model.model.endog
    
    # Validate positive values
    if np.any(y <= 0):
        raise ValueError(
            "All values in the response variable must be positive for Box-Cox transformation. "
            "Consider adding a constant or using a different transformation."
        )
    
    # Perform Box-Cox transformation
    y_transformed, fitted_lambda = boxcox(y)
    
    # Calculate lambdas from -3 to 3 for better CI accuracy
    lambdas = np.linspace(-3, 3, 100)
    log_likelihood = [boxcox_llf(lmbda, y) for lmbda in lambdas]
    
    # Plot lambdas from -2.1 to 2.1
    plot_lambdas = lambdas[(lambdas >= -2.1) & (lambdas <= 2.1)]
    plot_log_likelihood = [boxcox_llf(lmbda, y) for lmbda in plot_lambdas]
    
    # Calculate confidence interval
    max_log_likelihood = boxcox_llf(fitted_lambda, y)
    ci_cutoff = max_log_likelihood - 1.92  # Chi-squared distribution cutoff for 95% CI
    ci_lambdas = lambdas[np.array(log_likelihood) >= ci_cutoff]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Extract kwargs for plot
    plot_kwargs = {k: v for k, v in kwargs.items()}
    if 'color' not in plot_kwargs:
        plot_kwargs['color'] = color
    
    # Plot the log-likelihood curve
    plt.plot(plot_lambdas, plot_log_likelihood, 
            label='Log-Likelihood Function', **plot_kwargs)
    
    # Set axis limits
    plt.xlim([-2, 2])
    plt.ylim([min(plot_log_likelihood), 
              max(plot_log_likelihood) + 0.05 * (max(plot_log_likelihood) - min(plot_log_likelihood))])
    
    # Add vertical lines for lambda and CI if within range
    if -2 <= fitted_lambda <= 2:
        lambda_lower = ci_lambdas[0]
        lambda_upper = ci_lambdas[-1]
        
        plt.axvline(lambda_lower, color=color, linestyle='--', linewidth=1.5,
                   label=f'95% CI Lower: {lambda_lower:.4f}')
        plt.axvline(fitted_lambda, color=lcolor, linestyle='--', linewidth=2,
                   label=f'Best Lambda: {fitted_lambda:.4f}')
        plt.axvline(lambda_upper, color=color, linestyle='--', linewidth=1.5,
                   label=f'95% CI Upper: {lambda_upper:.4f}')
    else:
        import warnings
        warnings.warn(
            f"The fitted lambda ({fitted_lambda:.4f}) is outside the typical range of -2 to 2. "
            f"Confidence interval lines not plotted.",
            UserWarning
        )
    
    # Set labels and title with font sizes
    plt.xlabel(xlab, fontsize=LABEL_FONTSIZE)
    plt.ylabel(ylab, fontsize=LABEL_FONTSIZE)
    plt.title(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
    plt.legend(loc='lower right', fontsize=LEGEND_FONTSIZE, framealpha=0.9)
    plt.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()

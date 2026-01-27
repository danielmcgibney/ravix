import numpy as np
import pandas as pd


def format_sigfigs(x, sigfigs=6):
    """Format a number to a specified number of significant figures."""
    if pd.isna(x) or x == 0:
        return f"{x:>12.6f}"
    
    # Calculate the order of magnitude
    magnitude = int(np.floor(np.log10(abs(x))))
    
    # Calculate decimal places needed
    decimals = sigfigs - magnitude - 1
    
    # Ensure we don't have negative decimals for very large numbers
    decimals = max(0, decimals)
    
    # Format with the calculated decimals
    return f"{x:>12.{decimals}f}"


def format_r_style(x):
    """Format numbers using R-style logic."""
    if x == '' or pd.isna(x):
        return ''
    if abs(x) < 2e-16:
        return '< 2e-16'
    elif abs(x) < 0.0001:
        return f'{x:.2e}'
    elif abs(x) < 1:
        return f'{x:.6f}'.rstrip('0').rstrip('.')
    elif abs(x) < 1000:
        return f'{x:.4f}'.rstrip('0').rstrip('.')
    else:
        return f'{x:.2f}'


def format_pvalue(x):
    """Format p-values using R-style logic."""
    if x == '' or pd.isna(x):
        return ''
    if x < 2e-16:
        return '< 2e-16'
    elif x < 0.0001:
        return f'{x:.2e}'
    else:
        return f'{x:.6f}'.rstrip('0').rstrip('.')
      
def significance_code(p):
    """Returns the significance code for a given p-value."""
    try:
        p = float(p)
    except ValueError:
        return ''
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return ''


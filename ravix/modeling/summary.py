"""
Model Summary Generation Module

Provides comprehensive summary outputs for statsmodels regression models,
supporting multiple output formats (simple, R, STATA, coefficients, ANOVA).

LaTeX Output Policy:
-------------------
The `format='latex'` parameter produces LaTeX output with the following conventions:
- Summary outputs (simple, r): Full LaTeX with \\section* and \\subsection* headers,
  itemize lists, and proper table formatting. Good for standalone documents.
- Text-based outputs (stata, anova): Wrapped in \begin{verbatim} blocks to preserve
  formatting, since these are pre-formatted ASCII tables.
- Coefficient tables: LaTeX table only with optional section header.

Note: If you need tables without section headers (e.g., for inclusion in existing
documents), you can strip the headers or use format='df' and call .to_latex() yourself.

Implementation Notes:
--------------------
- _get_coefficient_table() builds the coefficient table directly from model.params,
  model.bse, model.tvalues/zvalues, and model.pvalues, preserving full numerical
  precision.  The old HTML-parsing approach rounded values before they reached the
  formatter; this approach avoids that entirely.
- All display paths use render_coef_table() for plain-text output and
  _format_coef_df_for_display() for LaTeX, guaranteeing identical formatting
  regardless of model type or output style.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

from ravix.modeling.format_utils import format_sigfigs, format_r_style, format_pvalue, significance_code
from ravix.modeling.print_anova_table import print_anova_table


# ============================================================================
# Utility Functions
# ============================================================================

def format_summary(summary_df, alpha=0.05):
    """
    Format the summary DataFrame by adding significance codes.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Raw summary DataFrame from statsmodels
    alpha : float
        Significance level for confidence intervals

    Returns
    -------
    pd.DataFrame
        Formatted summary DataFrame with significance codes

    Notes
    -----
    Retained for backward compatibility.  The main pipeline builds coefficient
    tables directly from model attributes via _get_coefficient_table().
    """
    # Format the P>|t| column if it exists (OLS models)
    if 'P>|t|' in summary_df.columns:
        summary_df['P>|t|'] = summary_df['P>|t|'].astype(float).map(format_pvalue)
        summary_df[' '] = summary_df['P>|t|'].apply(significance_code)

    # Format the P>|z| column if it exists (GLM/Logit models)
    if 'P>|z|' in summary_df.columns:
        summary_df['P>|z|'] = summary_df['P>|z|'].astype(float).map(format_pvalue)
        summary_df[' '] = summary_df['P>|z|'].apply(significance_code)

    return summary_df


# P-value column names recognised across all output styles
_PVALUE_COLS = frozenset(['P>|t|', 'P>|z|', 'Pr(>|t|)', 'Pr(>|z|)'])

# Significance column name (R-style blank header)
_SIG_COL = ' '


def _format_coef_df_for_display(summary_df):
    """
    Return a string-formatted copy of a coefficient DataFrame for LaTeX export.

    Used only by the ``format='latex'`` paths.  Plain-text rendering uses
    :func:`render_coef_table` directly on the numeric ``summary_df``.

    Applies consistent formatting to every column:
    - Numeric columns (estimates, std errors, t/z values): format_sigfigs(..., 6)
    - P-value columns: format_pvalue()
    - Significance-code column (' '): left unchanged

    The input DataFrame is never modified.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Numeric coefficient table as returned by _get_coefficient_table().

    Returns
    -------
    pd.DataFrame
        String-formatted copy ready for to_latex().
    """
    display_df = summary_df.copy()
    for col in display_df.columns:
        if col == _SIG_COL:
            continue
        elif col in _PVALUE_COLS:
            display_df[col] = display_df[col].apply(
                lambda x: format_pvalue(x) if isinstance(x, (int, float)) else x
            )
        else:
            display_df[col] = display_df[col].apply(
                lambda x: format_sigfigs(x, 6) if isinstance(x, (int, float)) else x
            )
    return display_df


# ---------------------------------------------------------------------------
# Console rendering — fixed-width manual renderer (no pandas to_string)
# ---------------------------------------------------------------------------

# Fixed column widths (characters).  Index width is a minimum; it grows to
# fit the longest variable name in the model.
_IDX_MIN = 12   # variable name column minimum width
_IDX_MAX = 16   # variable name column maximum width (truncate longer names)
_EST_W   = 9    # Estimate (fixed decimal or 4 sig fig sci notation)
_SE_W    = 10   # Std. Error (magnitude-aware decimals or sci notation)
_T_W     =  8   # t / z value (3 decimal places)
_P_W     = 10   # p-value (4 decimals or sci notation) - no stars
_SIG_W   =  4   # significance column: ' ' + stars (up to 3)
_COL_GAP = ' '  # single-space gap between columns
_EST_SE_GAP = '  ' # two-space gap between Estimate and Std. Error

# Reverse-mapping: R-style column label → canonical internal name used by
# render_coef_table.  _get_coefficient_table renames columns when called
# with r_style_labels=True; this table makes render_coef_table accept both.
_COEF_COL_ALIASES = {
    'Estimate':   'coef',
    'Std. Error': 'std err',
    't value':    't',
    'z value':    'z',
    'Pr(>|t|)':   'P>|t|',
    'Pr(>|z|)':   'P>|z|',
}


def format_pvalue_console(p: float) -> str:
    """
    Format a p-value for fixed-width console display (no stars).

    Returns just the numeric p-value formatted and right-aligned.
    Stars are handled separately in the significance column.

    Parameters
    ----------
    p : float
        Raw p-value from the model (must be in [0, 1]).

    Returns
    -------
    str
        Right-aligned p-value string fitting in _P_W = 10 characters
    """
    if p < 2e-16:
        pv_str = '< 2e-16'
    elif p < 1e-4:
        pv_str = f'{p:.2e}'
    else:
        pv_str = f'{p:.4f}'
    
    return f'{pv_str:>{_P_W}}'


def format_estimate_console(val: float) -> str:
    """
    Format coefficient estimate for console display using R-style logic.
    
    Uses scientific notation only when |val| < 1e-4 or |val| >= 1e5.
    Otherwise uses fixed decimal with magnitude-aware precision:
    larger magnitudes → fewer decimals, smaller magnitudes → more decimals.
    
    Parameters
    ----------
    val : float
        Coefficient estimate value
        
    Returns
    -------
    str
        Formatted string fitting in _EST_W = 12 characters
    """
    abs_val = abs(val)
    if abs_val < 1e-4 or abs_val >= 1e5:
        # Scientific notation with 4 significant figures
        return f'{val:.3e}'
    else:
        # Fixed decimal - magnitude-aware precision
        if abs_val >= 1000:
            return f'{val:.1f}'
        elif abs_val >= 100:
            return f'{val:.2f}'
        elif abs_val >= 10:
            return f'{val:.3f}'
        elif abs_val >= 1:
            return f'{val:.4f}'
        elif abs_val >= 0.1:
            return f'{val:.5f}'
        elif abs_val >= 0.01:
            return f'{val:.6f}'
        else:
            return f'{val:.7f}'


def format_stderr_console(val: float) -> str:
    """
    Format standard error for console display using R-style logic.
    
    Uses scientific notation only when |val| < 1e-4 or |val| >= 1e5.
    Otherwise uses fixed decimal with magnitude-aware precision.
    
    Parameters
    ----------
    val : float
        Standard error value
        
    Returns
    -------
    str
        Formatted string fitting in _SE_W = 10 characters
    """
    abs_val = abs(val)
    if abs_val < 1e-4 or abs_val >= 1e5:
        return f'{val:.3e}'
    else:
        # Fixed decimal - magnitude-aware precision
        if abs_val >= 1000:
            return f'{val:.1f}'
        elif abs_val >= 100:
            return f'{val:.2f}'
        elif abs_val >= 10:
            return f'{val:.3f}'
        elif abs_val >= 1:
            return f'{val:.4f}'
        elif abs_val >= 0.1:
            return f'{val:.5f}'
        elif abs_val >= 0.01:
            return f'{val:.6f}'
        else:
            return f'{val:.7f}'


def render_coef_table(df) -> str:
    """
    Render a numeric coefficient DataFrame as a fixed-width plain-text table.

    Accepts the raw ``summary_df`` produced by ``_get_coefficient_table()``
    (numeric floats, not pre-formatted strings).  Formatting and layout are
    both handled here so the table is independent of pandas rendering.

    Layout (R-style with magnitude-aware formatting)::

        <var name>    Estimate  Std. Error  t-value     p-value
        (Intercept)  20000.12      462.12    43.291   2.10e-16 ***
        income          3.1416      0.0988    31.801   4.56e-07 ***
        education      -0.00012   5.679e-05    -2.173     0.0298 *

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``_get_coefficient_table()``.  Must contain columns
        ``'coef'``, ``'std err'``, ``'t'`` or ``'z'``, ``'P>|t|'`` or
        ``'P>|z|'``, and ``' '`` (significance).  All numeric columns must
        hold floats.  R-style column labels are accepted via ``_COEF_COL_ALIASES``.

    Returns
    -------
    str
        Multi-line string ready for ``print()`` or inclusion in a
        ``\\begin{verbatim}`` block.
    """
    # Normalise column names: accept both raw names ('coef', 'std err', …)
    # and R-style labels ('Estimate', 'Std. Error', …) produced when
    # _get_coefficient_table is called with r_style_labels=True.
    df = df.rename(columns=_COEF_COL_ALIASES)

    # Detect whether this is a t-test or z-test model
    has_t    = 't' in df.columns
    stat_col = 't'      if has_t else 'z'
    p_col    = 'P>|t|'  if has_t else 'P>|z|'
    t_header = 't-value' if has_t else 'z-value'

    # Index width: at least _IDX_MIN, grows to fit the longest variable name
    # but capped at _IDX_MAX to prevent table overflow
    max_name_len = max(len(str(n)) for n in df.index)
    idx_w = max(_IDX_MIN, min(max_name_len, _IDX_MAX))
    
    # Truncate long variable names with ellipsis if they exceed _IDX_MAX
    def format_var_name(name):
        name_str = str(name)
        if len(name_str) > _IDX_MAX:
            return name_str[:_IDX_MAX-3] + '...'
        return name_str

    # Header row — significance column has no header (R convention)
    header = (
        f"{'':>{idx_w}}"
        f"{_COL_GAP}{'Estimate':>{_EST_W}}"
        f"{_EST_SE_GAP}{'Std. Error':>{_SE_W}}"
        f"{_COL_GAP}{t_header:>{_T_W}}"
        f"{_COL_GAP}{'p-value':>{_P_W}}"
    )

    # Data rows
    lines = [header]
    for name in df.index:
        est = format_estimate_console(df.loc[name, 'coef'])
        se  = format_stderr_console(df.loc[name, 'std err'])
        tv  = f"{df.loc[name, stat_col]:.3f}"
        pv  = format_pvalue_console(df.loc[name, p_col])
        sig = df.loc[name, _SIG_COL].strip()  # '' for non-significant rows
        
        formatted_name = format_var_name(name)
        line = (
            f"{formatted_name:<{idx_w}}"  # LEFT-aligned coefficient names
            f"{_COL_GAP}{est:>{_EST_W}}"
            f"{_EST_SE_GAP}{se:>{_SE_W}}"
            f"{_COL_GAP}{tv:>{_T_W}}"
            f"{_COL_GAP}{pv}"
            f"{_COL_GAP}{sig}"  # significance stars in separate column
        )
        lines.append(line)

    # rstrip every line so no trailing spaces; width driven by header (not sig-star rows)
    lines = [line.rstrip() for line in lines]
    return '\n'.join(lines)


# ============================================================================
# Main Summary Function
# ============================================================================

def summary(model, out='simple', alpha=0.05, level=None, format='text', **kwargs):
    """
    Generate and print a summary of a regression model.
    
    Produces formatted summaries similar to R, STATA, or statsmodels output,
    or displays specific components like coefficients or ANOVA tables.
    
    Parameters
    ----------
    model : Ravix regression result
        Fitted regression model (OLS, Logit, Poisson, or GLM)
    out : str, default='simple'
        Output format type:
        - 'simple': Clean, readable summary with coefficients and model stats
        - 'statsmodels' or 'stats': Native statsmodels summary
        - 'r': R-style summary with residual quantiles
        - 'stata': STATA-style tabular output
        - 'coefficients' or 'coef': Full coefficient table with SE, test statistics, p-values, and significance codes
        - 'confint' or 'ci': Confidence intervals table with estimates and CIs only
        - 'anova' or 'anova2': Type II ANOVA table
        - 'anova1': Type I (sequential) ANOVA table
    alpha : float, optional
        Significance level for hypothesis tests (0 < alpha < 1).
        If not provided, derived from level parameter (alpha = 1 - level).
        Default is 0.05 if neither alpha nor level is specified.
    level : float, optional
        Confidence level for confidence intervals (0 < level < 1).
        If not provided, derived from alpha parameter (level = 1 - alpha).
        Default is 0.95 if neither alpha nor level is specified.
    format : str, default='text'
        Output format:
        - 'text': Print formatted text to console (default behavior)
        - 'latex': Return LaTeX table string
        - 'dataframe' or 'df': Return pandas DataFrame
    **kwargs : dict
        Additional keyword arguments (reserved for future use)
        
    Returns
    -------
    None, str, or pd.DataFrame
        - If format='text': Prints to console, returns None
        - If format='latex': Returns LaTeX string
        - If format='dataframe' or 'df': Returns pandas DataFrame
        
    Raises
    ------
    ValueError
        If model type is unsupported, output format is invalid,
        or if both alpha and level are provided and don't sum to 1
        
    Examples
    --------
    >>> import ravix
    >>> model = ravix.ols("Y~X", data = df)
    >>> ravix.summary(model)  # Simple summary (text)
    >>> ravix.summary(model, out='r')  # R-style summary (text)
    >>> ravix.summary(model, out='coefficients', level=0.99)  # 99% CI (text)
    >>> ravix.summary(model, out='coefficients', alpha=0.01)  # Same as above
    >>> latex_str = ravix.summary(model, format='latex')  # LaTeX output
    >>> df = ravix.summary(model, format='df')  # DataFrame output
    """
    # Handle alpha and level parameters
    if level is None:
        # Only alpha provided (default or explicit)
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        level = 1 - alpha
    else:
        # level is provided
        if not (0 < level < 1):
            raise ValueError(f"level must be between 0 and 1, got {level}")
        # Check if alpha was also explicitly provided (not default)
        if alpha != 0.05:  # If alpha was changed from default
            # Both provided: ensure they sum to 1
            if not np.isclose(alpha + level, 1.0, atol=1e-10):
                raise ValueError(
                    f"When both alpha and level are provided, they must sum to 1. "
                    f"Got alpha={alpha}, level={level}, sum={alpha + level}"
                )
        else:
            # Only level provided, derive alpha
            alpha = 1 - level

    format = format.lower()
    if format not in ['text', 'latex', 'dataframe', 'df']:
        raise ValueError(f"format must be 'text', 'latex', 'dataframe', or 'df', got '{format}'")
    
    out = out.lower()
    
    # Handle native statsmodels output
    if out in ['statsmodels', 'stats']:
        if format == 'text':
            print(model.summary(alpha=alpha))
            return None
        elif format == 'latex':
            return model.summary(alpha=alpha).as_latex()
        else:  # dataframe/df
            # Return coefficient table as DataFrame
            return _get_coefficient_table(model, alpha, r_style_labels=False)
    
    # Dispatch to appropriate summary function based on model type
    model_type = _get_model_type(model)
    
    if model_type == "ols":
        return _print_ols_summary(model, out, alpha, format)
    elif model_type == "logit":
        return _print_logistic_summary(model, out, alpha, format)
    elif model_type == "poisson":
        return _print_poisson_summary(model, out, alpha, format)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# ============================================================================
# Format Output Helpers
# ============================================================================

def _handle_output(content, format='text'):
    """
    Centralized output handler for all summary formats.
    
    Parameters
    ----------
    content : str or pd.DataFrame
        Content to output. If DataFrame, will be converted to appropriate format.
        If string, assumes it's already in the target format (LaTeX string for latex, etc.)
    format : str
        Output format: 'text', 'latex', 'dataframe', or 'df'
        
    Returns
    -------
    None, str, or pd.DataFrame
    """
    if format == 'text':
        print(content)
        return None
    elif format == 'latex':
        if isinstance(content, pd.DataFrame):
            return content.to_latex()
        else:
            # Assume content is already LaTeX-formatted string
            return content
    else:  # dataframe or df
        if isinstance(content, pd.DataFrame):
            return content
        else:
            # For text content, create a simple DataFrame
            return pd.DataFrame({'Summary': [str(content)]})


def _format_anova_output(model, format='text', typ=2):
    """Format ANOVA table output."""
    anova_str = print_anova_table(model, typ=typ)
    
    if format in ['dataframe', 'df']:
        # Return a simple DataFrame representation
        return pd.DataFrame({'ANOVA': [anova_str]})
    
    # For LaTeX, wrap in verbatim since print_anova_table returns formatted text
    if format == 'latex':
        anova_str = "\\begin{verbatim}\n" + anova_str + "\n\\end{verbatim}"
    
    return _handle_output(anova_str, format)


# ============================================================================
# Model Type Detection
# ============================================================================

def _get_model_type(model):
    """
    Determine the type of regression model.
    
    Parameters
    ----------
    model : statsmodels regression result
        Fitted regression model
        
    Returns
    -------
    str
        Model type identifier: 'ols', 'logit', 'poisson', or 'unsupported'
    """
    # Check the underlying model object directly instead of wrapper type
    # This handles both RegressionResultsWrapper and GLMResultsWrapper
    
    if hasattr(model, 'model'):
        if isinstance(model.model, sm.OLS):
            return "ols"
        elif isinstance(model.model, sm.Logit):
            return "logit"
        elif isinstance(model.model, sm.GLM):
            if isinstance(model.model.family, sm.families.Binomial):
                return "logit"
            elif isinstance(model.model.family, sm.families.Poisson):
                return "poisson"
            # Other GLM families not yet supported
            return "unsupported"
    
    return "unsupported"


# ============================================================================
# OLS Summary Functions
# ============================================================================

def _print_ols_summary(model, out, alpha, format='text'):
    """
    Generate summary output for OLS regression models.
    
    Parameters
    ----------
    model : statsmodels OLS result
        Fitted OLS regression model
    out : str
        Output format type
    alpha : float
        Significance level
    format : str
        Output format: 'text', 'latex', 'dataframe', or 'df'
        
    Returns
    -------
    None, str, or pd.DataFrame
        Depends on format parameter
    """
    # Suppress specific warnings using context manager
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", sm.tools.sm_exceptions.ValueWarning)
        
        # Extract model statistics
        stats = _extract_ols_statistics(model, alpha)
        stats['alpha'] = alpha  # Add alpha for STATA output
        
        # Get formatted coefficient table (with R-style labels for simple and r outputs)
        use_r_labels = out in ['simple', 'r']
        summary_df = _get_coefficient_table(model, alpha, r_style_labels=use_r_labels)
    
    # Route to appropriate output format
    if out == 'r':
        return _format_r_style_summary(model, summary_df, stats, format)
    elif out == 'simple':
        return _format_simple_ols_summary(summary_df, stats, format)
    elif out in ['coefficients', 'coef']:
        return _format_coefficient_table(model, alpha, format)
    elif out in ['confint', 'ci']:
        return _format_confint_table(model, alpha, format)
    elif out in ['anova', 'anova2']:
        return _format_anova_output(model, format, typ=2)
    elif out == 'anova1':
        return _format_anova_output(model, format, typ=1)
    elif out == 'stata':
        return _format_stata_summary(model, summary_df, stats, format)
    else:
        raise ValueError(f"Unsupported output format: '{out}'")


def _extract_ols_statistics(model, alpha):
    """
    Extract and format key statistics from OLS model.
    
    Parameters
    ----------
    model : statsmodels OLS result
        Fitted OLS model
    alpha : float
        Significance level
        
    Returns
    -------
    dict
        Dictionary containing formatted model statistics
    """
    # Calculate statistics
    RSS = np.sum(model.resid**2)
    RSE = np.sqrt(RSS / model.df_resid)
    
    return {
        'n_obs': int(model.nobs),
        'df_model': int(model.df_model),
        'df_resid': int(model.df_resid),
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'f_p_value': model.f_pvalue,
        'aic': model.aic,
        'bic': model.bic,
        'RSE': RSE,
        'RSS': RSS,
        # Formatted versions
        'r_squared_fmt': f"{model.rsquared:.4f}",
        'adj_r_squared_fmt': f"{model.rsquared_adj:.4f}",
        'f_statistic_fmt': format_r_style(model.fvalue),
        'f_p_value_fmt': format_pvalue(model.f_pvalue),
        'aic_fmt': format_r_style(model.aic),
        'bic_fmt': format_r_style(model.bic),
        'RSE_fmt': format_r_style(RSE)
    }


def _get_coefficient_table(model, alpha, r_style_labels=False):
    """
    Build a coefficient table directly from model attributes.

    Parameters
    ----------
    model : statsmodels regression result
        Fitted regression model
    alpha : float
        Significance level (used only for type-checking; not applied here)
    r_style_labels : bool, default=False
        If True, rename columns to match R output style

    Returns
    -------
    pd.DataFrame
        Numeric coefficient table with columns:
        'coef', 'std err', 't'/'z', 'P>|t|'/'P>|z|', ' '
        (or R-style equivalents when r_style_labels=True).
        All numeric columns hold raw floats — no rounding applied.

    Notes
    -----
    Building from model.params / model.bse / model.tvalues / model.pvalues
    preserves full floating-point precision.  The previous HTML-parsing approach
    rounded values inside statsmodels' formatter before they reached this code.
    """
    # Build coefficient table directly from model attributes
    coef_data = {
        'coef': model.params,
        'std err': model.bse,
    }

    # Add appropriate test statistic and p-value columns
    if hasattr(model, 'tvalues'):
        coef_data['t'] = model.tvalues
        p_col_name = 'P>|t|'
        coef_data[p_col_name] = model.pvalues
    elif hasattr(model, 'zvalues'):
        coef_data['z'] = model.zvalues
        p_col_name = 'P>|z|'
        coef_data[p_col_name] = model.pvalues
    else:
        # Fallback for models without explicit test statistics
        p_col_name = 'P>|t|'
        coef_data[p_col_name] = model.pvalues

    summary_df = pd.DataFrame(coef_data)

    # Add significance codes; p-values stay numeric
    summary_df[_SIG_COL] = summary_df[p_col_name].apply(significance_code)

    # Rename columns to R-style if requested
    if r_style_labels:
        if 't' in summary_df.columns:
            column_mapping = {
                'coef': 'Estimate',
                'std err': 'Std. Error',
                't': 't value',
                'P>|t|': 'Pr(>|t|)',
            }
        elif 'z' in summary_df.columns:
            column_mapping = {
                'coef': 'Estimate',
                'std err': 'Std. Error',
                'z': 'z value',
                'P>|z|': 'Pr(>|z|)',
            }
        else:
            column_mapping = {}

        summary_df.rename(columns=column_mapping, inplace=True)

    return summary_df


def _format_simple_ols_summary(summary_df, stats, format='text'):
    """Format simple OLS summary with key statistics."""
    if format in ['dataframe', 'df']:
        result_df = summary_df.copy()
        result_df.attrs['r_squared'] = stats['r_squared']
        result_df.attrs['adj_r_squared'] = stats['adj_r_squared']
        result_df.attrs['f_statistic'] = stats['f_statistic']
        result_df.attrs['aic'] = stats['aic']
        result_df.attrs['bic'] = stats['bic']
        result_df.attrs['RSE'] = stats['RSE']
        return result_df
    
    # Build LaTeX output
    if format == 'latex':
        display_df = _format_coef_df_for_display(summary_df)
        latex_parts = []
        latex_parts.append("\\section*{Summary of OLS Regression Analysis}")
        latex_parts.append("\n\\subsection*{Coefficients}")
        latex_parts.append(display_df.to_latex())
        latex_parts.append("\n\\subsection*{Model Statistics}")
        latex_parts.append("\\begin{itemize}")
        latex_parts.append(f"\\item Residual Standard Error: {stats['RSE_fmt']}")
        latex_parts.append(f"\\item R-squared: {stats['r_squared_fmt']}")
        latex_parts.append(f"\\item Adjusted R-squared: {stats['adj_r_squared_fmt']}")
        latex_parts.append(f"\\item AIC: {stats['aic_fmt']}")
        latex_parts.append(f"\\item BIC: {stats['bic_fmt']}")
        latex_parts.append(f"\\item F-statistic: {stats['f_statistic_fmt']} on {stats['df_model']} and {stats['df_resid']} DF, p-value: {stats['f_p_value_fmt']}")
        latex_parts.append("\\end{itemize}")
        content = '\n'.join(latex_parts)
        return _handle_output(content, format)
    
    # Build text output
    coef_str = render_coef_table(summary_df)
    w = len(coef_str.splitlines()[0])
    output = []
    output.append("Summary of OLS Regression Analysis:")
    output.append("=" * w)
    output.append("\nCoefficients:")
    output.append("-" * w)
    output.append(coef_str)
    output.append("\nModel Statistics:")
    output.append("-" * w)
    output.append(f"Residual Std. Error: {stats['RSE_fmt']}")
    output.append(f"R-squared:      {stats['r_squared_fmt']:<16}AIC: {stats['aic_fmt']}")
    output.append(f"Adj. R-squared: {stats['adj_r_squared_fmt']:<16}BIC: {stats['bic_fmt']}")
    output.append(f"F-statistic: {stats['f_statistic_fmt']} on {stats['df_model']} and "
                 f"{stats['df_resid']} DF, p-value: {stats['f_p_value_fmt']}")
    output.append("=" * w)
    
    content = '\n'.join(output)
    return _handle_output(content, format)


def _format_r_style_summary(model, summary_df, stats, format='text'):
    """Format R-style summary with residual quantiles."""
    if format in ['dataframe', 'df']:
        # Return coefficient table for dataframe format
        result_df = summary_df.copy()
        result_df.attrs['r_squared'] = stats['r_squared']
        result_df.attrs['adj_r_squared'] = stats['adj_r_squared']
        result_df.attrs['f_statistic'] = stats['f_statistic']
        result_df.attrs['RSE'] = stats['RSE']
        return result_df
    
    # Calculate residual statistics
    resid_stats = [
        np.min(model.resid),
        np.percentile(model.resid, 25),
        np.median(model.resid),
        np.percentile(model.resid, 75),
        np.max(model.resid)
    ]
    
    # Build LaTeX output
    if format == 'latex':
        display_df = _format_coef_df_for_display(summary_df)
        latex_parts = []
        latex_parts.append("\\section*{R-Style Regression Summary}")
        latex_parts.append("\n\\subsection*{Residuals}")
        latex_parts.append("\\begin{verbatim}")
        latex_parts.append("    Min      1Q    Median     3Q      Max")
        latex_parts.append(" ".join(f"{x:8.4f}" for x in resid_stats))
        latex_parts.append("\\end{verbatim}")
        latex_parts.append("\n\\subsection*{Coefficients}")
        latex_parts.append(display_df.to_latex())
        latex_parts.append("\n\\textit{Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1}")
        latex_parts.append(f"\n\\noindent Residual standard error: {stats['RSE_fmt']} on {stats['df_resid']} degrees of freedom\\\\")
        latex_parts.append(f"R-squared: {stats['r_squared_fmt']}, Adjusted R-squared: {stats['adj_r_squared_fmt']}\\\\")
        latex_parts.append(f"F-statistic: {stats['f_statistic_fmt']} on {stats['df_model']} and {stats['df_resid']} DF, p-value: {stats['f_p_value_fmt']}")
        content = '\n'.join(latex_parts)
        return _handle_output(content, format)
    
    # Build text output
    output = []
    output.append("Residuals:")
    output.append("    Min      1Q    Median     3Q      Max")
    output.append(" ".join(f"{x:8.4f}" for x in resid_stats))
    output.append("")
    output.append("Coefficients:")
    output.append(render_coef_table(summary_df))
    output.append("---")
    output.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    output.append("")
    output.append(f"Residual standard error: {stats['RSE_fmt']} on {stats['df_resid']} degrees of freedom")
    output.append(f"R-squared: {stats['r_squared_fmt']}, Adjusted R-squared: {stats['adj_r_squared_fmt']}")
    output.append(f"F-statistic: {stats['f_statistic_fmt']} on {stats['df_model']} and "
                 f"{stats['df_resid']} DF, p-value: {stats['f_p_value_fmt']}")
    
    content = '\n'.join(output)
    return _handle_output(content, format)


def _format_coefficient_table(model, alpha, format='text'):
    """Format full coefficient table with estimates, SE, test statistics, p-values, and significance codes."""
    # Build coefficient dataframe with all statistics
    coef_data = {
        'Estimate': model.params,
        'Std. Error': model.bse
    }
    
    # Add appropriate test statistic column
    if hasattr(model, 'tvalues'):
        coef_data['t value'] = model.tvalues
        coef_data['Pr(>|t|)'] = model.pvalues
    elif hasattr(model, 'zvalues'):
        coef_data['z value'] = model.zvalues
        coef_data['Pr(>|z|)'] = model.pvalues
    
    coef_df = pd.DataFrame(coef_data)
    coef_df[_SIG_COL] = coef_df.iloc[:, -1].apply(significance_code)  # Last p-value column
    
    if format in ['dataframe', 'df']:
        return coef_df
    
    # Format for display using _format_coef_df_for_display
    coef_df_formatted = _format_coef_df_for_display(coef_df)
    
    # Build LaTeX output
    if format == 'latex':
        latex_parts = []
        latex_parts.append("\\section*{Coefficients}")
        latex_parts.append(coef_df_formatted.to_latex())
        latex_parts.append("\n\\textit{Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1}")
        content = '\n'.join(latex_parts)
        return _handle_output(content, format)
    
    # Build text output — render via render_coef_table for consistent fixed-width layout.
    # coef_df uses R-style column names so we pass it directly; render_coef_table's
    # _COEF_COL_ALIASES map will normalise them internally.
    coef_str = render_coef_table(coef_df)
    w = len(coef_str.splitlines()[0])
    output = []
    output.append("Coefficients:")
    output.append("=" * w)
    output.append(coef_str)
    output.append("---")
    output.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    output.append("=" * w)
    content = '\n'.join(output)
    return _handle_output(content, format)


def _format_confint_table(model, alpha, format='text'):
    """Format confidence interval table with estimates and CIs only."""
    conf_intervals = model.conf_int(alpha=alpha)
    level = 1 - alpha

    coef_df = pd.DataFrame({
        'Estimate': model.params,
        f'{level*100:.0f}% CI Lower': conf_intervals.iloc[:, 0],
        f'{level*100:.0f}% CI Upper': conf_intervals.iloc[:, 1]
    })
    
    if format in ['dataframe', 'df']:
        # Return numeric DataFrame for programmatic use
        return coef_df
    
    # Format to 6 significant figures for display (text/latex)
    coef_df_formatted = coef_df.copy()
    for col in coef_df_formatted.columns:
        coef_df_formatted[col] = coef_df_formatted[col].apply(lambda x: format_sigfigs(x, 6))
    
    # Build LaTeX output
    if format == 'latex':
        latex_parts = []
        latex_parts.append("\\section*{Confidence Intervals}")
        latex_parts.append(coef_df_formatted.to_latex())
        content = '\n'.join(latex_parts)
        return _handle_output(content, format)
    
    # Build text output
    output = []
    output.append("Confidence Intervals:")
    output.append("=" * 54)
    output.append(coef_df_formatted.to_string())
    output.append("=" * 54)
    content = '\n'.join(output)
    return _handle_output(content, format)


def _format_stata_summary(model, summary_df, stats, format='text'):
    """Format STATA-style summary with ANOVA table and coefficients."""
    if format in ['dataframe', 'df']:
        # Return coefficient table with confidence intervals
        conf_intervals = model.conf_int(alpha=stats.get('alpha', 0.05))
        coef_df = summary_df.copy()
        coef_df['CI Lower'] = conf_intervals.iloc[:, 0]
        coef_df['CI Upper'] = conf_intervals.iloc[:, 1]
        return coef_df
    
    # Build ANOVA header
    anova_header = _build_stata_anova_header(model, stats)
    
    # Build coefficient table
    conf_intervals = model.conf_int(alpha=stats.get('alpha', 0.05))
    level_pct = int((1 - stats.get('alpha', 0.05)) * 100)
    conf_label = f"{level_pct}% Conf. Interval"
    
    output = []
    output.append(anova_header)
    output.append("-" * 78)
    output.append(f"             |      Coef.   Std. Err.      t    P>|t|     [{conf_label}]")
    output.append("-" * 13 + "+" + "-" * 64)
    
    for row in summary_df.itertuples():
        coef = float(row[1])
        std_err = float(row[2])
        t_value = float(row[3])
        p_value = float(row[4])
        conf_low, conf_high = conf_intervals.loc[row.Index]
        var_name = row.Index
        
        output.append(f"{var_name:>12} | {coef:>10.4f}   {std_err:>8.4f}   "
                     f"{t_value:>7.2f}   {p_value:>6.4f}   "
                     f"[{conf_low:>8.4f}, {conf_high:>8.4f}]")
    
    output.append("-" * 13 + "+" + "-" * 64)
    content = '\n'.join(output)
    
    # For STATA output, use verbatim for LaTeX since it's formatted text
    if format == 'latex':
        content = "\\begin{verbatim}\n" + content + "\n\\end{verbatim}"
    
    return _handle_output(content, format)


def _build_stata_anova_header(model, stats):
    """Build STATA-style ANOVA header with model statistics."""
    # Calculate sum of squares
    ss_residual = stats['RSS']
    ss_model = np.sum((model.fittedvalues - np.mean(model.model.endog)) ** 2)
    ss_total = ss_residual + ss_model
    
    # Mean squares
    ms_residual = ss_residual / stats['df_resid']
    ms_model = ss_model / stats['df_model']
    df_total = stats['df_model'] + stats['df_resid']
    
    output = []
    output.append("-" * 78)
    output.append(f"             |      SS       df       MS              "
                 f"Number of obs = {stats['n_obs']:>3}")
    output.append("-" * 13 + "+" + "-" * 30 + "          " + 
                 f"F({stats['df_model']}, {stats['df_resid']})      = {stats['f_statistic']:.2f}")
    output.append(f"     Model   | {ss_model:10.6f}    {stats['df_model']:>2}  {ms_model:10.6f}           "
                 f"Prob > F      = {stats['f_p_value']:.4f}")
    output.append(f"  Residual   | {ss_residual:10.6f}    {stats['df_resid']:>2}  {ms_residual:10.6f}           "
                 f"R-squared     = {stats['r_squared']:>6.4f}")
    output.append("-" * 13 + "+" + "-" * 30 + "          " +
                 f"Adj R-squared = {stats['adj_r_squared']:.4f}")
    output.append(f"     Total   | {ss_total:10.6f}    {df_total:>2}  {ss_total/df_total:10.6f}          "
                 f"Root MSE      = {stats['RSE']:>6.4f}")
    output.append("-" * 78)
    
    return '\n'.join(output)


# ============================================================================
# Logistic Regression Summary Functions
# ============================================================================

def _print_logistic_summary(model, out, alpha, format='text'):
    """
    Generate summary output for logistic regression models.
    
    Parameters
    ----------
    model : statsmodels Logit result
        Fitted logistic regression model
    out : str
        Output format type
    alpha : float
        Significance level
    format : str
        Output format: 'text', 'latex', 'dataframe', or 'df'
        
    Returns
    -------
    None, str, or pd.DataFrame
        Depends on format parameter
    """
    # Suppress specific warnings using context manager
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", sm.tools.sm_exceptions.ConvergenceWarning)
        
        # Get coefficient table
        summary_df = _get_coefficient_table(model, alpha)
        
        # Extract statistics with robust handling of pseudo R-squared
        try:
            pseudo_r2 = model.pseudo_rsquared(kind='cs')
            pseudo_r2_fmt = format_r_style(pseudo_r2)
        except (AttributeError, TypeError):
            # Fallback if pseudo_rsquared not available (e.g., some GLM results)
            pseudo_r2 = None
            pseudo_r2_fmt = "N/A"
        
        stats = {
            'n_obs': int(model.nobs),
            'log_likelihood': model.llf,
            'aic': model.aic,
            'bic': model.bic,
            'pseudo_r_squared': pseudo_r2,
            'log_likelihood_fmt': format_r_style(model.llf),
            'aic_fmt': format_r_style(model.aic),
            'bic_fmt': format_r_style(model.bic),
            'pseudo_r_squared_fmt': pseudo_r2_fmt
        }
    
    # Route to appropriate output format
    if out == 'r':
        # R-style not yet implemented for logistic
        content = "R-style output not currently available for logistic regression."
        return _handle_output(content, format)
    elif out == 'simple':
        return _format_simple_logistic_summary(summary_df, stats, format)
    elif out in ['coefficients', 'coef']:
        return _format_coefficient_table(model, alpha, format)
    elif out in ['confint', 'ci']:
        return _format_confint_table(model, alpha, format)
    elif out == 'anova':
        content = "ANOVA table not applicable for logistic regression."
        return _handle_output(content, format)
    elif out == 'stata':
        content = "STATA-style output not currently available for logistic regression."
        return _handle_output(content, format)
    else:
        raise ValueError(f"Unsupported output format: '{out}'")


def _format_simple_logistic_summary(summary_df, stats, format='text'):
    """Format simple logistic regression summary."""
    if format in ['dataframe', 'df']:
        result_df = summary_df.copy()
        result_df.attrs['log_likelihood'] = stats['log_likelihood']
        result_df.attrs['pseudo_r_squared'] = stats['pseudo_r_squared']
        result_df.attrs['aic'] = stats['aic']
        result_df.attrs['bic'] = stats['bic']
        return result_df
    
    # Build LaTeX output
    if format == 'latex':
        display_df = _format_coef_df_for_display(summary_df)
        latex_parts = []
        latex_parts.append("\\section*{Summary of Logistic Regression Analysis}")
        latex_parts.append("\n\\subsection*{Coefficients (Log-Odds)}")
        latex_parts.append(display_df.to_latex())
        latex_parts.append("\n\\subsection*{Model Statistics}")
        latex_parts.append("\\begin{itemize}")
        latex_parts.append(f"\\item Log-Likelihood: {stats['log_likelihood_fmt']}")
        latex_parts.append(f"\\item Pseudo R-squared: {stats['pseudo_r_squared_fmt']}")
        latex_parts.append(f"\\item AIC: {stats['aic_fmt']}")
        latex_parts.append(f"\\item BIC: {stats['bic_fmt']}")
        latex_parts.append("\\end{itemize}")
        content = '\n'.join(latex_parts)
        return _handle_output(content, format)
    
    # Build text output
    coef_str = render_coef_table(summary_df)
    w = len(coef_str.splitlines()[0])
    output = []
    output.append("Summary of Logistic Regression Analysis:")
    output.append("=" * w)
    output.append("\nCoefficients (Log-Odds):")
    output.append("-" * w)
    output.append(coef_str)
    output.append("\nModel Statistics:")
    output.append("-" * w)
    output.append(f"Log-Likelihood: {stats['log_likelihood_fmt']:<16}AIC: {stats['aic_fmt']}")
    output.append(f"Pseudo R-squared: {stats['pseudo_r_squared_fmt']:<14}BIC: {stats['bic_fmt']}")
    output.append("=" * w)
    
    content = '\n'.join(output)
    return _handle_output(content, format)


# ============================================================================
# Poisson Regression Summary Functions
# ============================================================================

def _print_poisson_summary(model, out, alpha, format='text'):
    """
    Generate summary output for Poisson regression models.
    
    Parameters
    ----------
    model : statsmodels GLM result with Poisson family
        Fitted Poisson regression model
    out : str
        Output format type
    alpha : float
        Significance level
    format : str
        Output format: 'text', 'latex', 'dataframe', or 'df'
        
    Returns
    -------
    None, str, or pd.DataFrame
        Depends on format parameter
    """
    # Suppress specific warnings using context manager
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", sm.tools.sm_exceptions.ConvergenceWarning)
        
        # Get coefficient table
        summary_df = _get_coefficient_table(model, alpha)
        
        # Extract statistics with robust handling of pseudo R-squared
        try:
            pseudo_r2 = model.pseudo_rsquared(kind='cs')
            pseudo_r2_fmt = format_r_style(pseudo_r2)
        except (AttributeError, TypeError):
            # Fallback if pseudo_rsquared not available
            pseudo_r2 = None
            pseudo_r2_fmt = "N/A"
        
        # Calculate deviance statistics
        deviance = model.deviance
        null_deviance = model.null_deviance
        
        stats = {
            'n_obs': int(model.nobs),
            'log_likelihood': model.llf,
            'aic': model.aic,
            'bic': model.bic,
            'deviance': deviance,
            'null_deviance': null_deviance,
            'pseudo_r_squared': pseudo_r2,
            'df_model': int(model.df_model),
            'df_resid': int(model.df_resid),
            # Formatted versions
            'log_likelihood_fmt': format_r_style(model.llf),
            'aic_fmt': format_r_style(model.aic),
            'bic_fmt': format_r_style(model.bic),
            'deviance_fmt': format_r_style(deviance),
            'null_deviance_fmt': format_r_style(null_deviance),
            'pseudo_r_squared_fmt': pseudo_r2_fmt
        }
    
    # Route to appropriate output format
    if out == 'r':
        # R-style not yet implemented for Poisson
        content = "R-style output not currently available for Poisson regression."
        return _handle_output(content, format)
    elif out == 'simple':
        return _format_simple_poisson_summary(summary_df, stats, format)
    elif out in ['coefficients', 'coef']:
        return _format_coefficient_table(model, alpha, format)
    elif out in ['confint', 'ci']:
        return _format_confint_table(model, alpha, format)
    elif out == 'anova':
        content = "ANOVA table not applicable for Poisson regression."
        return _handle_output(content, format)
    elif out == 'stata':
        content = "STATA-style output not currently available for Poisson regression."
        return _handle_output(content, format)
    else:
        raise ValueError(f"Unsupported output format: '{out}'")


def _format_simple_poisson_summary(summary_df, stats, format='text'):
    """Format simple Poisson regression summary."""
    if format in ['dataframe', 'df']:
        result_df = summary_df.copy()
        result_df.attrs['log_likelihood'] = stats['log_likelihood']
        result_df.attrs['pseudo_r_squared'] = stats['pseudo_r_squared']
        result_df.attrs['aic'] = stats['aic']
        result_df.attrs['bic'] = stats['bic']
        result_df.attrs['deviance'] = stats['deviance']
        result_df.attrs['null_deviance'] = stats['null_deviance']
        return result_df
    
    # Build LaTeX output
    if format == 'latex':
        display_df = _format_coef_df_for_display(summary_df)
        latex_parts = []
        latex_parts.append("\\section*{Summary of Poisson Regression Analysis}")
        latex_parts.append("\n\\subsection*{Coefficients (Log-Rate)}")
        latex_parts.append(display_df.to_latex())
        latex_parts.append("\n\\subsection*{Model Statistics}")
        latex_parts.append("\\begin{itemize}")
        latex_parts.append(f"\\item Log-Likelihood: {stats['log_likelihood_fmt']}")
        latex_parts.append(f"\\item Deviance: {stats['deviance_fmt']}")
        latex_parts.append(f"\\item Null Deviance: {stats['null_deviance_fmt']}")
        latex_parts.append(f"\\item Pseudo R-squared: {stats['pseudo_r_squared_fmt']}")
        latex_parts.append(f"\\item AIC: {stats['aic_fmt']}")
        latex_parts.append(f"\\item BIC: {stats['bic_fmt']}")
        latex_parts.append(f"\\item Degrees of Freedom: {stats['df_model']} (Model), {stats['df_resid']} (Residual)")
        latex_parts.append("\\end{itemize}")
        content = '\n'.join(latex_parts)
        return _handle_output(content, format)
    
    # Build text output
    coef_str = render_coef_table(summary_df)
    w = len(coef_str.splitlines()[0])
    output = []
    output.append("Summary of Poisson Regression Analysis:")
    output.append("=" * w)
    output.append("\nCoefficients (Log-Rate):")
    output.append("-" * w)
    output.append(coef_str)
    output.append("\nModel Statistics:")
    output.append("-" * w)
    output.append(f"Log-Likelihood: {stats['log_likelihood_fmt']:<16}AIC: {stats['aic_fmt']}")
    output.append(f"Deviance: {stats['deviance_fmt']:<22}BIC: {stats['bic_fmt']}")
    output.append(f"Null Deviance: {stats['null_deviance_fmt']}")
    output.append(f"Pseudo R-squared: {stats['pseudo_r_squared_fmt']}")
    output.append(f"Degrees of Freedom: {stats['df_model']} (Model), {stats['df_resid']} (Residual)")
    output.append("=" * w)
    
    content = '\n'.join(output)
    return _handle_output(content, format)

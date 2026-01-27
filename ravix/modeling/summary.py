"""
Model Summary Generation Module

Provides comprehensive summary outputs for statsmodels regression models,
supporting multiple output formats (simple, R, STATA, coefficients, ANOVA).

LaTeX Output Policy:
-------------------
The `format='latex'` parameter produces LaTeX output with the following conventions:
- Summary outputs (simple, r): Full LaTeX with \section* and \subsection* headers,
  itemize lists, and proper table formatting. Good for standalone documents.
- Text-based outputs (stata, anova): Wrapped in \begin{verbatim} blocks to preserve
  formatting, since these are pre-formatted ASCII tables.
- Coefficient tables: LaTeX table only with optional section header.

Note: If you need tables without section headers (e.g., for inclusion in existing
documents), you can strip the headers or use format='df' and call .to_latex() yourself.

Implementation Notes:
--------------------
- _get_coefficient_table() uses HTML parsing of statsmodels summary tables.
  This is convenient but could be fragile across statsmodels versions.
  For production code, consider building tables directly from model.params,
  model.bse, model.tvalues/zvalues, and model.pvalues.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
from io import StringIO

from ravix.modeling.format_utils import format_sigfigs, format_r_style, format_pvalue, significance_code
from ravix.modeling.print_anova_table import print_anova_table


# ============================================================================
# Utility Functions
# ============================================================================

def format_summary(summary_df, alpha=0.05):
    """
    Format the summary DataFrame by removing CI columns and adding significance codes.
    
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
    """
    # Calculate confidence interval column names
    lower_ci = f'[{alpha/2:.3f}'
    upper_ci = f'{1 - alpha/2:.3f}]'
    
    # Remove the columns for the confidence intervals if they exist
    columns_to_drop = [lower_ci, upper_ci]
    existing_columns = [col for col in columns_to_drop if col in summary_df.columns]
    summary_df.drop(existing_columns, axis=1, inplace=True, errors='ignore')
    
    # Format the P>|t| column if it exists (OLS models)
    if 'P>|t|' in summary_df.columns:
        summary_df['P>|t|'] = summary_df['P>|t|'].astype(float).map(lambda x: f'{x:.6f}')
        summary_df[' '] = summary_df['P>|t|'].apply(significance_code)
    
    # Format the P>|z| column if it exists (GLM/Logit models)
    if 'P>|z|' in summary_df.columns:
        summary_df['P>|z|'] = summary_df['P>|z|'].astype(float).map(lambda x: f'{x:.6f}')
        summary_df[' '] = summary_df['P>|z|'].apply(significance_code)
    
    return summary_df


# ============================================================================
# Main Summary Function
# ============================================================================

def summary(model, out='simple', alpha = 0.05, format='text'):
    """
    Generate and print a summary of a regression model.
    
    Produces formatted summaries similar to R, STATA, or statsmodels output,
    or displays specific components like coefficients or ANOVA tables.
    
    Parameters
    ----------
    model : Ravix regression result
        Fitted regression model (OLS, Logit, or GLM)
    out : str, default='simple'
        Output format type:
        - 'simple': Clean, readable summary with coefficients and model stats
        - 'statsmodels' or 'stats': Native statsmodels summary
        - 'r': R-style summary with residual quantiles
        - 'stata': STATA-style tabular output
        - 'coefficients' or 'coef': Coefficient table with confidence intervals
        - 'anova': Overall ANOVA table
        - 'anova_lm': Type I (sequential) ANOVA table
    level : float, default=0.95
        Confidence level for confidence intervals (0 < level < 1)
    format : str, default='text'
        Output format:
        - 'text': Print formatted text to console (default behavior)
        - 'latex': Return LaTeX table string
        - 'dataframe' or 'df': Return pandas DataFrame
        
    Returns
    -------
    None, str, or pd.DataFrame
        - If format='text': Prints to console, returns None
        - If format='latex': Returns LaTeX string
        - If format='dataframe' or 'df': Returns pandas DataFrame
        
    Raises
    ------
    ValueError
        If model type is unsupported or output format is invalid
        
    Examples
    --------
    >>> import ravix
    >>> model = ravix.ols("Y~X", data = df)
    >>> ravix.summary(model)  # Simple summary (text)
    >>> ravix.summary(model, out='r')  # R-style summary (text)
    >>> ravix.summary(model, out='coefficients', level=0.99)  # 99% CI (text)
    >>> latex_str = ravix.summary(model, format='latex')  # LaTeX output
    >>> df = ravix.summary(model, format='df')  # DataFrame output
    """
    # Validate parameters
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    
    format = format.lower()
    if format not in ['text', 'latex', 'dataframe', 'df']:
        raise ValueError(f"format must be 'text', 'latex', 'dataframe', or 'df', got '{format}'")
    
    out = out.lower()
    level = round(1 - alpha, 5)
    
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
        return _print_ols_summary(model, out, level, alpha, format)
    elif model_type == "logit":
        return _print_logistic_summary(model, out, level, alpha, format)
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
        Model type identifier: 'ols', 'logit', or 'unsupported'
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
            # Other GLM families not yet supported
            return "unsupported"
    
    return "unsupported"


# ============================================================================
# OLS Summary Functions
# ============================================================================

def _print_ols_summary(model, out, level, alpha, format='text'):
    """
    Generate summary output for OLS regression models.
    
    Parameters
    ----------
    model : statsmodels OLS result
        Fitted OLS regression model
    out : str
        Output format type
    level : float
        Confidence level for intervals
    alpha : float
        Significance level (1 - level)
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
        return _format_coefficient_table(model, level, alpha, format)
    elif out == 'anova':
        return _format_anova_output(model, format)
    elif out == 'anova_lm':
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
    Extract and format coefficient table from model summary.
    
    Parameters
    ----------
    model : statsmodels regression result
        Fitted regression model
    alpha : float
        Significance level
    r_style_labels : bool, default=False
        If True, rename columns to match R output style
        
    Returns
    -------
    pd.DataFrame
        Formatted coefficient table with significance codes
        
    Notes
    -----
    This function uses HTML parsing of statsmodels summary tables, which is
    convenient but potentially fragile across statsmodels versions. A more
    robust approach would build the table directly from model.params, model.bse,
    model.tvalues/zvalues, and model.pvalues.
    """
    results_summary = model.summary(alpha=alpha)
    results_as_html = results_summary.tables[1].as_html()
    summary_df = pd.read_html(StringIO(results_as_html), header=0, index_col=0)[0]
    summary_df = format_summary(summary_df, alpha)
    
    # Rename columns to R-style if requested
    if r_style_labels:
        # Detect whether we have t-statistics or z-statistics
        if 't' in summary_df.columns:
            # OLS model uses t-statistics
            column_mapping = {
                'coef': 'Estimate',
                'std err': 'Std. Error',
                't': 't value',
                'P>|t|': 'Pr(>|t|)'
            }
        elif 'z' in summary_df.columns:
            # GLM/Logit models use z-statistics
            column_mapping = {
                'coef': 'Estimate',
                'std err': 'Std. Error',
                'z': 'z value',
                'P>|z|': 'Pr(>|z|)'
            }
        else:
            # Unknown format, skip renaming
            column_mapping = {}
        
        summary_df.rename(columns=column_mapping, inplace=True)
    
    return summary_df


def _format_simple_ols_summary(summary_df, stats, format='text'):
    """Format simple, clean OLS summary."""
    if format in ['dataframe', 'df']:
        # Return coefficient table with stats as attributes
        result_df = summary_df.copy()
        result_df.attrs['r_squared'] = stats['r_squared']
        result_df.attrs['adj_r_squared'] = stats['adj_r_squared']
        result_df.attrs['f_statistic'] = stats['f_statistic']
        result_df.attrs['aic'] = stats['aic']
        result_df.attrs['bic'] = stats['bic']
        return result_df
    
    # Build LaTeX output
    if format == 'latex':
        latex_parts = []
        latex_parts.append("\\section*{Summary of Regression Analysis}")
        latex_parts.append("\n\\subsection*{Coefficients}")
        latex_parts.append(summary_df.to_latex())
        latex_parts.append("\n\\subsection*{Model Statistics}")
        latex_parts.append("\\begin{itemize}")
        latex_parts.append(f"\\item R-squared: {stats['r_squared_fmt']}")
        latex_parts.append(f"\\item Adjusted R-squared: {stats['adj_r_squared_fmt']}")
        latex_parts.append(f"\\item AIC: {stats['aic_fmt']}")
        latex_parts.append(f"\\item BIC: {stats['bic_fmt']}")
        latex_parts.append(f"\\item F-statistic: {stats['f_statistic_fmt']} on {stats['df_model']} and {stats['df_resid']} DF, p-value: {stats['f_p_value_fmt']}")
        latex_parts.append("\\end{itemize}")
        content = '\n'.join(latex_parts)
        return _handle_output(content, format)
    
    # Build text output
    output = []
    output.append("Summary of Regression Analysis:")
    output.append("=" * 54)
    output.append("\nCoefficients:")
    output.append("-" * 54)
    output.append(str(summary_df))
    output.append("\nModel Statistics:")
    output.append("-" * 54)
    output.append(f"R-squared:      {stats['r_squared_fmt']:<16}AIC: {stats['aic_fmt']}")
    output.append(f"Adj. R-squared: {stats['adj_r_squared_fmt']:<16}BIC: {stats['bic_fmt']}")
    output.append(f"F-statistic: {stats['f_statistic_fmt']} on {stats['df_model']} and "
                 f"{stats['df_resid']} DF, p-value: {stats['f_p_value_fmt']}")
    output.append("=" * 54)
    
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
        latex_parts = []
        latex_parts.append("\\section*{R-Style Regression Summary}")
        latex_parts.append("\n\\subsection*{Residuals}")
        latex_parts.append("\\begin{verbatim}")
        latex_parts.append("    Min      1Q    Median     3Q      Max")
        latex_parts.append(" ".join(f"{x:8.4f}" for x in resid_stats))
        latex_parts.append("\\end{verbatim}")
        latex_parts.append("\n\\subsection*{Coefficients}")
        latex_parts.append(summary_df.to_latex())
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
    output.append(str(summary_df))
    output.append("---")
    output.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    output.append("")
    output.append(f"Residual standard error: {stats['RSE_fmt']} on {stats['df_resid']} degrees of freedom")
    output.append(f"R-squared: {stats['r_squared_fmt']}, Adjusted R-squared: {stats['adj_r_squared_fmt']}")
    output.append(f"F-statistic: {stats['f_statistic_fmt']} on {stats['df_model']} and "
                 f"{stats['df_resid']} DF, p-value: {stats['f_p_value_fmt']}")
    
    content = '\n'.join(output)
    return _handle_output(content, format)


def _format_coefficient_table(model, level, alpha, format='text'):
    """Format coefficient estimates with confidence intervals only."""
    conf_intervals = model.conf_int(alpha=alpha)
    
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
        latex_parts.append("\\section*{Coefficients}")
        latex_parts.append(coef_df_formatted.to_latex())
        content = '\n'.join(latex_parts)
        return _handle_output(content, format)
    
    # Build text output
    output = []
    output.append("Coefficients:")
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

def _print_logistic_summary(model, out, level, alpha, format='text'):
    """
    Generate summary output for logistic regression models.
    
    Parameters
    ----------
    model : statsmodels Logit result
        Fitted logistic regression model
    out : str
        Output format type
    level : float
        Confidence level for intervals
    alpha : float
        Significance level (1 - level)
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
        # Use consistent coefficient table formatting like OLS
        return _format_coefficient_table(model, level, alpha, format)
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
        latex_parts = []
        latex_parts.append("\\section*{Summary of Logistic Regression Analysis}")
        latex_parts.append("\n\\subsection*{Coefficients (Log-Odds)}")
        latex_parts.append(summary_df.to_latex())
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
    output = []
    output.append("Summary of Logistic Regression Analysis:")
    output.append("=" * 54)
    output.append("\nCoefficients (Log-Odds):")
    output.append("-" * 54)
    output.append(str(summary_df))
    output.append("\nModel Statistics:")
    output.append("-" * 54)
    output.append(f"Log-Likelihood: {stats['log_likelihood_fmt']:<16}AIC: {stats['aic_fmt']}")
    output.append(f"Pseudo R-squared: {stats['pseudo_r_squared_fmt']:<14}BIC: {stats['bic_fmt']}")
    output.append("=" * 54)
    
    content = '\n'.join(output)
    return _handle_output(content, format)

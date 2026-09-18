"""Nested OLS model comparison for Ravix."""

import numpy as np
import pandas as pd
from scipy import stats

from .format_utils import format_pvalue, format_r_style, format_sigfigs, significance_code


def compare(model1, model2, format="text"):
    """
    Compare two nested OLS models using a partial F-test.

    The models may be supplied in either order. Ravix automatically identifies
    the reduced and full models from their residual degrees of freedom, verifies
    that the models are nested and were fitted to the same response observations,
    and then performs the classical partial F-test.

    Parameters
    ----------
    model1, model2 : Ravix OLS results
        Two nested OLS models fitted to the same response observations.
    format : {"text", "latex", "dataframe", "df"}, default="text"
        Output format. ``"text"`` prints a formatted model-comparison table.
        ``"latex"`` returns a booktabs-style LaTeX table string (requires
        ``\\usepackage{booktabs}`` in the including document; does not
        require the optional ``jinja2`` package). ``"dataframe"`` or
        ``"df"`` returns the pandas DataFrame underlying the comparison.

    Returns
    -------
    None, str, or pandas.DataFrame
        - If ``format="text"``: prints the comparison and returns None.
        - If ``format="latex"``: returns a LaTeX string.
        - If ``format="dataframe"`` or ``format="df"``: returns a DataFrame.

    Notes
    -----
    The partial F-test evaluates whether the additional terms in the full model
    jointly improve fit relative to the reduced model. It is the classical OLS
    test and uses the full model's residual mean square as the denominator.

    Examples
    --------
    >>> reduced = ravix.ols("Sales ~ Price", data=df)
    >>> full = ravix.ols("Sales ~ Price + Advertising + Income", data=df)
    >>> ravix.compare(reduced, full)
    >>> table = ravix.compare(reduced, full, format="df")
    >>> latex = ravix.compare(reduced, full, format="latex")
    """
    output_format = str(format).lower()
    if output_format not in {"text", "latex", "dataframe", "df"}:
        raise ValueError(
            "format must be 'text', 'latex', 'dataframe', or 'df', "
            f"got '{format}'"
        )

    _validate_ols_model(model1, "model1")
    _validate_ols_model(model2, "model2")
    _validate_same_response(model1, model2)

    # The reduced model has more residual degrees of freedom because it uses
    # fewer independently estimated regression coefficients.
    if np.isclose(model1.df_resid, model2.df_resid):
        raise ValueError(
            "The models have the same residual degrees of freedom. A partial "
            "F-test requires a reduced model and a larger nested full model."
        )

    if model1.df_resid > model2.df_resid:
        reduced, full = model1, model2
    else:
        reduced, full = model2, model1

    _validate_nested(reduced, full)

    df_diff = float(reduced.df_resid - full.df_resid)
    if df_diff <= 0:
        raise ValueError("The full model must estimate more parameters than the reduced model.")

    sse_reduced = float(np.sum(np.asarray(reduced.resid, dtype=float) ** 2))
    sse_full = float(np.sum(np.asarray(full.resid, dtype=float) ** 2))
    sum_sq = sse_reduced - sse_full

    # Nested OLS models fitted to identical observations cannot have a larger
    # SSE for the full model, apart from tiny floating-point noise.
    tolerance = np.finfo(float).eps * max(1.0, abs(sse_reduced), abs(sse_full)) * 100
    if sum_sq < -tolerance:
        raise ValueError(
            "The apparent full model has a larger SSE than "
            "the reduced model. Verify that the models are nested and use the same data."
        )
    if abs(sum_sq) <= tolerance:
        sum_sq = 0.0

    mse_full = sse_full / float(full.df_resid)
    if mse_full <= 0:
        raise ValueError("The full model has a non-positive residual mean square.")

    f_stat = (sum_sq / df_diff) / mse_full
    p_value = float(stats.f.sf(f_stat, df_diff, full.df_resid))

    table = pd.DataFrame(
        {
            "Residual df": [float(reduced.df_resid), float(full.df_resid)],
            "SSE": [sse_reduced, sse_full],
            "df Difference": [np.nan, df_diff],
            "Sum Sq": [np.nan, sum_sq],
            "F": [np.nan, f_stat],
            "p-value": [np.nan, p_value],
            "Signif.": ["", significance_code(p_value)],
        },
        index=["Reduced", "Full"],
    )

    if output_format in {"dataframe", "df"}:
        return table

    if output_format == "latex":
        return _format_latex(table)

    print(_format_text(table, reduced, full))
    return None


def _validate_ols_model(model, name):
    """Validate that an object is an OLS regression result suitable for comparison."""
    required = ("model", "resid", "df_resid", "params")
    missing = [attr for attr in required if not hasattr(model, attr)]
    if missing:
        raise TypeError(
            f"{name} is not a fitted OLS model. Missing attributes: {', '.join(missing)}."
        )

    model_type = getattr(model, "model_type", None)
    statsmodels_name = model.model.__class__.__name__.lower()
    if model_type not in (None, "ols") or statsmodels_name != "ols":
        raise TypeError("compare() currently supports OLS models only.")


def _validate_same_response(model1, model2):
    """Ensure both models were fitted to the same response observations."""
    y1 = np.asarray(model1.model.endog)
    y2 = np.asarray(model2.model.endog)

    if y1.shape != y2.shape:
        raise ValueError(
            "The models were fitted to different numbers of response observations. "
            "Nested-model comparison requires the same observations."
        )

    try:
        same = np.allclose(y1.astype(float), y2.astype(float), equal_nan=True)
    except (TypeError, ValueError):
        same = np.array_equal(y1, y2)

    if not same:
        raise ValueError(
            "The models were fitted to different response observations. "
            "Nested-model comparison requires the same response data."
        )


def _validate_nested(reduced, full):
    """Verify that the reduced model's design space is contained in the full model."""
    x_reduced = np.asarray(reduced.model.exog, dtype=float)
    x_full = np.asarray(full.model.exog, dtype=float)

    if x_reduced.shape[0] != x_full.shape[0]:
        raise ValueError(
            "The models were fitted to different numbers of observations. "
            "Nested-model comparison requires the same rows."
        )

    rank_full = np.linalg.matrix_rank(x_full)
    rank_combined = np.linalg.matrix_rank(np.column_stack([x_full, x_reduced]))

    if rank_combined != rank_full:
        raise ValueError(
            "The models are not nested. Every term represented by the reduced "
            "model must be contained in the full model."
        )


def _format_text(table, reduced, full):
    """Create summary-style plain-text output for a nested-model comparison."""
    width = 88
    lines = [
        "Nested Model Comparison",
        "=" * width,
    ]

    reduced_formula = getattr(reduced, "formula", None)
    full_formula = getattr(full, "formula", None)
    if reduced_formula:
        lines.append(f"Reduced: {reduced_formula}")
    if full_formula:
        lines.append(f"Full:    {full_formula}")
    if reduced_formula or full_formula:
        lines.append("-" * width)

    header = (
        f"{'Model':<10}"
        f"{'Residual df':>13}"
        f"{'SSE':>14}"
        f"{'df Diff':>10}"
        f"{'Sum Sq':>14}"
        f"{'F':>12}"
        f"{'p-value':>15}"
    )
    lines.append(header)
    lines.append("-" * width)

    for label in ["Reduced", "Full"]:
        row = table.loc[label]

        residual_df = _format_df(row["Residual df"])
        sse = format_sigfigs(row["SSE"], 6)

        if label == "Reduced":
            df_diff = ""
            sum_sq = ""
            f_value = ""
            p_value = ""
        else:
            df_diff = _format_df(row["df Difference"])
            sum_sq = format_sigfigs(row["Sum Sq"], 6)
            f_value = format_r_style(row["F"])
            p_value = format_pvalue(row["p-value"])
            sig = significance_code(row["p-value"])
            if sig:
                p_value = f"{p_value} {sig}"

        lines.append(
            f"{label:<10}"
            f"{residual_df:>13}"
            f"{sse:>14}"
            f"{df_diff:>10}"
            f"{sum_sq:>14}"
            f"{f_value:>12}"
            f"{p_value:>15}"
        )

    lines.append("=" * width)
    lines.append("Partial F-test: additional terms in the full model are tested jointly.")
    return "\n".join(lines)


def _format_df(value):
    """Format degrees of freedom without an unnecessary decimal when integral."""
    if pd.isna(value):
        return ""
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return format_sigfigs(value, 6)


def _format_latex(table):
    """Render the comparison table as a booktabs-style LaTeX tabular.

    Built by hand instead of calling ``DataFrame.to_latex()``: pandas has
    routed ``to_latex()`` through its Styler implementation since 2.0, which
    imports jinja2 purely to render numbers into a ``tabular`` block --
    jinja2 is not, and shouldn't need to be, a declared ravix dependency for
    that. The table here has a fixed, simple shape (numeric columns, two
    rows), so a templating engine adds nothing a plain string join can't do.

    Every column is formatted with the same helper ``_format_text`` uses for
    that column (``format_sigfigs`` for SSE/Sum Sq, ``format_r_style`` for F,
    ``format_pvalue`` + ``significance_code`` for p-value), so a number reads
    identically whether ``format="text"`` or ``format="latex"`` produced it.
    The presentation-oriented LaTeX output appends significance stars
    directly to the p-value, matching the text output, rather than carrying
    the DataFrame's separate ``Signif.`` column into the table; the
    DataFrame output (``format="dataframe"``/``"df"``) keeps ``p-value``
    numeric and ``Signif.`` separate for downstream analysis.

    Requires ``\\usepackage{booktabs}`` in the including LaTeX document, to
    match the table style pandas' own LaTeX export used.
    """
    columns = [col for col in table.columns if col != "Signif."]
    column_format = "l" + "r" * len(columns)

    def escape(text):
        # Only "&", "%", and "_" can plausibly appear here (formula-derived
        # column labels aren't included in this table; only the fixed column
        # names and formatted numbers/stars are), but escape defensively
        # rather than assume.
        for char, replacement in (
            ("\\", r"\textbackslash{}"),
            ("&", r"\&"),
            ("%", r"\%"),
            ("_", r"\_"),
            ("#", r"\#"),
        ):
            text = text.replace(char, replacement)
        return text

    def format_cell(row, column):
        value = row[column]
        if pd.isna(value):
            return ""
        if column in ("Residual df", "df Difference"):
            return _format_df(value)
        if column in ("SSE", "Sum Sq"):
            return format_sigfigs(value, 6)
        if column == "F":
            return format_r_style(value)
        if column == "p-value":
            p_text = format_pvalue(value)
            sig = significance_code(value)
            return f"{p_text} {sig}" if sig else p_text
        return f"{float(value):.6g}"

    header_cells = ["Model"] + [escape(str(col)) for col in columns]
    lines = [
        f"\\begin{{tabular}}{{{column_format}}}",
        "\\toprule",
        " & ".join(header_cells) + r" \\",
        "\\midrule",
    ]
    for label in table.index:
        row = table.loc[label]
        cells = [escape(str(label))] + [format_cell(row, col) for col in columns]
        lines.append(" & ".join(cells) + r" \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)

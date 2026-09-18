"""Regression tests for Ravix bugs fixed after 1.0.1.

These tests protect package behavior that is not specific to the textbook API.
They cover issues reproduced in Lab 2, the Lead Generation case study, and
video_engagement.csv, plus the boxplot categorical-type deprecation.

Coefficient-name formatting is intentionally unchanged for textbook compatibility.
Dataset packaging checks are outside this suite's scope.
"""

from __future__ import annotations

import sys
import warnings
from functools import wraps

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba_array
import numpy as np
import pandas as pd
import pytest

from ravix import (
    abline,
    barplot,
    boxplot,
    bsr,
    compare,
    get_data,
    intervals,
    logistic,
    ols,
    plot,
    plot_cor,
    predict,
    robust,
    stepwise,
)
from ravix.modeling.format_utils import format_r_style
from ravix.modeling.parse_formula import parse_formula


@pytest.mark.parametrize("kind", ["series", "dataframe", "numeric_formula", "categorical_formula"])
def test_boxplot_avoids_vert_and_preserves_medians(kind, education_df, monkeypatch):
    from matplotlib.axes import Axes
    original = Axes.bxp
    medians = []
    axes_used = []

    @wraps(original)
    def strict_bxp(self, bxpstats, *args, **kwargs):
        assert "vert" not in kwargs, "Deprecated vert reached Matplotlib"
        assert kwargs["orientation"] == "vertical"
        medians.extend(stat["med"] for stat in bxpstats)
        axes_used.append(self)
        return original(self, bxpstats, *args, **kwargs)

    monkeypatch.setattr(Axes, "bxp", strict_bxp)
    monkeypatch.setattr(plt, "show", lambda: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if kind == "series":
            boxplot(education_df.Salary)
            expected = [education_df.Salary.median()]
        elif kind == "dataframe":
            boxplot(education_df[["Salary", "Jobs"]])
            expected = education_df[["Salary", "Jobs"]].median().tolist()
        elif kind == "numeric_formula":
            boxplot("Salary ~ Jobs", data=education_df)
            expected = education_df[["Salary", "Jobs"]].median().tolist()
        else:
            boxplot("Salary ~ Education", data=education_df)
            expected = education_df.groupby("Education").Salary.median().tolist()
    np.testing.assert_allclose(sorted(medians), sorted(expected))
    assert all("bxp" not in ax.__dict__ for ax in axes_used)
    assert not [w for w in caught if "vert" in str(w.message).lower()]


def test_boxplot_adapter_restores_axes_after_error(monkeypatch):
    from ravix.plots.boxplot import _seaborn_boxplot
    import seaborn as sns
    fig, ax = plt.subplots()
    original = ax.bxp
    def fail(**kwargs):
        raise RuntimeError("render failed")
    monkeypatch.setattr(sns, "boxplot", fail)
    with pytest.raises(RuntimeError, match="render failed"):
        _seaborn_boxplot(ax=ax)
    assert ax.bxp == original
    assert "bxp" not in ax.__dict__
    plt.close(fig)


@pytest.mark.parametrize("response", ["Y^(-1", "Y^-1)", "Y^(2", "Y^2)", "Y^1.2.3"])
def test_response_power_rejects_malformed_syntax(response):
    with pytest.raises(ValueError, match="Invalid response power expression"):
        parse_formula(f"{response} ~ X", pd.DataFrame({"Y": [1., 2.], "X": [1., 2.]}))


@pytest.mark.parametrize("power", [-0.5, -1.5, 0.5, 1.5])
def test_response_fractional_power_rejects_negative_bases(power):
    with pytest.raises(ValueError, match="negative value"):
        parse_formula(f"Y^{power} ~ X", pd.DataFrame({"Y": [-1., 2.], "X": [1., 2.]}))


@pytest.mark.parametrize("response", ["Y^-1", "Y^(-1)", "Y**-1"])
def test_response_negative_power_rejects_zero(response):
    with pytest.raises(ValueError, match="zero value"):
        parse_formula(f"{response} ~ X", pd.DataFrame({"Y": [0., 2.], "X": [1., 2.]}))


@pytest.mark.parametrize("exponent", ["-1", "(-2)", "+2", "(+2)", "0", "2.0"])
def test_response_integer_powers_preserve_negative_bases(exponent):
    df = pd.DataFrame({"Y": [-2., 3.], "X": [1., 2.]})
    y, x = parse_formula(f"Y^{exponent} ~ X", df)
    np.testing.assert_allclose(y, df.Y ** float(exponent.strip("()")))
    pd.testing.assert_frame_equal(x, parse_formula("Y ~ X", df)[1])


@pytest.mark.parametrize("exponent", ["-.5", "(-0.5)", "+.5", "1.5"])
def test_response_fractional_powers_preserve_positive_bases(exponent):
    df = pd.DataFrame({"Y": [1., 4.], "X": [1., 2.]})
    y, _ = parse_formula(f"Y^{exponent} ~ X", df)
    np.testing.assert_allclose(y, df.Y ** float(exponent.strip("()")))


@pytest.mark.parametrize(
    ("formula", "new_data"),
    [
        ("Y^-1 ~ X^2", {"X": [4.0]}),
        ("Y**-1 ~ X**2", {"X": [4.0]}),
        ("log(Y) ~ log(X)", {"X": [4.0]}),
        ("sqrt(Y) ~ X", {"X": [4.0]}),
    ],
)
def test_predict_does_not_require_transformed_response_column(formula, new_data):
    df = pd.DataFrame({"Y": [2.0, 4.0, 8.0, 16.0], "X": [1.0, 2.0, 3.0, 4.0]})
    model = ols(formula, data=df)
    result = predict(model, pd.DataFrame(new_data))
    assert result.shape == (1,)
    assert np.isfinite(result).all()


def test_predict_transformed_response_preserves_fit_time_categories():
    df = pd.DataFrame(
        {"Y": [10.0, 12.0, 18.0, 20.0], "X": [1.0, 2.0, 3.0, 4.0],
         "Group": ["A", "B", "A", "B"]}
    )
    model = ols("log(Y) ~ X + Group", data=df)
    result = predict(model, pd.DataFrame({"X": [2.5], "Group": ["B"]}))
    assert result.shape == (1,)
    assert np.isfinite(result).all()


def test_predict_transformed_response_with_dot_formula_excludes_no_predictors():
    df = pd.DataFrame({"Y": [2.0, 4.0, 8.0, 16.0], "X": [1., 2., 3., 4.], "Z": [4., 1., 2., 5.]})
    model = ols("log(Y) ~ .", data=df)
    result = predict(model, pd.DataFrame({"X": [2.5], "Z": [2.5]}))
    assert result.shape == (1,)
    assert np.isfinite(result).all()


def test_predict_exact_single_row_categorical_notebook_call():
    df = get_data("job_changes.csv")
    model = ols("Salary ~ Jobs + Education", data=df)
    new_employee = pd.DataFrame({"Jobs": [4], "Education": ["Masters"]})
    result = predict(model, new_employee)
    np.testing.assert_allclose(np.asarray(result), [100.931353], rtol=1e-6)


def _run_selector(selector, formula, data, metric="bic"):
    if selector == "bsr":
        return bsr(formula, data=data, max_var=8, metric=metric)
    return stepwise(
        formula,
        data=data,
        direction="forward",
        metric=metric,
        verbose=False,
    )


@pytest.mark.parametrize("selector", ["bsr", "stepwise"])
def test_selected_numeric_model_predicts_from_raw_data(selector):
    rng = np.random.default_rng(12)
    x1 = np.linspace(0, 10, 60)
    df = pd.DataFrame(
        {
            "Y": 5 + 3 * x1 + rng.normal(0, 0.4, 60),
            "X1": x1,
            "X2": rng.normal(size=60),
        }
    )
    model = _run_selector(selector, "Y ~ X1 + X2", df, metric="aic")
    result = predict(model, pd.DataFrame({"X1": [2.5], "X2": [0.1]}))
    assert result.shape == (1,)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("selector", ["bsr", "stepwise"])
def test_selected_complete_categorical_model_predicts_one_row(selector):
    rng = np.random.default_rng(21)
    group = np.tile(["A", "B", "C"], 30)
    df = pd.DataFrame(
        {
            "Y": 10 + 20 * (group == "B") + 40 * (group == "C")
            + rng.normal(0, 0.3, 90),
            "Group": group,
        }
    )
    model = _run_selector(selector, "Y ~ Group", df)
    assert set(model.model.exog_names) == {"Intercept", "Group_B", "Group_C"}
    result = predict(model, pd.DataFrame({"Group": ["C"]}))
    assert result.shape == (1,)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("selector", ["bsr", "stepwise"])
def test_selected_partial_categorical_model_rejects_prediction_clearly(selector):
    rng = np.random.default_rng(34)
    group = np.tile(["A", "B", "C"], 30)
    df = pd.DataFrame(
        {
            "Y": 10 + 50 * (group == "B") + rng.normal(0, 0.3, 90),
            "Noise": rng.normal(size=90),
            "Group": group,
        }
    )
    model = _run_selector(selector, "Y ~ Noise + Group", df)
    assert model.model.exog_names == ["Intercept", "Group_B"]
    with pytest.raises(
        ValueError,
        match="selected predictors include only some levels.*Group",
    ):
        predict(model, pd.DataFrame({"Noise": [0.0], "Group": ["B"]}))


@pytest.fixture
def education_df() -> pd.DataFrame:
    """Categorical data matching the calls used in the current Lab 2."""
    return pd.DataFrame(
        {
            "Salary": [48, 57, 61, 73, 82, 91, 104, 116],
            "Jobs": [1, 2, 3, 3, 4, 5, 6, 7],
            "Education": [
                "HS",
                "Bachelors",
                "Masters",
                "HS",
                "Bachelors",
                "Masters",
                "HS",
                "Masters",
            ],
        }
    )


@pytest.fixture
def video_engagement_df() -> pd.DataFrame:
    """Two-promoter data matching video_engagement.csv's column structure."""
    return pd.DataFrame(
        {
            "Likes": [120, 180, 240, 310, 390, 470, 560, 650],
            "Promoter": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "Age": [2, 4, 7, 11, 16, 22, 29, 37],
            "Sentiment": [-0.30, 0.15, -0.05, 0.35, 0.10, 0.55, 0.40, 0.75],
        }
    )


@pytest.fixture
def lead_generation_df() -> pd.DataFrame:
    """Positive data supporting every transformation in the lead-generation case."""
    x1 = np.arange(1.0, 9.0)
    x2 = np.exp(np.linspace(0.2, 1.6, 8))
    x3 = np.exp(np.linspace(0.1, 1.2, 8))
    x4 = np.arange(5.0, 45.0, 5.0)
    return pd.DataFrame(
        {
            "X1": x1,
            "Y1": np.exp(2.0 + 0.18 * x1),
            "X2": x2,
            "Y2": 25.0 + 8.0 * np.log(x2),
            "X3": x3,
            "Y3": np.exp(1.2 + 0.65 * np.log(x3)),
            "X4": x4,
            "Y4": 1.0 / (0.01 + 0.0005 * x4),
        }
    )


@pytest.fixture
def simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [2.0, 2.0, 5.0, 1.0, 3.0, 0.5, 7.0],
            "y": [7.0, 5.0, 12.0, 4.0, 8.0, 1.0, 12.0],
        }
    )


def test_abline_transformed_predictor_without_ravix_plot_explains_error(
    lead_generation_df,
):
    model = ols("Y2 ~ log(X2)", data=lead_generation_df)
    fig, ax = plt.subplots()
    ax.scatter(lead_generation_df.X2, lead_generation_df.Y2)
    with pytest.raises(ValueError, match=r"plot was not generated with ravix\.plot\(\)"):
        abline(model, ax=ax)
    assert not ax.lines
    plt.close(fig)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("education_dtype", ["object", "category"])
def test_boxplot_categorical_formula_has_no_dtype_deprecation(
    education_df, education_dtype, monkeypatch
):
    """Both string and categorical grouping columns avoid the deprecated dtype API."""
    df = education_df.copy()
    df["Education"] = df["Education"].astype(education_dtype)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        boxplot("Salary ~ Education", data=df)

    dtype_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, (FutureWarning, DeprecationWarning))
        and "is_categorical_dtype" in str(item.message)
    ]
    assert dtype_warnings == []


def test_barplot_categorical_formula_has_no_future_warning(education_df, monkeypatch):
    """The exact categorical barplot call in Lab 2 should be warning-free."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = barplot("Salary ~ Education", data=education_df)

    future_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, FutureWarning)
    ]
    assert result is None
    assert future_warnings == []


def test_barplot_categorical_series_counts_levels(education_df, monkeypatch):
    """barplot(df.Education) should create one frequency bar per level."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "clf", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)

    result = barplot(education_df.Education)
    ax = plt.gca()
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    heights = [patch.get_height() for patch in ax.patches]
    plotted_counts = dict(zip(labels, heights))

    assert result is None
    assert set(plotted_counts) == {"HS", "Bachelors", "Masters"}
    assert plotted_counts == pytest.approx(
        education_df.Education.value_counts().to_dict()
    )


def test_coefficient_summary_omits_significance_codes_key(simple_df, capsys):
    """out='coef' keeps the coefficient table but omits its explanatory key."""
    reg = ols("y ~ x", data=simple_df)
    reg.summary(out="coef")
    out = capsys.readouterr().out

    assert "Coefficients:" in out
    assert "Estimate" in out
    assert "Signif. codes:" not in out


def test_plot_cor_formula_includes_all_categorical_dummies(video_engagement_df):
    """Both Promoter levels must appear in the video-engagement correlation plot."""
    fig, ax = plot_cor(
        "Likes ~ Promoter + Age + Sentiment",
        data=video_engagement_df,
        style=4,
        show=False,
    )
    labels = [tick.get_text() for tick in ax.get_xticklabels()]

    assert labels == ["Likes", "Promoter_A", "Promoter_B", "Age", "Sentiment"]
    plt.close(fig)


def test_scatterplot_matrix_honors_per_observation_colors(
    video_engagement_df, monkeypatch
):
    """The color array must propagate to every off-diagonal scatter panel."""
    colors = np.where(video_engagement_df.Promoter == "A", "darkred", "darkblue")
    expected_rgb = to_rgba_array(colors)[:, :3]

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)

    result = plot(
        "Likes ~ Age + Sentiment",
        data=video_engagement_df,
        color=colors,
    )
    fig = plt.gcf()
    scatter_axes = [ax for ax in fig.axes if ax.collections]

    assert result is None
    assert len(scatter_axes) == 6
    for ax in scatter_axes:
        actual_rgb = ax.collections[0].get_facecolors()[:, :3]
        assert actual_rgb.shape[0] == len(video_engagement_df)
        np.testing.assert_allclose(actual_rgb, expected_rgb)


def test_formula_negative_exponent_response_lead_generation(lead_generation_df):
    """The inverse-response syntax used in the lead-generation case must parse."""
    model = ols("Y4^-1 ~ X4", data=lead_generation_df)

    np.testing.assert_allclose(
        np.asarray(model.model.endog),
        np.asarray(lead_generation_df["Y4"], dtype=float) ** -1,
    )


@pytest.mark.parametrize(
    ("formula", "raw_x", "log_x"),
    [
        ("log(Y1) ~ X1", "X1", False),
        ("Y2 ~ log(X2)", "X2", True),
        ("log(Y3) ~ log(X3)", "X3", True),
        ("Y4^-1 ~ X4", "X4", False),
    ],
)
def test_abline_handles_lead_generation_transformations(
    lead_generation_df, formula, raw_x, log_x
):
    """Overlay the fitted line in the same transformed coordinates as plot()."""
    reg = ols(formula, data=lead_generation_df)
    fig, ax = plot(formula, data=lead_generation_df, show=False)
    offsets = np.asarray(ax.collections[0].get_offsets(), dtype=float)
    x_expected = np.asarray(lead_generation_df[raw_x], dtype=float)
    if log_x:
        x_expected = np.log(x_expected)

    np.testing.assert_allclose(offsets[:, 0], x_expected)
    np.testing.assert_allclose(offsets[:, 1], np.asarray(reg.model.endog))

    xlim_before = ax.get_xlim()
    ylim_before = ax.get_ylim()
    lines_before = len(ax.lines)
    result = abline(reg, ax=ax)

    assert result is None
    assert len(ax.lines) == lines_before + 1
    assert ax.get_xlim() == xlim_before
    assert ax.get_ylim() == ylim_before

    x_line = np.asarray(ax.lines[-1].get_xdata(), dtype=float)
    y_line = np.asarray(ax.lines[-1].get_ydata(), dtype=float)
    slope_name = next(name for name in reg.params.index if name != "Intercept")
    expected_line = reg.params["Intercept"] + reg.params[slope_name] * x_line

    assert x_line.min() == pytest.approx(x_expected.min())
    assert x_line.max() == pytest.approx(x_expected.max())
    np.testing.assert_allclose(y_line, expected_line)
    plt.close(fig)


# ---------------------------------------------------------------------------
# robust(): HC0-HC3 heteroskedasticity-robust inference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
def test_robust_matches_statsmodels_get_robustcov_results(cov_type):
    """robust() must reproduce statsmodels' own get_robustcov_results exactly."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=40)
    y = 2.0 + 3.0 * x + rng.normal(scale=1.0 + np.abs(x), size=40)
    df = pd.DataFrame({"y": y, "x": x})
    model = ols("y ~ x", data=df)

    fitted = robust(model, type=cov_type)
    expected = model.get_robustcov_results(cov_type=cov_type, use_t=True)

    np.testing.assert_allclose(np.asarray(fitted.params), np.asarray(expected.params))
    np.testing.assert_allclose(np.asarray(fitted.bse), np.asarray(expected.bse))
    np.testing.assert_allclose(np.asarray(fitted.params), np.asarray(model.params))


def test_robust_changes_se_but_not_coefficients():
    """Robust covariance changes standard errors, never the point estimates."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=50)
    y = 1.0 + 2.0 * x + rng.normal(scale=1.0 + 2.0 * np.abs(x), size=50)
    df = pd.DataFrame({"y": y, "x": x})
    model = ols("y ~ x", data=df)
    fitted = robust(model, type="HC3")

    np.testing.assert_allclose(np.asarray(fitted.params), np.asarray(model.params))
    assert not np.allclose(np.asarray(fitted.bse), np.asarray(model.bse))


def test_robust_summary_reports_covariance_type(simple_df, capsys):
    """summary(out='coef') must announce which robust covariance type was used."""
    model = ols("y ~ x", data=simple_df)
    fitted = robust(model, type="HC1")
    fitted.summary(out="coef")
    out = capsys.readouterr().out
    assert "Robust covariance: HC1" in out


def test_robust_dataframe_summary_records_covariance_type(simple_df):
    """The dataframe summary output must carry the covariance type as metadata."""
    model = ols("y ~ x", data=simple_df)
    fitted = robust(model, type="HC2")
    coef_df = fitted.summary(out="coef", format="dataframe")
    assert coef_df.attrs.get("robust_covariance") == "HC2"


def test_robust_rejects_non_ols_model():
    """robust() is OLS-only; logistic models must be rejected clearly."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=40)
    p = 1 / (1 + np.exp(-(0.5 * x)))
    y = (rng.random(40) < p).astype(int)
    df = pd.DataFrame({"y": y, "x": x})
    model = logistic("y ~ x", data=df)
    with pytest.raises(TypeError, match="OLS models only"):
        robust(model)


# ---------------------------------------------------------------------------
# compare(): nested-model partial F-test
# ---------------------------------------------------------------------------

def test_compare_matches_anova_lm():
    """compare()'s F-stat and p-value must match statsmodels' anova_lm."""
    from statsmodels.stats.anova import anova_lm
    rng = np.random.default_rng(5)
    x1 = rng.normal(size=60)
    x2 = rng.normal(size=60)
    y = 1.0 + 2.0 * x1 + rng.normal(size=60)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    reduced = ols("y ~ x1", data=df)
    full = ols("y ~ x1 + x2", data=df)

    result = anova_lm(reduced, full)
    table = compare(reduced, full, format="df")

    np.testing.assert_allclose(table.loc["Full", "F"], result["F"].iloc[1])
    np.testing.assert_allclose(table.loc["Full", "p-value"], result["Pr(>F)"].iloc[1])


def test_compare_is_order_invariant():
    """compare() must identify reduced/full itself regardless of argument order."""
    rng = np.random.default_rng(9)
    x1 = rng.normal(size=50)
    x2 = rng.normal(size=50)
    y = 1.0 + 2.0 * x1 + 0.5 * x2 + rng.normal(size=50)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    reduced = ols("y ~ x1", data=df)
    full = ols("y ~ x1 + x2", data=df)

    table_a = compare(reduced, full, format="df")
    table_b = compare(full, reduced, format="df")
    pd.testing.assert_frame_equal(table_a, table_b)


def test_compare_rejects_same_residual_df():
    """Two models with identical residual df cannot form a partial F-test."""
    rng = np.random.default_rng(13)
    x = rng.normal(size=30)
    y = 1.0 + x + rng.normal(size=30)
    df = pd.DataFrame({"y": y, "x": x})
    model1 = ols("y ~ x", data=df)
    model2 = ols("y ~ x", data=df)
    with pytest.raises(ValueError, match="same residual degrees of freedom"):
        compare(model1, model2)


def test_compare_rejects_non_nested_models():
    """Models of different size whose terms aren't a subset must be rejected."""
    rng = np.random.default_rng(17)
    x1 = rng.normal(size=40)
    x2 = rng.normal(size=40)
    x3 = rng.normal(size=40)
    y = 1.0 + x1 + rng.normal(size=40)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    small = ols("y ~ x1", data=df)
    other = ols("y ~ x2 + x3", data=df)
    with pytest.raises(ValueError, match="not nested"):
        compare(small, other)


def test_compare_rejects_mismatched_observation_counts():
    """Models fitted to different numbers of rows cannot be compared."""
    rng = np.random.default_rng(19)
    x = rng.normal(size=40)
    z = rng.normal(size=40)
    y = 1.0 + x + rng.normal(size=40)
    df_full = pd.DataFrame({"y": y, "x": x, "z": z})
    reduced = ols("y ~ x", data=df_full.iloc[:30])
    full = ols("y ~ x + z", data=df_full)
    with pytest.raises(ValueError, match="different numbers of"):
        compare(reduced, full)


def test_compare_latex_matches_text_formatting():
    """LaTeX output must fold significance into p-value (no separate Signif.
    column) and format SSE/Sum Sq/F consistently with the text output."""
    rng = np.random.default_rng(23)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    y = 1.0 + 5.0 * x1 + 5.0 * x2 + rng.normal(scale=0.1, size=200)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    reduced = ols("y ~ x1", data=df)
    full = ols("y ~ x1 + x2", data=df)

    table = compare(reduced, full, format="df")
    latex = compare(reduced, full, format="latex")

    assert "Signif." not in latex
    assert format_r_style(table.loc["Full", "F"]) in latex


def test_compare_text_prints_and_returns_none(capsys):
    """format='text' (the default) prints the comparison and returns None."""
    rng = np.random.default_rng(29)
    x1 = rng.normal(size=40)
    x2 = rng.normal(size=40)
    y = 1.0 + x1 + rng.normal(size=40)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    reduced = ols("y ~ x1", data=df)
    full = ols("y ~ x1 + x2", data=df)

    result = compare(reduced, full)
    out = capsys.readouterr().out

    assert result is None
    assert "Nested Model Comparison" in out
    assert "Partial F-test" in out


# ---------------------------------------------------------------------------
# barplot(): agg=None ambiguity guard
# ---------------------------------------------------------------------------

def test_barplot_agg_none_rejects_ambiguous_no_formula_input(monkeypatch):
    """Multiple numeric columns with multiple rows can't be plotted unaggregated."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(plt, "clf", lambda *a, **k: None)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    with pytest.raises(ValueError, match="each variable must have only one value"):
        barplot(df, agg=None)


def test_barplot_agg_none_rejects_ambiguous_formula_input(monkeypatch):
    """The same guard applies to a numeric-predictor formula, not just raw data."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(plt, "clf", lambda *a, **k: None)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    df = pd.DataFrame(
        {
            "Salary": [48, 57, 61, 73, 82, 91, 104, 116],
            "Jobs": [1, 2, 3, 3, 4, 5, 6, 7],
        }
    )
    with pytest.raises(ValueError, match="each variable must have only one value"):
        barplot("Salary ~ Jobs", data=df, agg=None)


def test_barplot_agg_none_rejects_duplicate_categories(monkeypatch):
    """A categorical formula with repeated category values is equally ambiguous."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(plt, "clf", lambda *a, **k: None)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    df = pd.DataFrame(
        {
            "Salary": [48, 57, 61, 73, 82, 91, 104, 116],
            "Education": ["HS", "Bachelors", "Masters", "HS", "Bachelors", "Masters", "HS", "Masters"],
        }
    )
    with pytest.raises(ValueError, match="each category must appear only once"):
        barplot("Salary ~ Education", data=df, agg=None)


def test_barplot_agg_none_single_column_uses_named_index_labels(monkeypatch):
    """A single-column, multi-row frame labels bars from the column/index names."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(plt, "clf", lambda *a, **k: None)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    df = pd.DataFrame(
        {"Sales": [10.0, 20.0, 30.0]},
        index=pd.Index(["Q1", "Q2", "Q3"], name="Store"),
    )
    barplot(df, agg=None)
    ax = plt.gca()
    assert ax.get_ylabel() == "Sales"
    assert ax.get_xlabel() == "Store"


def test_barplot_agg_none_single_column_defaults_to_observation_label(monkeypatch):
    """With an unnamed index, the fallback label is 'Observation', not 'Variable'."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(plt, "clf", lambda *a, **k: None)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    df = pd.DataFrame({"Sales": [10.0, 20.0, 30.0]})
    barplot(df, agg=None)
    ax = plt.gca()
    assert ax.get_ylabel() == "Sales"
    assert ax.get_xlabel() == "Observation"


# ---------------------------------------------------------------------------
# stepwise(): no-intercept guard and all-candidates-failed RuntimeError
# ---------------------------------------------------------------------------

def test_stepwise_forward_rejects_no_intercept_formula():
    """Forward selection can't start from an empty, intercept-less model."""
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x1": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="cannot start forward selection from an empty model"):
        stepwise("y ~ x1 - 1", data=df, direction="forward")


def test_stepwise_backward_skips_removal_that_would_empty_no_intercept_model():
    """Backward elimination must skip (not attempt) removing the last term of a
    no-intercept model, since that would leave nothing to fit."""
    df = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x1": [1.0, 2.0, 3.0, 4.0, 6.0]}
    )
    model = stepwise("y ~ x1 - 1", data=df, direction="backward", verbose=False)
    assert model.step.final_variables == ["x1"]
    assert any(
        "leaving nothing to fit" in candidate["reason"]
        for candidate in model.step.failed_candidates
    )


def test_stepwise_raises_when_every_candidate_fails_at_a_step(monkeypatch):
    """If every candidate model attempted at a step raises, that must surface as
    a RuntimeError rather than being silently treated as 'no improvement'."""
    real_module = sys.modules["ravix.modeling.stepwise"]
    original_fit = real_module._fit_matrices

    def flaky_fit(Y, X, *args, **kwargs):
        if X.shape[1] >= 2:
            raise ValueError("synthetic failure")
        return original_fit(Y, X, *args, **kwargs)

    monkeypatch.setattr(real_module, "_fit_matrices", flaky_fit)

    rng = np.random.default_rng(41)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=30),
            "x1": rng.normal(size=30),
            "x2": rng.normal(size=30),
        }
    )
    with pytest.raises(RuntimeError, match=r"All \d+ candidate model\(s\) failed to fit"):
        stepwise("y ~ x1 + x2", data=df, direction="forward", verbose=False)


def test_stepwise_all_failed_error_carries_failed_candidates(monkeypatch):
    """The raised RuntimeError must expose .failed_candidates with every failure,
    since the function never reaches a return (and thus no model.step) here."""
    real_module = sys.modules["ravix.modeling.stepwise"]
    original_fit = real_module._fit_matrices

    def flaky_fit(Y, X, *args, **kwargs):
        if X.shape[1] >= 2:
            raise ValueError("synthetic failure")
        return original_fit(Y, X, *args, **kwargs)

    monkeypatch.setattr(real_module, "_fit_matrices", flaky_fit)

    rng = np.random.default_rng(47)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=30),
            "x1": rng.normal(size=30),
            "x2": rng.normal(size=30),
        }
    )
    with pytest.raises(RuntimeError) as excinfo:
        stepwise("y ~ x1 + x2", data=df, direction="forward", verbose=False)

    failed = excinfo.value.failed_candidates
    assert len(failed) == 2
    assert {c["variable"] for c in failed} == {"x1", "x2"}
    assert all("synthetic failure" in c["reason"] for c in failed)


def test_stepwise_partial_candidate_failure_does_not_raise(monkeypatch):
    """A failure recorded for one candidate must not block selection when other
    candidates at that step still fit successfully."""
    real_module = sys.modules["ravix.modeling.stepwise"]
    original_fit = real_module._fit_matrices

    def flaky_fit(Y, X, *args, **kwargs):
        column_names = list(getattr(X, "columns", []))
        if "x2" in column_names and len(column_names) > 2:
            raise ValueError("synthetic failure")
        return original_fit(Y, X, *args, **kwargs)

    monkeypatch.setattr(real_module, "_fit_matrices", flaky_fit)

    rng = np.random.default_rng(53)
    x1 = rng.normal(size=50)
    df = pd.DataFrame(
        {
            "y": 5 + 3 * x1 + rng.normal(scale=0.2, size=50),
            "x1": x1,
            "x2": rng.normal(size=50),
            "x3": rng.normal(size=50),
        }
    )
    model = stepwise(
        "y ~ x1 + x2 + x3",
        data=df,
        direction="forward",
        metric="aic",
        max_steps=2,
        verbose=False,
    )
    assert "x1" in model.step.final_variables
    assert any(c["variable"] == "x2" for c in model.step.failed_candidates)


# ===========================================================================
# Additional 1.0.3 regression coverage merged from the latest staged suite
# ===========================================================================


# ---------------------------------------------------------------------------
# 1.0.3 prediction intervals and response-scale prediction
# ---------------------------------------------------------------------------

def test_predict_intervals_share_formula_pipeline_with_point_prediction():
    x = np.arange(1, 13, dtype=float)
    group = np.array(["A", "B", "C"] * 4)
    df = pd.DataFrame(
        {
            "Y": 5 + 2 * np.log(x) + 3 * (group == "B") + 5 * (group == "C"),
            "X": x,
            "Group": group,
        }
    )
    model = ols("Y ~ log(X) + Group", data=df)
    new_data = pd.DataFrame({"X": [7.0], "Group": ["B"]})

    point = predict(model, new_data)
    ci = predict(model, new_data, interval="confidence")
    legacy_ci = intervals(model, new_data, interval="confidence")

    np.testing.assert_allclose(ci["Prediction"].to_numpy(), np.asarray(point))
    pd.testing.assert_frame_equal(ci, legacy_ci)

def test_model_predict_accepts_interval_arguments():
    df = pd.DataFrame(
        {"Y": [3.1, 5.2, 7.0, 8.9, 11.2, 12.8], "X": [1, 2, 3, 4, 5, 6]}
    )
    model = ols("Y ~ X", data=df)
    result = model.predict(pd.DataFrame({"X": [3.5]}), interval="prediction", level=0.95)

    assert list(result.columns) == ["Prediction", "Lower Bound", "Upper Bound"]
    assert result.loc[0, "Lower Bound"] < result.loc[0, "Prediction"] < result.loc[0, "Upper Bound"]

def test_predict_response_scale_back_transforms_log_response():
    x = np.arange(1.0, 8.0)
    df = pd.DataFrame({"X": x, "Y": np.exp(1.0 + 0.25 * x)})
    reg = ols("log(Y) ~ X", data=df)
    new_data = pd.DataFrame({"X": [3.5]})

    model_scale = predict(reg, new_data)
    response_scale = predict(reg, new_data, scale="response")

    np.testing.assert_allclose(response_scale, np.exp(np.asarray(model_scale)))

def test_predict_formula_less_model_accepts_named_design_matrix():
    import statsmodels.api as sm

    x = np.arange(1.0, 8.0)
    X = pd.DataFrame({"Intercept": 1.0, "x": x})
    y = 2.0 + 3.0 * x
    raw = sm.OLS(y, X).fit()

    result = predict(raw, pd.DataFrame({"x": [4.5]}))
    np.testing.assert_allclose(np.asarray(result), [15.5])



# ---------------------------------------------------------------------------
# 1.0.3 grouped categorical model selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("selector", ["bsr", "stepwise"])
def test_group_categorical_selects_all_dummy_levels_together(selector):
    rng = np.random.default_rng(34)
    group = np.tile(["A", "B", "C"], 30)
    df = pd.DataFrame(
        {
            "Y": 10 + 50 * (group == "B") + rng.normal(0, 0.3, 90),
            "Noise": rng.normal(size=90),
            "Group": group,
        }
    )

    if selector == "bsr":
        model = bsr(
            "Y ~ Noise + Group",
            data=df,
            max_var=2,
            metric="bic",
            group_categorical=True,
        )
        assert model.bsr.best_features == ["Group"]
        assert model.bsr.groups["Group"] == ["Group_B", "Group_C"]
    else:
        model = stepwise(
            "Y ~ Noise + Group",
            data=df,
            direction="forward",
            metric="bic",
            group_categorical=True,
        )
        assert model.step.final_variables == ["Group"]
        assert model.step.groups["Group"] == ["Group_B", "Group_C"]

    assert model.model.exog_names == ["Intercept", "Group_B", "Group_C"]
    result = predict(model, pd.DataFrame({"Noise": [0.0], "Group": ["B"]}))
    assert np.isfinite(result).all()

def test_bsr_grouped_categorical_counts_as_one_max_var_term():
    group = np.tile(["A", "B", "C"], 20)
    df = pd.DataFrame(
        {
            "Y": 10 + 20 * (group == "B") + 40 * (group == "C"),
            "X": np.linspace(0, 1, 60),
            "Group": group,
        }
    )
    model = bsr(
        "Y ~ X + Group",
        data=df,
        max_var=1,
        metric="bic",
        group_categorical=True,
    )

    assert all(len(features) == 1 for features in model.bsr.results["Features"])
    assert ("Group",) in set(model.bsr.results["Features"])



# ---------------------------------------------------------------------------
# 1.0.3 public API exports
# ---------------------------------------------------------------------------

def test_public_robust_and_compare_exports_are_callable():
    import ravix

    assert callable(ravix.robust)
    assert callable(ravix.compare)



# ---------------------------------------------------------------------------
# 1.0.3 plot_cor DataFrame dummy handling and wide-matrix regression coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", [1, 2, 3, 4])
def test_plot_cor_wide_matrix_sets_tick_positions_before_labels(style):
    """Wide numeric matrices should render without tick/label length mismatches."""
    rng = np.random.default_rng(123)
    df = pd.DataFrame(
        rng.normal(size=(25, 60)),
        columns=[f"V{i}" for i in range(60)],
    )

    fig, ax = plot_cor(df, style=style, show=False, dummy=False)

    assert len(ax.get_xticks()) == 60
    assert len(ax.get_yticks()) == 60
    assert len(ax.get_xticklabels()) == 60
    assert len(ax.get_yticklabels()) == 60
    plt.close(fig)

def test_plot_cor_direct_dataframe_dummy_default_keeps_all_levels_and_input_unchanged():
    """Direct DataFrame input should dummy-code categoricals without dropping a level."""
    df = pd.DataFrame(
        {
            "Score": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "Group": ["A", "B", "C", "A", "B", "C"],
        }
    )
    original = df.copy(deep=True)

    fig, ax = plot_cor(df, style=4, show=False)
    labels = [tick.get_text() for tick in ax.get_xticklabels()]

    assert labels == ["Score", "Group_A", "Group_B", "Group_C"]
    pd.testing.assert_frame_equal(df, original)
    plt.close(fig)

def test_plot_cor_dummy_false_excludes_categorical_columns():
    """dummy=False should preserve the historical numeric-only DataFrame behavior."""
    df = pd.DataFrame(
        {
            "X": [1.0, 2.0, 3.0, 4.0],
            "Y": [4.0, 3.0, 2.0, 1.0],
            "Group": ["A", "B", "A", "B"],
        }
    )

    fig, ax = plot_cor(df, style=4, show=False, dummy=False)
    labels = [tick.get_text() for tick in ax.get_xticklabels()]

    assert labels == ["X", "Y"]
    plt.close(fig)

def test_plot_cor_high_cardinality_dummy_note_is_cleared(monkeypatch):
    """High-cardinality dummy expansion should show and then clear only its own hint."""
    import importlib

    module = importlib.import_module("ravix.plots.plot_cor")
    sentinel = object()
    calls = {"shown": 0, "cleared": []}

    def fake_show():
        calls["shown"] += 1
        return sentinel

    def fake_clear(status):
        calls["cleared"].append(status)

    monkeypatch.setattr(module, "_show_temporary_dummy_note", fake_show)
    monkeypatch.setattr(module, "_clear_temporary_dummy_note", fake_clear)

    n = 31
    df = pd.DataFrame(
        {
            "Value": np.arange(n, dtype=float),
            "Customer": [f"C{i}" for i in range(n)],
        }
    )

    fig, _ = module.plot_cor(df, style=4, show=False)

    assert calls["shown"] == 1
    assert calls["cleared"] == [sentinel]
    plt.close(fig)



# ---------------------------------------------------------------------------
# 1.0.3 compare() LaTeX dependency/presentation regression coverage
# ---------------------------------------------------------------------------

def test_compare_latex_matches_presentation_rules_without_dataframe_to_latex(monkeypatch):
    """LaTeX output should be self-contained and place significance stars by p-values."""
    x = np.arange(1.0, 21.0)
    z = np.tile([0.0, 1.0], 10)
    noise = np.array([
        0.1, -0.3, 0.2, 0.4, -0.2, 0.3, -0.1, 0.5, -0.4, 0.2,
        0.0, -0.2, 0.4, -0.1, 0.3, -0.3, 0.2, 0.1, -0.2, 0.4,
    ])
    df = pd.DataFrame({"y": 5 + 2 * x + 4 * z + noise, "x": x, "z": z})
    reduced = ols("y ~ x", data=df)
    full = ols("y ~ x + z", data=df)

    def fail_to_latex(*args, **kwargs):
        raise AssertionError("compare() should not call DataFrame.to_latex()")

    monkeypatch.setattr(pd.DataFrame, "to_latex", fail_to_latex)

    latex = compare(reduced, full, format="latex")

    assert r"\begin{tabular}" in latex
    assert r"\toprule" in latex
    assert "Signif." not in latex
    assert "***" in latex
    assert "SSE" in latex

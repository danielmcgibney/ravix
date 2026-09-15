"""Ebook compatibility tests for Ravix.

This suite is based on the Ravix syntax and workflows used in
*Applied Linear Regression for Business Analytics with Python*.

Run from the Ravix project root with:

    python -m pytest -q tests/test_textbook_compatibility.py

or place this file in ``tests/`` and run:

    python -m pytest -q

The tests intentionally focus on the public Ravix interface used in the ebook:
data access, formula syntax, OLS modeling, summaries, prediction/intervals,
exploratory plots, diagnostics, transformations, and variable selection.

All ebook syntax covered by this suite is expected to work with Ravix 1.0.2.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.outliers_influence import OLSInfluence

import ravix
from ravix import (
    abline,
    barplot,
    box_cox,
    boxplot,
    bsr,
    get_data,
    hist,
    intervals,
    ncv,
    ols,
    plot,
    plot_bsr,
    plot_cor,
    predict,
    qq,
    shapiro,
    stepwise,
    vif,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_df() -> pd.DataFrame:
    """Stable positive-response data for OLS, diagnostics, and Box-Cox tests."""
    rng = np.random.default_rng(637)
    n = 60
    x1 = np.linspace(1.0, 10.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    x3 = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.6, n)
    y = 50.0 + 3.0 * x1 - 2.0 * x2 + 0.5 * x3 + noise
    return pd.DataFrame({"Y": y, "X1": x1, "X2": x2, "X3": x3})


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Small simple-regression data shaped like examples used in the ebook."""
    x = np.array([2.0, 2.0, 5.0, 1.0, 3.0, 0.5, 7.0])
    y = np.array([7.0, 5.0, 12.0, 4.0, 8.0, 1.0, 12.0])
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def quadratic_df() -> pd.DataFrame:
    x = np.linspace(1.0, 10.0, 40)
    y = 20.0 + 4.0 * x - 0.25 * x**2 + np.sin(x)
    return pd.DataFrame({"Y": y, "X": x})


@pytest.fixture
def interaction_df() -> pd.DataFrame:
    x = np.arange(1.0, 31.0)
    group = np.tile([0.0, 1.0], 15)
    y = 10.0 + 2.0 * x + 4.0 * group + 1.5 * x * group
    return pd.DataFrame({"Y": y, "X": x, "GroupA": group})


@pytest.fixture(autouse=True)
def close_figures():
    """Keep plotting tests isolated and memory-light."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Package/data interface used in the ebook
# ---------------------------------------------------------------------------


def test_ravix_version_is_exposed():
    assert isinstance(ravix.__version__, str)
    assert ravix.__version__


def test_get_data_lists_bundled_datasets(capsys):
    result = get_data()
    out = capsys.readouterr().out

    assert result is None
    assert "Available data files in Ravix" in out
    assert "betas.csv" in out
    assert "house_prices.csv" in out
    assert "insurance.csv" in out


def test_get_data_loads_ebook_dataset():
    df = get_data("betas.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "SPY" in df.columns


# ---------------------------------------------------------------------------
# Formula syntax documented in the ebook/appendix
# ---------------------------------------------------------------------------


def test_formula_plus(linear_df):
    model = ols("Y ~ X1 + X2", data=linear_df)
    assert list(model.params.index) == ["Intercept", "X1", "X2"]


def test_formula_dot(linear_df):
    model = ols("Y ~ .", data=linear_df)
    assert set(model.params.index) == {"Intercept", "X1", "X2", "X3"}


def test_formula_minus_excludes_predictor(linear_df):
    model = ols("Y ~ . - X3", data=linear_df)
    assert "X3" not in model.params.index
    assert {"Intercept", "X1", "X2"}.issubset(model.params.index)


def test_formula_interaction_colon(interaction_df):
    model = ols("Y ~ X + GroupA + X:GroupA", data=interaction_df)
    assert "X:GroupA" in model.params.index


def test_formula_star_expands_main_effects_and_interaction(interaction_df):
    model = ols("Y ~ X*GroupA", data=interaction_df)
    assert {"X", "GroupA", "X:GroupA"}.issubset(model.params.index)


def test_formula_exponent_transformation(quadratic_df):
    model = ols("Y ~ X + X**2", data=quadratic_df)
    assert "X^2" in model.params.index


def test_formula_remove_intercept(linear_df):
    model = ols("Y ~ X1 - 1", data=linear_df)
    assert "Intercept" not in model.params.index
    assert "X1" in model.params.index


# ---------------------------------------------------------------------------
# OLS modeling and summary calls used throughout the ebook
# ---------------------------------------------------------------------------


def test_ols_returns_fitted_model(simple_df):
    reg = ols("y ~ x", data=simple_df)
    assert hasattr(reg, "params")
    assert hasattr(reg, "resid")
    assert hasattr(reg, "fittedvalues")
    assert "x" in reg.params.index


def test_default_summary_matches_ravix_ols_output(simple_df, capsys):
    reg = ols("y ~ x", data=simple_df)
    reg.summary()
    out = capsys.readouterr().out

    assert "Summary of OLS Regression Analysis" in out
    assert "Coefficients:" in out
    assert "t-value" in out
    assert "R-squared" in out
    assert "F-statistic" in out


@pytest.mark.filterwarnings(
    "ignore:omni_normtest is not valid with less than 8 observations.*"
)
def test_statsmodels_summary_ebook_call(simple_df, capsys):
    """Regression guard for reg.summary(out='statsmodels')."""
    reg = ols("y ~ x", data=simple_df)
    reg.summary(out="statsmodels")
    out = capsys.readouterr().out

    assert "OLS Regression Results" in out
    assert "R-squared" in out


def test_confint_summary_default_95_percent(simple_df, capsys):
    reg = ols("y ~ x", data=simple_df)
    reg.summary(out="confint")
    out = capsys.readouterr().out

    assert "Confidence Intervals" in out
    assert "95% CI Lower" in out
    assert "95% CI Upper" in out


def test_confint_summary_level_argument(simple_df, capsys):
    reg = ols("y ~ x", data=simple_df)
    reg.summary(out="confint", level=0.95)
    out = capsys.readouterr().out
    assert "95% CI Lower" in out


def test_confint_summary_alpha_argument(simple_df, capsys):
    reg = ols("y ~ x", data=simple_df)
    reg.summary(out="confint", alpha=0.10)
    out = capsys.readouterr().out
    assert "90% CI Lower" in out


def test_summary_anova_ebook_call(linear_df, capsys):
    reg = ols("Y ~ X1 + X2 + X3", data=linear_df)
    reg.summary(out="anova")
    out = capsys.readouterr().out

    assert "Regression" in out
    assert "Residual" in out
    assert "Sum Sq" in out
    assert "F" in out


def test_summary_anova1_ebook_call(linear_df, capsys):
    reg = ols("Y ~ X1 + X2 + X3", data=linear_df)
    reg.summary(out="anova1")
    out = capsys.readouterr().out

    assert "X1" in out
    assert "X2" in out
    assert "X3" in out
    assert "Residual" in out


# ---------------------------------------------------------------------------
# Fitted-model attributes accessed directly in the ebook
# ---------------------------------------------------------------------------


def test_model_attributes_used_in_ebook(simple_df):
    reg = ols("y ~ x", data=simple_df)

    assert np.isfinite(reg.params["x"])
    assert 0.0 <= reg.rsquared <= 1.0
    assert np.isfinite(reg.rsquared_adj)
    assert np.isfinite(reg.aic)
    assert np.isfinite(reg.bic)
    assert len(reg.resid) == len(simple_df)
    assert len(reg.fittedvalues) == len(simple_df)


def test_ols_influence_workflow_used_in_diagnostics(linear_df):
    reg = ols("Y ~ X1 + X2 + X3", data=linear_df)
    influence = OLSInfluence(reg)

    assert len(influence.hat_matrix_diag) == len(linear_df)
    assert len(influence.resid_studentized_internal) == len(linear_df)
    assert len(influence.resid_studentized_external) == len(linear_df)
    assert len(influence.cooks_distance[0]) == len(linear_df)


# ---------------------------------------------------------------------------
# Prediction and interval syntax used in Chapter 6
# ---------------------------------------------------------------------------


def test_predict_new_dataframe(simple_df):
    reg = ols("y ~ x", data=simple_df)
    new_x = pd.DataFrame({"x": [2.5, 4.0]})

    preds = predict(reg, new_x)

    assert len(preds) == 2
    assert np.all(np.isfinite(preds))


def test_predict_without_new_data_returns_fitted_values(simple_df):
    reg = ols("y ~ x", data=simple_df)
    preds = predict(reg)
    np.testing.assert_allclose(np.asarray(preds), np.asarray(reg.fittedvalues))


def test_confidence_intervals_ebook_call(simple_df):
    reg = ols("y ~ x", data=simple_df)
    new_x = pd.DataFrame({"x": [2.5]})

    result = intervals(reg, new_x, interval="confidence")

    assert list(result.columns) == ["Prediction", "Lower Bound", "Upper Bound"]
    assert result.loc[0, "Lower Bound"] < result.loc[0, "Prediction"] < result.loc[0, "Upper Bound"]


def test_prediction_intervals_are_wider_than_confidence_intervals(simple_df):
    reg = ols("y ~ x", data=simple_df)
    new_x = pd.DataFrame({"x": [2.5]})

    ci = intervals(reg, new_x, interval="confidence", level=0.95)
    pi = intervals(reg, new_x, interval="prediction", alpha=0.05)

    ci_width = ci.loc[0, "Upper Bound"] - ci.loc[0, "Lower Bound"]
    pi_width = pi.loc[0, "Upper Bound"] - pi.loc[0, "Lower Bound"]
    assert pi_width > ci_width


# ---------------------------------------------------------------------------
# Exploratory and diagnostic plotting used in the ebook
# ---------------------------------------------------------------------------


def test_plot_basic_scatterplot(linear_df):
    result = plot("Y ~ X1", data=linear_df, show=False)
    assert isinstance(result, tuple)
    fig, ax = result
    assert fig is not None
    assert ax is not None


def test_plot_scatterplot_with_labels(linear_df):
    fig, ax = plot(
        "Y ~ X1",
        data=linear_df,
        xlab="Predictor",
        ylab="Response",
        title="Scatterplot",
        show=False,
    )
    assert ax.get_xlabel() == "Predictor"
    assert ax.get_ylabel() == "Response"
    assert ax.get_title() == "Scatterplot"


def test_plot_multiple_predictors_ebook_matrix_call(linear_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    # The ebook uses the same formula-style call for a scatterplot matrix.
    plot("Y ~ X1 + X2 + X3", data=linear_df, show=False)


def test_plot_dot_scatterplot_matrix_call(linear_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    plot("Y ~ .", data=linear_df, show=False)


def test_plot_regression_residuals(linear_df):
    reg = ols("Y ~ X1 + X2 + X3", data=linear_df)
    result = plot(reg, color="orange", show=False)
    assert isinstance(result, tuple)


def test_hist_one_variable(linear_df):
    result = hist(linear_df.Y, show=False)
    assert isinstance(result, tuple)


def test_hist_formula_multiple_variables(linear_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    hist("Y ~ X1 + X2", data=linear_df, norm=False, bins=5, show=False)


def test_hist_dot_multiple_variables(linear_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    hist("Y ~ .", data=linear_df, show=False)


def test_hist_regression_residuals(linear_df):
    reg = ols("Y ~ X1 + X2 + X3", data=linear_df)
    result = hist(reg, color="orange", show=False)
    assert isinstance(result, tuple)


@pytest.mark.filterwarnings(
    "ignore:vert: bool will be deprecated in a future version.*:PendingDeprecationWarning"
)
def test_boxplot_vector(linear_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    assert boxplot(linear_df.Y) is None


@pytest.mark.filterwarnings(
    "ignore:vert: bool will be deprecated in a future version.*:PendingDeprecationWarning"
)
def test_boxplot_dataframe(linear_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    assert boxplot(linear_df[["Y", "X1", "X2"]]) is None


@pytest.mark.filterwarnings(
    "ignore:vert: bool will be deprecated in a future version.*:PendingDeprecationWarning"
)
def test_boxplot_formula(linear_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    assert boxplot("Y ~ X1 + X2", data=linear_df) is None


def test_barplot_formula(linear_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    assert barplot("Y ~ X1 + X2", data=linear_df) is None


def test_barplot_aggregation_ebook_call(linear_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    assert barplot("Y ~ X1 + X2", data=linear_df, agg="mean") is None


def test_plot_cor_dataframe(linear_df):
    result = plot_cor(linear_df, show=False)
    assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# abline(), including the 1.0.1 axis-scale guard and 1.0.2 clean return value
# ---------------------------------------------------------------------------


def test_plot_then_abline_primary_layering_api(linear_df):
    """Regression guard for the documented plot(...); abline(model) workflow."""
    reg = ols("Y ~ X1", data=linear_df)

    result = plot("Y ~ X1", data=linear_df)
    assert result is None

    fig = plt.gcf()
    ax = plt.gca()
    fig_num = fig.number
    xlim_before = ax.get_xlim()
    ylim_before = ax.get_ylim()
    scatter_count = len(ax.collections)
    lines_before = len(ax.lines)

    result = abline(reg)

    assert result is None
    assert plt.gcf().number == fig_num
    assert len(ax.collections) == scatter_count
    assert len(ax.lines) == lines_before + 1
    assert ax.lines[-1].axes is ax
    assert ax.get_xlim() == xlim_before
    assert ax.get_ylim() == ylim_before


def test_plot_show_false_abline_explicit_axes(linear_df):
    """Manual figure management should remain compatible with abline as well."""
    reg = ols("Y ~ X1", data=linear_df)
    fig, ax = plot("Y ~ X1", data=linear_df, show=False)
    lines_before = len(ax.lines)

    result = abline(reg, ax=ax)

    assert result is None
    assert len(ax.lines) == lines_before + 1
    assert ax.lines[-1].axes is ax


def test_abline_explicit_intercept_and_slope_does_not_rescale(linear_df):
    fig, ax = plt.subplots()
    ax.scatter(linear_df.X1, linear_df.Y)
    xlim_before = ax.get_xlim()
    ylim_before = ax.get_ylim()
    lines_before = len(ax.lines)

    result = abline(a=10.0, b=4.0, ax=ax)

    assert result is None
    assert len(ax.lines) == lines_before + 1
    assert ax.lines[-1].axes is ax
    assert ax.get_xlim() == xlim_before
    assert ax.get_ylim() == ylim_before


def test_abline_fitted_linear_model_does_not_rescale(linear_df):
    reg = ols("Y ~ X1", data=linear_df)
    fig, ax = plt.subplots()
    ax.scatter(linear_df.X1, linear_df.Y)
    xlim_before = ax.get_xlim()
    ylim_before = ax.get_ylim()
    lines_before = len(ax.lines)

    result = abline(reg, ax=ax)

    assert result is None
    assert len(ax.lines) == lines_before + 1
    assert ax.lines[-1].axes is ax
    assert ax.get_xlim() == xlim_before
    assert ax.get_ylim() == ylim_before


def test_abline_quadratic_model_ebook_call(quadratic_df):
    quad = ols("Y ~ X + X**2", data=quadratic_df)
    fig, ax = plt.subplots()
    ax.scatter(quadratic_df.X, quadratic_df.Y)
    xlim_before = ax.get_xlim()
    ylim_before = ax.get_ylim()
    lines_before = len(ax.lines)

    result = abline(quad, ax=ax)

    assert result is None
    assert len(ax.lines) == lines_before + 1
    assert ax.lines[-1].axes is ax
    assert ax.get_xlim() == xlim_before
    assert ax.get_ylim() == ylim_before


# ---------------------------------------------------------------------------
# Diagnostics and transformation functions used in Chapter 8
# ---------------------------------------------------------------------------


def test_vif_formula(linear_df):
    result = vif("Y ~ X1 + X2 + X3", data=linear_df)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"X1", "X2", "X3"}
    assert "VIF" in result.index


def test_vif_plot_true(linear_df, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    result = vif("Y ~ X1 + X2 + X3", data=linear_df, plot=True)
    assert isinstance(result, pd.DataFrame)


def test_ncv_model_call(linear_df):
    reg = ols("Y ~ X1 + X2 + X3", data=linear_df)
    result = ncv(reg, return_dict=True)

    assert {"test_statistic", "p_value", "df", "alpha"}.issubset(result)
    assert 0.0 <= result["p_value"] <= 1.0


def test_ncv_printed_ebook_output(linear_df, capsys):
    reg = ols("Y ~ X1 + X2 + X3", data=linear_df)
    ncv(reg, alpha=0.05)
    out = capsys.readouterr().out
    assert "Nonconstant Variance Test" in out
    assert "p-value" in out


def test_qq_regression_model(linear_df):
    reg = ols("Y ~ X1 + X2 + X3", data=linear_df)
    result = qq(reg, show=False)
    assert isinstance(result, tuple)


def test_shapiro_model_call(linear_df):
    reg = ols("Y ~ X1 + X2 + X3", data=linear_df)
    result = shapiro(reg, return_dict=True)

    assert {"test_statistic", "p_value", "alpha"}.issubset(result)
    assert 0.0 <= result["p_value"] <= 1.0


def test_shapiro_printed_ebook_output(linear_df, capsys):
    reg = ols("Y ~ X1 + X2 + X3", data=linear_df)
    shapiro(reg)
    out = capsys.readouterr().out
    assert "Shapiro-Wilk Test for Normality" in out
    assert "p-value" in out


def test_box_cox_regression_model(linear_df, monkeypatch):
    reg = ols("Y ~ X1 + X2 + X3", data=linear_df)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    assert box_cox(reg) is None


# ---------------------------------------------------------------------------
# Variable selection used in Chapter 9
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["backward", "forward", "both"])
def test_stepwise_directions(linear_df, direction):
    model = stepwise(
        "Y ~ X1 + X2 + X3",
        data=linear_df,
        direction=direction,
        verbose=False,
    )

    assert hasattr(model, "step")
    assert model.step.direction == direction
    assert isinstance(model.step.final_variables, list)


def test_stepwise_verbose_ebook_call(linear_df, capsys):
    model = stepwise(
        "Y ~ X1 + X2 + X3",
        data=linear_df,
        direction="forward",
        verbose=True,
    )
    out = capsys.readouterr().out

    assert hasattr(model, "step")
    assert "Initial AIC" in out
    assert "Final AIC" in out


def test_bsr_default_aic(linear_df):
    model = bsr("Y ~ X1 + X2 + X3", data=linear_df, max_var=3)

    assert hasattr(model, "bsr")
    assert model.bsr.metric == "aic"
    assert not model.bsr.results.empty
    assert 1 in model.bsr.best_by_k


@pytest.mark.parametrize("metric", ["aic", "bic", "adjr2"])
def test_bsr_metrics_used_in_ebook(linear_df, metric):
    model = bsr(
        "Y ~ X1 + X2 + X3",
        data=linear_df,
        max_var=3,
        metric=metric,
    )
    assert model.bsr.metric == metric


@pytest.mark.parametrize("plot_type", ["line", "bar", "predictors"])
def test_plot_bsr_modes(linear_df, plot_type, monkeypatch):
    model = bsr("Y ~ X1 + X2 + X3", data=linear_df, max_var=3, metric="aic")
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    assert plot_bsr(model, type=plot_type, top_n=5) is None


# ---------------------------------------------------------------------------
# One compact end-to-end workflow matching the ebook's canonical sequence
# ---------------------------------------------------------------------------


def test_ebook_end_to_end_workflow(linear_df):
    # Explore
    fig1, ax1 = plot("Y ~ X1", data=linear_df, show=False)
    fig2, ax2 = hist(linear_df.Y, show=False)

    # Fit and inspect
    reg = ols("Y ~ X1 + X2 + X3", data=linear_df)
    summary_df = reg.summary(format="df")

    # Predict and diagnose
    new_x = pd.DataFrame({"X1": [5.0], "X2": [0.0], "X3": [0.0]})
    pred = predict(reg, new_x)
    ci = intervals(reg, new_x, interval="confidence")
    ncv_result = ncv(reg, return_dict=True)
    shapiro_result = shapiro(reg, return_dict=True)

    assert fig1 is not None and ax1 is not None
    assert fig2 is not None and ax2 is not None
    assert isinstance(summary_df, pd.DataFrame)
    assert len(pred) == 1
    assert len(ci) == 1
    assert 0.0 <= ncv_result["p_value"] <= 1.0
    assert 0.0 <= shapiro_result["p_value"] <= 1.0

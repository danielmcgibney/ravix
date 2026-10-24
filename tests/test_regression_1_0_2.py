"""Regression tests for features/fixes documented in Ravix 1.0.2.

Run cumulatively with every later release: python -m pytest tests/ -q
Dependencies: installed local/development ravix, pytest, numpy, pandas, matplotlib.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import ravix


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def data():
    x = np.arange(1.0, 19.0)
    return pd.DataFrame({
        "x": x,
        "z": 2 + (x % 5),
        "y": 4 + 2.5 * x + 0.1 * np.sin(x),
        "group": pd.Categorical(["A", "B", "C"] * 6),
    })


@pytest.mark.parametrize("response", ["y^-1", "y^(-1)", "y**-1"])
def test_signed_response_exponents_fit(response, data):
    model = ravix.ols(f"{response} ~ x", data=data)
    assert np.isfinite(np.asarray(model.params, dtype=float)).all()


@pytest.mark.parametrize("formula", ["y^-1 ~ x", "log(y) ~ x", "y ~ log(x)", "log(y) ~ log(x)"])
def test_abline_uses_plotted_transform_coordinates_without_shifting_limits(formula, data):
    model = ravix.ols(formula, data=data)
    fig, ax = ravix.plot(formula, data=data, show=False)
    try:
        before = (ax.get_xlim(), ax.get_ylim())
        previous = len(ax.lines)
        result = ravix.abline(model, ax=ax)
        assert result is None  # 1.0.2: no stray Line2D output in notebooks
        assert len(ax.lines) == previous + 1
        np.testing.assert_allclose(ax.get_xlim(), before[0])
        np.testing.assert_allclose(ax.get_ylim(), before[1])
        x_line = np.asarray(ax.lines[-1].get_xdata(), dtype=float)
        y_line = np.asarray(ax.lines[-1].get_ydata(), dtype=float)
        assert np.isfinite(y_line).all()
        assert np.isfinite(x_line).all()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("bad_y", [-1.0, -4.0])
def test_fractional_response_power_rejects_negative_values(bad_y, data):
    sample = data.copy()
    sample.loc[0, "y"] = bad_y
    with pytest.raises(ValueError):
        ravix.ols("y^0.5 ~ x", data=sample)


@pytest.mark.parametrize("formula", ["y^(-1 ~ x", "y^abc ~ x"])
def test_invalid_response_power_gives_clear_error(formula, data):
    with pytest.raises((ValueError, SyntaxError)):
        ravix.ols(formula, data=data)


def test_coefficient_only_text_suppresses_significance_legend(data, capsys):
    model = ravix.ols("y ~ x + z", data=data)
    ravix.summary(model, out="coef", format="text")
    text = capsys.readouterr().out
    assert "Estimate" in text
    assert "Signif. codes" not in text


def test_coefficient_only_latex_suppresses_significance_legend(data):
    model = ravix.ols("y ~ x + z", data=data)
    latex = ravix.summary(model, out="coef", format="latex")
    assert isinstance(latex, str) and "\\begin{tabular}" in latex
    assert "Signif. codes" not in latex


def test_categorical_series_barplot_displays_category_counts(monkeypatch):
    observed = {}
    def record():
        ax = plt.gca()
        observed["heights"] = [patch.get_height() for patch in ax.patches]
        observed["labels"] = [label.get_text() for label in ax.get_xticklabels()]
    monkeypatch.setattr(plt, "show", record)
    series = pd.Series(pd.Categorical(["B", "A", "B", "C", "B", "A"]), name="group")
    ravix.barplot(series)
    assert set(observed["labels"]) == {"A", "B", "C"}
    assert sorted(observed["heights"]) == [1, 2, 3]


def test_categorical_formula_barplot_renders(monkeypatch, data):
    recorded = []
    monkeypatch.setattr(plt, "show", lambda: recorded.append(len(plt.gca().patches)))
    ravix.barplot("y ~ group", data=data)
    assert recorded and recorded[-1] >= 3


def test_categorical_dummy_levels_in_correlation_plot(data):
    fig, ax = ravix.plot_cor(data[["y", "group"]], show=False)
    try:
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert {"group_A", "group_B", "group_C"}.issubset(set(labels))
    finally:
        plt.close(fig)


def test_single_row_prediction_with_categorical_fit_levels(data):
    model = ravix.ols("y ~ x + group", data=data)
    # Fit-time levels are retained even though new data has only one level.
    pred = ravix.predict(model, pd.DataFrame({"x": [5.5], "group": ["C"]}))
    assert np.asarray(pred).shape == (1,)
    assert np.isfinite(np.asarray(pred, dtype=float)).all()


def test_bsr_selected_numeric_model_predicts(data):
    model = ravix.bsr("y ~ x", data=data, max_var=1)
    pred = ravix.predict(model, pd.DataFrame({"x": [4.5]}))
    assert len(np.asarray(pred).ravel()) == 1
    assert np.isfinite(np.asarray(pred, dtype=float)).all()


def test_stepwise_selected_numeric_model_predicts(data):
    model = ravix.stepwise("y ~ x", data=data, direction="backward", metric="aic")
    pred = ravix.predict(model, pd.DataFrame({"x": [4.5]}))
    assert len(np.asarray(pred).ravel()) == 1
    assert np.isfinite(np.asarray(pred, dtype=float)).all()


def test_categorical_scatter_matrix_accepts_point_colors(monkeypatch, data):
    observed = []
    def record():
        fig = plt.gcf()
        observed.extend(len(ax.collections) for ax in fig.axes)
    monkeypatch.setattr(plt, "show", record)
    ravix.plot(data[["x", "z", "y"]], color=["red", "blue"] * 9)
    assert observed and max(observed) > 0

"""Regression tests for fixes documented in Ravix 1.0.3.

These complement, rather than replace, any existing 1.0.3 regression tests.
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
    x = np.linspace(2, 15, 36)
    return pd.DataFrame({
        "x": x,
        "z": 2 + np.cos(x),
        "y": 15 + 0.8 * x + 0.4 * np.sin(x),
        "group": ["A", "B", "C"] * 12,
    })


@pytest.mark.parametrize("formula", [
    "y^-1 ~ x", "log(y) ~ x", "sqrt(y) ~ x", "log(y) ~ log(x)",
])
def test_transformed_response_predict_only_requires_predictors(data, formula):
    model = ravix.ols(formula, data=data)
    new = pd.DataFrame({"x": [3.0, 6.0]})
    pred = np.asarray(ravix.predict(model, new), dtype=float)
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()


def test_transformed_response_dot_formula_predict_without_y(data):
    fit_data = data[["y", "x", "z"]]
    model = ravix.ols("log(y) ~ .", data=fit_data)
    new = pd.DataFrame({"x": [4.0], "z": [1.2]})
    result = ravix.predict(model, new)
    assert np.asarray(result).size == 1
    assert np.isfinite(np.asarray(result, dtype=float)).all()


def test_log_response_backtransform_matches_exp_model_predictions(data):
    model = ravix.ols("log(y) ~ x", data=data)
    new = pd.DataFrame({"x": [3.0, 8.0]})
    on_model_scale = np.asarray(ravix.predict(model, new, scale="model"), dtype=float)
    original_scale = np.asarray(ravix.predict(model, new, scale="response"), dtype=float)
    np.testing.assert_allclose(original_scale, np.exp(on_model_scale), rtol=1e-12)


def test_transformed_response_prediction_intervals_return_expected_columns(data):
    model = ravix.ols("log(y) ~ x", data=data)
    result = ravix.predict(model, pd.DataFrame({"x": [5.0, 6.0]}), interval="confidence")
    assert list(result.columns) == ["Prediction", "Lower Bound", "Upper Bound"]
    assert (result["Lower Bound"] <= result["Prediction"]).all()
    assert (result["Prediction"] <= result["Upper Bound"]).all()


@pytest.mark.parametrize("which", ["A", "B", "C"])
def test_predict_single_category_even_when_other_levels_absent(data, which):
    model = ravix.ols("y ~ x + group", data=data)
    pred = ravix.predict(model, pd.DataFrame({"x": [4.0], "group": [which]}))
    assert np.asarray(pred).size == 1
    assert np.isfinite(np.asarray(pred, dtype=float)).all()


def test_bsr_grouped_multilevel_predicts(data):
    # group_categorical=True selects all dummy levels as one term.
    model = ravix.bsr("y ~ group", data=data, max_var=1, group_categorical=True)
    assert hasattr(model, "formula")
    pred = ravix.predict(model, pd.DataFrame({"group": ["C"]}))
    assert np.asarray(pred).size == 1


def test_bsr_partial_multilevel_categoricals_fail_with_actionable_error(data):
    # Ungrouped selection of exactly one dummy cannot be recreated with formula.
    model = ravix.bsr("y ~ group", data=data, max_var=1, group_categorical=False)
    with pytest.raises(ValueError, match="categorical|levels|subset|partial"):
        ravix.predict(model, pd.DataFrame({"group": ["A"]}))


def test_boxplot_with_pandas_extension_string_categories(monkeypatch, data):
    frame = data.copy()
    frame["group"] = frame["group"].astype(pd.StringDtype())
    seen = []
    monkeypatch.setattr(plt, "show", lambda: seen.append(len(plt.gca().patches)))
    ravix.boxplot("y ~ group", data=frame)
    assert seen  # Pandas 3 string dtype must reach the grouped-boxplot path.


def test_boxplot_with_pandas_categorical_dtype(monkeypatch, data):
    frame = data.copy()
    frame["group"] = pd.Categorical(frame["group"])
    seen = []
    monkeypatch.setattr(plt, "show", lambda: seen.append(plt.gca().get_xticklabels()))
    ravix.boxplot("y ~ group", data=frame)
    assert seen


def test_boxplot_is_callable_after_many_repeated_calls(monkeypatch, data):
    monkeypatch.setattr(plt, "show", lambda: None)
    for _ in range(3):
        ravix.boxplot("y ~ x + z", data=data)
    assert plt.get_fignums() == []

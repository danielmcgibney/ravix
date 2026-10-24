"""Regression tests for Ravix 1.0.4.

Run against the assembled 1.0.4 source (most tests fail against PyPI 1.0.3,
by design). Run cumulatively with the 1.0.2 and 1.0.3 files:
    python -m pytest tests/ -q
LaTeX compilation tests require pdflatex and skip if it is unavailable.
"""

import re
import shutil
import subprocess

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
def continuous():
    x = np.linspace(0.5, 10.0, 48)
    return pd.DataFrame({
        "x": x,
        "z": 2 * x + np.sin(x),
        "y": 4 + 1.8 * x + 0.3 * np.cos(1.3 * x),
    })


@pytest.fixture
def binary_groups():
    return pd.DataFrame({
        "salary": [40.0, 60.0, 100.0, 120.0, 50.0, 90.0],
        "churn": [0, 0, 1, 1, 0, 1],
        "dept": ["A", "A", "A", "B", "B", "B"],
    })


def _capture_bars(monkeypatch):
    """Snapshot pyplot-based barplot just before it closes its figure."""
    snapshot = {}
    def record():
        ax = plt.gca()
        snapshot["labels"] = [t.get_text() for t in ax.get_xticklabels()]
        snapshot["heights"] = [p.get_height() for p in ax.patches]
    monkeypatch.setattr(plt, "show", record)
    return snapshot


# ---------------------------------------------------------------------------
# barplot()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["integer", "boolean", "filtered"])
def test_barplot_groups_binary_predictor(monkeypatch, binary_groups, kind):
    snapshot = _capture_bars(monkeypatch)
    frame = binary_groups.copy()
    if kind == "boolean":
        frame["churn"] = frame["churn"].astype(bool)
    elif kind == "filtered":
        frame = frame.loc[frame.churn == 1].copy()
    ravix.barplot("salary ~ churn", data=frame)
    expected = [50.0, 310.0 / 3] if kind != "filtered" else [310.0 / 3]
    assert len(snapshot["heights"]) == len(expected)
    np.testing.assert_allclose(sorted(snapshot["heights"]), sorted(expected), rtol=1e-8)


def test_barplot_binary_response_remains_numeric(monkeypatch, binary_groups):
    snapshot = _capture_bars(monkeypatch)
    ravix.barplot("churn ~ dept", data=binary_groups)
    assert len(snapshot["heights"]) == 2
    np.testing.assert_allclose(sorted(snapshot["heights"]), [1 / 3, 2 / 3])


def test_barplot_continuous_predictor_warns_without_reclassifying(monkeypatch, continuous):
    snapshot = _capture_bars(monkeypatch)
    with pytest.warns(UserWarning, match="pd.cut|category"):
        ravix.barplot("y ~ x", data=continuous)
    assert len(snapshot["heights"]) == 2  # two variable summaries, not 48 groups


def test_barplot_closes_figures_when_validation_fails(binary_groups):
    before = set(plt.get_fignums())
    with pytest.raises(ValueError):
        ravix.barplot("salary ~ dept", data=binary_groups, agg=None)
    assert set(plt.get_fignums()) == before


# ---------------------------------------------------------------------------
# plot_cor()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", [1, 2, 3, 4])
def test_plot_cor_without_colorbar(style, continuous):
    fig, ax = ravix.plot_cor(continuous[["x", "y"]], style=style,
                             cbar=False, show=False)
    try:
        assert ax.collections and ax.collections[0].colorbar is None
    finally:
        plt.close(fig)


@pytest.mark.parametrize("style", [1, 2, 3, 4])
def test_plot_cor_respects_custom_colorbar_options(style, continuous):
    fig, ax = ravix.plot_cor(continuous[["x", "y"]], style=style,
                             cbar_kws={"label": "Custom correlation", "shrink": 0.7},
                             show=False)
    try:
        bar = ax.collections[0].colorbar
        assert bar is not None
        assert bar.ax.get_ylabel() == "Custom correlation"
        assert bar.formatter(0.5) == "0.50"  # default tick format kept (merged, not replaced)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("style", [1, 2, 3, 4])
def test_plot_cor_respects_gridline_options(style, continuous):
    fig, ax = ravix.plot_cor(continuous[["x", "y"]], style=style,
                             linewidths=3.0, linecolor="magenta", show=False)
    try:
        mesh = ax.collections[0]
        assert np.allclose(mesh.get_linewidths(), 3.0)
        rgba = mesh.get_edgecolors()[0]
        np.testing.assert_allclose(rgba, [1.0, 0.0, 1.0, 1.0])
    finally:
        plt.close(fig)


def test_plot_cor_clips_correlation_rounding_errors(monkeypatch):
    original = pd.DataFrame.corr
    def overshot(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result.iloc[0, 1] = 1 + 1e-12
        result.iloc[1, 0] = 1 + 1e-12
        return result
    monkeypatch.setattr(pd.DataFrame, "corr", overshot)
    frame = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    fig, ax = ravix.plot_cor(frame, style=1, show=False)
    try:
        assert ax.collections  # no RGBA alpha error for slightly >1 correlations
    finally:
        plt.close(fig)


def test_plot_cor_retains_boolean_dummy_levels():
    frame = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0], "flag": [False, True, False, True]})
    fig, ax = ravix.plot_cor(frame, show=False)
    try:
        ticks = [t.get_text() for t in ax.get_xticklabels()]
        assert {"flag_False", "flag_True"}.issubset(set(ticks))
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot(): categorical positions and the logistic abline() overlay
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["string", "bool", "categorical"])
def test_plot_uses_consistent_binary_positions(kind):
    values = ["Retained", "Churned", "Retained", "Churned"]
    if kind == "bool":
        values = [True, False, True, False]
    elif kind == "categorical":
        values = pd.Categorical(values, categories=["Retained", "Churned"], ordered=True)
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "event": values})
    fig, ax = ravix.plot(frame[["x", "event"]], show=False)
    try:
        labels = [t.get_text() for t in ax.get_yticklabels()]
        if kind == "categorical":
            assert labels == ["Retained", "Churned"]
        elif kind == "bool":
            assert labels == ["False", "True"]
        else:
            assert labels == ["Churned", "Retained"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind", ["string", "boolean", "ordered-categorical"])
def test_scatter_points_for_modeled_level_are_at_y_one(kind):
    # Modeled level deliberately appears FIRST, which triggered the original
    # Matplotlib first-appearance ordering error in older Ravix releases.
    names = ["Retained", "Churned"] * 16
    if kind == "boolean":
        event = [name == "Retained" for name in names]
    elif kind == "ordered-categorical":
        event = pd.Categorical(names, categories=["Churned", "Retained"], ordered=True)
    else:
        event = names
    frame = pd.DataFrame({"x": np.linspace(-3, 3, len(names)), "event": event})
    model = ravix.logistic("event ~ x", data=frame)
    modeled_level = (model._response_event[1]
                     if getattr(model, "_response_event", None) is not None else True)
    fig, ax = ravix.plot(frame[["x", "event"]], show=False)
    try:
        offsets = np.vstack([np.asarray(c.get_offsets(), dtype=float) for c in ax.collections
                             if len(c.get_offsets())])
        assert offsets.shape[0] == len(frame), "Plot must draw all observations"
        actual = {round(float(x), 12): float(y) for x, y in offsets}
        expected = {round(float(x), 12): float(value == modeled_level)
                    for x, value in zip(frame["x"], frame["event"])}
        assert actual == expected, "Modeled event must plot at y=1, other class at y=0"
        n_lines = len(ax.lines)
        ravix.abline(model, ax=ax)
        assert len(ax.lines) == n_lines + 1
        curve = np.asarray(ax.lines[-1].get_ydata(), dtype=float)
        assert np.isfinite(curve).all()
        assert np.all((0 <= curve) & (curve <= 1))
    finally:
        plt.close(fig)


def test_logistic_curve_rises_toward_plotted_modeled_level():
    # Outcome depends on x, so the curve has a direction to check: it must
    # approach 1 where the points at y = 1 (the modeled level) are.
    rng = np.random.default_rng(7)
    x = np.linspace(-3, 3, 40)
    state = np.where(x + rng.normal(0, 1.0, len(x)) > 0, "Retained", "Churned")
    state[0] = "Retained"  # first row is the modeled level, as in the original notebook
    frame = pd.DataFrame({"x": x, "state": state})
    model = ravix.logistic("state ~ x", data=frame)
    fig, ax = ravix.plot("state ~ x", data=frame, show=False)
    try:
        ravix.abline(model, ax=ax)
        y_positions = np.asarray(ax.collections[0].get_offsets()[:, 1], dtype=float)
        np.testing.assert_array_equal(y_positions[state == "Retained"], 1.0)
        curve = ax.lines[-1]
        cx = np.asarray(curve.get_xdata(), dtype=float)
        cy = np.asarray(curve.get_ydata(), dtype=float)
        assert cy[np.argmax(cx)] > 0.5 > cy[np.argmin(cx)]
    finally:
        plt.close(fig)


def test_plot_matrix_categorical_diagonal_counts_and_no_categorical_lines(monkeypatch):
    snapshots = []
    def record():
        fig = plt.gcf()
        axes = fig.axes
        snapshots.append({
            "diagonal_bars": len(axes[0].patches),
            "off_diagonal_lines": len(axes[1].lines),
        })
    monkeypatch.setattr(plt, "show", record)
    frame = pd.DataFrame({
        "event": ["Retained", "Churned", "Churned", "Retained", "Retained", "Churned"],
        "x": [1.0, 2.1, 3.2, 4.3, 5.4, 6.5],
        "z": [2.0, 1.0, 3.0, 1.5, 4.0, 2.5],
    })
    ravix.plot(frame, diag="density", lines=True)
    assert snapshots and snapshots[0]["diagonal_bars"] == 2
    assert snapshots[0]["off_diagonal_lines"] == 0


# ---------------------------------------------------------------------------
# logistic(): response validation and the modeled level
# ---------------------------------------------------------------------------

def _logistic_test_data(y):
    y = list(y)
    return pd.DataFrame({"x": np.linspace(-3, 3, len(y)), "y": y})


def test_logistic_rejects_missing_string_response():
    frame = pd.DataFrame({
        "x": np.arange(12.0),
        "state": (["Churned", "Retained"] * 6),
    })
    frame.loc[5, "state"] = None
    with pytest.raises(ValueError, match="missing|NaN"):
        ravix.logistic("state ~ x", data=frame)


@pytest.mark.parametrize("values, match", [
    ([1, 2] * 16, r"coded 0/1.*\{1, 2\}.*== 2"),
    ([0, 2] * 16, r"coded 0/1.*\{0, 2\}"),
    ([-1, 1] * 16, r"coded 0/1.*\{-1, 1\}"),
    ([0.0, .25, .75, 1.0] * 8, "Proportions are not supported"),
    ([.2, .8] * 16, "Proportions are not supported"),
    ([0] * 32, "both 0 and 1"),
    ([1] * 32, "both 0 and 1"),
    ([True] * 32, "both 0 and 1"),
], ids=["1-2", "0-2", "minus1-1", "proportions-mixed", "proportions-only",
        "all-zero", "all-one", "all-true"])
def test_logistic_rejects_non_binary_or_degenerate_numeric_outcomes(values, match):
    """Only 0/1 outcomes with both classes observed are valid for fitting."""
    with pytest.raises(ValueError, match=match):
        ravix.logistic("y ~ x", data=_logistic_test_data(values))


@pytest.mark.parametrize("values, match", [
    ([1, 2] * 16, r"coded 0/1.*\{1, 2\}"),
    ([0, 2] * 16, r"coded 0/1.*\{0, 2\}"),
    ([-1, 1] * 16, r"coded 0/1.*\{-1, 1\}"),
    ([0.1, 0.9] * 16, "Proportions are not supported"),
    ([0] * 32, "both 0 and 1"),
    ([1] * 32, "both 0 and 1"),
    ([0.0, 1.0] * 4 + [np.nan] + [0.0, 1.0] * 4, "missing values"),
    ([0.0, 1.0] * 4 + [np.inf] + [0.0, 1.0] * 4, "infinite values"),
], ids=["1-2", "0-2", "minus1-1", "proportions", "all-zero", "all-one", "nan", "inf"])
def test_fit_matrices_rejects_invalid_logistic_outcomes_directly(values, match):
    # The matrix path used by stepwise()/bsr() applies the same checks. The
    # messages are matched so that statsmodels' own failures (e.g. its NaN
    # "first guess on the deviance function" error) cannot pass for them.
    from ravix.modeling.fit import _fit_matrices
    design = pd.DataFrame({"Intercept": np.ones(len(values)), "x": np.linspace(-2, 2, len(values))})
    with pytest.raises(ValueError, match=match):
        _fit_matrices(pd.Series(values, name="y"), design, method="logistic")


@pytest.mark.parametrize("routine", ["stepwise", "bsr"])
def test_selection_routines_reject_miscoded_logistic_response(routine):
    rng = np.random.default_rng(1040)
    x = np.linspace(-3, 3, 60)
    frame = pd.DataFrame({"x": x, "z": np.cos(x),
                          "y": rng.binomial(1, 1 / (1 + np.exp(-x))) + 1})
    # bsr() reports per-subset failures inside a RuntimeError; the 0/1 message
    # must still reach the user.
    with pytest.raises((ValueError, RuntimeError), match="coded 0/1"):
        getattr(ravix, routine)("y ~ x + z", data=frame, method="logistic")


@pytest.mark.parametrize("kind", ["float", "boolean", "text", "categorical"])
def test_logistic_valid_binary_codings_fit_identically(kind):
    rng = np.random.default_rng(1041)
    x = np.linspace(-3, 3, 60)
    y = rng.binomial(1, 1 / (1 + np.exp(-x)))
    reference = ravix.logistic("y ~ x", data=pd.DataFrame({"x": x, "y": y})).params
    coded = {
        "float": y.astype(float),
        "boolean": y.astype(bool),
        "text": np.where(y == 1, "Yes", "No"),
        "categorical": pd.Categorical(np.where(y == 1, "Yes", "No")),
    }[kind]
    frame = pd.DataFrame({"x": x, "y": coded})
    model = ravix.logistic("y ~ x", data=frame)
    np.testing.assert_allclose(model.params.values, reference.values, rtol=1e-10)
    assert np.isfinite(np.asarray(ravix.predict(model, frame[["x"]]), dtype=float)).all()


def test_logistic_categorical_ignores_unused_level_and_fits_two_observed():
    rng = np.random.default_rng(1042)
    x = np.linspace(-3, 3, 48)
    text = np.where(rng.binomial(1, 1 / (1 + np.exp(-x))) == 1, "Yes", "No")
    categorical = pd.Categorical(text, categories=["Unused", "No", "Yes"], ordered=True)
    model = ravix.logistic("y ~ x", data=pd.DataFrame({"x": x, "y": categorical}))
    assert model._response_event == ("y", "Yes")
    reference = ravix.logistic("y ~ x", data=pd.DataFrame({"x": x, "y": text}))
    np.testing.assert_allclose(model.params.values, reference.params.values, rtol=1e-10)


def test_logistic_three_observed_categories_names_each_in_error():
    labels = ["Leave", "Remain", "Unknown"] * 12
    frame = pd.DataFrame({"x": np.linspace(-2, 2, len(labels)), "y": labels})
    with pytest.raises(ValueError, match=r"exactly two categories.*'Leave', 'Remain', 'Unknown'"):
        ravix.logistic("y ~ x", data=frame)


def test_logistic_single_level_text_response_error():
    frame = pd.DataFrame({"x": np.linspace(-2, 2, 24), "y": "Yes"})
    with pytest.raises(ValueError, match=r"found only 1: \['Yes'\]"):
        ravix.logistic("y ~ x", data=frame)


def test_logistic_summary_explains_modeled_category(capsys):
    x = np.linspace(-4, 4, 28)
    # Interleave outcomes to avoid perfect separation; start with Retained.
    labels = ["Retained", "Churned"] * 14
    frame = pd.DataFrame({"x": x, "state": labels})
    model = ravix.logistic("state ~ x", data=frame)
    ravix.summary(model, format="text")
    output = capsys.readouterr().out
    assert "Modeling: P(state = 'Retained')" in output
    table = ravix.summary(model, format="df")
    assert table.attrs["modeled_event"] == "P(state = 'Retained')"


# ---------------------------------------------------------------------------
# predict(): Boolean predictors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["ols", "logistic", "poisson"])
def test_prediction_with_boolean_and_numeric_predictors(method):
    x = np.linspace(-2, 2, 56)
    b = np.arange(56) % 3 == 0
    rng = np.random.default_rng(104)
    if method == "ols":
        y = 2 + x + 0.5 * b + rng.normal(scale=0.2, size=len(x))
    elif method == "logistic":
        probability = 1 / (1 + np.exp(-(0.8 * x + 0.4 * b)))
        y = rng.binomial(1, probability)
    else:
        y = rng.poisson(np.exp(0.2 + 0.3 * x + 0.2 * b))
    frame = pd.DataFrame({"x": x, "flag": b, "y": y})
    model = getattr(ravix, method)("y ~ x + flag", data=frame)
    new = pd.DataFrame({"x": [0.1, 0.2], "flag": [False, True]})
    prediction = np.asarray(ravix.predict(model, new))
    assert prediction.dtype.kind == "f"
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()


def test_predict_raw_matrix_mixed_boolean_and_numeric():
    # No formula: the encoded newdata path used by some selection models.
    from ravix.modeling.fit import _fit_matrices
    y = pd.Series([1.0, 3.0, 3.1, 5.2, 5.0, 7.0])
    design = pd.DataFrame({"Intercept": 1.0, "x": [0., 1., 1., 2., 2., 3.],
                           "flag_True": [False, False, True, False, True, True]})
    fitted = _fit_matrices(y, design.astype(float), method="ols")
    new = pd.DataFrame({"x": [1.5], "flag_True": [True]})
    pred = np.asarray(ravix.predict(fitted, new))
    assert pred.shape == (1,)
    assert pred.dtype.kind == "f"


# ---------------------------------------------------------------------------
# summary(): full names, own-line fallback, width setting
# ---------------------------------------------------------------------------

@pytest.fixture
def summary_width_model():
    x = np.linspace(-3, 3, 96)
    medium = "advertising_channel_length"  # 26 characters: fits a row at 80, not at 65
    very_long = "very_long_predictor_name_" + "extended_" * 9  # longer than 80 characters
    frame = pd.DataFrame({
        "x": x,
        medium: np.sin(x * 1.31),
        very_long: np.cos(x * 0.87),
    })
    frame["y"] = (4 + 1.2 * frame["x"] + 0.5 * frame[medium]
                  - 0.3 * frame[very_long] + 0.02 * np.sin(7 * x))
    model = ravix.ols(f"y ~ x + {medium} + {very_long}", data=frame)
    return model, medium, very_long


def _coefficient_lines(model, capsys):
    ravix.summary(model, out="coef", format="text")
    output = capsys.readouterr().out
    assert "Estimate" in output
    # The table is the lines from the column header to the next separator.
    lines = output.splitlines()
    header = next(i for i, line in enumerate(lines) if "Estimate" in line and "Std. Error" in line)
    body = []
    for line in lines[header:]:
        if line and set(line.strip()) in ({"="}, {"-"}):
            break
        body.append(line)
    return body


def test_summary_default_width_80_without_truncating(summary_width_model, capsys):
    # No theme change: the built-in theme defines no summary_width, so 80 applies.
    model, medium, very_long = summary_width_model
    lines = _coefficient_lines(model, capsys)
    assert not any("..." in line for line in lines)
    assert next(line for line in lines if line.lstrip().startswith(medium)).strip() != medium, (
        "A medium-length name that fits in 80 columns must remain on its coefficient row"
    )
    assert very_long in lines, "An overlong name must appear untruncated on its own line"
    assert re.match(r"^\s+[-+0-9.]", lines[lines.index(very_long) + 1]), (
        "The numeric coefficient row must follow the overlong name"
    )
    assert all(len(line) <= 80 for line in lines if line != very_long), (
        "Ordinary coefficient-table rows, including the header, must fit 80 characters"
    )


def test_summary_short_coefficient_preserves_one_line_layout(summary_width_model, capsys):
    model, _, _ = summary_width_model
    lines = _coefficient_lines(model, capsys)
    rows = [line for line in lines if re.match(r"^x\s+[-+0-9.]", line)]
    assert len(rows) == 1
    assert "x" not in [line.strip() for line in lines]  # no split-off name


def test_render_coef_table_short_names_match_1_0_3_exactly():
    # Golden text generated with ravix 1.0.3's render_coef_table. Names of up
    # to 16 characters (1.0.3's cap; 'Region_Southwest' is exactly 16) must
    # render byte-for-byte as before, so existing textbook output is unchanged.
    from ravix.modeling.summary import render_coef_table
    coefficients = pd.DataFrame({
        "coef": [2.0, 1.2345, -0.25],
        "std err": [.35, .07654, .05],
        "t": [5.714, 16.128, -5.0],
        "P>|t|": [.000004, .0023, 1e-20],
        " ": ["***", "**", "***"],
    }, index=["(Intercept)", "marketing_spend", "Region_Southwest"])
    expected = (
        "                  Estimate  Std. Error  t-value    p-value\n"
        "(Intercept)         2.0000     0.35000    5.714   4.00e-06 ***\n"
        "marketing_spend     1.2345    0.076540   16.128     0.0023 **\n"
        "Region_Southwest  -0.25000    0.050000   -5.000    < 2e-16 ***"
    )
    assert render_coef_table(coefficients) == expected


def test_summary_theme_width_65_is_honored(summary_width_model, capsys):
    # The documented route: a registered theme with summary_width
    from ravix.plots._theme import register_theme, get_theme, theme_context
    register_theme("test_summary_width_65", {**get_theme(), "summary_width": 65})
    model, medium, very_long = summary_width_model
    with theme_context("test_summary_width_65"):
        lines = _coefficient_lines(model, capsys)
    assert medium in lines, "A name fitting 80 but not 65 must move to its own line"
    assert very_long in lines
    assert all(len(line) <= 65 for line in lines if line not in {medium, very_long})


# ---------------------------------------------------------------------------
# summary(format="latex"): escaping, compilation, no Jinja2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("out", ["simple", "coef", "confint"])
@pytest.mark.parametrize("method", ["ols", "logistic", "poisson"])
def test_latex_escapes_special_characters(method, out):
    rng = np.random.default_rng(444)
    n = 56
    x = np.linspace(-2, 2, n)
    group = np.resize(["A", "B", "C", "A"], n)
    if method == "ols":
        y = 3.0 + 0.9 * x + (group == "B") + rng.normal(0, 0.2, n)
    elif method == "logistic":
        prob = 1 / (1 + np.exp(-x))
        y = rng.binomial(1, prob)
    else:
        y = rng.poisson(np.exp(0.3 + 0.2 * x))
    model = getattr(ravix, method)("y ~ x + group", data=pd.DataFrame({"y": y, "x": x, "group": group}))
    latex = ravix.summary(model, out=out, format="latex")
    assert "\\begin{tabular}" in latex
    assert "group\\_" in latex  # default dummy labels contain underscores
    if out == "confint":
        assert "\\%" in latex  # confidence-interval header must compile


@pytest.fixture
def special_character_models():
    """Realistic, nonseparated models with problematic LaTeX coefficient names."""
    rng = np.random.default_rng(20261024)
    n = 96
    x = np.linspace(-2.5, 2.5, n)
    groups = np.resize(["North|East", "South<West", "Base_A"], n)
    b = (groups == "North|East").astype(float)
    data = pd.DataFrame({"x": x, "group": groups})
    ols_data = data.assign(y=3 + x + .35 * b + rng.normal(0, .2, n))
    probs = 1 / (1 + np.exp(-(.4 * x + .3 * b)))
    logistic_data = data.assign(y=rng.binomial(1, probs))
    poisson_data = data.assign(y=rng.poisson(np.exp(.25 + .2 * x + .1 * b)))
    return {
        "ols": ravix.ols("y ~ x + group", data=ols_data),
        "logistic": ravix.logistic("y ~ x + group", data=logistic_data),
        "poisson": ravix.poisson("y ~ x + group", data=poisson_data),
    }


def _compile_latex(tmp_path, content):
    document = ("\\documentclass{article}\n\\usepackage{booktabs}\n"
                "\\begin{document}\n" + content + "\n\\end{document}\n")
    tex = tmp_path / "regression.tex"
    tex.write_text(document, encoding="utf-8")
    proc = subprocess.run(
        ["pdflatex", "-halt-on-error", "-interaction=nonstopmode",
         "-output-directory", str(tmp_path), str(tex)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stdout[-2500:] + "\n" + proc.stderr[-1000:]


@pytest.mark.parametrize("method", ["ols", "logistic", "poisson"])
@pytest.mark.parametrize("out", ["simple", "coef", "confint"])
def test_latex_escapes_angle_bracket_and_bar_in_coefficient_names(special_character_models, method, out):
    latex = ravix.summary(special_character_models[method], out=out, format="latex")
    assert r"\textless{}" in latex, "Must escape '<' to render correctly under OT1 encoding"
    assert r"\textbar{}" in latex, "Must escape '|' to render correctly under OT1 encoding"
    assert r"\_" in latex  # categorical dummy names contain underscores


@pytest.mark.parametrize("method", ["ols", "logistic", "poisson"])
def test_latex_escapes_p_value_header_and_threshold(method):
    # Headers ('Pr(>|t|)') and p-values ('< 2e-16') are escaped too, not just
    # names. Strong effects make every model report '< 2e-16'.
    x = np.linspace(-3, 3, 400)
    rng = np.random.default_rng(55)
    y = {"ols": 1 + 3 * x + rng.normal(0, 0.1, x.size),
         "logistic": rng.binomial(1, 1 / (1 + np.exp(-2 * x))),
         "poisson": rng.poisson(np.exp(0.5 + 0.8 * x))}[method]
    model = getattr(ravix, method)("y ~ x", data=pd.DataFrame({"x": x, "y": y}))
    latex = ravix.summary(model, format="latex")
    stat = "t" if method == "ols" else "z"
    assert f"Pr(\\textgreater{{}}\\textbar{{}}{stat}\\textbar{{}})" in latex
    assert "\\textless{} 2e-16" in latex
    assert "Pr(>|" not in latex and "< 2e-16" not in latex


@pytest.mark.parametrize("method", ["ols", "logistic", "poisson"])
@pytest.mark.parametrize("out", ["simple", "coef", "confint"])
def test_latex_does_not_depend_on_dataframe_to_latex(monkeypatch, special_character_models, method, out):
    def forbidden(*args, **kwargs):
        raise AssertionError("LaTeX summaries must not invoke DataFrame.to_latex (Jinja2 dependency)")
    monkeypatch.setattr(pd.DataFrame, "to_latex", forbidden)
    latex = ravix.summary(special_character_models[method], out=out, format="latex")
    assert "\\begin{tabular}" in latex


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
@pytest.mark.parametrize("method", ["ols", "logistic", "poisson"])
@pytest.mark.parametrize("out", ["simple", "coef", "confint"])
def test_latex_output_compiles_all_model_types(tmp_path, special_character_models, method, out):
    latex = ravix.summary(special_character_models[method], out=out, format="latex")
    _compile_latex(tmp_path, latex)


@pytest.fixture
def logistic_model_with_escaped_event():
    rng = np.random.default_rng(3214)
    n = 80
    x = np.linspace(-2, 2, n)
    grp = np.resize(["Base", "North|East", "South<West", "Base"], n)
    p = 1 / (1 + np.exp(-.5 * x))
    outcome = np.where(rng.binomial(1, p) == 1, "Buy_%", "Avoid")
    frame = pd.DataFrame({"state": outcome, "x": x, "group": grp})
    model = ravix.logistic("state ~ x + group", data=frame)
    assert getattr(model, "_response_event", (None, None))[1] == "Buy_%"
    return model


def test_latex_logistic_modeled_event_escapes_special_characters(logistic_model_with_escaped_event):
    latex = ravix.summary(logistic_model_with_escaped_event, out="simple", format="latex")
    assert "Modeling:" in latex
    assert r"Buy\_\%" in latex
    assert r"\textless{}" in latex and r"\textbar{}" in latex


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_latex_logistic_special_modeled_event_compiles(tmp_path, logistic_model_with_escaped_event):
    latex = ravix.summary(logistic_model_with_escaped_event, out="simple", format="latex")
    _compile_latex(tmp_path, latex)

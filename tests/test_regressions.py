"""Regression tests for Ravix bugs fixed after 1.0.1.

These tests protect package behavior that is not specific to the textbook API.
They cover issues reproduced in Lab 2, the Lead Generation case study, and
video_engagement.csv, plus the boxplot categorical-type deprecation.

Coefficient-name formatting is intentionally unchanged for textbook compatibility.
Dataset packaging checks are outside this suite's scope.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba_array
import numpy as np
import pandas as pd
import pytest

from ravix import abline, barplot, boxplot, ols, plot, plot_cor


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

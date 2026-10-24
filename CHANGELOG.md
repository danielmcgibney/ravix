# Changelog

## 1.0.4 - 2026-10-24

### Fixed
- Fixed `plot_cor()` failing with "RGBA values should be within 0-1 range" in style 1 under Pandas < 3.0, which can return perfectly correlated pairs (for example, the two dummies of a two-level categorical variable) as slightly beyond ±1. Correlations are now clipped to [-1, 1] before plotting.
- Fixed `plot_cor(cbar=False)` raising an `AttributeError` in all four styles. Custom `cbar_kws` settings (such as colorbar labels) and Seaborn `linewidths`/`linecolor` options are now respected across all four styles; provided colorbar settings are merged with the defaults rather than overwritten.
- Fixed `barplot("Salary ~ Churn", data=df)` when the grouping variable is Boolean or numeric with one or two distinct nonmissing values (for example, Churn coded 0/1): it is now treated as a grouping variable, including when filtering leaves one observed level. **This changes the output** of such calls, which previously plotted side-by-side means of both variables. Response variables are not reclassified, so `barplot("Churn ~ Dept", data=df)` still plots churn rates by department.
- Fixed `barplot()` leaving an empty figure open when validation or plotting raises an exception.
- Fixed categorical scatterplot axes in `plot()` so text, Boolean, and Pandas categorical values use sorted positions (category order for a Pandas categorical), matching the level `logistic()` models as 1, rather than first-appearance order. This prevents string-coded binary responses from appearing reversed relative to a logistic probability curve added by `abline()`, without changing fitted probabilities or coefficients.
- Fixed categorical ordering in `plot()` scatterplot matrices, including tick labels and diagonal frequency plots. Categorical diagonal panels use counts instead of a continuous density estimate, and regression lines are not fitted through categorical codes.
- Fixed `logistic()` handling of missing text/categorical responses by rejecting missing outcomes before dummy encoding, rather than silently treating them as the reference category.
- Fixed `logistic()` accepting numeric responses other than 0/1 (including `{0, 2}`, `{1, 2}`, and fractional responses), which could produce meaningless fits. Numeric and Boolean responses must contain both outcomes, 0 and 1, and must not contain missing or infinite values. Validation also applies to the matrix-based logistic fitting path used by `stepwise()` and `bsr()`.
- Fixed logistic regression with Pandas categorical responses that retain unused category levels after filtering: only observed levels now count toward the two-class requirement. Responses with only one observed category or more than two now produce informative errors that name the levels found.
- Fixed `predict()` with Boolean predictors (including dummies from `pd.get_dummies`) mixed with numeric predictors: prediction design matrices are now converted to `float64`, as at fit time, so logistic and Poisson predictions no longer fail and OLS predictions return floating-point rather than object-dtype results. Both formula-based and raw-matrix prediction paths use the conversion.
- Fixed LaTeX regression summaries failing to compile when coefficient names, categorical labels, or column headers contain special characters, including the `_` in dummy names and the `%` in confidence-interval headers. `<`, `>`, and `|` also now render correctly with LaTeX's default font encoding.

### Improved
- Improved `barplot()` warnings for a formula with a single numeric predictor that has three or more distinct values: the message explains that separate summaries of the response and predictor are plotted, not the response grouped by the predictor, and suggests explicit categorical conversion or `pd.cut()` where appropriate. Multiple numeric predictors continue to produce side-by-side summaries.
- Recorded the modeled category for logistic regressions with text/categorical responses and displayed it in the default summary (for example, `Modeling: P(Churn = 'Retained')`). The modeled event is also included in LaTeX output and DataFrame summary attributes. Existing encoding rules and numerical results are unchanged.
- Improved text coefficient tables in `summary()` to preserve full predictor and interaction names instead of truncating them. Long names appear on a separate line when necessary; table width defaults to 80 characters and can be changed with the theme's `summary_width` setting.
- Improved LaTeX table generation to avoid relying on Pandas `DataFrame.to_latex()` and its optional Jinja2 dependency, maintaining compatibility across Pandas versions.

### Tests and compatibility
- Added `test_regression_1_0_4.py` (104 tests) covering the 1.0.4 plotting, logistic-response, prediction, summary, and LaTeX fixes, including `pdflatex` compilation for OLS, logistic, and Poisson summaries. Together with the 1.0.2 and 1.0.3 regression files, all 139 regression tests pass under Pandas 2.2.3 and 3.0.5.
- Pending before release: a full-suite run including the 64 textbook compatibility cases.

## 1.0.3 - 2026-09-17

### Fixed
- Fixed `predict()` for models with transformed responses, including formulas such as `Salary^-1 ~ Jobs**2`, `log(Y) ~ log(X)`, and `sqrt(Y) ~ X`. Prediction now parses only the predictor side of the stored formula, so new data no longer need to contain the response variable.
- Fixed `predict()` with transformed responses and dot formulas so the supplied predictor columns are retained without introducing a dummy response column.
- Updated `boxplot()` for compatibility with current and future Matplotlib releases by translating Seaborn's deprecated `vert=` argument to `orientation=` when supported. The legacy path remains available for older Matplotlib versions.
- Fixed categorical formula boxplots under Pandas 3, where text columns use a dedicated string dtype rather than `object`, ensuring calls such as `boxplot("Salary ~ Education", data=df)` continue to produce one box per category.
- Fixed misleading `barplot(..., agg=None)` output when a bar would combine multiple observations: it now raises a clear error instead of letting Seaborn silently plot means. A single numeric column without aggregation still produces one bar per observation, with more informative axis labels. (Previously omitted from this entry.)

### Tests and compatibility
- Added automated coverage for transformed-response prediction, including negative-power, logarithmic, square-root, categorical, and dot-formula cases.
- Added an exact regression test for single-row categorical prediction using `job_changes.csv`.
- Added automated prediction tests for numeric and complete-categorical models returned by `bsr()` and `stepwise()`, plus clear rejection of models that select only part of a categorical variable's dummy columns.
- Added boxplot compatibility tests verifying that `vert=` does not reach supported Matplotlib versions, boxplot medians are preserved across all supported input styles, and axes state is restored after rendering errors.
- Validated the updated source with 117 passing tests: 64 textbook compatibility cases and 53 regression cases.

## 1.0.2 - 2026-09-14

### Fixed
- Updated `boxplot()` to use the current Pandas categorical-type API, eliminating deprecation warnings and improving compatibility with future Pandas releases.
- Updated `barplot()` to eliminate the Pandas/Seaborn `FutureWarning` produced by categorical formula plots such as `barplot("Salary ~ Education", data=df)`.
- Fixed `barplot()` so a categorical Series, such as `barplot(df.Education)`, produces a frequency bar plot without raising an error.
- Updated `abline()` to return `None` after adding a regression line, preventing Jupyter from displaying the underlying Matplotlib `Line2D` object.
- Fixed `abline()` overlays for single-predictor OLS models with transformed responses or predictors, including `log(Y1) ~ X1`, `Y2 ~ log(X2)`, `log(Y3) ~ log(X3)`, and `Y4^-1 ~ X4`. Lines use the plotted coordinates without applying transformations twice and preserve existing axis limits.
- Clarified `abline()`'s error message when the target axes lack Ravix plot metadata (i.e. the plot wasn't generated with `ravix.plot()`), explaining why prediction failed and how to recreate the plot. Transformed predictors on such external plots remain unsupported by the fallback.
- Added signed response exponents to formula parsing, including `Y^-1`, `Y^(-1)`, and `Y**-1`.
- Fixed plotting's intercept handling so a negative response exponent is not mistaken for an intercept-removal instruction.
- Added clear errors for malformed response-power expressions, including unmatched parentheses and invalid numeric exponents.
- Added validation rejecting negative response values with non-integer exponents, preventing silent `NaN` results. Integer powers of negative values remain supported, and negative powers continue to reject zero denominators.
- Removed the significance-codes legend from coefficient-only text and LaTeX summaries (`reg.summary(out="coef")`). Coefficient names and significance stars remain unchanged.
- Fixed `plot()` so the `color` argument is applied to observations throughout scatterplot matrices, including calls such as `plot("Likes ~ Age + Sentiment", data=df, color=colors)`.
- **Fixed `predict()` for categorical predictors when new data doesn't include every category level seen at fit time** -- most commonly, predicting for a single new observation. Previously this always raised `ValueError: The following required columns are missing: {...}` for any single-row prediction on a model with a categorical predictor of more than one non-baseline level, regardless of which category the new observation had (including the baseline category itself). `ols()`/`logistic()`/`poisson()` now capture each categorical predictor's fit-time categories, in the exact order used for dummy encoding, on the fitted model; `predict()` reuses them so every trained dummy column is reproduced -- as an all-zero column where appropriate -- no matter which categories are present in the new data.
- **Enabled `predict()` on `bsr()` and `stepwise()` results**, previously unsupported -- these paths never attached a `.formula` to the returned model, so `predict()` failed with a bare `AttributeError`. Support is conditional: `bsr()`/`stepwise()` select individual dummy columns as independent candidates, so a result can include only *some* levels of a categorical predictor, and no formula can reproduce that. `predict()` now works on a `bsr()`/`stepwise()` result whenever its selected predictors can be losslessly re-expressed as a formula; when they can't, it raises a specific `ValueError` naming the affected variable and its missing levels, instead of the previous uninformative `AttributeError`.

### Added
- Added `marketing_strategies.csv`, a dataset for comparing the effectiveness of multiple marketing strategies.

### Tests and compatibility
- Added regression coverage for the plotting, summary, and formula fixes, including 22 response-power validation cases and an external-plot error-message check for `abline()`.
- Verified that `plot_cor()` includes all categorical indicators, including both promoter categories in `video_engagement.csv`; the supplied implementation already supported this behavior.
- Preserved textbook coefficient-name formatting.
- Corrected three malformed warning filters in the textbook compatibility suite.
- Validated the updated source with 99 passing tests: 64 textbook compatibility cases and 35 regression cases.
- The `predict()` categorical fix and the `bsr()`/`stepwise()` formula-reconstruction fix were validated manually for 1.0.2. Dedicated automated coverage was added for 1.0.3.

## 1.0.1 - 2026-08-21

### Fixed
- Added support for `level` as an alternative to `alpha` in regression summaries.
- Added a warning when both `alpha` and `level` are specified; `level` takes precedence.
- Preserved clean confidence-level display by rounding the derived alpha value.
- Fixed `plot()` behavior so `plot(...); abline(model)` correctly layers the regression line on the existing plot.
- Improved `abline()` axis handling so added lines do not change existing plot limits.
- Corrected bundled dataset references to use `betas.csv`.

### Improved
- Updated regression summary output and model-summary wrappers.
- Added textbook compatibility tests covering Ravix functionality used in the companion textbook.
- Added `HELP.md` with an ebook-focused Ravix reference.
- Improved plotting documentation and error messages.
- Updated package metadata to use the SPDX `MIT` license expression.
- Updated packaging configuration and excluded development files from distributions.

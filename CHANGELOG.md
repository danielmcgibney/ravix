# Changelog

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
- The `predict()`/categorical fix and the `bsr()`/`stepwise()` formula-reconstruction fix were validated by running the full 99-test suite (no regressions) and by targeted manual verification against hand-computed predictions, including single-row prediction on a non-baseline category, on the baseline category itself, on a model with an interaction term, and on a pre-existing custom-ordered `Categorical` column; `bsr()`/`stepwise()` were checked against both a full-categorical selection and an engineered dataset that forces a partial-categorical selection. **No dedicated `pytest` cases were added to `tests/` for these two fixes** -- this coverage is not yet part of the automated suite. Given this changelog entry is dated as a release rather than marked Unreleased, confirm whether that's an acceptable gap for this date or whether these two entries should move to 1.0.3 pending test coverage.

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

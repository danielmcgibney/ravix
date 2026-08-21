# Changelog

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

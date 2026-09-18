# Ravix Help: Ebook Reference

Companion reference for *Applied Linear Regression for Business Analytics with Python*

This file documents the Ravix functions and fitted-model attributes used in the ebook, together with a small number of post-publication additions that extend the same workflow. It remains intentionally narrower than the full Ravix API.

---

## 1. Installation and Importing

Install Ravix in Jupyter or Google Colab:

```python
!pip install ravix
```

Import the package:

```python
import ravix
```

or use an alias:

```python
import ravix as rav
```

Check the installed version:

```python
ravix.__version__
```

The ebook also commonly imports individual functions:

```python
from ravix import ols, plot, hist
```

A convenient import block for the Ravix functions used in the ebook is:

```python
from ravix import (
    get_data,
    ols,
    summary,
    predict,
    intervals,
    robust,
    compare,
    plot,
    hist,
    boxplot,
    barplot,
    plot_cor,
    abline,
    qq,
    vif,
    ncv,
    shapiro,
    box_cox,
    stepwise,
    bsr,
    plot_bsr,
)
```

---

## 2. Ravix Formula Notation

Many Ravix functions use formulas written as strings.

```python
"Y ~ X"
```

The response variable is placed to the left of `~`, and predictor variables are placed to the right.

| Syntax | Meaning | Example |
|---|---|---|
| `~` | Separate response and predictors | `"Y ~ X"` |
| `+` | Add predictors | `"Y ~ X1 + X2"` |
| `.` | Include all other dataframe columns | `"Y ~ ."` |
| `-` | Exclude a predictor | `"Y ~ . - X3"` |
| `:` | Interaction only | `"Y ~ X1:X2"` |
| `*` | Main effects plus interaction | `"Y ~ X1*X2"` |
| `**` | Exponent transformation | `"Y ~ X + X**2"` |
| `-1` | Remove the intercept | `"Y ~ X - 1"` |

Variables may come from a dataframe supplied through `data=` or, when appropriate, directly from the Python environment.

---

## 3. Typical Ravix Workflow

A common workflow in the ebook is:

```python
from ravix import get_data, plot, hist, ols, ncv, shapiro

# Load data
df = get_data("house_prices.csv")

# Explore
plot("Price ~ Rooms", data=df)
hist(df.Price)

# Fit model
reg = ols("Price ~ Rooms + Income + TaxRate + Commercial", data=df)

# Summarize
reg.summary()

# Diagnose
plot(reg)
hist(reg)
ncv(reg)
shapiro(reg)
```

---

# DATA

## 4. `get_data()`

Load datasets bundled with Ravix.

### Syntax

```python
get_data(filename=None)
```

### List available datasets

```python
from ravix import get_data
get_data()
```

### Load a dataset

```python
from ravix import get_data

df = get_data("betas.csv")
df.head()
```

The returned object is a Pandas dataframe.

---

# EXPLORATORY PLOTTING

## 5. `plot()`

Creates scatterplots, scatterplot matrices, fitted-model plots, and regression diagnostic plots depending on the input.

### Basic scatterplot

```python
from ravix import plot

plot("Y ~ X", data=df)
```

### Add labels and a title

```python
plot(
    "Profit ~ Sales",
    data=df,
    xlab="Sales (Billions $)",
    ylab="Profits (Billions $)",
    title="Scatterplot of the Top 200 Companies"
)
```

### Scatterplot matrix

Use multiple predictors:

```python
plot("Price ~ Rooms + Income + TaxRate + Commercial", data=df)
```

or use `.` to include all other variables:

```python
plot("Sales ~ .", data=numeric_df)
```

### Add fitted lines to a scatterplot matrix

For a multi-variable scatterplot matrix, use `lines=True`:

```python
plot(
    "Y ~ X1 + X2 + X3 + X4",
    data=df,
    color="blue",
    lines=True
)
```

For a single X-Y scatterplot, fit the model separately and layer the regression line with `abline()`:

```python
reg = ols("Y ~ X", data=df)
plot("Y ~ X", data=df)
abline(reg)
```

### Residual plot from a fitted OLS model

```python
plot(reg)
```

### Common arguments

```python
plot(
    input_data,
    data=None,
    color="blue",
    lines=False,
    smooth=True,
    res="resid",
    title=None,
    xlab=None,
    ylab=None,
    psize=50,
    alpha=1.0,
    figsize=None,
    show=True,
    diag="label"
)
```

---

## 6. `hist()`

Creates histograms for variables or model residuals.

### One variable

```python
from ravix import hist

hist(df.Sales)
```

### Multiple variables using a formula

```python
hist("Sales ~ Calls", data=df)
```

```python
hist("Sales ~ .", data=numeric_df)
```

### Choose the number of bins

```python
hist("Sales ~ Calls", data=df, norm=False, bins=5)
```

### Histogram of model residuals

```python
hist(reg)
```

### Common arguments

```python
hist(
    input_data,
    data=None,
    bins=30,
    color="blue",
    norm=True,
    layout="matrix",
    title=None,
    xlab=None,
    ylab="Frequency",
    figsize=None,
    show=True
)
```

`norm=True` overlays a normal curve where appropriate.

---

## 7. `boxplot()`

Creates box plots for a variable, dataframe, or formula.

### One variable

```python
from ravix import boxplot

boxplot(df.Calls)
```

### Multiple variables in one dataframe

```python
boxplot(
    df,
    title="Boxplot",
    xlab="Variable",
    ylab="Value"
)
```

### Formula input

```python
boxplot("Sales ~ Calls", data=df)
```

### Common arguments

```python
boxplot(
    formula=None,
    data=None,
    color="blue",
    title="Boxplots of Variables",
    xlab="Variable",
    ylab="Value",
    figsize=None
)
```

---

## 8. `barplot()`

Creates bar plots, including aggregated summaries of dataframe columns.

### Basic use

```python
from ravix import barplot

barplot(
    "SPY ~ .",
    means_df,
    xlab="Stock Ticker",
    ylab="Mean Monthly Returns",
    title="Mean Monthly Stock Returns"
)
```

### Aggregate within `barplot()`

```python
barplot(
    "SPY ~ .",
    df,
    agg="mean",
    xlab="Stock Ticker",
    ylab="Mean Monthly Returns"
)
```

### Common arguments

```python
barplot(
    formula=None,
    data=None,
    color="blue",
    title="Barplots of Variables",
    xlab="Variable",
    ylab="Value",
    agg="mean",
    horizontal=False,
    figsize=None
)
```

`agg="mean"` is the default for data that require aggregation. If `agg=None` is used with ambiguous multi-row, multi-column numeric input, Ravix raises a `ValueError` rather than silently averaging the data. For unaggregated raw distributions, `boxplot()` is usually more appropriate.

---

## 9. `plot_cor()`

Creates a color-coded correlation matrix.

### Syntax

```python
from ravix import plot_cor

plot_cor(df)
```

For direct DataFrame input, categorical, string, and boolean columns are converted to dummy variables by default. Every category level is retained (`drop_first=False`).

```python
plot_cor(df, dummy=True)
```

To restrict the correlation matrix to numeric columns only:

```python
plot_cor(df, dummy=False)
```

Formula input also retains every dummy level and is not changed by the `dummy` argument:

```python
plot_cor("Y ~ X1 + Region", data=df)
```

If dummy encoding creates many columns, Ravix may temporarily display:

```text
Note: Consider dummy=False to avoid long processing times.
```

### Common arguments

```python
plot_cor(
    formula,
    data=None,
    style=1,
    title="",
    xlab="",
    ylab="",
    figsize=None,
    show=True,
    dummy=True
)
```

The `style` argument supports four preset visual styles. With `show=False`, the function returns `(fig, ax)` for manual figure control.

---

# MODELING

## 10. `ols()`

Fits an ordinary least squares regression model.

### Syntax

```python
ols(formula, data=None, **kwargs)
```

### Simple linear regression

```python
from ravix import ols

reg = ols("Y ~ X", data=df)
reg.summary()
```

### Multiple regression

```python
reg = ols(
    "Price ~ Rooms + Income + TaxRate + Commercial",
    data=df
)
reg.summary()
```

### All predictors

```python
reg = ols("Price ~ .", data=df)
```

### Quadratic model

```python
quad = ols("Sales ~ Employment + Employment**2", data=df)
quad.summary()
```

### Interaction

```python
reg = ols(
    "Sales ~ Mktg_MC + ToyA + Mktg_MC:ToyA",
    data=df
)
```

The returned model object contains the regression coefficients, fitted values, residuals, model-fit statistics, and the Ravix `summary()` method.

---

## 11. `summary()`

Produces regression summaries and related tables.

The ebook most commonly uses the method form:

```python
reg.summary()
```

The function form is also available:

```python
from ravix import summary
summary(reg)
```

### Default Ravix summary

```python
reg.summary()
```

Shows coefficient estimates, standard errors, t-values, p-values, residual standard error, R-squared, adjusted R-squared, AIC, BIC, and the overall F-test.

### Native statsmodels summary

```python
reg.summary(out="statsmodels")
```

### Coefficient confidence intervals

```python
reg.summary(out="confint")
```

Specify the confidence level:

```python
reg.summary(out="confint", level=0.95)
```

or equivalently specify alpha:

```python
reg.summary(out="confint", alpha=0.05)
```

### ANOVA table

```python
reg.summary(out="anova")
```

### Sequential / Type I ANOVA

```python
reg.summary(out="anova1")
```

### Confidence level and significance level

If neither `alpha` nor `level` is specified, Ravix uses the conventional defaults:

- confidence level = `0.95`
- significance level (`alpha`) = `0.05`

You may specify either one:

```python
reg.summary(out="confint", level=0.90)
reg.summary(out="confint", alpha=0.10)
```

If both are specified, Ravix issues a warning and `level` takes precedence.

### Full signature

```python
summary(
    model,
    out="simple",
    alpha=None,
    level=None,
    format="text"
)
```

Useful ebook values for `out`:

| `out` | Result |
|---|---|
| `"simple"` | Default Ravix model summary |
| `"statsmodels"` | Native statsmodels summary |
| `"confint"` or `"ci"` | Coefficient confidence intervals |
| `"anova"` | ANOVA table |
| `"anova1"` | Sequential / Type I ANOVA table |

---

## 12. `predict()`

Generates point predictions and, optionally, confidence or prediction intervals from a fitted model.

### Syntax

```python
predict(
    model,
    newX=None,
    interval=None,
    level=None,
    alpha=None,
    scale="model"
)
```

### Point prediction

```python
import pandas as pd
from ravix import predict

new_data = pd.DataFrame({"X": [2500]})
predict(reg, new_data)
```

The new dataframe should contain the predictor variable names used to fit the model. Ravix reuses fit-time categorical levels when reconstructing the prediction design matrix.

### Confidence interval for the mean response

```python
predict(reg, new_data, interval="confidence")
```

### Prediction interval for an individual future response

```python
predict(reg, new_data, interval="prediction")
```

Specify the confidence level with either `level` or `alpha`:

```python
predict(reg, new_data, interval="confidence", level=0.95)
predict(reg, new_data, interval="prediction", alpha=0.05)
```

### Transformed-response models

By default, predictions are returned on the scale used to fit the model:

```python
predict(reg, new_data, scale="model")
```

For models whose response was transformed with `log()`, `sqrt()`, `inverse()`/`inv()`, or a supported power transformation, predictions can be returned on the original response scale:

```python
predict(reg, new_data, scale="response")
```

When an interval is requested, its endpoints are also back-transformed. This is a direct inverse transformation and does not apply a bias correction such as log-response smearing.

---

## 13. `intervals()`

Calculates confidence or prediction intervals for new predictor values. `intervals()` is retained for textbook and backward compatibility and uses the same prediction pipeline as `predict()`.

### Syntax

```python
intervals(
    model,
    newX,
    interval="confidence",
    level=None,
    alpha=None
)
```

### Confidence interval for the mean response

```python
from ravix import intervals

intervals(reg, new_data, interval="confidence")
```

### Prediction interval for an individual future response

```python
intervals(reg, new_data, interval="prediction")
```

### Set confidence level

```python
intervals(reg, new_data, interval="confidence", level=0.95)
```

or:

```python
intervals(reg, new_data, interval="prediction", alpha=0.05)
```

Equivalent new code may use:

```python
predict(reg, new_data, interval="confidence")
predict(reg, new_data, interval="prediction")
```

---

# USEFUL FITTED-MODEL ATTRIBUTES

## 14. `robust()`

Returns an OLS result with heteroskedasticity-robust standard errors. The fitted coefficients, fitted values, and residuals are unchanged; inference is recomputed using a heteroskedasticity-consistent covariance estimator.

### Syntax

```python
robust(model, type="HC3", use_t=True)
```

### Example

```python
from ravix import robust

reg = ols("Sales ~ Advertising + Price", data=df)
rob = robust(reg)
rob.summary()
```

`HC3` is the default. The supported covariance estimators are:

```text
"HC0"
"HC1"
"HC2"
"HC3"
```

The returned object remains a Ravix-compatible OLS result, so standard attributes and methods can be used:

```python
rob.params
rob.bse
rob.tvalues
rob.pvalues
rob.conf_int()
```

`robust()` does not modify the original fitted model.

---

## 15. `compare()`

Compares two nested OLS models using the classical partial F-test. The models may be supplied in either order; Ravix identifies the reduced and full models automatically.

### Syntax

```python
compare(model1, model2, format="text")
```

### Example

```python
from ravix import compare

reduced = ols("Sales ~ Price", data=df)
full = ols("Sales ~ Price + Advertising + Income", data=df)

compare(reduced, full)
```

The comparison uses SSE notation and tests whether the additional terms in the full model jointly improve fit. Ravix verifies that the models are nested and use matching response observations.

### Output formats

```python
compare(reduced, full, format="text")
table = compare(reduced, full, format="df")
latex = compare(reduced, full, format="latex")
```

Text and LaTeX output place Ravix significance stars next to the p-value. DataFrame output keeps the p-value numeric and stores significance codes separately in the `Signif.` column. The LaTeX output uses a booktabs-style table and does not require `jinja2`.

---

## 16. Model Attributes Used in the Ebook

Once a model has been fitted:

```python
reg = ols("Y ~ X", data=df)
```

useful values can be accessed directly.

### Coefficients

```python
reg.params
```

Specific coefficient:

```python
reg.params["X"]
```

### R-squared

```python
reg.rsquared
```

### Adjusted R-squared

```python
reg.rsquared_adj
```

### Residuals

```python
reg.resid
```

### Fitted values

```python
reg.fittedvalues
```

### AIC

```python
reg.aic
```

### BIC

```python
reg.bic
```

These are useful when calculations need to be performed directly rather than read from `reg.summary()`.

---

# ADDING FITTED LINES

## 17. `abline()`

Adds a fitted line or curve to an existing Ravix plot.

### Add the line from a fitted model

```python
from ravix import plot, abline

plot("Sales ~ Employment", data=df)
abline(reg)
```

This also works with models containing polynomial terms:

```python
plot("Sales ~ Employment", data=df)
abline(quad)
```

### Supply an intercept and slope directly

```python
abline(-907.5484, 9.3253, color="darkblue")
```

or explicitly:

```python
abline(a=-907.5484, b=9.3253, color="darkblue")
```

### Common arguments

```python
abline(
    model=None,
    a=None,
    b=None,
    color="black",
    linetype="-",
    linewidth=1.5,
    label=None,
    ax=None
)
```

In Ravix 1.0.1, `abline()` overlays the line without changing the existing x- or y-axis scale.

The standard Ravix layering syntax is:

```python
plot("Y ~ X", data=df)
abline(reg)
```

For explicit Matplotlib axis control:

```python
import matplotlib.pyplot as plt
from ravix import plot, abline

fig, ax = plot("Y ~ X", data=df, show=False)
abline(reg, ax=ax)
plt.show()
```

---

# MULTICOLLINEARITY AND DIAGNOSTICS

## 18. `vif()`

Calculates variance inflation factors for predictor variables.

### Syntax

```python
vif(formula=None, data=None, plot=False, ...)
```

### Example

```python
from ravix import vif

vif("Y ~ X1 + X2 + X3", data=df)
```

### Plot the VIF values

```python
vif(
    "Price ~ Rooms + Income + TaxRate + Commercial",
    data=df,
    plot=True
)
```

---

## 19. `ncv()`

Performs Ravix's nonconstant variance test on a fitted model.

### Syntax

```python
ncv(model, alpha=0.05, return_dict=False)
```

### Example

```python
from ravix import ncv

ncv(reg)
```

Specify alpha if desired:

```python
ncv(reg, alpha=0.05)
```

---

## 20. `qq()`

Creates a Q-Q plot for assessing normality.

### Model residuals

```python
from ravix import qq

qq(reg)
```

### Common arguments

```python
qq(
    input_data,
    data=None,
    res="resid",
    level=0.95,
    color="blue",
    lcolor="red",
    line_type="45",
    layout="matrix",
    title=None,
    xlab="Theoretical Quantiles",
    ylab="Sample Quantiles",
    figsize=None,
    grid=True,
    show=True
)
```

For approximately normal residuals, points should lie reasonably close to the reference line.

---

## 21. `shapiro()`

Performs the Shapiro-Wilk test for normality.

### Syntax

```python
shapiro(input_data, alpha=0.05, return_dict=False)
```

### Fitted model

```python
from ravix import shapiro

shapiro(reg)
```

The function tests the model residuals when given a fitted regression object.

---

## 22. `box_cox()`

Displays the Box-Cox log-likelihood profile to help identify a response-variable transformation.

### Syntax

```python
box_cox(
    model,
    color="blue",
    lcolor="red",
    title="Log-Likelihood for Box-Cox Transformation",
    xlab="Lambda",
    ylab="Log-Likelihood",
    figsize=(10, 6)
)
```

### Example

```python
from ravix import box_cox

box_cox(reg)
```

The response variable must contain positive values for a Box-Cox transformation.

---

# VARIABLE SELECTION

## 23. `stepwise()`

Performs iterative variable selection.

### Syntax

```python
stepwise(
    formula,
    data,
    method="ols",
    direction="backward",
    metric="aic",
    threshold_in=0.05,
    threshold_out=0.10,
    max_steps=100,
    verbose=False,
    group_categorical=False
)
```

### Backward elimination

Backward elimination is the default:

```python
from ravix import stepwise

BE = stepwise("Price ~ .", data=df)
```

Show the steps:

```python
BE = stepwise("Price ~ .", data=df, verbose=True)
```

### Forward selection

```python
FS = stepwise(
    "Price ~ .",
    data=df,
    direction="forward",
    verbose=True
)
```

### Stepwise regression in both directions

```python
SW = stepwise(
    "Price ~ .",
    data=df,
    direction="both",
    verbose=True
)
```

The returned object is a fitted regression model, so the usual model methods and attributes can be used:

```python
SW.summary()
SW.aic
SW.bic
SW.params
```

### Grouping categorical predictors

By default, `group_categorical=False` preserves Ravix's historical behavior: individual dummy columns created from a multilevel categorical predictor may enter or leave the model separately.

```python
SW = stepwise(
    "Y ~ X1 + Region + Department",
    data=df,
    group_categorical=False
)
```

Set `group_categorical=True` when the categorical predictor should be treated as one selection term. All dummy columns generated for that categorical main effect then enter or leave together.

```python
SW = stepwise(
    "Y ~ X1 + Region + Department",
    data=df,
    group_categorical=True
)
```

This option changes only the model-selection grouping. It does not change the underlying regression coding of the categorical predictor.

---

## 24. `bsr()`

Performs best subsets regression.

### Syntax

```python
bsr(
    formula,
    data,
    max_var=8,
    metric="aic",
    method="ols",
    group_categorical=False
)
```

### Example

```python
from ravix import bsr

BSR = bsr("Price ~ .", data=df)
```

### Consider more predictors

```python
BSR = bsr("Price ~ .", data=df, max_var=13)
```

### Choose a metric

```python
BSR = bsr("Price ~ .", data=df, metric="bic")
```

Metrics used in the ebook include:

```text
"aic"
"bic"
"adjr2"
"rmse"
```

### Grouping categorical predictors

As with `stepwise()`, the default `group_categorical=False` allows individual dummy columns from a multilevel categorical predictor to be selected separately.

```python
BSR = bsr("Y ~ .", data=df, group_categorical=False)
```

With `group_categorical=True`, the dummy columns generated by a categorical main effect are treated as a single selection term. The entire categorical predictor is either included or excluded, and it counts as one term toward `max_var`.

```python
BSR = bsr("Y ~ .", data=df, max_var=6, group_categorical=True)
```

### Summary of the selected model

```python
BSR.summary()
```

---

## 25. `plot_bsr()`

Plots results from a best subsets regression object.

### Syntax

```python
plot_bsr(
    model,
    type="predictors",
    top_n=5,
    color="darkgreen",
    title=None,
    xlab=None,
    ylab=None,
    figsize=None
)
```

### Predictor-inclusion plot

```python
from ravix import plot_bsr

plot_bsr(BSR)
```

### Line plot of model-selection metric

```python
plot_bsr(BSR, type="line")
```

### Show more candidate models

```python
plot_bsr(BSR, top_n=10)
```

Useful `type` values include:

```text
"predictors"
"line"
"bar"
```

---

# STATSMODELS INFLUENCE MEASURES USED IN THE EBOOK

## 26. `OLSInfluence`

The ebook also uses the Statsmodels `OLSInfluence` object for leverage, standardized/studentized residuals, and Cook's distance. These are Statsmodels commands rather than Ravix functions.

```python
from statsmodels.stats.outliers_influence import OLSInfluence

influence = OLSInfluence(reg)
```

### Leverage

```python
hat_values = influence.hat_matrix_diag
```

### Standardized residuals

```python
standardized_resid = influence.resid_studentized_internal
```

### Studentized residuals

```python
studentized_resid = influence.resid_studentized_external
```

### Cook's distance

```python
cooks_d = influence.cooks_distance[0]
```

---

# QUICK REFERENCE

## 27. Ebook Ravix Functions at a Glance

| Function | Primary purpose |
|---|---|
| `get_data()` | List or load bundled datasets |
| `plot()` | Scatterplots, scatterplot matrices, residual plots |
| `hist()` | Histograms and residual histograms |
| `boxplot()` | Box plots |
| `barplot()` | Bar plots and aggregated bar plots |
| `plot_cor()` | Correlation heatmap with optional categorical dummy encoding |
| `ols()` | Fit ordinary least squares regression |
| `summary()` | Model summary, confidence intervals, ANOVA |
| `predict()` | Point predictions and optional confidence/prediction intervals |
| `intervals()` | Backward-compatible confidence and prediction intervals |
| `robust()` | HC0-HC3 heteroskedasticity-robust OLS inference |
| `compare()` | Partial F-test for nested OLS models |
| `abline()` | Overlay fitted line/curve on a plot |
| `vif()` | Variance inflation factors |
| `ncv()` | Nonconstant variance test |
| `qq()` | Q-Q plot |
| `shapiro()` | Shapiro-Wilk normality test |
| `box_cox()` | Box-Cox transformation guidance |
| `stepwise()` | Backward, forward, or bidirectional selection; optional categorical grouping |
| `bsr()` | Best subsets regression; optional categorical grouping |
| `plot_bsr()` | Best subsets visualization |

---

## 28. Compact Example

```python
import pandas as pd
from ravix import (
    get_data, plot, hist, ols, predict, intervals, robust, compare,
    vif, ncv, qq, shapiro
)

# Load data
df = get_data("house_prices.csv")

# Explore
plot("Price ~ Rooms + Income + TaxRate + Commercial", data=df)
hist(df.Price)

# Check multicollinearity
vif("Price ~ Rooms + Income + TaxRate + Commercial", data=df)

# Fit model
reg = ols("Price ~ Rooms + Income + TaxRate + Commercial", data=df)
reg.summary()

# Access model values
print(reg.params)
print(reg.rsquared)
print(reg.rsquared_adj)

# Predict
new_data = pd.DataFrame({
    "Rooms": [5],
    "Income": [60000],
    "TaxRate": [3.0],
    "Commercial": [10]
})

predict(reg, new_data)
predict(reg, new_data, interval="confidence")
predict(reg, new_data, interval="prediction")

# Robust inference, when appropriate
rob = robust(reg)
rob.summary()

# Diagnose
plot(reg)
hist(reg)
qq(reg)
ncv(reg)
shapiro(reg)
```

---

## Scope of This Help File

This reference follows the Ravix functionality used in *Applied Linear Regression for Business Analytics with Python* and also documents selected post-publication additions that extend the same statistics-first workflow, including robust inference, nested-model comparison, enhanced prediction, categorical grouping in model selection, and categorical handling in correlation plots. Ravix may contain additional functionality beyond this reference.

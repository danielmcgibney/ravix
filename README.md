# Ravix

Ravix is a Python package for regression analysis and data visualization. It provides tools for model fitting, prediction, and various types of plots to help visualize your data particularly for regression analysis.

## Features

- Model fitting and prediction with convenient formula notation
- Streamlined code for plotting (boxplot, histogram, scatter plot, etc.)
- Regression analysis diagnostic tools
- Integration with popular libraries like `pandas` and `statsmodels`

## Installation

You can install the package using `pip`:

```sh
pip install ravix
```

## Usage

Here are some examples of how to use the key functions in the package.

### Importing the Package

To use the functions provided by the package, import it as follows:

```python
import ravix
```

### Getting Data

There are multiple datasets available from Ravix and are easily attained using the `get_data` function.  The datasets currently available are:

* AirBnb.csv
* Betas.csv
* Charges.csv
* Employment.csv
* HousePrices.csv
* HR_retention.csv
* MarketingToys.csv
* Sales.csv
* Top200.csv
* Twitter.csv
* Youtube.csv

See [Applied Linear Regression for Business Analytics with Python](www.businessregression.com) for details regarding these datasets.  Sample import example:

```python
import ravix

# Load data from ravix
df = ravix.get_data("Betas.csv")

# Format the data (for later)
df.drop(columns = df.columns[0], inplace=True)
```

### Model Fitting and Prediction

Ravix formula supports formula functionality similar to R. Fit a model with a formula:

```python
# Fit model with formula 
model = ravix.ols("SPY ~ .", df)
```

Summary types are specified using the `out` argument.  Different summaries are available including:

* simple (default)
* statsmodels 
* R
* ANOVA
* coefficients (coef)

```python
# Generate a model summary
ravix.summary(model)
```

### Making Predictions

A Statsmodels object is created by default.  From this object, the predict function can be used. Since `df` is the dataframe used to fit the model, the following lines produce the same result.

```python
# Make predictions
ravix.predict(model, df)

# Produce fitted values
ravix.predict(model)
```

### General Plotting

Plotting code is streamlined and built on top of Seaborn and MatPlotLib. Samples provided below.

```python
# Generate a boxplot
ravix.boxplot("SPY ~ .", df)

# Generate a histogram
ravix.hist(df.SPY)

# Multiple histograms
ravix.hists("SPY ~ .",data = df)

# Scatter plot
ravix.plot("MSFT ~ SPY", data = df)

# Multiple Scatter plots
ravix.plot("SPY ~ .", data = df)

# Correlation Plot
ravix.plot_cor(df)
```

### Required Fixes

Based on current testing, the following fixes are required:

1. Ensure global scope accessibility for variables.
2. Adjust summary spacing.
3. Provide compatibility with `scikit-learn`.
4. Implement AI-generated summaries.
5. Allow for additional plotting customization (using kwargs).
6. Review and improve diagnostic plots.
7. Provide support for logistic regression and other GLMs.
8. Provide support for automatic dummy variable retrieval.
9. Plots should work without formulas.

## Contributing

We welcome contributions to Ravix! If you find a bug or have a feature request, please open an issue on [GitHub](https://github.com/danmcgib/ravix). You can also contribute by:

1. Forking the repository
2. Creating a new branch (`git checkout -b feature-branch`)
3. Committing your changes (`git commit -am 'Add some feature'`)
4. Pushing to the branch (`git push origin feature-branch`)
5. Creating a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We would like to thank all contributors and users of Ravix for their support and feedback.
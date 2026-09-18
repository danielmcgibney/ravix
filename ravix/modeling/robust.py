"""Heteroskedasticity-robust inference for Ravix OLS models."""

import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .predict import predict
from .summary import summary as summary_function


_VALID_COV_TYPES = {"HC0", "HC1", "HC2", "HC3"}


def robust(model, type="HC3", use_t=True):
    """
    Return an OLS result with heteroskedasticity-robust standard errors.

    The fitted coefficients, fitted values, and residuals are unchanged. The
    covariance matrix is replaced with a heteroskedasticity-consistent
    estimator, which changes the standard errors, test statistics, p-values,
    confidence intervals, and covariance-aware model tests.

    Parameters
    ----------
    model : Ravix OLS result
        A model fitted with ``ravix.ols()`` or another Ravix routine that
        returns an OLS result.
    type : {"HC0", "HC1", "HC2", "HC3"}, default="HC3"
        Heteroskedasticity-consistent covariance estimator. ``HC3`` is the
        default because it provides a useful small-sample correction.
    use_t : bool, default=True
        If True, use the Student t distribution for coefficient inference.
        If False, use the standard normal distribution.

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        A new Ravix-compatible OLS result. Standard result attributes such as
        ``params``, ``bse``, ``tvalues``, ``pvalues``, and ``conf_int()``
        reflect the requested robust covariance estimator.

    Notes
    -----
    ``robust()`` does not modify the original fitted model.

    Examples
    --------
    >>> reg = ravix.ols("Sales ~ Advertising + Price", data=df)
    >>> rob = ravix.robust(reg)
    >>> rob.params
    >>> rob.bse
    >>> rob.summary()

    >>> rob_hc1 = ravix.robust(reg, type="HC1")
    """
    if not hasattr(model, "model") or not isinstance(model.model, sm.OLS):
        raise TypeError("robust() currently supports OLS models only.")

    if not isinstance(type, str):
        raise TypeError("type must be one of 'HC0', 'HC1', 'HC2', or 'HC3'.")

    cov_type = type.upper()
    if cov_type not in _VALID_COV_TYPES:
        allowed = ", ".join(sorted(_VALID_COV_TYPES))
        raise ValueError(f"Unknown robust covariance type '{type}'. Choose from: {allowed}.")

    if not isinstance(use_t, bool):
        raise TypeError("use_t must be True or False.")

    # statsmodels returns a bare OLSResults object here. Re-wrap it so labeled
    # pandas objects such as params and bse are preserved for Ravix users.
    robust_result = model.get_robustcov_results(cov_type=cov_type, use_t=use_t)
    fitted = RegressionResultsWrapper(robust_result)

    # Preserve Ravix metadata needed by formula-based prediction and by models
    # returned from selection routines.
    for attr in (
        "formula",
        "_categorical_levels",
        "_predict_unsupported_reason",
        "model_type",
    ):
        if hasattr(model, attr):
            value = getattr(model, attr)
            if attr == "_categorical_levels" and isinstance(value, dict):
                value = dict(value)
            setattr(fitted, attr, value)

    fitted.model_type = getattr(model, "model_type", "ols")
    fitted._ravix_robust = True
    fitted._ravix_cov_type = cov_type

    # Preserve native statsmodels methods before installing the Ravix wrappers.
    fitted._statsmodels_predict = fitted.predict
    fitted._statsmodels_summary = fitted.summary

    def predict_wrapper(newdata=None, *args, **kwargs):
        """Generate predictions using the standard Ravix prediction interface."""
        if newdata is None and not args and not kwargs:
            return fitted._statsmodels_predict()
        return predict(fitted, newdata, *args, **kwargs)

    fitted.predict = predict_wrapper

    def summary_wrapper(out="simple", alpha=None, level=None, format="text"):
        """Generate a Ravix summary using robust inference quantities."""
        return summary_function(
            fitted,
            out=out,
            alpha=alpha,
            level=level,
            format=format,
        )

    fitted.summary = summary_wrapper

    return fitted

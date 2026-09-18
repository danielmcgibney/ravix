import numpy as np
import pandas as pd

from .parse_formula import parse_formula, _parse_response


def _validate_interval_args(interval=None, level=None, alpha=None):
    """Validate interval options and return the statsmodels alpha value."""
    if interval is None:
        if level is not None or alpha is not None:
            raise ValueError("'level' and 'alpha' require interval='confidence' or interval='prediction'.")
        return None

    if interval not in ("confidence", "prediction"):
        raise ValueError(
            f"Invalid interval type '{interval}'. Must be 'confidence' or 'prediction'."
        )

    if level is not None and alpha is not None:
        raise ValueError(
            "Cannot specify both 'level' and 'alpha'. Use 'level' for confidence "
            "level (e.g., 0.95) or 'alpha' for significance level (e.g., 0.05)."
        )

    if level is None and alpha is None:
        return 0.05

    if level is not None:
        if not (0 < level < 1):
            raise ValueError(f"level must be between 0 and 1 (exclusive), got {level}")
        return 1 - level

    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be between 0 and 1 (exclusive), got {alpha}")
    return alpha


def _validate_scale(scale):
    """Validate prediction scale."""
    if scale not in ("model", "response"):
        raise ValueError("scale must be either 'model' or 'response'.")
    return scale


_INTERCEPT_NAMES = ("Intercept", "const", "intercept")


def _prepare_newdata_raw_matrix(model, newX, model_columns):
    """Build a prediction design matrix for a model with no stored formula.

    This path covers models fit directly from matrices, and models returned
    by ``bsr()``/``stepwise()`` when the selected terms couldn't be
    reconstructed into a formula (e.g. a selection that includes only some
    levels of a categorical variable -- see
    ``_attach_formula_if_reconstructible``).

    Unlike the formula path, this does not encode categorical variables or
    apply predictor transformations from raw values: ``newX`` must already
    contain one column per non-intercept entry in ``model_columns``, matched
    by name. An intercept column is added automatically, under whichever name
    the model actually used, if the model has one and ``newX`` doesn't already
    supply it. Columns are then selected and reordered by name to match
    ``model_columns`` exactly -- callers are not required to supply columns in
    any particular order.
    """
    model_intercept = next((name for name in _INTERCEPT_NAMES if name in model_columns), None)
    has_intercept_col = any(col in _INTERCEPT_NAMES for col in newX.columns)

    if model_intercept and not has_intercept_col:
        newX = newX.copy()
        newX.insert(0, model_intercept, 1)

    missing_cols = set(model_columns) - set(newX.columns)
    if missing_cols:
        reason = getattr(model, "_predict_unsupported_reason", None)
        why = f" ({reason})" if reason else ""
        raise ValueError(
            f"This model has no stored formula{why}, so predict() cannot encode "
            "newX automatically. newX must already contain one column per model "
            "term, matching model.model.exog_names by name. Missing columns: "
            f"{sorted(missing_cols)}."
        )

    return newX[list(model_columns)]


def _prepare_newdata(model, newX):
    """Build a prediction design matrix from new data.

    Uses the model's stored formula when available: categorical encoding and
    predictor transformations are re-derived from raw column values via
    ``parse_formula()``, using the categorical levels captured at fit time.

    When the model has no stored formula, falls back to a raw design-matrix
    contract instead of raising -- see ``_prepare_newdata_raw_matrix``. This
    keeps ``predict()`` usable on models that ``ols()``/``bsr()``/
    ``stepwise()`` couldn't attach a reconstructible formula to, at the cost
    of requiring newX to already be encoded to match the model's columns.
    """
    if not isinstance(newX, pd.DataFrame):
        newX = pd.DataFrame(newX)

    newX = newX.reset_index(drop=True)
    model_columns = model.model.exog_names

    if not hasattr(model, "formula"):
        return _prepare_newdata_raw_matrix(model, newX, model_columns)

    formula = model.formula
    if "~" not in formula:
        raise ValueError(
            "predict() isn't supported for this model: its stored formula does not "
            "contain a '~' separating response and predictors."
        )

    # Prediction needs only the right-hand side. Parsing the full formula would
    # incorrectly require the response column in newX, especially when the
    # response itself was transformed.
    predictor_formula = "~" + formula.split("~", 1)[1]
    categorical_levels = dict(getattr(model, "_categorical_levels", None) or {})

    try:
        _, transformed_X = parse_formula(
            predictor_formula,
            newX,
            drop_first=False,
            categorical_levels=categorical_levels,
        )
    except KeyError as exc:
        raise ValueError(f"Missing required variable in newX: {str(exc)}") from exc

    transformed_X = transformed_X.reset_index(drop=True)

    missing_cols = set(model_columns) - set(transformed_X.columns)
    if missing_cols:
        raise ValueError(f"The following required columns are missing: {missing_cols}")

    return transformed_X[model_columns]


def _get_response_transform(model):
    """Return the response transformation stored in the model formula."""
    formula = getattr(model, "formula", None)
    if not formula or "~" not in formula:
        return None, None

    response_term = formula.split("~", 1)[0].strip()
    if not response_term:
        return None, None

    # Keep response parsing consistent with parse_formula().
    response_term = response_term.replace("**", "^").replace(" ", "")
    _, resp_func, resp_power = _parse_response(response_term)
    return resp_func, resp_power


def _back_transform_array(values, resp_func, resp_power):
    """Back-transform numeric values from model scale to original response scale."""
    arr = np.asarray(values, dtype=float)

    if resp_func is None:
        return arr.copy()

    if resp_func == "log":
        return np.exp(arr)

    if resp_func == "sqrt":
        if np.any(arr < 0):
            raise ValueError(
                "Cannot back-transform to the original response scale because the model "
                "produced negative values on the sqrt(response) scale."
            )
        return np.square(arr)

    if resp_func == "inverse":
        if np.any(arr == 0):
            raise ValueError(
                "Cannot back-transform to the original response scale because a prediction "
                "is zero on the inverse(response) scale."
            )
        return 1.0 / arr

    if resp_func == "power":
        power = float(resp_power)

        if power == 0:
            raise ValueError("A response transformed as Y^0 cannot be uniquely back-transformed.")
        if power == 1:
            return arr.copy()

        if power.is_integer():
            exponent = int(power)

            # Even integer powers lose the sign of the original response, so the
            # inverse is not unique without an additional domain assumption.
            if abs(exponent) % 2 == 0:
                raise ValueError(
                    f"Cannot uniquely back-transform a response modeled as Y^{power:g}: "
                    "an even integer power loses the sign of the original response."
                )

            if exponent < 0 and np.any(arr == 0):
                raise ValueError(
                    "Cannot back-transform to the original response scale because a prediction "
                    f"is zero on the Y^{power:g} scale."
                )

            # Signed root preserves real-valued inverses for odd integer powers,
            # including negative odd powers such as Y^-1 and Y^-3.
            return np.sign(arr) * np.power(np.abs(arr), 1.0 / power)

        # parse_formula requires non-negative original responses for non-integer
        # powers. Their transformed values therefore cannot be negative.
        if np.any(arr < 0):
            raise ValueError(
                "Cannot back-transform to the original response scale because the model "
                f"produced negative values on the Y^{power:g} scale."
            )
        if power < 0 and np.any(arr == 0):
            raise ValueError(
                "Cannot back-transform to the original response scale because a prediction "
                f"is zero on the Y^{power:g} scale."
            )
        return np.power(arr, 1.0 / power)

    raise ValueError(f"Unsupported response transformation '{resp_func}'.")


def _restore_type(original, transformed):
    """Preserve pandas Series/index information when back-transforming."""
    if isinstance(original, pd.Series):
        return pd.Series(transformed, index=original.index, name=original.name)
    if isinstance(original, pd.DataFrame):
        return pd.DataFrame(transformed, index=original.index, columns=original.columns)
    return transformed


def _back_transform(values, model):
    """Back-transform predictions using the response term in the fitted formula."""
    resp_func, resp_power = _get_response_transform(model)
    transformed = _back_transform_array(values, resp_func, resp_power)
    return _restore_type(values, transformed)


def _validate_interval_back_transform(lower, upper, model):
    """Reject interval back-transforms that cross a transformation singularity."""
    resp_func, resp_power = _get_response_transform(model)

    has_singularity_at_zero = (
        resp_func == "inverse"
        or (resp_func == "power" and resp_power is not None and float(resp_power) < 0)
    )

    if has_singularity_at_zero:
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)
        if np.any((lower_arr <= 0) & (upper_arr >= 0)):
            raise ValueError(
                "Cannot express this interval as one finite interval on the original response "
                "scale because it crosses zero on a reciprocal/negative-power scale."
            )


def _native_predict(model, transformed_X):
    """Call the model's underlying point-prediction method.

    Ravix's own fit()/robust() wrappers replace ``model.predict`` with a
    Ravix-flavored wrapper and stash the original statsmodels method under
    ``_statsmodels_predict``. A model that never passed through one of those
    wrappers -- e.g. a bare ``sm.OLS(y, X).fit()`` -- has no such attribute,
    so fall back to the model's own ``.predict()`` in that case.
    """
    predict_fn = getattr(model, "_statsmodels_predict", None) or model.predict
    return predict_fn(transformed_X)


def predict(model, newX=None, interval=None, level=None, alpha=None, scale="model"):
    """
    Generate predictions from a fitted Ravix model, optionally with intervals.

    Parameters
    ----------
    model : object
        Fitted Ravix regression model.
    newX : pandas.DataFrame, optional
        New predictor values. If omitted, fitted observations are used.
    interval : {"confidence", "prediction"}, optional
        If omitted, return point predictions only. ``"confidence"`` returns
        intervals for the mean response; ``"prediction"`` returns intervals
        for individual future observations when supported by the fitted model.
    level : float, optional
        Confidence level between 0 and 1. Cannot be used together with alpha.
    alpha : float, optional
        Significance level between 0 and 1. Cannot be used together with level.
    scale : {"model", "response"}, default="model"
        Scale on which predictions are returned. ``"model"`` preserves the
        fitted response scale. ``"response"`` back-transforms predictions to
        the original response units when the left-hand side of the formula uses
        ``log()``, ``sqrt()``, ``inverse()``/``inv()``, or a power transformation.
        Requires a model with a stored formula (see Notes).

        For nonlinear response transformations, ``scale="response"`` performs
        direct inverse transformation. It does not apply a bias correction (for
        example, a smearing correction for log-response models).

    Returns
    -------
    array-like or pandas.DataFrame
        Point predictions when ``interval`` is omitted. When an interval is
        requested, returns columns ``Prediction``, ``Lower Bound``, and
        ``Upper Bound``. With ``scale="response"``, all three are reported on
        the original response scale.

    Notes
    -----
    New data is encoded one of two ways, chosen automatically:

    - **Model has a stored formula** (the usual case for ``ols()``,
      ``logistic()``, ``poisson()``, and most ``bsr()``/``stepwise()``
      results): ``newX`` should contain raw, untransformed predictor columns,
      exactly as passed to the fit formula. Categorical encoding and any
      predictor transformations are re-derived automatically.
    - **Model has no stored formula** (built directly from matrices, or
      returned by ``bsr()``/``stepwise()`` when the selected terms couldn't be
      re-expressed as a formula -- e.g. only some levels of a categorical
      variable were selected): ``newX`` must already be encoded to match
      ``model.model.exog_names`` -- one column per model term, matched by
      name. An intercept column is added automatically if the model has one.
      Categorical encoding and predictor transformations are the caller's
      responsibility in this case, and ``scale="response"`` is unavailable
      since the response transformation can't be read back from a formula
      that doesn't exist.

    Examples
    --------
    >>> reg = ols("log(Sales) ~ Advertising", data=df)
    >>> predict(reg, new_data)                         # log(Sales) scale
    >>> predict(reg, new_data, scale="response")      # Sales scale
    >>> predict(reg, new_data, interval="prediction", scale="response")
    """
    interval_alpha = _validate_interval_args(interval, level, alpha)
    scale = _validate_scale(scale)

    if scale == "response" and not hasattr(model, "formula"):
        raise ValueError(
            "scale='response' requires a model with a stored formula, since the "
            "response transformation is read from the formula's left-hand side. "
            "This model has no stored formula; use scale='model' instead."
        )

    if newX is None:
        if interval is None:
            result = model.fittedvalues
            return _back_transform(result, model) if scale == "response" else result
        prediction_result = model.get_prediction()
    else:
        transformed_X = _prepare_newdata(model, newX)
        if interval is None:
            result = _native_predict(model, transformed_X)
            return _back_transform(result, model) if scale == "response" else result
        prediction_result = model.get_prediction(transformed_X)

    summary_frame = prediction_result.summary_frame(alpha=interval_alpha)

    if interval == "confidence":
        lower_col = "mean_ci_lower"
        upper_col = "mean_ci_upper"
    else:
        lower_col = "obs_ci_lower"
        upper_col = "obs_ci_upper"
        if lower_col not in summary_frame.columns or upper_col not in summary_frame.columns:
            raise ValueError(
                "Prediction intervals for individual observations are not available "
                "for this model type."
            )

    if "mean" not in summary_frame.columns:
        raise ValueError("This model type does not provide prediction summary intervals.")

    prediction = summary_frame["mean"]
    lower = summary_frame[lower_col]
    upper = summary_frame[upper_col]

    if scale == "response":
        _validate_interval_back_transform(lower, upper, model)

        prediction = _back_transform(prediction, model)
        lower_bt = _back_transform(lower, model)
        upper_bt = _back_transform(upper, model)

        # Some inverse transformations are decreasing, so transformed lower and
        # upper endpoints can reverse order on the original response scale.
        lower_values = np.minimum(np.asarray(lower_bt), np.asarray(upper_bt))
        upper_values = np.maximum(np.asarray(lower_bt), np.asarray(upper_bt))
        lower = pd.Series(lower_values, index=summary_frame.index)
        upper = pd.Series(upper_values, index=summary_frame.index)

    return pd.DataFrame(
        {
            "Prediction": prediction,
            "Lower Bound": lower,
            "Upper Bound": upper,
        }
    )

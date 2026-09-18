from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

from .predict import predict


def intervals(
    model,
    newX: Union[pd.DataFrame, np.ndarray],
    interval: Literal["confidence", "prediction"] = "confidence",
    level: Optional[float] = None,
    alpha: Optional[float] = None,
) -> pd.DataFrame:
    """
    Calculate confidence or prediction intervals for new observations.

    This function is retained for textbook and backward compatibility. Interval
    calculations now use the same formula-aware prediction pipeline as
    :func:`predict`, so categorical variables and predictor transformations are
    handled consistently by both functions.

    Parameters
    ----------
    model : object
        Fitted Ravix regression model.
    newX : pandas.DataFrame or numpy.ndarray
        New predictor values.
    interval : {"confidence", "prediction"}, default="confidence"
        ``"confidence"`` returns an interval for the mean response;
        ``"prediction"`` returns an interval for an individual future
        observation when supported by the fitted model.
    level : float, optional
        Confidence level between 0 and 1. Defaults to 0.95 when neither level
        nor alpha is supplied. Cannot be used together with alpha.
    alpha : float, optional
        Significance level between 0 and 1. Cannot be used together with level.

    Returns
    -------
    pandas.DataFrame
        Columns: ``Prediction``, ``Lower Bound``, and ``Upper Bound``.

    Notes
    -----
    ``intervals()`` is a compatibility wrapper around ``predict()``. New code
    may equivalently use ``predict(model, newX, interval=...)``.
    """
    return predict(
        model,
        newX,
        interval=interval,
        level=level,
        alpha=alpha,
    )

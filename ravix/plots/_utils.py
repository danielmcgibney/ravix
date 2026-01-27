# ravix/plots/_utils.py
"""
Internal utilities shared across plotting modules.

Centralizes helper functions duplicated across:
- plot.py (scatter + residuals)
- hist.py (histograms + residual hist)
- qq.py (Q-Q plots)
"""

from __future__ import annotations

from typing import Any, List, Optional, Union
import re
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba


# ======================================================================
# Formula / data helpers
# ======================================================================

def _ensure_no_intercept(formula: str) -> str:
    """Add +0 to formula if user hasn't already controlled intercept."""
    f = formula.strip()
    if "~" not in f:
        return f

    f_nospace = f.replace(" ", "")
    if "+0" in f_nospace or "-1" in f_nospace:
        return f

    # Keep readable; parse_formula should handle spaces
    return f + " +0"


def _clean_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame for plotting:
      - replace +/-inf with NaN
      - keep numeric columns only
    """
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.select_dtypes(include=[np.number])
    return out


def _as_1d_numeric(values: Any) -> np.ndarray:
    """Coerce to 1D numeric array and drop non-finite values."""
    v = np.asarray(values).ravel()
    v = v[np.isfinite(v)]
    return v


# ======================================================================
# Color / small plotting helpers (used by plot.py)
# ======================================================================

def _normalize_models(model: Optional[Union[Any, List[Any]]]) -> List[Any]:
    """Return list of models (empty list if None)."""
    if model is None:
        return []
    if isinstance(model, list):
        return model
    return [model]


def _normalize_line_colors(lcolor: Union[str, List[str]], n: int) -> List[str]:
    """
    Normalize line colors to match number of models.

    Raises
    ------
    ValueError
        If lcolor is a list with wrong length
    """
    if n == 0:
        return []
    if isinstance(lcolor, list):
        if len(lcolor) != n:
            raise ValueError("If lcolor is a list, its length must match the number of models.")
        return lcolor
    return [lcolor] * n


def _coerce_color_str(c: Union[str, List[Any], np.ndarray], default: str = "blue") -> str:
    """Return c if string color, otherwise return default."""
    return c if isinstance(c, str) else default


def _is_color_like(x: Any) -> bool:
    """Check if a value is a valid matplotlib color."""
    try:
        to_rgba(x)
        return True
    except (ValueError, TypeError):
        return False


def _all_color_like(arr: np.ndarray) -> bool:
    """Check if all elements in array are valid matplotlib colors."""
    try:
        return all(_is_color_like(x) for x in arr)
    except Exception:
        return False

# ======================================================================
# Model diagnostics helpers (used by plot.py / hist.py / qq.py)
# ======================================================================

def _detect_model_type(model: Any) -> str:
    """
    Detect model type for appropriate residual handling.

    Returns
    -------
    str
        One of: "linear", "logistic", "glm", "other"
    """
    cls = model.__class__.__name__.lower()

    fam = None
    if hasattr(model, "family"):
        fam = model.family
    elif hasattr(model, "model") and hasattr(model.model, "family"):
        fam = model.model.family

    if fam is not None:
        fam_name = str(fam).lower()
        if "binomial" in fam_name:
            return "logistic"
        return "glm"

    if "logit" in cls or "binaryresults" in cls:
        return "logistic"

    if any(k in cls for k in ["ols", "wls", "gls", "regressionresults"]):
        return "linear"

    return "other"


def _get_residuals(model: Any, res_type: str) -> np.ndarray:
    """
    Extract residuals from fitted model.

    Parameters
    ----------
    model : Any
        Fitted model object
    res_type : str
        Type of residuals: "resid", "pearson", "deviance", "anscombe",
        "response", "studentized"

    Returns
    -------
    np.ndarray
        Residual array
    """
    res_type = str(res_type).lower().strip()

    if res_type == "resid":
        if hasattr(model, "resid"):
            return np.asarray(model.resid)
        raise AttributeError("Model does not expose `.resid` attribute.")

    mapping = {
        "pearson": "resid_pearson",
        "deviance": "resid_deviance",
        "anscombe": "resid_anscombe",
        "response": "resid_response",
    }

    if res_type in mapping:
        attr = mapping[res_type]
        if hasattr(model, attr):
            return np.asarray(getattr(model, attr))
        if hasattr(model, "model") and hasattr(model.model, attr):
            return np.asarray(getattr(model.model, attr))
        raise AttributeError(
            f"{res_type} residuals not available for this model. Model must have `.{attr}` attribute."
        )

    if res_type == "studentized":
        try:
            infl = model.get_influence()
        except AttributeError as e:
            raise AttributeError(
                "Model does not have `get_influence()` method. Studentized residuals require influence diagnostics."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Could not compute influence diagnostics: {e}") from e

        if hasattr(infl, "resid_studentized_external"):
            return np.asarray(infl.resid_studentized_external)
        if hasattr(infl, "resid_studentized_internal"):
            return np.asarray(infl.resid_studentized_internal)
        raise AttributeError("Influence object does not have studentized residual attributes.")

    raise ValueError(
        f"Invalid residual type '{res_type}'. "
        "Valid types: resid, pearson, deviance, anscombe, response, studentized."
    )

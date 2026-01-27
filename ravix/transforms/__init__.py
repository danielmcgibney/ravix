"""
Data transformations.

This submodule contains functions that modify the response or predictors
to address modeling issues (e.g., non-normality, non-constant variance).
"""

from .box_cox import box_cox

__all__ = [
    "box_cox",
]

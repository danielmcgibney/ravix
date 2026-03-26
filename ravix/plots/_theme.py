"""
ravix.plots._theme
==================
Lightweight theme system for Ravix plots.

Usage
-----
>>> from ravix.plots._theme import set_theme, get_theme, theme_context

>>> set_theme("print")          # switch globally
>>> set_theme("colab")          # back to default
>>> with theme_context("print"):
...     plot("y ~ x", data=df)  # only this call uses "print"

Built-in themes
---------------
"colab"   (default) – compact sizes tuned for Google Colab inline output
"print"             – larger sizes for PDF / LaTeX / textbook figures
"default"           – matplotlib defaults; no size overrides, no figsize override
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Tuple

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ThemeDict = Dict[str, Any]

# ---------------------------------------------------------------------------
# Built-in theme definitions
# ---------------------------------------------------------------------------
# `figsize` is the *fallback* used when the caller passes figsize=None.
# Setting it to None means "use matplotlib's rcParams['figure.figsize']".
# Font size keys accept None, which propagates to matplotlib (uses rc default).

_THEMES: Dict[str, ThemeDict] = {
    "colab": {
        "title_fontsize": 10,
        "label_fontsize": 10,
        "tick_fontsize":  9,
        "diag_fontsize":  10,
        "figsize":        (7, 5),
    },
    "print": {
        "title_fontsize": 14,
        "label_fontsize": 12,
        "tick_fontsize":  11,
        "diag_fontsize":  12,
        "figsize":        (8, 6),
    },
    "default": {
        "title_fontsize": None,
        "label_fontsize": None,
        "tick_fontsize":  None,
        "diag_fontsize":  None,
        "figsize":        None,   # matplotlib's own default (6.4 x 4.8)
    },
}

# ---------------------------------------------------------------------------
# Active theme state
# ---------------------------------------------------------------------------
_active_theme: str = "colab"


def set_theme(name: str) -> None:
    """
    Set the global Ravix theme.

    Parameters
    ----------
    name : str
        One of ``"colab"`` (default), ``"print"``, ``"default"``, or the name
        of a theme registered via :func:`register_theme`.

    Raises
    ------
    ValueError
        If *name* is not a recognised theme.
    """
    global _active_theme
    _validate_name(name)
    _active_theme = name


def get_theme() -> ThemeDict:
    """Return a copy of the currently active theme dict."""
    return dict(_THEMES[_active_theme])


def register_theme(name: str, theme: ThemeDict) -> None:
    """
    Register a custom theme.

    Parameters
    ----------
    name : str
        Identifier for the new theme.
    theme : dict
        A dict with any subset of the keys used by built-in themes.
        Missing keys are filled in from the ``"default"`` theme.

    Examples
    --------
    >>> register_theme("slides", {"title_fontsize": 18, "label_fontsize": 14,
    ...                            "tick_fontsize": 12, "diag_fontsize": 14,
    ...                            "figsize": (12, 8)})
    >>> set_theme("slides")
    """
    base = dict(_THEMES["default"])
    base.update(theme)
    _THEMES[name] = base


@contextmanager
def theme_context(name: str) -> Generator[None, None, None]:
    """
    Context manager: temporarily switch to *name* for the duration of the
    ``with`` block, then restore the previous theme.

    Examples
    --------
    >>> with theme_context("print"):
    ...     plot("y ~ x", data=df)   # uses "print" theme
    >>> # back to whatever was active before
    """
    _validate_name(name)
    global _active_theme
    previous = _active_theme
    _active_theme = name
    try:
        yield
    finally:
        _active_theme = previous


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_figsize(figsize: Optional[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Return *figsize* if provided, otherwise fall back to the theme's figsize,
    then to ``matplotlib.rcParams['figure.figsize']``.
    """
    if figsize is not None:
        return figsize
    theme_size = get_theme().get("figsize")
    if theme_size is not None:
        return theme_size
    import matplotlib.pyplot as plt
    return tuple(plt.rcParams["figure.figsize"])  # type: ignore[return-value]


def _validate_name(name: str) -> None:
    if name not in _THEMES:
        known = ", ".join(f'"{k}"' for k in _THEMES)
        raise ValueError(
            f"Unknown theme {name!r}. "
            f"Available themes: {known}. "
            "Use register_theme() to add a custom theme."
        )

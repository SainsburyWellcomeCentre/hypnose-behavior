"""Figure destinations for the behavioural derivatives tree, plus the shared
styling/saving re-exported from hypnose-helpers.

What lives here is the part hypnose-helpers cannot know: which *scope* of figure
belongs at which level of the tree.
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

from hypnose_helpers.viz.styles import (  # noqa: F401  (re-exported for existing callers)
    nature_style, poster_style, presentation_style, use_style, use_presentation_style,
    nice_x_locator, _presentation_active, _resolve_style,
)
from hypnose_helpers.viz.save import (  # noqa: F401
    set_size, strip_legends, _coerce_list, _unique_sorted, _format_span,
)
from hypnose_helpers.viz.save import save_figure as _save_figure
from hypnose_helpers.viz.metadata import read_figure_metadata  # noqa: F401  (re-exported)
from hypnose_helpers.provenance import provenance as _provenance


# This module deliberately applies NO style at import: two packages mutating global
# rcParams at module scope means whoever imports last silently wins. Apply a style
# explicitly at the top of a notebook or script:
#
#     use_style()                  # nature (the default)
#     use_style("presentation")    # presentation (also caps y-ticks)
#     with plt.rc_context(nature_style()): ...   # scoped, as scripts/modelling does
#
# `pdf.fonttype`/`ps.fonttype` = 42 (editable PDF text rather than Type 3) is part of
# every style dict, and save_figure enforces it regardless, so saved PDFs are safe even
# when no style has been applied.


# Optional hook letting a *consuming* repo with a different derivatives layout
# reuse save_figure without wrapping it. Registered once at import; save_figure
# then resolves through it instead of resolve_figure_dir(). None = use the
# default hypnose-behavior-analysis layout below.
_FIGURE_DIR_RESOLVER = None

# Subdirectory for figures drawn from SLEAP tracking. Lives here rather than in
# `visualization/movement/` so the four modules using it share a leaf instead of
# importing a peer; they already import `save_figure` from here, so it costs no edge.
MOVEMENT_FIGURES_SUBDIR = "movement_figures"


def resolve_figure_dir(subjids, dates=None) -> Path:
    """Determine where to save figures based on subject/session scope.

    Rules:
    - Multiple subjects: figures at derivatives_root / "figures".
    - Single subject, multiple sessions: figures at subject_dir / "figures".
    - Single subject, single session: figures at session_dir / "figures".
    """
    # Imported here rather than at module scope: paths.py requires Python 3.10+,
    # and consumers that supply their own figure directory (see
    # set_figure_dir_resolver) never reach this function.
    from hypnose_behavior.io.layout import derivatives

    subj_list = _coerce_list(subjids)
    date_list = _coerce_list(dates)

    if len(subj_list) == 0:
        raise ValueError("At least one subjid is required to resolve figure path")

    if len(subj_list) > 1:
        fig_dir = derivatives.root / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        return fig_dir

    # Single subject
    subj_dir = derivatives.subject_dir(subj_list[0])

    if len(date_list) == 1:
        fig_dir = derivatives.find_session(subj_list[0], date=date_list[0]).path / "figures"
    else:
        fig_dir = subj_dir / "figures"

    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save_figure(fig, save_name: str, *, subjids, dates=None, subdir=None,
                fig_dir=None, provenance=None, **kwargs):
    """Save a figure into the behavioural derivatives tree.

    Resolves the destination, then delegates to `hypnose_helpers.viz.save.save_figure`.
    Directory resolution, most specific first: an explicit `fig_dir` wins; then a resolver
    registered by a consuming repo; otherwise this repo's subject/session layout.

    - This is a `save_figure` wrapper, so it MUST pass `skip_modules=(__name__,)`.
      Without it the provenance walk stops at this frame and the record names this
      function instead of the plotter that called it. See DECISIONS.md section 9.
    """
    if fig_dir is None:
        if _FIGURE_DIR_RESOLVER is not None:
            fig_dir = _FIGURE_DIR_RESOLVER(subjids, dates)
        else:
            fig_dir = resolve_figure_dir(subjids, dates)
    if provenance is None:
        provenance = _provenance(skip_modules=(__name__,))
    return _save_figure(fig, save_name, fig_dir=fig_dir, subjids=subjids, dates=dates,
                        subdir=subdir, provenance=provenance, **kwargs)

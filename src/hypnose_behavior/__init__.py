"""Behavioural analysis for the Hypnose task.

**The public surface is `hypnose_behavior.api`.** It is one explicit,
hand-maintained module naming what the other repos in the family may use; read its
docstring before importing anything else from here.

    from hypnose_behavior.api import session
    s = session(57, 20260709)

The four entry points below are also reachable straight off the package, because
one line is the right cost for the commonest thing anyone does:

    import hypnose_behavior
    s = hypnose_behavior.session(57, 20260709)

### This file imports nothing, and that is load-bearing

**No eager re-exports here or in any package `__init__.py`** (follow-up item 2;
`docs/DECISIONS.md` sections 3 and 31). They are what keep `frames.py` and
`parameters.py` importable as leaves: an eager import here would make
`import hypnose_behavior.frames` pull matplotlib, harp, aeon and dotmap, paid by every
downstream repo including the ones pinned to Python 3.9 that `frames.py` is kept
importable for.

So the four names are forwarded lazily, through PEP 562's module `__getattr__`, which
runs only on **attribute access** for a name not already bound. `import
hypnose_behavior` and `import hypnose_behavior.frames` therefore cost exactly what they
cost before this file had a body -- measured, 39 and 614 modules -- and
`hypnose_behavior.session` pays for the analysis stack at the moment it is used.

Four names, not everything `api` exports. The package root is a shortcut to the handle,
not a second copy of the API surface -- two spellings of one thing is how the two come
to disagree (section 27).

**A name not on the list raises `AttributeError`**, which is what keeps
`from hypnose_behavior import frames` working: Python asks for the attribute first and
falls back to importing the submodule only when that raises.
"""

_FORWARDED_TO_API = frozenset({"session", "sessions", "Session", "metric_names"})


def __getattr__(name):
    if name in _FORWARDED_TO_API:
        from hypnose_behavior import api
        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}. The public "
                         f"surface is hypnose_behavior.api.")


def __dir__():
    return sorted(set(globals()) | _FORWARDED_TO_API)

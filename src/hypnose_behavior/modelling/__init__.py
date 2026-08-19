"""Behavioural sequence modelling.

Pure numeric model code fitted to binary trial sequences: numpy arrays in, numpy arrays
and plain dicts out. No file I/O, no plotting, no path handling -- those belong in the
calling script (see ``scripts/modelling/``) and, for figures, in
``hypnose_behavior.visualization.modelling``.

Each analysis is one subpackage. The first is ``switchpoint`` -- the LONG -> SHORT strategy
change (see ``hypnose_behavior.modelling.switchpoint``).
"""

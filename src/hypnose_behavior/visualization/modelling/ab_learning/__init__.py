"""Figures for the A/B learning analysis (see ``hypnose_behavior.modelling.ab_learning``).

One module per analysis block, mirroring the modelling package:

- ``diagnostics``    -- initiation rate, accuracy after failed attempts, lose-shift split.
- ``log_regression`` -- the fitted per-session log-odds and the within/overnight gains.

``_common`` holds what they share: the loader guard, the figure and axis setup, the
session ticks, the title and the colour slots.
"""

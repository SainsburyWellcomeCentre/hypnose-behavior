"""Figures for the A/B learning analysis (see ``hypnose_behavior.modelling.ab_learning``).

One module per analysis block, mirroring the modelling package:

- ``diagnostics``    -- initiation rate, accuracy after failed attempts, lose-shift split.
- ``log_regression`` -- the fitted per-session log-odds and the within/overnight gains.
- ``session_gain``   -- first/last-window gains and accuracy by position in a session.
- ``cumulative``     -- running counts on task time, and excess correct by row index.
- ``state_space``    -- the random walk's p_k and expected excess over the data, with the
  learning trial and the counted changes.
- ``sigmoid``        -- the sigmoid fits, their lnL surface, and their midpoints on task
  time.
- ``step``           -- the step fit on the data and its posterior over the switch trial.

``_common`` holds what they share: the loader guard, the figure and axis setup, the
session ticks, the reference lines and the colour slots.
"""

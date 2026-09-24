"""Sudden vs. gradual learning of the A/B odour discrimination.

One module per role:

- ``data``        -- the per-animal trial and failed-attempt frames, odour-discrimination
  runs only, and the choice frame the models are fitted to.
- ``diagnostics`` -- whether completed trials alone describe the learning: initiation
  rate, accuracy on failed attempts, and the lose-shift split of completed trials.
- ``log_regression`` -- logistic regression of accuracy on session and on position within
  a session, and the split of the total change into within-session and overnight parts.
- ``session_gain`` -- raw-accuracy gains within and across sessions, and accuracy by
  position in a session.
- ``cumulative``  -- running counts of initiations, excess correct and rewards.
- ``sigmoid``     -- single-sigmoid fits of accuracy, engagement and reward rate, with
  every separated optimum.

Figures live in ``hypnose_behavior.visualization.modelling.ab_learning``.
"""

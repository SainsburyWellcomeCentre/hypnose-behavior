"""Metric definitions, grouped by behavioural construct.

`metric_analysis` is the single definition site for every metric in the package.

    accuracy      correct choices, response rate, the rolling reward fraction
    false_alarm   every FA-labelled quantity, incl. FA port bias and latency
    sequence      completion, abortion by odor and by position
    hidden_rule   hidden-rule performance, detection, and the HR split
    sampling      how long the animal spent at each odor port
    timing        response and reward latencies
    common        the predicates, rate reduction and frame slicing they share

plus `../movement/` and `../sing_rew_metrics.py`.

- **Grouping is by construct, not by frame.** Which frame a metric consumes is a
  decorator argument, so `fa_latency_from_pokeout` sits with the false alarms it
  measures rather than with the other latencies.
- Every metric is a pure `f(frame) -> value` **core** plus a thin
  `*_session(results)` **wrapper** that prints and returns the same value. `run.py`
  calls the wrappers; another granularity calls the core through
  `resolvers.by_group` / `over_windows`. See DECISIONS.md section 1.
- A metric registered but absent from `run.REPORT` is **not** in the metrics
  fingerprint. Check `REPORT` before assuming a metric change is gated.
"""

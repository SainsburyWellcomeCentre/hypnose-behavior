# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Rendering a metrics dict as text.

Mirrors ``trial_classification/summary.py``. Formatting is a property of the report,
not of the metric, so it lives apart from the definitions.
"""

import pandas as pd

__all__ = ["save_merged_metrics_txt", "format_fa_abortion_tables"]


def save_merged_metrics_txt(metrics, header, txt_path, pretty_print_str=None):
    """
    Save merged metrics to a txt file with a header and formatted output.
    """
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(header + "\n\n")
        if pretty_print_str is not None:
            f.write(pretty_print_str)
        else: 
            for k, v in metrics.items():
                if isinstance(v, (tuple, list)) and len(v) == 3 and isinstance(v[0], (int, float)):
                    # Standard metric: numerator, denominator, value
                    num, denom, val = v
                    f.write(f"{k.replace('_',' ').title()}: {num}/{denom} = {val:.3f}\n")
                elif isinstance(v, dict) and "bias" in v and "n_fa" in v and "n_ab" in v:
                    # FA Odor Bias
                    f.write(f"{k.replace('_',' ').title()}:\n")
                    for od in v["bias"]:
                        bias = v["bias"][od]
                        n_fa = v["n_fa"].get(od, 0)
                        n_ab = v["n_ab"].get(od, 0)
                        f.write(f"  {od}: {n_fa}/{n_ab} FA, Bias: {bias:.3f}\n")
                elif isinstance(v, dict):
                    f.write(f"{k.replace('_',' ').title()}:\n")
                    for subk, subv in v.items():
                        if isinstance(subv, float):
                            f.write(f"  {subk}: {subv:.3f}\n")
                        else:
                            f.write(f"  {subk}: {subv}\n")
                elif isinstance(v, pd.Series):
                    f.write(f"{k.replace('_',' ').title()}:\n")
                    for idx, val in v.items():
                        f.write(f"  {idx}: {val:.3f}\n")
                elif isinstance(v, float):
                    f.write(f"{k.replace('_',' ').title()}: {v:.3f}\n")
                else:
                    f.write(f"{k.replace('_',' ').title()}: {v}\n")


# ---- fa_abortion_stats: numbers in, readable tables out ------------------------
#
# `fa_abortion_stats` returns numbers -- counts as `int`, rates as `float`, positions
# as `int` -- and all formatting happens here, so changing how the report reads cannot
# change what was measured.

# Everything else in those tables is a count rendered as "n (fraction of the
# row's abortions)". Deriving the subtype columns rather than naming them keeps
# this in step with the metric: add an FA subtype there and it renders here.
_FA_STRUCTURAL = ("Odor", "Position", "Total Abortions", "Reached Trials",
                  "Abortion Rate", "FA Abortions", "FA Abortion Rate")


def _fa_shared_columns(frame):
    total = frame["Total Abortions"].to_numpy()
    cols = {
        "Total Abortions": total,
        "FA Abortion Rate": [f"{n}/{d} ({n / d:.2f})"
                             for n, d in zip(frame["FA Abortions"].to_numpy(), total)],
    }
    for col in frame.columns:
        if col in _FA_STRUCTURAL:
            continue
        cols[col] = [f"{c} ({c / d:.2f})"
                     for c, d in zip(frame[col].to_numpy(), total)]
    return cols


def _format_fa_table(frame, keys):
    if frame.empty:
        return frame
    return pd.DataFrame({**{k: frame[k].to_numpy() for k in keys},
                         **_fa_shared_columns(frame)})


def _format_fa_position_table(frame):
    if frame.empty:
        return frame
    total = frame["Total Abortions"].to_numpy()
    reached = frame["Reached Trials"].to_numpy()
    rate = frame["Abortion Rate"].to_numpy()
    cols = {
        "Position": frame["Position"].to_numpy(),
        "Total Abortions": total,
        "Reached Trials": reached,
        "Abortion Rate": [f"{n}/{d} ({v:.2f})" if d > 0 else "N/A"
                          for n, d, v in zip(total, reached, rate)],
        # Report-only: the same number as "Abortion Rate", which the metric returns
        # once rather than twice.
        "Abortion Rate Value": rate,
    }
    cols.update({k: v for k, v in _fa_shared_columns(frame).items()
                 if k != "Total Abortions"})
    return pd.DataFrame(cols)


def format_fa_abortion_tables(df_odor, df_pos, df_out):
    """Render `fa_abortion_stats`' three numeric tables for a text report.

    Column names and order are what the report has always had, so the txt is
    unchanged except that positions print as `2` rather than `2.0` -- they are
    integers now.
    """
    return (_format_fa_table(df_odor, ["Odor"]),
            _format_fa_position_table(df_pos),
            _format_fa_table(df_out, ["Odor", "Position"]))

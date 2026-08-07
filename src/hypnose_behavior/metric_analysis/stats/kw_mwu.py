# Defers evaluation of PEP-604 annotations, keeping this module importable on
# Python 3.9 for repos pinned there.
from __future__ import annotations

"""Kruskal-Wallis across groups, then Holm-corrected pairwise Mann-Whitney U.

One module per test, per judgement call 6 of the Phase 4a metric audit: a single
`stats.py` accumulates unrelated procedures and their assumptions, and the next
test added is what makes it a grab bag.

It lived inside `plot_movement_analysis_statistics` in `visualization/`, which
made it unreachable from anywhere else and invisible to anything that wanted to
report the same comparison. Nothing about it is specific to movement data.
"""

import numpy as np
from scipy.stats import kruskal, mannwhitneyu

__all__ = ["kw_mwu_by_group"]


def kw_mwu_by_group(df, value_col, group_col="condition", min_pair_n=5):
    """Run Kruskal-Wallis across groups, then pairwise Mann-Whitney U with Holm correction if KW is significant.

    Returns dict with keys:
      - kruskal: {"stat": H, "p": p, "n_per_group": {...}, "groups": [...]} or None
      - pairwise: list of {g1, g2, n1, n2, u_stat, p_raw, p_corr} (only if KW significant and n>=min_pair_n in both).
    """
    if df is None or df.empty or value_col not in df.columns or group_col not in df.columns:
        return {"kruskal": None, "pairwise": []}

    clean_df = df[[group_col, value_col]].dropna()
    clean_df = clean_df[np.isfinite(clean_df[value_col].astype(float))]
    if clean_df.empty:
        return {"kruskal": None, "pairwise": []}

    groups = {}
    for g, sub in clean_df.groupby(group_col):
        vals = sub[value_col].astype(float).to_numpy()
        if vals.size > 0:
            groups[g] = vals

    if len(groups) < 2:
        return {"kruskal": None, "pairwise": []}

    # Kruskal-Wallis
    try:
        kw_stat, kw_p = kruskal(*groups.values())
    except Exception:
        return {"kruskal": None, "pairwise": []}

    kruskal_res = {
        "stat": float(kw_stat),
        "p": float(kw_p),
        "n_per_group": {k: int(len(v)) for k, v in groups.items()},
        "groups": list(groups.keys()),
    }

    # Pairwise only if significant
    pairwise = []
    if kw_p < 0.05:
        pairs = [("rewarded", "unrewarded"), ("rewarded", "fa"), ("unrewarded", "fa")]
        raw_ps = []
        stats_tmp = []
        for g1, g2 in pairs:
            v1 = groups.get(g1)
            v2 = groups.get(g2)
            n1 = len(v1) if v1 is not None else 0
            n2 = len(v2) if v2 is not None else 0
            if v1 is None or v2 is None or n1 < min_pair_n or n2 < min_pair_n:
                continue
            try:
                u_stat, p_raw = mannwhitneyu(v1, v2, alternative="two-sided")
            except Exception:
                continue
            raw_ps.append(p_raw)
            stats_tmp.append({"g1": g1, "g2": g2, "n1": n1, "n2": n2, "u_stat": float(u_stat), "p_raw": float(p_raw)})

        # Holm-Bonferroni on the collected raw p-values
        m = len(raw_ps)
        if m > 0:
            order = np.argsort(raw_ps)
            adjusted = np.empty(m)
            max_adj = 0.0
            for rank, idx in enumerate(order):
                adj = raw_ps[idx] * (m - rank)
                adj = min(adj, 1.0)
                max_adj = max(max_adj, adj) # this should enfore monotonicity, as each p_corr should be >= the previous one
                adjusted[idx] = max_adj
            # map back
            for i, entry in enumerate(stats_tmp):
                entry["p_corr"] = float(adjusted[i])
                pairwise.append(entry)

    return {"kruskal": kruskal_res, "pairwise": pairwise}

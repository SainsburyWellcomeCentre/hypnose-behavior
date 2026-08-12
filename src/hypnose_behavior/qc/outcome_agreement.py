"""How often do the three outcome derivations actually disagree?

rewarded / unrewarded / timeout is derived three times in this codebase, by code that shares
nothing:

  A  ``classify_trials``        -- which ``completed_sequence_*`` frame a trial is appended to,
                                   from supply pulses and reward pokes in protocol-specific
                                   windows.
  B  ``analyze_response_times`` -- the ``response_time_category`` column, from supply pulses and
                                   reward pokes in *its own* windows, and additionally requiring
                                   that a response time could be computed at all.
  C  ``save_results._derive_outcome`` -- re-derived from the saved ``total_supply_count`` /
                                   ``total_reward_pokes`` / ``await_reward_time`` columns.

Phase 6a's brief proposes merging them. "They share no code and can drift" is a hypothesis
about drift, not a measurement -- and ``DECISIONS.md`` section 13 records what happened the last
time look-alike helpers were merged without checking: five of seven were different rules wearing
the same name, one pair disagreeing on 63.8% of trials.

This script measures the three against each other, per trial, on the regression sessions.

    python src/hypnose_behavior/qc/outcome_agreement.py
    python src/hypnose_behavior/qc/outcome_agreement.py 061:20260729

It reports two different things, and the distinction is the point:

  **conflict** -- both rules named a category and the categories differ. A merge would change a
                  value, and which rule is right is a scientific question.
  **coverage** -- one rule named a category and the other returned nothing. A merge would fill a
                  gap rather than overwrite a decision.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from hypnose_behavior.qc import _common  # noqa: E402
from hypnose_behavior.io.save_results import _derive_outcome  # noqa: E402

# A calls it 'timeout'; B and C call the same thing 'timeout_delayed'. Compare on a common
# vocabulary so a pure naming difference is not counted as a disagreement.
CANON = {
    'timeout': 'timeout',
    'timeout_delayed': 'timeout',
    'rewarded': 'rewarded',
    'unrewarded': 'unrewarded',
    'false_response': 'false_response',
}


def _canon(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return CANON.get(str(value), str(value))


def _key(row):
    """Trials are numbered per run, so a trial is identified by (run_id, trial_id)."""
    run = row.get('run_id')
    return (1 if pd.isna(run) else int(run), row.get('trial_id'))


def _classify_trials_outcomes(classification) -> dict:
    """A: (run_id, trial_id) -> outcome, from which completed_sequence_* frame a trial is in."""
    frames = {
        'rewarded': 'completed_sequence_rewarded',
        'unrewarded': 'completed_sequence_unrewarded',
        'timeout': 'completed_sequence_reward_timeout',
        'false_response': 'completed_sequence_false_response',
    }
    out = {}
    for label, key in frames.items():
        df = classification.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty and 'trial_id' in df.columns:
            for _, row in df.iterrows():
                out[_key(row)] = label
    return out


def _response_time_outcomes(classification) -> dict:
    """B: (run_id, trial_id) -> response_time_category as analyze_response_times produced it.

    Read from ``completed_sequences_with_response_times``, which is the per-trial table merged
    onto the completed frame and *before* ``_derive_outcome`` overwrites anything.
    """
    df = classification.get('completed_sequences_with_response_times')
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    if 'trial_id' not in df.columns or 'response_time_category' not in df.columns:
        return {}
    return {_key(row): row.get('response_time_category') for _, row in df.iterrows()}


def measure_session(subjid, date) -> pd.DataFrame:
    """Run one session and return its per-trial A/B/C outcomes with the deciding evidence."""
    from hypnose_behavior.trial_classification.run import analyze_session_multi_run_by_id_date

    with tempfile.TemporaryDirectory(prefix="hyp_outcome_") as tmp:
        _common._redirect_derivatives(Path(tmp))
        with contextlib.redirect_stdout(io.StringIO()):
            payload = analyze_session_multi_run_by_id_date(
                str(subjid), str(date), verbose=False, save=True, print_summary=False,
                save_csv=True)   # reads trial_data.csv below; ask for it explicitly

        matches = list(Path(tmp).glob(f"**/ses-*_date-{date}/saved_analysis_results/trial_data.csv"))
        if not matches:
            raise FileNotFoundError(f"trial_data.csv not found for sub-{subjid} {date}")
        trial_data = pd.read_csv(matches[0])

    classification = payload.get('classification') or {}
    a_map = _classify_trials_outcomes(classification)
    b_map = _response_time_outcomes(classification)

    rows = []
    for _, row in trial_data.iterrows():
        k = _key(row)
        rows.append({
            'run_id': k[0],
            'trial_id': k[1],
            'A_classify_trials': _canon(a_map.get(k)),
            'B_response_times': _canon(b_map.get(k)),
            'C_derive_outcome': _canon(_derive_outcome(row)),
            'is_completed': k in a_map,
            'sequence_rewarded': row.get('sequence_rewarded'),
            'total_supply_count': row.get('total_supply_count'),
            'total_reward_pokes': row.get('total_reward_pokes'),
        })
    return pd.DataFrame(rows)


PAIRS = [('A_classify_trials', 'B_response_times'),
         ('A_classify_trials', 'C_derive_outcome'),
         ('B_response_times', 'C_derive_outcome')]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="limit to these 'subjid:date' keys")
    ap.add_argument("--examples", type=int, default=6, help="conflicting trials to print per pair")
    args = ap.parse_args()

    import yaml
    sessions = yaml.safe_load((_HERE.parent / "sessions.yml").read_text())["sessions"]
    if args.sessions:
        targets = set(args.sessions)
        sessions = [s for s in sessions
                    if f"{str(s['subjid']).zfill(3)}:{s['date']}" in targets
                    or f"{s['subjid']}:{s['date']}" in targets]

    print(f"Measuring outcome agreement across {len(sessions)} session(s)\n")
    print("  A = classify_trials (completed_sequence_* frames)")
    print("  B = analyze_response_times (response_time_category)")
    print("  C = save_results._derive_outcome (supply/poke counts)\n")

    all_rows = []
    for s in sessions:
        subjid, date, label = s["subjid"], s["date"], s.get("label", "")
        try:
            df = measure_session(subjid, date)
        except Exception as e:
            print(f"  [ERROR] sub-{subjid} {date} ({label}): {type(e).__name__}: {e}")
            continue
        df['session'] = f"sub-{subjid}_{date}"
        all_rows.append(df)
        n_completed = int(df['is_completed'].sum())
        conflicts = {}
        for x, y in PAIRS:
            both = df[df[x].notna() & df[y].notna()]
            conflicts[f"{x[0]}v{y[0]}"] = int((both[x] != both[y]).sum())
        print(f"  sub-{subjid} {date} ({label}): {len(df)} trials, {n_completed} completed"
              f" -- conflicts " + ", ".join(f"{k}={v}" for k, v in conflicts.items()))

    if not all_rows:
        print("\nNo sessions measured.")
        return 1

    df = pd.concat(all_rows, ignore_index=True)
    total = len(df)
    completed = int(df['is_completed'].sum())

    print("\n" + "=" * 78)
    print(f"TOTALS: {total} trials across {df['session'].nunique()} sessions "
          f"({completed} completed by classify_trials)")
    print("=" * 78)

    grand_conflicts = 0
    for x, y in PAIRS:
        both = df[df[x].notna() & df[y].notna()]
        disagree = both[both[x] != both[y]]
        grand_conflicts += len(disagree)
        only_x = int((df[x].notna() & df[y].isna()).sum())
        only_y = int((df[x].isna() & df[y].notna()).sum())

        pct = (len(disagree) / len(both) * 100.0) if len(both) else 0.0
        print(f"\n{x} vs {y}")
        print(f"  both defined : {len(both)}")
        print(f"  CONFLICT     : {len(disagree)} ({pct:.2f}% of jointly-defined)")
        print(f"  coverage     : {x} only = {only_x}, {y} only = {only_y}")

        if len(disagree):
            print("  conflict shapes:")
            for (a, b), n in Counter(zip(disagree[x], disagree[y])).most_common():
                print(f"    {a!s:>16} -> {b!s:<16} {n}")
            print(f"  first {args.examples} conflicting trials:")
            for _, r in disagree.head(args.examples).iterrows():
                print(f"    {r['session']} run{r['run_id']} trial {r['trial_id']}: "
                      f"{x}={r[x]} {y}={r[y]} "
                      f"(supply={r['total_supply_count']}, pokes={r['total_reward_pokes']}, "
                      f"seq_rewarded={r['sequence_rewarded']})")

        if only_x or only_y:
            print("  coverage shapes:")
            for col, other in ((x, y), (y, x)):
                gap = df[df[col].notna() & df[other].isna()]
                if len(gap):
                    counts = Counter(gap[col])
                    inner = ", ".join(f"{k}={v}" for k, v in counts.most_common())
                    print(f"    {col} defined where {other} is null: {inner}")

    print("\n" + "=" * 78)
    if grand_conflicts == 0:
        print("VERDICT: no conflicts. Where two rules both name a category, they always agree.")
        print("         Any merge changes coverage only -- see the coverage lines above.")
    else:
        print(f"VERDICT: {grand_conflicts} conflicting trial(s). A merge WOULD change values.")
        print("         Which rule is correct is a scientific decision, not a refactoring one.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

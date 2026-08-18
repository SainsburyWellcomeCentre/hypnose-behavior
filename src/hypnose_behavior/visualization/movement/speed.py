# Defers evaluation of PEP-604 annotations (`X | None`), keeping this module
# importable on Python 3.9 for repos pinned there (hypnose-eeg-preprocessing).
from __future__ import annotations

"""Cue-port speed figures from SLEAP tracking.

Carved out of ``movement_analysis_utils.py`` in restructure_2 Phase 10
(follow-up Item 1). Source-only move -- no behaviour change.

Both **read** ``speed_analysis.parquet``, written by
``metric_analysis.movement.speed_analysis.compute_speed_analysis``. Neither
computes it: a session without that file is reported and skipped. The threshold
has one definition -- ``metric_analysis.movement.speed_analysis.speed_threshold``
-- because a plotted threshold that disagrees with the one used to compute the
saved latencies is invisible in any output (audit finding 7).
"""

import pandas as pd
import matplotlib.pyplot as plt
from hypnose_behavior.frames import position_entries_by_trial
from hypnose_behavior.io.paths import get_derivatives_root
from hypnose_behavior.utils.helpers import (
    _filter_session_dirs,
    _get_from_cache,
    _update_cache,
    session_selectors,
)
from hypnose_behavior.io.layout import (
    derivatives,
    normalize_subjid,
)
from hypnose_behavior.io.loaders import (
    _load_position_data,
    _load_trial_views,
)
from hypnose_behavior.visualization.prep import smooth_xy
from hypnose_behavior.io.tracking import _load_tracking_and_behavior
from hypnose_behavior.metric_analysis.movement.speed_analysis import speed_threshold
from hypnose_behavior.io.save import save_figure
import re
import numpy as np
from hypnose_behavior.io.save import MOVEMENT_FIGURES_SUBDIR



def plot_epoch_speeds_by_condition(
    subjid,
    dates=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    bin_ms: int = 100,
    fa_label_filter=None,
    mode: str = "mean",
    threshold: bool = True,
    threshold_alpha: float = 10.0,
    threshold_beta: float = 10.0,
    figsize=(8, 5),
    save: bool = False,
    verbose: bool = True,
    return_paths: bool = False,
):
    """Plot cue-port speed epochs from precomputed speed_analysis.parquet.

    Uses outputs from compute_speed_analysis (same parameters) to build per-session, per-condition
    per-trial traces with session mean overlay and optional threshold lines. Violin plots are omitted.
    Figures can optionally be saved into the movement_figures subdirectory when `save=True`.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )

    saved_paths = []

    def _slugify(value: str) -> str:
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_")
        return slug.lower() or "figure"

    def _save_fig(fig, save_name: str, date_scope):
        if not save:
            return
        try:
            out_path = save_figure(
                fig,
                save_name,
                subjids=[subjid],
                dates=date_scope,
                subdir=MOVEMENT_FIGURES_SUBDIR,
            )
            saved_paths.append(out_path)
            if verbose:
                print(f"[plot_epoch_speeds_by_condition] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_epoch_speeds_by_condition] Failed to save figure '{save_name}': {exc}")

    if mode not in {"max", "mean"}:
        raise ValueError("mode must be 'max' or 'mean'")

    # Normalize FA labels: accept comma-separated string or any iterable of labels (used at compute time)
    if fa_label_filter is None:
        fa_labels = {"fa_time_in"}
    elif isinstance(fa_label_filter, str):
        parts = re.split(r"[;,]", fa_label_filter)
        fa_labels = {p.strip().lower() for p in parts if p.strip()}
    else:
        try:
            fa_labels = {str(s).strip().lower() for s in fa_label_filter if str(s).strip()}
        except TypeError:
            fa_labels = {str(fa_label_filter).strip().lower()}

    subj_str = normalize_subjid(subjid)
    derivatives_dir = get_derivatives_root()
    subj_dir = derivatives.subject_dir(subjid)

    ses_dirs = _filter_session_dirs(subj_dir, dates, **select)
    if not ses_dirs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")

    bin_s = bin_ms / 1000.0
    baseline_window = (-0.15, -0.05)

    per_session = []
    combined_data = {"rewarded": [], "unrewarded": [], "fa": []}

    for ses_dir in ses_dirs:
        date_str = ses_dir.name.split("_date-")[-1]
        results_dir = ses_dir / "saved_analysis_results"
        if not results_dir.exists():
            continue

        df_speed = _get_from_cache(subjid, date_str, kind="speed_analysis")
        if df_speed is None:
            path_speed = results_dir / "speed_analysis.parquet"
            if not path_speed.exists():
                raise FileNotFoundError(f"Missing speed_analysis.parquet for {date_str}; run compute_speed_analysis first")
            df_speed = pd.read_parquet(path_speed)
            _update_cache(subjid, [date_str], {date_str: df_speed.copy()}, kind="speed_analysis")

        df_speed = df_speed.copy()
        conds_with_data = [c for c in ["rewarded", "unrewarded", "fa"] if not df_speed[df_speed["condition"] == c].empty]
        if not conds_with_data:
            continue

        # Baseline stats from stored speeds
        baseline_mask = (df_speed["bin_mid_s"] >= baseline_window[0]) & (df_speed["bin_mid_s"] <= baseline_window[1])
        baseline_vals = df_speed.loc[baseline_mask, "speed"].dropna().to_numpy()
        # One definition of the threshold, shared with compute_speed_analysis --
        # which is what produced the latencies this figure draws (finding 7).
        stats = speed_threshold(baseline_vals, alpha=threshold_alpha,
                                beta=threshold_beta, enabled=threshold)
        baseline_mean, baseline_sd = stats["mu"], stats["sigma"]
        thr_alpha_mu = stats["alpha_mu"]
        thr_mu_plus_beta_sigma = stats["mu_plus_beta_sigma"]
        thr_max = stats["max_alpha_mu_mu_plus_beta_sigma"]

        figs_by_cond = {}
        for cond in conds_with_data:
            sub = df_speed[df_speed["condition"] == cond].copy()
            if sub.empty:
                continue
            # Trial-wise traces
            trials = []
            trial_arrays = []
            mids_all = np.sort(sub["bin_mid_s"].unique())
            fig_t, ax_t = plt.subplots(figsize=figsize)

            for tid, g in sub.groupby("trial_index"):
                g = g.sort_values("bin_mid_s")
                mids = g["bin_mid_s"].to_numpy(float)
                speeds = g["speed"].to_numpy(float)
                if mids.size and speeds.size:
                    ax_t.plot(mids, speeds, color="gray", alpha=0.2)
                trials.append((tid, mids, speeds))

                arr_full = np.full_like(mids_all, np.nan, dtype=float)
                mid_to_idx = {m: i for i, m in enumerate(mids_all)}
                for m, s in zip(mids, speeds):
                    idx = mid_to_idx.get(m)
                    if idx is not None:
                        arr_full[idx] = s
                trial_arrays.append(arr_full)

            if trial_arrays:
                stack = np.vstack(trial_arrays)
                mean_speeds = np.nanmean(stack, axis=0)
                ax_t.plot(mids_all, mean_speeds, color="blue", linewidth=2, label="session mean")

            if threshold and baseline_mean is not None:
                ax_t.axhline(baseline_mean, color="red", linestyle="-", linewidth=1.5, label="baseline μ")
                if thr_max is not None:
                    ax_t.axhline(thr_max, color="#2F4F4F", linestyle="--", linewidth=1.4, label=f"max(αμ, μ+βσ), α={threshold_alpha:g}, β={threshold_beta:g}")

            ax_t.set_title(f"{cond} — sub {subjid}, {date_str} ({mode})")
            ax_t.set_xlabel("Time from last poke-out (s)")
            ax_t.set_ylabel("Speed (units/s)")
            ax_t.legend()
            fig_t.tight_layout()
            figs_by_cond[cond] = fig_t

            date_scope = [int(date_str)] if str(date_str).isdigit() else [date_str]
            save_name = f"epoch_speeds_{_slugify(cond)}_{_slugify(mode)}_{_slugify(date_str)}"
            _save_fig(fig_t, save_name, date_scope)

            if trial_arrays:
                combined_data[cond].append((date_str, mids_all, np.nanmean(np.vstack(trial_arrays), axis=0)))

        per_session.append({
            "date": date_str,
            "fig_traces": figs_by_cond,
            "baseline": {
                "mu": baseline_mean,
                "sigma": baseline_sd,
                "alpha": threshold_alpha,
                "beta": threshold_beta,
                "alpha_mu": thr_alpha_mu,
                "mu_plus_beta_sigma": thr_mu_plus_beta_sigma,
                "max_alpha_mu_mu_plus_beta_sigma": thr_max,
            } if threshold else None,
        })

    combined_figs = {}
    if len(per_session) > 1:
        colors = plt.cm.tab10.colors
        for idx, cond in enumerate(["rewarded", "unrewarded", "fa"]):
            if not combined_data[cond]:
                continue
            fig, ax = plt.subplots(figsize=figsize)
            for j, (date_str, mids, session_mean) in enumerate(combined_data[cond]):
                ax.plot(mids, session_mean, color=colors[j % len(colors)], label=date_str)
            ax.set_title(f"Session means — {cond} ({mode})")
            ax.set_xlabel("Time from last poke-out (s)")
            ax.set_ylabel("Speed (units/s)")
            ax.legend()
            fig.tight_layout()
            combined_figs[cond] = fig

            date_scope = []
            if combined_data[cond]:
                for date_str, *_ in combined_data[cond]:
                    date_scope.append(int(date_str) if str(date_str).isdigit() else date_str)
            save_name = f"epoch_speeds_combined_{_slugify(cond)}_{_slugify(mode)}"
            _save_fig(fig, save_name, date_scope or dates)

    result = {"per_session": per_session, "combined": combined_figs}
    if save and return_paths:
        return result, saved_paths
    return result



def plot_traces_with_speed_threshold(
    subjid,
    dates=None,
    *,
    ses=None,
    index=None,
    date_range=None,
    ses_range=None,
    index_range=None,
    xlim=None,
    ylim=None,
    position_units="cm",
    arena_size_cm=50.0,
    fa_types="FA_time_in",
    pre_buffer_s: float = 0.2,
    smooth_window: int = 5,
    figsize=(10, 8),
    invert_y: bool = True,
    save: bool = False,
    verbose: bool = True,
    return_paths: bool = False,
):
    """Plot spatial traces for rewarded, unrewarded, and FA trials with a speed threshold marker.

    For the selected sessions, builds three figures (rewarded, unrewarded, fa). Traces are overlaid
    across sessions. Each trial trace gets a black dot at the `speed_threshold_time` recorded in
    that session's ``speed_analysis.parquet``.

    **This reads; it does not compute.** A session without that file is reported and skipped --
    run ``scripts/run_speed_analysis.py`` for it first. Until 2026-08-18 this recomputed the
    threshold in memory instead, which was a second derivation of a quantity
    ``metric_analysis.movement`` owns (the section 14 hazard), and which no gate could reach:
    measured, ``plot_regression``'s case takes the saved branch, so the recompute ran zero times.
    The docstring also claimed the recomputed result was "saved + cached"; it never was.

    ``bin_ms`` / ``mode`` / ``threshold_alpha`` / ``threshold_beta`` went with it. They only ever
    fed the recompute, so keeping them would have silently accepted a threshold setting that
    decides nothing -- they belong to ``compute_speed_analysis``, which is where they still are.

    Parameters
    ----------
    subjid : int
        Subject ID.
    dates : list | tuple | None
        Dates list or inclusive range; None uses all available for the subject.
    xlim, ylim : tuple | None
        Pixel axis limits. When position_units="cm", each limit range is mapped
        to 0-arena_size_cm on the displayed axis.
    position_units : {"cm", "px"}
        Display position coordinates in centimetres or raw pixels.
    arena_size_cm : float
        Physical size represented by the xlim/ylim ranges when displaying cm.
    fa_types : str | Iterable
        FA labels to include (default "FA_time_in"). Case-insensitive; accepts comma/semicolon list.
    pre_buffer_s : float
        Seconds of trace to draw before last poke-out (default 0.2).
    smooth_window : int
        Rolling window (frames) for smoothing X/Y before speed computation and plotting.
    figsize : tuple
        Figure size for each condition plot.
    invert_y : bool
        If True, invert Y-axis to match video coordinates.
    save : bool
        When True, saves each generated figure into movement_figures via save_figure().
    verbose : bool
        If True, logs save successes/failures.
    return_paths : bool
        When True and save is enabled, returns list of saved file paths alongside the figures.

    Returns
    -------
    dict with keys "rewarded", "unrewarded", "fa" mapping to matplotlib figures. When
    return_paths is True, also returns the list of saved file paths.

    ``ses`` / ``index`` / ``date_range`` / ``ses_range`` / ``index_range`` narrow the
    selection further; they intersect with ``dates`` and with each other, and ``index``
    is the subject's gap-free chronological rank (`utils.helpers.session_selectors`).
    """
    select = session_selectors(
        ses=ses, index=index, date_range=date_range,
        ses_range=ses_range, index_range=index_range,
    )

    saved_paths = []

    def _slugify(value: str) -> str:
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(value)).strip("_")
        return slug.lower() or "figure"

    def _save_fig(fig, save_name: str, date_scope):
        if not save:
            return
        try:
            out_path = save_figure(
                fig,
                save_name,
                subjids=[subjid],
                dates=date_scope,
                subdir=MOVEMENT_FIGURES_SUBDIR,
            )
            saved_paths.append(out_path)
            if verbose:
                print(f"[plot_traces_with_speed_threshold] Saved figure to {out_path}")
        except Exception as exc:
            if verbose:
                print(f"[plot_traces_with_speed_threshold] Failed to save figure '{save_name}': {exc}")

    # Color palette consistent with plot_trial_traces_by_mode
    port_colors = {1: "#FF6B6B", 2: "#4ECDC4"}
    port_colors_fa = {1: "#FF8E8E", 2: "#7EE9DF"}
    aborted_color = "#555555"

    # Normalize FA labels
    fa_label_display = "FA"
    if isinstance(fa_types, str):
        if fa_types.lower() == "all":
            fa_label_display = "all"
            def fa_filter_fn(lbl):
                return str(lbl).lower().startswith("fa_") if pd.notna(lbl) else False
        else:
            fa_set = {s.strip().lower() for s in re.split(r"[;,]", fa_types) if s.strip()}
            fa_label_display = ", ".join(sorted(fa_set)) if fa_set else "selected"
            def fa_filter_fn(lbl):
                return str(lbl).lower() in fa_set if pd.notna(lbl) else False
    else:
        fa_set = {str(s).strip().lower() for s in fa_types}
        fa_label_display = ", ".join(sorted(fa_set)) if fa_set else "selected"
        def fa_filter_fn(lbl):
            return str(lbl).lower() in fa_set if pd.notna(lbl) else False

    # `mode` used to lead this suffix, so saved names read `..._mean_fa_time_in_...`.
    # It is gone with the recompute, and dropping it from the name is the point
    # rather than a side effect: the aggregation mode is now a property of the
    # saved `speed_analysis.parquet`, chosen when that file was computed. A figure
    # that no longer aggregates anything cannot know it, so keeping it in the
    # filename would label the output with a mode that may not be the one the
    # thresholds actually came from.
    suffix_parts = []
    if fa_label_display:
        suffix_parts.append(_slugify(fa_label_display))
    if smooth_window > 1:
        suffix_parts.append(f"smooth{smooth_window}")
    save_suffix = "_".join(filter(None, suffix_parts))

    if position_units not in {"cm", "px"}:
        raise ValueError("position_units must be either 'cm' or 'px'")
    if position_units == "cm" and arena_size_cm <= 0:
        raise ValueError("arena_size_cm must be positive")

    subj_str = normalize_subjid(subjid)
    derivatives_dir = get_derivatives_root()
    subj_dir = derivatives.subject_dir(subjid)

    ses_dirs = _filter_session_dirs(subj_dir, dates, **select)
    if not ses_dirs:
        raise FileNotFoundError(f"No sessions found for subject {subjid} with given dates")


    def _safe_dt(val):
        try:
            return pd.to_datetime(val)
        except Exception:
            return pd.NaT

    def _last_poke_out_scanning_back(entries):
        """Scan **back by position** to the first non-null `poke_odor_end`.

        A different rule from `_last_poke_out_by_position` above, which takes the
        last entry and accepts its null -- `DECISIONS.md` sections 13 and 14 are
        about not merging helpers that differ, so both survive the Phase 7b.4b move
        onto `position_data` unchanged.
        """
        for poke in reversed(entries or []):
            dt_val = _safe_dt(poke.get("poke_odor_end"))
            if pd.notna(dt_val):
                return dt_val

        return pd.NaT

    def _end_time(row, cond):
        if cond == "rewarded":
            return _safe_dt(row.get("first_supply_time")) or _safe_dt(row.get("sequence_end"))
        if cond == "unrewarded":
            return _safe_dt(row.get("first_reward_poke_time"))
        if cond == "fa":
            return _safe_dt(row.get("fa_time")) or _safe_dt(row.get("sequence_end"))
        return _safe_dt(row.get("sequence_end"))

    def _infer_port_with_odor_fallback(row):
        # Try explicit port fields first
        for col in [
            "response_port", "rewarded_port", "reward_port", "supply_port",
            "choice_port", "port", "fa_port", "last_reward_port", "odor_port",
        ]:
            if col in row and pd.notna(row[col]):
                try:
                    return int(row[col])
                except Exception:
                    try:
                        return int(float(row[col]))
                    except Exception:
                        continue
        # Try odor-number style fields
        for col in ["last_odor_num", "odor_num", "odor_index", "odor_position"]:
            if col in row and pd.notna(row[col]):
                try:
                    val = int(row[col])
                    if val == 2:
                        return 2
                    if val == 1:
                        return 1
                except Exception:
                    continue
        # Try odor labels
        odor = str(row.get("last_odor_name") or row.get("last_odor") or row.get("odor_name") or row.get("odor") or "").strip().lower()
        if odor in {"b", "odorb", "odor_b", "2", "portb", "port_b"}:
            return 2
        if odor in {"a", "odora", "odor_a", "1", "porta", "port_a"}:
            return 1
        return None

    def _port_for_coloring(row, cond):
        """Choose plotting port by explicit behavior columns first.

        - FA trials: use fa_port
        - Rewarded trials: use first_supply_port
        - Unrewarded trials: use first_reward_poke_port
        Falls back to generic inference if needed.
        """
        if cond == "fa":
            preferred_col = "fa_port"
        elif cond == "unrewarded":
            preferred_col = "first_reward_poke_port"
        else:
            preferred_col = "first_supply_port"

        if preferred_col in row and pd.notna(row[preferred_col]):
            try:
                return int(row[preferred_col])
            except Exception:
                try:
                    return int(float(row[preferred_col]))
                except Exception:
                    pass
        return _infer_port_with_odor_fallback(row)

    def _category_from_row(row):
        odor = str(row.get("last_odor_name") or row.get("last_odor") or "A")
        if odor in {"A", "OdorA", "1"}:
            return "A"
        if odor in {"B", "OdorB", "2"}:
            return "B"
        return "A"

    traces = {"rewarded": [], "unrewarded": [], "fa": []}
    markers = {"rewarded": [], "unrewarded": [], "fa": []}
    for ses_dir in ses_dirs:
        date_str = ses_dir.name.split("_date-")[-1]
        results_dir = ses_dir / "saved_analysis_results"
        if not results_dir.exists():
            continue
        skipped_no_poke_end = []
        analysis_path = results_dir / "speed_analysis.parquet"

        trial_data = None
        use_saved_thresholds = False

        cached_df = _get_from_cache(subjid, date_str, kind="speed_analysis")
        if cached_df is None and analysis_path.exists():
            try:
                cached_df = pd.read_parquet(analysis_path)
                _update_cache(subjid, [date_str], {date_str: cached_df.copy()}, kind="speed_analysis")
            except Exception as e:
                print(f"Failed to read {analysis_path.name}: {e}")

        if cached_df is not None:
            # Extract per-trial threshold times from per-bin records
            thr_series = (cached_df.dropna(subset=["speed_threshold_time"])
                                       .drop_duplicates(subset=["trial_index"])
                                       .set_index("trial_index")["speed_threshold_time"])
        else:
            thr_series = None

        views = _load_trial_views(results_dir)
        trial_data = views.get("trial_data", pd.DataFrame()).copy()
        if trial_data.empty:
            print(f"No trial_data for {date_str}; skipping")
            continue
        for c in ["sequence_start", "sequence_end", "first_supply_time", "first_reward_poke_time", "fa_time", "speed_threshold_time"]:
            if c in trial_data.columns:
                trial_data[c] = pd.to_datetime(trial_data[c], errors="coerce")
        # `in_poke_times` is the flag matching `position_poke_times`, the blob the two
        # `_last_poke_out_scanning_back` call sites read before Phase 7b.4b (section 2).
        pokes_by_trial = position_entries_by_trial(
            _load_position_data(results_dir, trial_data), "in_poke_times")

        if thr_series is not None:
            trial_data["speed_threshold_time"] = trial_data.index.map(thr_series)
            use_saved_thresholds = True

        # Load tracking per session
        try:
            tracking, _ = _load_tracking_and_behavior(subjid, date_str)
        except Exception as e:
            print(f"Skipping {date_str}: tracking load failed ({e})")
            continue

        tracking = tracking.copy()
        tracking["time"] = pd.to_datetime(tracking["time"], errors="coerce")
        tracking = tracking.dropna(subset=["time"]).reset_index(drop=True)
        for cand in [("centroid_x", "centroid_y"), ("X", "Y")]:
            if cand[0] in tracking.columns and cand[1] in tracking.columns:
                tracking["X"] = tracking[cand[0]]
                tracking["Y"] = tracking[cand[1]]
                break
        tracking = tracking.dropna(subset=["X", "Y"])
        tracking = tracking.loc[:, ~tracking.columns.duplicated()]
        if tracking.empty:
            continue
        tracking = smooth_xy(tracking, smooth_window)

        if use_saved_thresholds:
            # Strictly use stored threshold times; no recomputation
            for idx, row in trial_data.iterrows():
                rtc = str(row.get("response_time_category", "")).lower()
                is_aborted = bool(row.get("is_aborted", False))
                fa_label = str(row.get("fa_label", "")).lower()

                if rtc == "rewarded" and not is_aborted:
                    cond = "rewarded"
                elif rtc == "unrewarded" and not is_aborted:
                    cond = "unrewarded"
                elif fa_label.startswith("fa_") and fa_filter_fn(fa_label):
                    cond = "fa"
                else:
                    continue

                t_zero = _last_poke_out_scanning_back(
                    pokes_by_trial.get(row.get("global_trial_id")))
                if pd.isna(t_zero):
                    trial_id = row.get("trial_id", idx) if hasattr(row, "get") else idx
                    skipped_no_poke_end.append(trial_id)
                    continue
                t_end = _end_time(row, cond)
                if pd.isna(t_end) or t_end <= t_zero:
                    continue

                start_dt = t_zero - pd.Timedelta(seconds=pre_buffer_s)
                seg = tracking[(tracking["time"] >= start_dt) & (tracking["time"] <= t_end)].copy()
                if len(seg) < 2 or {"X", "Y", "time"} - set(seg.columns):
                    continue
                t_rel = (seg["time"] - t_zero).dt.total_seconds().to_numpy()
                if not np.isfinite(t_rel).all() or np.ptp(t_rel) == 0:
                    continue
                x = seg["X"].to_numpy()
                y = seg["Y"].to_numpy()

                marker = None
                thr_time = row.get("speed_threshold_time") if "speed_threshold_time" in trial_data.columns else pd.NaT
                if pd.notna(thr_time):
                    nearest_idx = int(np.argmin(np.abs((seg["time"] - thr_time).dt.total_seconds())))
                    marker = (x[nearest_idx], y[nearest_idx])

                port = _port_for_coloring(row, cond)
                if cond == "fa":
                    color = port_colors_fa.get(port, port_colors_fa[1])
                else:
                    color = port_colors.get(port, port_colors[1 if _category_from_row(row) == "A" else 2])

                traces[cond].append({"x": x, "y": y, "color": color, "session": date_str})
                if marker is not None:
                    markers[cond].append({"xy": marker, "color": "black", "session": date_str})
            if skipped_no_poke_end:
                print(f"Warning [{date_str}]: skipped trials with no poke_odor_end in position_poke_times: {skipped_no_poke_end}")
            # done with this session
            continue

        # No `speed_analysis.parquet` for this session. Until 2026-08-18 this fell
        # through to recomputing the threshold in memory -- a second derivation of a
        # quantity `metric_analysis.movement` owns (section 14), reached by no gate
        # (measured: `plot_regression`'s case takes the branch above, so this ran zero
        # times), and whose result the docstring wrongly claimed was saved.
        print(f"[plot_traces_with_speed_threshold] No speed_analysis.parquet for "
              f"{date_str}; skipping this session. Run "
              f"scripts/run_speed_analysis.py --subjids {subjid} --dates {date_str} first.")
        continue

    figs = {}

    # No session yielded a trace -- every one of them was missing its
    # `speed_analysis.parquet`. Return empty rather than falling through: the
    # axis-scaling below needs finite coordinate limits and raises
    # "Cannot convert to cm without finite x/y coordinate limits" without them,
    # which reports a units problem for what is really missing input. Caught by
    # the probe for this change, not by any gate -- `plot_regression`'s case runs
    # on a session that has the file, so it never reaches here.
    if not any(traces.values()):
        print("[plot_traces_with_speed_threshold] No speed analysis available for any "
              "requested session; nothing to plot.")
        return (figs, saved_paths) if (save and return_paths) else figs

    def _coord_limits(axis):
        vals = []
        for cond_traces in traces.values():
            for tr in cond_traces:
                arr = np.asarray(tr[axis], dtype=float)
                vals.extend(arr[np.isfinite(arr)])
        for cond_markers in markers.values():
            for mk in cond_markers:
                xy = mk.get("xy")
                if xy is None:
                    continue
                val = xy[0] if axis == "x" else xy[1]
                if np.isfinite(val):
                    vals.append(float(val))
        if not vals:
            return None
        return float(np.nanmin(vals)), float(np.nanmax(vals))

    def _normalize_limits(limits, axis):
        if limits is None:
            limits = _coord_limits(axis)
        if limits is None:
            return None
        lo, hi = limits
        return float(lo), float(hi)

    x_source_limits = _normalize_limits(xlim, "x")
    y_source_limits = _normalize_limits(ylim, "y")
    x_display_lim = x_source_limits
    y_display_lim = y_source_limits

    def _scale_axis(values, limits):
        values = np.asarray(values, dtype=float)
        if position_units == "px":
            return values
        lo, hi = limits
        span = hi - lo
        if span == 0:
            return np.full_like(values, np.nan, dtype=float)
        scaled = (values - lo) / span * arena_size_cm
        return np.clip(scaled, 0.0, arena_size_cm)

    def _x_display(values):
        return _scale_axis(values, x_source_limits)

    def _y_display(values):
        return _scale_axis(values, y_source_limits)

    if position_units == "cm":
        if x_source_limits is None or y_source_limits is None:
            raise ValueError("Cannot convert to cm without finite x/y coordinate limits")
        x_display_lim = tuple(float(v) for v in _x_display(x_source_limits))
        y_display_lim = tuple(float(v) for v in _y_display(y_source_limits))
        if verbose:
            print(
                "[plot_traces_with_speed_threshold] position mapping: "
                f"x px {x_source_limits} -> cm {x_display_lim}; "
                f"y px {y_source_limits} -> cm {y_display_lim}"
            )

    for cond, label in [("rewarded", "Rewarded"), ("unrewarded", "Unrewarded"), ("fa", "False Alarms")]:
        if not traces[cond]:
            continue
        fig, ax = plt.subplots(figsize=figsize)
        for tr in traces[cond]:
            ax.plot(_x_display(tr["x"]), _y_display(tr["y"]), color=tr["color"])
        for mk in markers[cond]:
            ax.scatter(_x_display([mk["xy"][0]])[0], _y_display([mk["xy"][1]])[0], color="black", zorder=5)
        ax.set_title(f"{label} traces with speed-threshold crossing")
        unit_label = "cm" if position_units == "cm" else "px"
        ax.set_xlabel(f"X Position ({unit_label})")
        ax.set_ylabel(f"Y Position ({unit_label})")
        if x_display_lim is not None:
            ax.set_xlim(x_display_lim)
        if y_display_lim is not None:
            ax.set_ylim(y_display_lim)
        if position_units == "cm":
            tick_values = np.linspace(0.0, arena_size_cm, 6)
            ax.set_xticks(tick_values)
            ax.set_yticks(tick_values)
        if invert_y:
            ax.invert_yaxis()
        if position_units == "cm":
            tick_values = np.linspace(0.0, arena_size_cm, 6)
            ax.set_yticks(tick_values)
            if invert_y:
                ax.set_yticklabels([f"{v:g}" for v in tick_values[::-1]])
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        figs[cond] = fig

        cond_dates_raw = sorted({tr.get("session") for tr in traces[cond] if tr.get("session")})
        date_scope = []
        for date_str in cond_dates_raw:
            date_scope.append(int(date_str) if str(date_str).isdigit() else date_str)
        save_name = f"speed_threshold_traces_{_slugify(cond)}"
        if save_suffix:
            save_name = f"{save_name}_{save_suffix}"
        _save_fig(fig, save_name, date_scope or dates)

    if save and return_paths:
        return figs, saved_paths
    return figs

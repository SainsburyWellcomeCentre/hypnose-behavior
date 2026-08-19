"""Fast lookup indices over a classification dict.

``build_classification_index`` flattens the ``completed_sequence_*`` / aborted / non-initiated
frames into by-trial-id lookups so downstream code can reach a trial without scanning every
frame. Called once after classification and again after the response-time merge.
"""
from __future__ import annotations

import pandas as pd


def build_classification_index(classification: dict) -> dict: # Classification function for easier dictionary access later on
    """
    Build convenient lookup indices over classification outputs.
    Provides:
      - by_trial: trial_id -> full row dict (completed_with_RT preferred, else completed, else aborted_detailed)
      - categories.completed.*_ids: lists of trial_ids for major completed categories (and HR variants)
      - sets.*: quick sets of IDs for initiated, completed, aborted
      - aborted: re-exposes the aborted_index (by_position/by_odor/by_type/by_fa_label)
    """

    idx = {'by_trial': {}, 'categories': {'completed': {}}, 'sets': {}, 'aborted': {}}

    # Prefer completed_with_RT for richer rows
    comp_df = classification.get('completed_sequences_with_response_times')
    if not isinstance(comp_df, pd.DataFrame) or comp_df.empty:
        comp_df = classification.get('completed_sequences', pd.DataFrame())

    ab_det = classification.get('aborted_sequences_detailed')
    ab_df = ab_det if isinstance(ab_det, pd.DataFrame) else classification.get('aborted_sequences', pd.DataFrame())

    # by_trial: completed first (wins), then aborted to fill missing ones
    if isinstance(comp_df, pd.DataFrame) and not comp_df.empty and 'trial_id' in comp_df:
        for _, r in comp_df.iterrows():
            tid = r.get('trial_id')
            if pd.notna(tid):
                idx['by_trial'][tid] = r.to_dict()
    if isinstance(ab_df, pd.DataFrame) and not ab_df.empty and 'trial_id' in ab_df:
        for _, r in ab_df.iterrows():
            tid = r.get('trial_id')
            if pd.notna(tid) and tid not in idx['by_trial']:
                idx['by_trial'][tid] = r.to_dict()

    # Completed category ID lists
    def ids_from(name):
        df = classification.get(name, pd.DataFrame())
        return [] if not isinstance(df, pd.DataFrame) or df.empty or 'trial_id' not in df else list(df['trial_id'])

    c = idx['categories']['completed']
    c['rewarded_ids'] = ids_from('completed_sequence_rewarded')
    c['unrewarded_ids'] = ids_from('completed_sequence_unrewarded')
    c['timeout_ids'] = ids_from('completed_sequence_reward_timeout')

    # Single-reward protocol: completed non-rewarded ("no-go") sequences. Empty for default protocol.
    c['false_response_ids'] = ids_from('completed_sequence_false_response')

    c['hr_rewarded_ids'] = ids_from('completed_sequence_HR_rewarded')
    c['hr_unrewarded_ids'] = ids_from('completed_sequence_HR_unrewarded')
    c['hr_timeout_ids'] = ids_from('completed_sequence_HR_reward_timeout')

    c['hr_missed_rewarded_ids'] = ids_from('completed_sequence_HR_missed_rewarded')
    c['hr_missed_unrewarded_ids'] = ids_from('completed_sequence_HR_missed_unrewarded')
    c['hr_missed_timeout_ids'] = ids_from('completed_sequence_HR_missed_reward_timeout')

    # Sets for quick membership tests
    idx['sets']['initiated_ids'] = (
        set(classification['initiated_sequences']['trial_id']) 
        if isinstance(classification.get('initiated_sequences'), pd.DataFrame) 
        and 'trial_id' in classification['initiated_sequences'] else set()
    )
    idx['sets']['completed_ids'] = set(comp_df['trial_id']) if isinstance(comp_df, pd.DataFrame) and 'trial_id' in comp_df else set()
    idx['sets']['aborted_ids'] = (
        set(classification['aborted_sequences']['trial_id']) 
        if isinstance(classification.get('aborted_sequences'), pd.DataFrame) 
        and 'trial_id' in classification['aborted_sequences'] else set()
    )

    # Aborted sub-index (already built by abortion_classification)
    ab_index = classification.get('aborted_index')
    if isinstance(ab_index, dict):
        idx['aborted'] = ab_index
    else:
        # Minimal fallback
        idx['aborted'] = {'by_trial': {}, 'by_position': {}, 'by_odor': {}, 'by_type': {}, 'by_fa_label': {}}
        if isinstance(ab_df, pd.DataFrame) and not ab_df.empty:
            try:
                idx['aborted']['by_trial'] = ab_df.set_index('trial_id', drop=False).apply(lambda r: r.to_dict(), axis=1).to_dict()
            except Exception:
                idx['aborted']['by_trial'] = {r['trial_id']: r.to_dict() for _, r in ab_df.dropna(subset=['trial_id']).iterrows()}
            def group_ids(col):
                out = {}
                if col in ab_df.columns:
                    for k, g in ab_df.groupby(col):
                        out[k] = list(g.sort_values('sequence_start')['trial_id']) if 'sequence_start' in g else list(g['trial_id'])
                return out
            for col, key in [('last_odor_position','by_position'), ('last_odor_name','by_odor'), ('abortion_type','by_type'), ('fa_label','by_fa_label')]:
                idx['aborted'][key] = group_ids(col)

    return idx

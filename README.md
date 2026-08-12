# Hypnose Analysis

This repository is utilised for processing and visualising data acquired from Hypnose Harris lab project.

## Features

- Extracting mouse interactions with ports, olfactometer commands and synchronisation with cameras/EEG recordings

- Analysing and visualising behavioral metrics

## Repository structure

```
configs/                 user-facing setup configs (rig/olfactometer .yml)
data/rawdata             symlink to the read-only data on the server; all output -> derivatives
notebooks/               analysis/visualisation notebooks (import from src; no definitions)
scripts/                 terminal entry points (thin CLI wrappers; no analysis logic)
src/hypnose_behavior/
    io/                  data loading, saving, paths (readers, loaders, load_results, save, save_results, paths)
    trial_classification/ trial detection + classification, in three layers: leaves
                         (windows, outcome, params, hidden_rule, index) <- workers
                         (detect_trials, classify_trials, response_times,
                         aborted_trials) <- run; plus detect_stage/settings, merge,
                         summary. Workers never import each other. 
    metric_analysis/     behavioural metric calculation: metrics/ (definitions, one
                         module per behavioural construct), run/merge/summary
                         (orchestration), frames, resolvers, registry
    visualization/       figure-making (valve/poke, metrics, pred-seq, movement)
    utils/               small shared helpers
    qc/                  quality control: data validation + golden-master regression tools (see below)
    resources/device_schemas/  harp schemas (behavior.yml, olfactometer.yml), loaded as package data
```

The importable package is `hypnose_behavior` (e.g. `from hypnose_behavior.trial_classification.run import batch_analyze_sessions`).

## How to Use

1. Clone the repository

Within your working directory use a terminal to clone the repo to your local folder:

```git clone github.com/SainsburyWellcomeCentre/hypnose-behavior-analysis```

2. Create and activate the conda environment using the environment.yml file

```conda env create -f environment.yml```
```conda activate hypnose-behavior-analysis```

> **`hypnose-helpers` is a required dependency** and is not on PyPI, so install it from a
> clone before this package. It owns the data-location mechanism (`io/paths.py`), the
> figure styles and the shared layout/selector parsing:
>
> ```
> git clone github.com/SainsburyWellcomeCentre/hypnose-helpers
> pip install -e /path/to/hypnose-helpers
> ```
>
> Without it, `import hypnose_behavior.io.paths` fails outright — this is a hard
> dependency, not the optional/lazy kind hypnose-somnotate uses.

> **Installing by hand?** Use the `behavioral` extra — `pip install -e ".[behavioral]"`.
> A bare `pip install -e .` gives the *base* install (analysis code, figure styles,
> data-location helpers) but omits the behavioural/video stack (`swc-aeon`,
> `harp-python`, `moviepy`, `opencv-python`), so the loaders and readers will fail with
> `ModuleNotFoundError: aeon` / `harp`. If that happens, rerun with the extra.
> `environment.yml` already uses it, so the command above is complete on its own.
>
> The split exists because `swc-aeon==0.1.0` requires Python ≥3.11. Keeping it out of
> the base dependencies lets repos pinned to older Python — `hypnose-eeg-preprocessing`
> is on 3.9 via pomegranate/somnotate — install this package and reuse
> `hypnose_behavior.io.paths` and `hypnose_behavior.io.save` (the shared figure styles), which need
> nothing from that stack.

3. Add the environment as a kernel to run notebooks

```python -m ipykernel install --user --name=hypnose-behavior-analysis --display-name="Hypnose Analysis"```

5. Symlink: 

Directories in this repo are resolved with a symlink inside /data pointing to the mounted server containing data. Depending on local structure of mounting the server, the symlink may need adjusting. 

1. Windows (requires Ceph server mounted at Z:):

- Open a PowerShell Terminal as Administrator

- cd into hypnose-behavior-analysis (repo main folder)

- Remove any possible existing items in the symlink folder by running  ```Remove-Item -LiteralPath .\data\rawdata -Recurse -Force```

- Non-persistently map the mounted server for this session by running ```net use Z: "\\ceph-gw02.hpc.swc.ucl.ac.uk\harris" /persistent:no```
 
- Confirm path exists by running ```Test-Path "Z:\hypnose\rawdata" ``` should return True

- Create SymLink to ceph data storage by running ```New-Item -ItemType SymbolicLink -Path ".\data\rawdata" -Target "Z:\hypnose\rawdata" ```

- SymLink should appear in the repo folder

2. macOS (requires server mounted as /Volumes/harris)

- cd into hyopnose-analysis (repo main folder)

- remove existing symlink or folders by running ```rm -rf ./data/rawdata```

- create data directory by running ```mkdir -p ./data```

- create the synmlink to ceph by running: ```ln -s /Volumes/harris/hypnose/rawdata ./data/rawdata```

## File Copying using robocop: 

For local analysis, use files copied from ceph to local machine. Use robocopy in powershell: robocopy "Z:\hypnose\rawdata\sub-045_id-284" "E:\rawdata\sub-045_id-284" /E /MT:32 /R:5 /W:5 /LOG:C:\Users\HarrisLab\Desktop\robocopy_log.txt /TEE /NP
Check log file for files that failed transfer. 

## Running Analysis: 

The analysis consists of two parts: **trial classification** and **behavioral metric calculation**. Both can be run either from the notebooks or from the terminal scripts.

### From the terminal (scripts/)

```
python scripts/run_trial_classification.py --subjids 53 --dates 20260528
python scripts/run_metrics_analysis.py     --subjids 53 --dates 20260528
python scripts/batch_process.py            --subjids 53 --date-range 20260501 20260531
```

`--subjids` and `--dates` are optional (omit to run all); use `--date-range START END` for an inclusive range. Run trial classification before metric analysis (metrics read the saved classification results). The scripts validate that data exists first (`hypnose_behavior.io.validate.validate_subject`) and are thin wrappers over `hypnose_behavior.trial_classification.run.batch_analyze_sessions` and `hypnose_behavior.metric_analysis.run.batch_run_all_metrics_with_merge`.

1. Trial Classification

The trial_classification notebook runs the trial classification. The functions it uses live in `src/hypnose_behavior/trial_classification/`: `run.py` for the session/batch entry points, and `detect_trials.py` / `classify_trials.py` / `response_times.py` / `aborted_trials.py` for the stages themselves. 

batch_analyze_sessions can run on any combination of dates and subjids to run analysis on several subjects or dates at ones. If one parameter is None, it will run on all subjects for date(s) provided or all dates for subject(s) provided. Results are saved as json and csv combination. A summary txt file is saved per session analyzed. 

plot_valve_and_poke_events can be used to visualize all valve states, with option to specify a time window. 

cut_video can be used to cut a short video of the experiment with a defined time window. 

1.1. Output

The trial classification returns a trial_data.parquet / .csv file containing the following data columns: 

Trial Identification

    - global_trial_id (int): Count of all trials across runs, starting at 0
    - trial_id (int): Non-global trial ID per run, starts at 0 per run
    - run_id (int): Run number the data is from, starting at 1
    - attempt_number (int): Which attempt of poking the first odor started the sequence

Sequence Timing

    - initiation_sequence_time (timestamp ISO 8601): Time the initiation sequence starts for this trial. From here, the mouse can poke
    - sequence_start (timestamp ISO 8601): Time the mouse successfully initiated a sequence
    - sequence_start_corrected (timestamp ISO 8601): Corrected sequence start if sequence contained position 1 attempts. For most trials, similar to normal sequence start
    - sequence_end (timestamp ISO 8601): Time the sequence ends (either reward port poke or begin of next sequence)
    - timestamp (timestamp ISO 8601): Same as sequence_start; the time the successful poke for starting the sequence occurred

Odor Sequence Information

    - odor_name (string): Name of the first odor (e.g., OdorA)
    - odor_sequence (list): List of all odors in the sequence from first to last
    - num_odors (int): Number of odors in this sequence
    - last_odor (string): Last odor in odor_sequence, which is when the animal left the odor cue port
    - sequence_name (string): Name of the protocol used in this run
    - continuous_poke_time_ms (float): Time the first odor was poked to initiate the sequence
    - required_min_sampling_time_ms (float): Minimum poke time required for the first odor to initiate the sequence
    - minimum_sampling_time_ms_by_odor (dict): Dictionary containing all odors and their respective minimum sampling times

Detailed Event Data

    - position_valve_times (dict): Dictionary of positions containing position name, odor presented (odor_name), valve start time (valve_start), valve end time (valve_end), valve duration (valve_duration_ms), and required minimum sampling time (required_min_sampling_time_ms)
    - position_poke_times (dict): Dictionary of positions containing position name, odor presented (odor_name), poke time for that odor (poke_time_ms), poke odor start (poke_odor_start), poke odor end (poke_odor_end), time of first poke in (poke_first_in), and required minimum sampling time (required_min_sampling_time_ms)
    - presentations (list): Contains index_in_trial, odor_name, valve_start, valve_end, valve_duration_ms, poke_time_ms, poke_first_in, required_min_sampling_time_ms, and is_last_event for all indices in the sequence
    - last_event_index (int): Index where the last event (odor) appeared

Hidden Rule Information

    - hidden_rule_location (int): Currently always 0. Do not use this information
    - hidden_rule_locations (list): All locations the hidden rule odor can appear
    - hidden_rule_positions (list): Index-corrected list of hidden rule locations (+1 for each index)
    - hit_hidden_rule (boolean): Whether the hidden rule appeared in the sequence
    - hidden_rule_hit_indices (list): The index where the hidden rule appeared
    - hidden_rule_hit_positions (list): The position where the hidden rule was hit (similar to index + 1)
    - hidden_rule_success (boolean): Whether the animal successfully completed the hidden rule trial (got to await_reward state by leaving at the hidden rule odor; can still be unrewarded or timeout)
    - hidden_rule_success_position (string): Same as hidden_rule_hit_position, but only in trials where hidden_rule_success is True
    - enough_odors_for_hr (boolean): Whether enough odors were presented for hidden rule to be possible

Reward Information

    - await_reward_time (timestamp ISO 8601): Timestamp of when await reward state was triggered
    - first_supply_time (timestamp ISO 8601): Time point the supply port was activated the first time (reward delivered)
    - first_supply_port (int): In rewarded trials, which supply port the animal poked first (1 for A, 2 for B)
    - first_supply_odor_identity (string): In rewarded trials, the identity of the first supply port poked (A or B)
    - supply1_count (int): In rewarded trials, 1 or 0 indicating whether first poke happened at supply port 1
    - supply2_count (int): In rewarded trials, 1 or 0 indicating whether first poke happened at supply port 2
    - total_supply_count (int): In rewarded trials, 1 or NaN indicating whether a supply port poke happened
    - poke_window_end (timestamp ISO 8601): Time when the reward window ended, indicating timeout or unrewarded trial
    - response_time_ms (float): In rewarded, unrewarded, or timeout trials, time between the animal's LAST odor cue port poke out before the reward poke, and that reward poke. This is the movement latency -- if the animal returns to the cue port after the sequence completes (resampling, or checking for another odor), that resampling time is not charged to the response.
    - completed_window_latency_ms (float): The same reward poke measured from AwaitReward, i.e. from where the experiment PC starts its response-time counter. This is what the response window is built on and what decides unrewarded vs timeout; response_time_ms answers how fast the animal moved, this one answers whether it responded in time. The two differ whenever the animal was still at the cue port when the counter started.
    - response_time_category (string): Categorical label for completed sequences: "rewarded", "unrewarded", or "timeout_delayed"

Non-Rewarded Trial Information

    - port1_pokes_count (int): In non-rewarded completed trials (unrewarded or timeout), number of pokes in port 1
    - port2_pokes_count (int): In non-rewarded completed trials, number of pokes in port 2
    - total_reward_pokes (int): In unrewarded trials, count of pokes in the wrong supply port
    - first_reward_poke_time (timestamp ISO 8601): In unrewarded trials, time of supply port poke
    - first_reward_poke_port (int): In unrewarded trials, port number of supply port poke
    - first_reward_poke_odor_identity (string): In unrewarded trials, identity of incorrect reward port poke (A or B)

Aborted Trial Information

    - is_aborted (boolean): Identifier if trial was aborted or not
    - abortion_type (string): Type of abortion: "initiation_abortion" or "reinitiation_abortion"
    - abortion_time (timestamp ISO 8601): Time point abortion happened (poke out of cue port)
    - last_odor_position (int): In aborted trials, position where the last odor in sequence appeared
    - last_odor_name (string): In aborted trials, name of last odor
    - last_odor_valve_duration_ms (float): In aborted trials, duration of last odor valve
    - last_odor_poke_time_ms (float): In aborted trials, poke duration of last odor
    - last_required_min_sampling_time_ms (float): In aborted trials, required minimum sampling time for that odor, defining abortion classification (initiation or re-initiation)

False Alarm Information

    - fa_label (string): In aborted trials, false alarm classification: "fa_time_in" (within response time window), "fa_time_out" (up to 3x the response time window), "fa_late" (later than that), or "nFA" (no false alarm)
    - fa_time (timestamp ISO 8601): In false alarm trials, when the false alarm happened (supply port poke)
    - fa_window_latency_ms (float): Time between the abortion (last cue port poke out inside the trial) and poking either supply port. This is the window-relative time and it is what fa_label buckets.
    - fa_response_time_ms (float): The same false alarm measured from the animal's last cue port poke out before the poke. Excludes any resampling between giving up and false-alarming, which the two differ by on ~44% of false alarms.
    - fa_port (int): In false alarm trials, port ID (1 for A, 2 for B) of first supply port poke

Single-Reward Protocol / False Response Information

    These columns are only populated for sessions using the new "single-reward" protocol, where
    NOT all candidate sequences are rewarded at their final position (e.g. `singrew-task-stage1`,
    where only specific full sequences such as OdorC-OdorF-OdorA and OdorG-OdorE-OdorB are
    rewarded, or single-odor go/no-go `FreeRun_StageN` stages). The protocol is detected from the
    Schema: a sequence is "rewarded" iff the chosen item at its final position has rewarded=True;
    the session is flagged single-reward iff at least one candidate sequence is not. For the
    default protocol (every sequence rewarded at the end) none of these columns are added and all
    other behaviour/output is unchanged.

    - sequence_rewarded (boolean): Whether THIS trial's full presented odor sequence exactly matches one of the schema's rewarded sequences. Rewarded-type sequences are classified exactly as in the default protocol (rewarded / unrewarded / timeout, or false alarm if aborted). Non-rewarded ("no-go") sequences get the false_response columns below. NOTE: this is an exact full-sequence match; for a partial/aborted sequence it is False simply because the partial sequence is not a complete rewarded one. Use `reward_determinacy` (below) to reason about partial sequences.
    - reward_determinacy (string): Whether the trial's reward outcome is already determined by the presented (possibly partial) odor_sequence, judged against ALL candidate sequences in the schema (`allSequences`, rewarded and non-rewarded). Looks at every candidate that starts with the presented prefix: "rewarded" (all such candidates are rewarded), "nonrewarded" (all non-rewarded), "ambiguous" (both kinds still possible — the animal could not yet know), or "off_protocol" (no candidate starts with this prefix). A completed sequence is always "rewarded"/"nonrewarded"; an aborted prefix may be "ambiguous". This is schema-driven so it adapts to any protocol. For aborted trials this distinguishes a correct abortion ("nonrewarded"), an incorrect abortion ("rewarded"), and a genuinely ambiguous abortion ("ambiguous").
    - determinacy_position (int): The 1-based position at which the outcome first became determined while reading the presented odors left to right (i.e. the earliest prefix length whose matching candidates are all the same reward type). Empty when never determined within the presented odors ("ambiguous") or "off_protocol". Informative even for completed trials (e.g. position 1 means the first odor alone already fixed the outcome).
    - determined_final_odor (string): The final (last-position) odor the sequence is bound to end at, but ONLY when every candidate matching the presented prefix shares the same last odor (i.e. the eventual reward port is already guaranteed, e.g. "this sequence must end in OdorA"). Empty when the final odor is not yet pinned down or "off_protocol". This is strictly finer than reward_determinacy: a sequence can be reward-determined yet have an ambiguous final odor (e.g. a prefix whose candidates are all non-rewarded but end A vs B). Lets an early/aborted leave be scored against the port the sequence was bound to end at.
    - false_response (boolean): For COMPLETED non-rewarded sequences only. True if the animal went to a reward port before the next trial initiation (analogous to a false alarm, but for a completed no-go sequence); False if it correctly withheld.
    - fr_label (string): False-response classification, mirroring fa_label: "FR_time_in" (reward poke within the response time window), "FR_time_out" (up to 3x the response time window), "FR_late" (later than that, before the next trial), or "nFR" (no reward poke, i.e. correct withholding). NOTE: when the schema's responseTime is effectively unlimited (e.g. 99999 s) every reward poke falls into FR_time_in; the timing buckets are only informative with a finite responseTime.
    - fr_time (timestamp ISO 8601): In false_response trials, when the false response happened (first reward port poke).
    - fr_window_latency_ms (float): In false_response trials, time between sequence completion (await_reward) and the first reward port poke. Window-relative, and what fr_label buckets.
    - fr_response_time_ms (float): The same poke measured from the animal's last cue port poke out before it. Excludes resampling after the sequence completed, which happens on ~57% of false responses.
    - fr_port (int): In false_response trials, port ID (1 for A, 2 for B) of the first reward port poke.
    - fr_odor_identity (string): In false_response trials, identity of the first reward port poked (A or B).
    - fr_window_end (timestamp ISO 8601): End of the search window for a false response (first cue-port poke after the next trial initiation).

    For single-reward sessions, response_time_category (and response_time_ms) are left empty for
    non-rewarded-type completions, so existing decision/choice-accuracy metrics that rely on
    response_time_category remain meaningful (rewarded/unrewarded/timeout) for rewarded-type
    sequences only. Completed non-rewarded sequences are also collected in the
    `completed_sequence_false_response` table.


2. Behavioral Metric Calculation

The metrics_analysis notebook runs the behavioral metric calculation. The definitions live in `metric_analysis/metrics/`, one module per behavioural construct (`accuracy`, `false_alarm`, `sequence`, `hidden_rule`, `sampling`, `timing`), plus `movement.py` and `sing_rew_metrics.py`.

To add another metric: write a pure `f(frame) -> value` core in the module for its construct, decorate it `@metric(frame="trials" | "position_data" | "trials+position_data")`, and give it a thin `*_session(results)` wrapper decorated `@session_metric(core)` if it should print. To have it saved to `metrics_*.json`, give the core a `title=` and add its name to `REPORT` in `metric_analysis/run.py`. 

batch_run_all_metrics_with_merge can run on any combination of dates and subjids. Further, a protocol filter can be applied to only run on sessions under same protocol (within or across subjects). 

Results are saved per session and merged for all sessions analyzed, either within the subject directory, or in the merged directory at the subject directory level for multi-subject runs.

Results are saved as a json and csv file combination with a summary txt file. 

## Quality control (`src/hypnose_behavior/qc/`)

The `qc` package holds tools to run **after major changes** to confirm the analysis output is unaffected (or to mark what changed, if intended). Run them in the project conda environment.

- **`validate.py`** — `validate_subject()`: pre-flight check that rawdata exists for a subject/date; flags missing dates in a range without aborting. Used by the terminal scripts.

- **`regression.py`** — golden-master value regression. For a fixed set of coverage sessions (`sessions.yml`) it fingerprints `trial_data` (canonical CSV) and the metrics dict and md5-compares against stored baselines in `fixtures/`. It reads the read-only rawdata and writes only to a temp dir (never the server).
  ```
  python src/hypnose_behavior/qc/regression.py            # compare against fixtures (exit 0 = GREEN)
  python src/hypnose_behavior/qc/regression.py --generate # regenerate baselines (only when a change is intended)
  ```

- **`verify_scripts.py`** — runs the actual terminal scripts via subprocess and md5-checks their `trial_data` + metrics against the same fixtures (covers the CLI arg wiring, which the function-level regression does not).
  ```
  python src/hypnose_behavior/qc/verify_scripts.py
  ```

- **`check_imports.py`** — static checker: disassembles every function and flags any referenced global that isn't imported (catches missing-import NameErrors that only surface at call time). Run on the whole package or a single module/file.
  ```
  python src/hypnose_behavior/qc/check_imports.py                                   # whole package
  python src/hypnose_behavior/qc/check_imports.py hypnose_behavior.trial_classification.run  # one module
  ```

A RED regression means output changed — revert/fix, or, if the change was intended, regenerate the fixtures in a separate reviewed commit. Fixtures are only valid in the pinned environment recorded in `fixtures/env.json`.
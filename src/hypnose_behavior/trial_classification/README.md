# Trial Classification (`src/hypnose_behavior/trial_classification/`)

Turns a session's raw harp streams into `trial_data` — one row per trial — plus the
per-position record and the non-initiated tables, written by
[`io/save_results.py`](../io/save_results.py) into `saved_analysis_results/`.

```
python scripts/run_trial_classification.py --subjids 59 --dates 20260612
```

The layer is three deep and the workers must not import each other
(`docs/DECISIONS.md` §17): `detect_*` find the trials, `classify_trials` /
`aborted_trials` / `response_times` score them, `run.py` orchestrates and `merge.py`
joins runs.

---

## Adding a new field — read this first

**There are two grains, they are declared in two different places, and neither updates
the other.** Choosing the wrong one is the mistake this section exists to prevent.

| your value is… | grain | declare it in |
|---|---|---|
| one value **per trial** | trial | a `TrialRecord` class in [`io/protocol_schema.py`](../io/protocol_schema.py) |
| one value **per position within a trial** | trial × position | a field list in [`frames.py`](../frames.py) |

If you are unsure: ask whether the value could differ between position 2 and position 3
of the *same* trial. If yes it is per-position; if no it is per-trial.

---

## A per-trial field

### 1. Declare it on the right record

`io/protocol_schema.py` holds three classes, and which one you pick decides which
sessions carry the column:

- **`_TrialRecordBase`** — every protocol writes it. Use this unless you know otherwise.
- **`StandardTrialRecord`** — the standard protocol's fixed response deadline.
- **`SingleRewardTrialRecord`** — extends `StandardTrialRecord`, because a single-reward
  session uses **both** scorers.
- **`OdourDiscriminationTrialRecord`** — deliberately does *not* extend the standard one.

Every field defaults to `None`. Declaring it on the base rather than a subclass is the
safe default; declaring it too narrowly gives an `AttributeError` at the assignment site
(the classes are `slots=True`), which is loud and easy to fix.

**Do not declare `run_id`, `is_aborted` or `global_trial_id`.** They are assembled
downstream (`ASSEMBLED_COLUMNS`), and declaring them would make `merge._with_run_id`
mint a phantom all-null `run_id_original` on every merged session (§21).

### 2. Assign it in `classify_trials`

```python
trial_rec.my_new_field = value
```

`slots=True` means a typo raises at the assignment site instead of silently inventing a
column of NaNs — which is exactly what the free-form dict used to do.

### 3. If it is a datetime, add it to `DATETIME_FIELDS`

**This one bites silently.** A column of nothing but `None` carries no type, so pandas
infers `object`, and one empty run turns the merged column `object` — after which
`to_csv` writes `…806000` where it used to write `…806`. Measured: 289 cells moved on
the only multi-run session in the fixture set, and it is **invisible on single-run
sessions** (§21). Cast losslessly; never `errors="coerce"`.

### 4. What happens automatically

`trial_data_columns(mode)` derives from `fields(cls)`, so once declared:

- `save_results`' conform creates the column on every session of that mode, so the
  table's shape is a function of the protocol rather than of which branches this
  session's trials happened to take;
- `load_results.py`'s schema check starts warning about older files that lack it.

### 5. Nothing to do in `position_data`

A per-trial field is **not** part of the per-position table and must not be added to the
`frames.py` field lists. The one exception is a trial-level column deliberately
denormalised onto every position row (`_TRIAL_COLUMNS`, currently `last_event_index`
only) — do that only when a per-position metric genuinely needs it, and say why.

---

## A per-position field

### 1. Write it into the blob

In `classify_trials`: `_position_poke_times`, `_position_valve_times`, or
`_build_presentations`.

### 2. Add it to the matching list in `frames.py`

| the field lives in… | add it to |
|---|---|
| `position_poke_times` (or both, poke wins) | `_POKE_FIELDS` |
| `position_valve_times` (or both, valve wins) | `_VALVE_FIELDS` |
| `presentations` **only** | `_PRES_ONLY_FIELDS` |
| nowhere — deliberately dropped | `KNOWN_UNCARRIED_FIELDS`, **with a reason** |

> **Not `_PRES_FIELDS`.** It documents what that blob happens to carry and the builder
> **does not read it**. Adding a field there and nowhere else leaves it dropped — a
> mistake the guard itself made once, and the reason `_PRES_ONLY_FIELDS` exists (§27).

### 3. You cannot forget step 2

`build_position_data` copies a **whitelist**; anything else is dropped with no error and
no empty column. So `save_results` calls it with `strict=True` and you get:

```
UncarriedPositionFieldError: position blob field(s) my_new_field are not carried
into `position_data` and would be silently dropped. Add them to `_POKE_FIELDS` /
`_VALVE_FIELDS` / `_PRES_ONLY_FIELDS` in `frames.py` (NOT `_PRES_FIELDS`, which the
builder does not read), or to `KNOWN_UNCARRIED_FIELDS` with a reason the information
is not lost.
```

Read paths only **warn**, deliberately: `build_position_data` also runs over sessions
saved long before these lists existed, and refusing to read historical data would be a
worse failure than dropping a field from it (§2, §27).

A field belongs in `KNOWN_UNCARRIED_FIELDS` only when the information is **not lost**
— "nothing reads it today" is not a reason, because the point is the reader that does
not exist yet.

---

## Either way: it is an intended output change

New columns move the golden master. Expect:

```
$PY -u src/hypnose_behavior/qc/regression.py
  [RED] ... + added column: my_new_field
```

in `trial_data_columns` for a per-trial field, or in `position_data_columns` for a
per-position one. Confirm the diff shows **only** your field — `+ added`, with zero
`~ changed` and zero `- removed` — then regenerate deliberately:

```
$PY -u src/hypnose_behavior/qc/regression.py --generate
```

Run `qc/position_data_lossless.py` too if you touched a blob: it checks that every blob
field is recoverable from `position_data` with an equal value, which is the precondition
for dropping the blob columns from `trial_data`.

**A `~ changed` you did not intend is a stop, not a regenerate.**

---

## See also

- [`docs/DECISIONS.md`](../../../docs/DECISIONS.md) §21 (the record classes and their
  traps), §24 (`position_data` is a projection), §27 (the uncarried-field guard).
- [`qc/README.md`](../qc/README.md) — the gates and what each one can and cannot see.

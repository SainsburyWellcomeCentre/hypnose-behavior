# configs

## Data location (`data_locations.yml` + `data_locations.local.yml`)

Where the pipeline reads `rawdata` and writes `derivatives`, chosen per machine —
no symlink required.

- **`data_locations.yml`** (shared): named profiles. Each has `rawdata`
  (required) and `derivatives` (optional — defaults to the sibling of `rawdata`).
  Add machines/drives here.
- **`data_locations.local.yml`** (git-ignored, per machine): one line, `active: <profile>`.
  Set it with the CLI, never commit it.

```bash
python scripts/set_data_location.py --list          # show profiles
python scripts/set_data_location.py server-mac      # activate one (writes the local file)
python scripts/set_data_location.py --show          # print the resolved roots (+ warns if missing)
```

`hypnose_behavior.io.paths` resolves the roots as: **HYPNOSE_\* env vars → active profile → legacy
`data/rawdata` symlink**. So env vars (used by the QC sandbox / CI) override temporarily without
touching your config; the active profile is the normal everyday selection; the symlink is a fallback.

A running kernel caches the paths — after switching, **restart the kernel** or call
`hypnose_behavior.io.paths.reload()`. Terminal runs pick up the new choice automatically.

## Other setup configs

User-facing setup `.yml` (e.g. rig/olfactometer) can also live here.

Note: the harp **device schemas** (`behavior.yml`, `olfactometer.yml`) consumed by the analysis
code remain *package data* under `src/hypnose_behavior/resources/device_schemas/`, loaded via
`importlib.resources` — they are not duplicated here.

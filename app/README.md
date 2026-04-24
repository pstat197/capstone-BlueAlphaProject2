# Streamlit UI

The `app/` package is the presentation layer. All simulation math and config loading live under `scripts/`. The UI builds a YAML-shaped dict in session state, merges it with widget overrides, runs `app.pipeline_runner.run_pipeline` through `app.cache.run_with_cache`, and renders Plotly results.

For pipeline math, file layout, `budget_shifts`, toggle semantics, and correlation theory, see the root [README.md](../README.md).

---

## Setup and run

From the repository root (editable install avoids `PYTHONPATH` issues):

```bash
pip install -e .
streamlit run app/streamlit_app.py
```

`streamlit_app.py` inserts the repo root on `sys.path` before importing `scripts`. Editable install is still recommended for `pytest` and `python -m scripts.main`.

---

## Session state and first load

- **`sim_config`:** canonical configuration dict. Initialized from **`load_example_text()`** in `ui_yaml_io.py`, which reads repo-root **`example.yaml`** when present (`app/paths.EXAMPLE_YAML_PATH`), otherwise a small inline YAML fallback.
- **`config_collapsed`:** after a successful run, set `True` so the main configuration UI hides and a compact **Results** view appears.
- **`yaml_manual_edit`:** when `True`, the Advanced YAML textarea is not overwritten each rerun by a silent merge from the form.
- **`last_df`, `last_run_id`, `last_cache_hit`, `last_hash`, `last_corr_results`, `last_error`:** outcomes of the latest **Run simulation** attempt.

---

## Settings (sidebar and popover)

The same controls are rendered twice with different widget key prefixes (`sb_` vs `pop_`): **Night mode**, **Colorblind-safe chart colors** (Wong-style palette for Plotly), **Reset to example.yaml** (reloads example config, clears broad widget keys, resets top-level simulation inputs, queues YAML dump), **Clear simulation cache** (deletes files under `app/.cache/runs/`).

Night and colorblind flags are mirrored into canonical keys **`night_mode`** and **`colorblind_charts`** via `on_change` callbacks.

---

## Simulation settings (main page)

Single row of widgets: **Week range** (`week_range_num`, minimum 1, no hard maximum), **Run Name** (`run_identifier_input`, labels downloads and captions), **Random seed** (`seed_input`, bound to reproducibility and cache hashing together with the rest of the config). Changes call **`yaml_sync_from_form`** so the Advanced YAML panel can refresh from the merged form state on the next rerun.

---

## Tabs

### Channels

- Caption points users to per-channel **reference** expanders (noise, saturation, adstock) with formulas from **`ui_help_markdown.py`**.
- **Add channel:** text field + button appends a row built from **`default_channel_dict()`** with a unique **`channel_name`**, clears channel-scoped widget keys, reruns.
- **`render_channel_widgets`** (`ui_channel_form.py`): one expander per channel, schema-driven numeric and select fields, curve previews where applicable, **Availability** (enabled, adstock/saturation toggles, pause rules table mapping to YAML `enabled` / `off_ranges` / `sticky_pause_ranges`), **trend slope**, **seasonality** block (`ui_seasonality_panel.py`: one radio among none, repeating cycle, sin, random Fourier, comma pattern; collapsed expander **How seasonality works (all modes)**; cycle path uses cycle length, `st.data_editor` on multipliers, Plotly line chart).
- **Global effect switches** below the list: two checkboxes write **`adstock.global`** and **`saturation.global`** on `sim_config`.

### Correlations

- **`render_correlations_section`** (`ui_correlations.py`): table of pairwise channel pairs and ρ, aligned with `sim_config["correlations"]`. Channel name dropdowns track live renames from the Channels tab where possible. Editing rows updates session state for merge into YAML.

### Advanced

- **`advanced_yaml`:** full-dump text area. If **`yaml_manual_edit`** is false, each script run sets the textarea from **`merge_ui_into_config(schema, silent=True)`** so it tracks Channels + Correlations widgets without opening expanders.
- **`yaml_mark_dirty`:** marks manual edit when the user types in the textarea.
- **Apply YAML to form:** `yaml.safe_load` the textarea into **`sim_config`**, then **`_resync_form_from_sim_config`** (clears `clear_channel_widget_keys`, reapplies correlation UI rows from dict, sets flags so week range / seed / run id sync from `sim_config`), **`st.rerun`**.

---

## Run simulation and errors

Primary button runs **`merge_ui_into_config(schema)`** (warnings as `st.warning`), requires a non-empty **`channel_list`**, then **`run_with_cache(merged, run_pipeline)`**.

- **Success:** warnings shown, dataframe and metadata stored, **`sim_config`** replaced with the merged copy (so the run matches what was simulated), **`pending_yaml_dump`** refreshes Advanced YAML, **`config_collapsed`** true, rerun. Next view: compact results or full results block.
- **Failure:** **`st.error`** with message; **`last_df`** cleared; **`config_collapsed`** false so the user stays in the editor.

---

## Results panel (`ui_results.py`)

Shown when **`last_df`** exists: either **compact** (title **Results**, wide **Edit configuration** + **Download CSV**) or **Latest results** under the form when the config panel is expanded.

Toolbar caption: run id, whether the row was **served from cache** or newly computed, and a short config hash suffix.

**Configuration (YAML snapshot)** expander: read-only **`yaml_dump(sim_config)`**.

Three sub-tabs:

1. **Chart view:** **Series scope** (all-channel totals or one channel's revenue, spend, impressions). **Overlay series** checkbox switches to min-max normalized overlay (Plotly). Respects night mode and colorblind palette.
2. **Correlation analysis:** spend correlation heatmap, pairwise summary (observed spend ρ, configured log-copula ρ in parentheses, drift labels), rolling ρ chart for a selected pair, multicollinearity-style bars. On **cache hits**, correlation structures are rebuilt from the CSV spend columns plus **`sim_config`** when schema allows.
3. **Data preview:** first 25 rows, rounded for display.

If cached CSV predates per-channel **`_revenue`** columns, the UI instructs to clear cache or re-run.

---

## Caching

`run_with_cache` hashes the merged config (including **`CACHE_VERSION`** in `cache.py`). Hits skip **`run_pipeline`**; misses run the pipeline and save CSV + JSON metadata when the dataframe passes schema validation. Bump **`CACHE_VERSION`** when output columns or semantics change for the same logical config.

---

## Related files

| File | Role |
|------|------|
| `ui_config_merge.py` | Collect per-channel overrides, seasonality, toggles, correlations; `merge_ui_into_config`; `clear_widget_keys` / `clear_channel_widget_keys`. |
| `ui_form_state.py` | Shared parsing helpers for numeric fields and select keys. |
| `ui_helpers.py` | `get_at`, `apply_overrides` path helpers. |
| `ui_channel_toggles.py` | Merge availability tables into `sim_config`. |
| `default_channel.py` | Template dict for newly added channels. |
| `ui_schema.yaml` | Drives generic per-channel field widgets. |

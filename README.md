# capstone-BlueAlphaProject2

Marketing media simulation: sample weekly spend per channel, map spend to impressions, apply saturation and adstock, add baseline revenue with optional trend and seasonality, add noise, and export a wide CSV. A Streamlit app edits the same YAML-shaped config, merges widget state, runs the pipeline with disk caching, and plots results.

Additional narrative (Google Doc): [Documentation of Code](https://docs.google.com/document/d/1glQWezaB3eBH13Mxp2eR0Y7SM1zAaVAaS1qHFy2uu-o/edit?usp=sharing)

---

## Repository layout

| Area | Role |
|------|------|
| `scripts/config/` | `default.yaml`, `loader.py` (`load_config`, `load_config_from_dict`, `apply_seed_append_expansion`), `defaults.py`, RNG helpers. Merges user YAML over defaults, fills missing channel fields, optional `number_of_channels` expansion, expands **`budget_shifts_auto_mode`** / **`correlations_auto_mode`** into lists, validates `correlations`. |
| `scripts/synth_input_classes/` | `InputConfigurations` and `Channel` dataclasses: parsing from dict, `normalize_seasonality_config` on each channel, global adstock/saturation flags. |
| `scripts/revenue_simulation/` | `revenue_generation.py` (saturation, adstock, ROI, baseline, seasonality, noise), `seasonality_fit.py` (Fourier fit, sin to Fourier, evaluation). |
| `scripts/spend_simulation/` | `spend_generation.py` (independent gamma per cell **or** correlated lognormal draw from YAML `correlations`, then `budget_shifts`, then toggle masks), `correlation_analysis.py`, `pairwise_summary.py`. |
| `scripts/impressions_simulation/` | `impressions_generation.py` (CPM, impression noise, masks). |
| `scripts/main.py` | `run_simulation`, `construct_csv`, CLI entry that reads YAML (see [Config loading paths](#config-loading-paths)). |
| `app/` | `streamlit_app.py` (layout, tabs, run/cache), `ui_channel_form.py`, `ui_config_merge.py`, `ui_seed_extras.py` (simulation-settings seed-append modes), `ui_seasonality_panel.py`, `ui_correlations.py`, `ui_budget_shifts.py`, `ui_results.py` (charts, correlation panel, data preview), `ui_help_markdown.py`, `ui_schema.yaml`, `theme.py`, `default_channel.py`, `cache.py`, `pipeline_runner.py` (calls `load_config_from_dict` + `run_simulation`, no Streamlit). |
| `tests/` | Pytest modules mirroring config, spend, impressions, revenue, pipeline, toggles, correlations, seasonality merge/fit, UI toggle helpers. |
| `example.yaml` | Sample multi-channel config for local runs. |

---

## Setup

From the project root, create a virtual environment (recommended) and install the project in **editable** mode so `scripts` and `app` resolve without setting `PYTHONPATH`:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -e .
```

Dependencies are listed in `pyproject.toml` and mirrored in `requirements.txt`: `numpy>=1.25`, `pandas>=2.0`, `PyYAML>=6.0`, `streamlit>=1.28`, `plotly>=5.18`. Python 3.10+.
Install test-only tooling with extras: `pip install -e ".[tests]"`.

### Streamlit UI

From the project root:

```bash
streamlit run app/streamlit_app.py
```

Shorter UI-specific notes live in [app/README.md](app/README.md). The app prepends the repo root to `sys.path`; editable install is still recommended for tests and `python -m scripts.main`.

---

## Table of contents

- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Config loading paths](#config-loading-paths)
- [Running the pipeline](#running-the-pipeline)
- [Running tests](#running-tests)
- [End-to-end data flow](#end-to-end-data-flow)
- [Configuration and YAML](#configuration-and-yaml)
- [Stage: spend generation](#stage-spend-generation)
- [Stage: impressions](#stage-impressions)
- [Stage: revenue](#stage-revenue)
- [Seasonality and `seasonality_fit`](#seasonality-and-seasonality_fit)
- [CSV output](#csv-output)
- [Spend correlations](#spend-correlations)
- [Streamlit: merge, YAML, cache](#streamlit-merge-yaml-cache)
- [Results view (after a run)](#results-view-after-a-run)
- [Channel availability and effect toggles](#channel-availability-and-effect-toggles)

---

## Config loading paths

There are two equivalent ways a dict/YAML becomes an `InputConfigurations` object:

1. **`scripts.config.loader.load_config(path)` or `load_config_from_dict(user_dict)`** (used by tests, `app/pipeline_runner.run_pipeline`, and any code that should match the UI): reads `scripts/config/default.yaml`, deep-merges the user dict on top, fills missing per-channel keys from the default channel template, optionally grows `channel_list` to `number_of_channels` with noised numerics, expands **`budget_shifts_auto_mode`** / **`correlations_auto_mode`** into concrete `budget_shifts` / `correlations` via **`apply_seed_append_expansion`**, validates `correlations` channel names / `rho` ranges / PSD compatibility, initializes the global RNG from `seed` when present, then calls `InputConfigurations.from_yaml_dict(merged, default_channel_template=...)`.

2. **`python -m scripts.main` with a YAML file** (see [Running the pipeline](#running-the-pipeline)): now uses **`load_config`** internally, so CLI runs match Streamlit and test behavior (default merge + auto-mode expansion + validations).

---

## Running the pipeline

```bash
python -m scripts.main example.yaml
# or
python -m scripts.main -c path/to/config.yaml
```

Writes a timestamped CSV under `output/` and prints a short preview plus correlation report text when spend correlation analysis runs.

---

## Running tests

**Preferred (full suite under `tests/`, including seasonality merge, fit, UI helpers, and any newer modules):**

```bash
pytest tests/ -q
```

`python test.py` is now a thin wrapper around `pytest tests/ -q`.

Single file examples:

```bash
python -m pytest tests/test_config.py -q
python -m pytest tests/test_pipeline.py -q
```

---

## End-to-end data flow

```
YAML (or merged dict)
    → load_config_from_dict / load_config  →  merged dict
    → apply_seed_append_expansion (auto modes → concrete budget_shifts / correlations)
    → InputConfigurations.from_yaml_dict     →  InputConfigurations + Channel objects

InputConfigurations
    → generate_spend(config)
        → spend_matrix [weeks × channels]

    → generate_impressions(config, spend_matrix)
        → impressions_matrix [weeks × channels]

    → generate_revenue(config, impressions_matrix)
        → revenue_matrix [weeks × channels]

    → construct_csv(config, spend_matrix, impressions_matrix, revenue_matrix)
        → pandas DataFrame → CSV (CLI: output/; UI: cache + download)
```

Weekly spend correlations are computed from `spend_matrix` inside `run_simulation` and returned alongside the DataFrame for reporting and UI tables.

---

## Configuration and YAML

- **Top-level keys** typically include `run_identifier`, `week_range`, `seed`, `channel_list`, optional `correlations`, optional **`budget_shifts_auto_mode`** / **`correlations_auto_mode`** (seed-append intent; expanded in the loader), optional `adstock` / `saturation` global sections, optional `number_of_channels`, and optional **`budget_shifts`** (manual rules applied after the base spend draw: `type: scale` multiplies all channels in an inclusive 1-based week range; `type: scale_channel` multiplies one `channel_name` in a week range; `type: reallocate` moves a fraction from `from_channel` to `to_channel` each week in `[start_week, end_week]` when `end_week` is set, otherwise from `start_week` through the end of the horizon). See `example.yaml` comments and `tests/test_spend_generation.py`.
- **Each channel** (under `channel_list` as `- channel: { ... }`) includes at least: `channel_name`, `cpm`, `spend_range`, `true_roi`, `baseline_revenue`, `trend_slope` (linear drift on baseline per week), `seasonality_config` (often `{}`), `saturation_config`, `adstock_decay_config`, `spend_sampling_gamma_params`, `noise_variance`, plus optional availability and effect toggles (see [Channel availability and effect toggles](#channel-availability-and-effect-toggles)).
- **`scripts/config/default.yaml`** is the canonical default channel template used by the loader for fills and generated channels.

---

## Streamlit: seeded random `budget_shifts` and `correlations` (YAML intent + loader expansion)

**Random append (same run seed)** lives under **Simulation settings** in `app/ui_seed_extras.py` (outside tabs so Streamlit always records the choice). The merged config and YAML snapshots store **intent** in two optional top-level keys:

- **`budget_shifts_auto_mode`:** `none` | `global` | `global_and_channel` (defaults in `scripts/config/default.yaml`).
- **`correlations_auto_mode`:** `none` | `random`.

`app/ui_config_merge.py` writes them beside manual **`budget_shifts`** / **`correlations`**. **`load_config_from_dict`** (`scripts/config/loader.py`) runs **`apply_seed_append_expansion`**, which turns those modes into concrete rules using **`seed`**, **`week_range`**, and channel names, then **drops** the mode keys before **`InputConfigurations.from_yaml_dict`**. The same YAML therefore reproduces the same effective lists. Generators use **`SeedSequence([seed, 0xB05E11AF])`** and **`0xC0FFA1AB`**, so they do not perturb the main spend RNG sequence.

After **Run simulation**, `sim_config` and the results YAML snapshot use a **deep copy** of the merged dict; `yaml_dump` sanitizes NumPy scalars. The snapshot shows the two **`*_auto_mode`** keys plus manual lists (not the post-expansion-only view inside `InputConfigurations`). Disk cache hashing (`app/cache.py` **`canonical_config_hash`**) includes those keys so cache hits match intent.

### Random `budget_shifts` (`scripts/spend_simulation/budget_shift_auto.py`)

- **UI labels:** *None*; *Global — random all-channel scales + bounded reallocates*; *Global + per-channel — also random single-channel scales*.
- **RNG:** `numpy.random.default_rng(numpy.random.SeedSequence([seed, 0xB05E11AF]))`.
- **What gets built:** A pseudo-random number of rules (for example 2–4 in *Global*, 3–6 in *Global + per-channel*): **`scale`**, **`reallocate`**, and in *Global + per-channel* often **`scale_channel`**; if that mode finishes without any `scale_channel`, one may be **appended** when at least one channel exists (see `budget_shift_auto.py`).
- **Order:** Manual rules normalized first, auto rules concatenated after; `_apply_budget_shifts` runs the combined list in order.

### Random `correlations` (`scripts/spend_simulation/correlation_auto.py`)

- **UI:** **Extra correlation pairs** — *None* vs *Random pairs (same seed)*.
- **RNG:** `numpy.random.default_rng(numpy.random.SeedSequence([seed, 0xC0FFA1AB]))`.
- **What gets built:** **Sorted** channel names, all unordered pairs shuffled, random `K ∈ {1, …, min(3, #pairs)}`, each **ρ ~ Uniform(−0.92, 0.92)** (**copula correlation in log-spend**, same meaning as manual sliders).
- **Conflict policy:** Auto pairs are added only when the unordered pair is **not** already manual; **manual ρ wins**.

---

## Stage: spend generation

**Code:** `scripts/spend_simulation/spend_generation.py`

- **If `correlations` is empty:** each week and channel is an **independent** draw from that channel’s gamma(`shape`, `scale`), then clipped to `spend_range`.
- **If `correlations` is non-empty:** spend is drawn **jointly** each week: channel marginals are matched to the same gammas in **lognormal** space (`mu`, `sigma` per channel), a correlation matrix is built from the YAML pairwise `rho` values (unspecified pairs are 0), and **`rng.multivariate_normal(mu, cov, size=num_weeks)`** produces log-spend; **`exp`** yields positive spend, then per-channel **`spend_range`** clipping. The YAML `rho` is therefore a **linear correlation in log-spend** (Gaussian coupling), not the same number as a Pearson correlation of **dollar spend** after `exp` and clipping. The UI and `correlation_analysis` explain both “configured copula ρ” and “observed spend ρ”.
- After the draw: optional **`budget_shifts`** mutate the matrix in place (global scale, per-channel `scale_channel`, or `reallocate` between channels by name), then everything is clipped again to ranges and non-negative.
- **Toggles:** fully disabled channels and deterministic or sticky pause rules zero spend where configured. Sticky pauses use a reproducible RNG branch from `(seed, channel_index)`.

---

## Stage: impressions

**Code:** `scripts/impressions_simulation/impressions_generation.py`

For each channel `c` and week `w`: **`base = (spend[w,c] / CPM[c]) * 1000`**. Noise is **`Normal(0, sigma^2)`** with **`sigma = sqrt(noise_variance["impression"]) * base`** (0 variance gives zero noise). **`impressions = max(base + noise, 0)`**. The same spend-allowed mask as in spend generation is applied again as a safety net for off weeks and fully disabled channels.

---

## Stage: revenue

**Code:** `scripts/revenue_simulation/revenue_generation.py`

For each channel and week, pipeline order on the impression series is:

1. **Saturation** (if global and per-channel saturation are on): `linear` (default slope 1 scales impressions), `hill`, or `diminishing_returns` as configured in `saturation_config`.
2. **Adstock** (if global and per-channel adstock are on): `linear` (uniform window of length `lag+1`), `geometric` / `exponential` decay kernel, or `weighted` custom FIR.
3. **ROI scaling:** `transformed_impressions * true_roi` (this is the media-attributed component).
4. **Baseline:** `baseline_revenue + trend_slope * week_index`, then multiplied by the **seasonality** multiplier series when `seasonality_config` is non-empty (see [Seasonality and `seasonality_fit`](#seasonality-and-seasonality_fit)).
5. **Revenue noise:** Gaussian with standard deviation `sqrt(noise_variance["revenue"]) * abs(revenue)` when variance is positive.

Fully disabled channels return zeros for the entire run (no baseline, no noise, no echo). Deterministic off weeks zero spend and impressions upstream; adstock can still carry past activity forward, so off weeks may show non-zero revenue from decay plus baseline and noise (Policy A in the toggle documentation below).

---

## Seasonality and `seasonality_fit`

**Code:** `scripts/revenue_simulation/seasonality_fit.py`, used from `input_configurations.py` (normalize on load) and from the Streamlit merge path for table and pattern inputs.

**Role in the model:** Seasonality multiplies the **baseline** path only (after trend). It does not change spend or impressions.

**Normalization at load:** `sin` and `categorical` (comma-separated pattern) specs in YAML are converted to **deterministic Fourier** configs (`type: fourier` with `period`, `intercept`, and `coefficients` harmonics) so simulation stays reproducible. Random Fourier (`K`, `scale`, no precomputed coefficients) remains stochastic in shape, but now uses per-channel deterministic seed fallback from the run seed when `seasonality_config.seed` is not set.

**Runtime (`_seasonality` in `revenue_generation.py`):** Supports normalized `fourier` (deterministic evaluation or random draw), `hybrid` components, and defensively still accepts `sin` / `categorical` dicts by re-normalizing and recursing. Evaluation of fitted curves uses `evaluate_deterministic_fourier`.

**Streamlit UI:** One radio per channel selects **none**, **repeating cycle** (table of multipliers per week in a cycle, preview chart, merged as fitted Fourier), **sin**, **random Fourier**, or **comma pattern**. A single collapsed expander documents all modes (`app/ui_help_markdown.py`). Widget state merges into `seasonality_config` via `app/ui_config_merge.py` (`_collect_seasonality_overrides`).

---

## CSV output

**Code:** `scripts/main.py` (`construct_csv`)

Each row is one week. Columns include `week`, total `revenue` (sum of per-channel revenue that week), for each channel `{name}_impressions`, `{name}_spend`, `{name}_revenue`, then `total_impressions` and `total_spend`. Channel blocks are ordered consistently for readability.

---

## Spend correlations

**Code:** `scripts/spend_simulation/correlation_analysis.py`, invoked from `scripts.main.run_simulation` after `generate_spend`.

The analysis works on the **simulated spend matrix** (dollar levels per week per channel): a **static Pearson matrix** over the full run, **rolling Pearson** correlations with default window **12 weeks** (capped by run length; see `analyze_spend_correlations` in `correlation_analysis.py`), per-channel average absolute correlation, and a **`pairwise_summary`** (from `pairwise_summary.py`) that labels drift in rolling spend-ρ between early and late windows.

YAML **`correlations`** is a list of `{ "channels": ["A", "B"], "rho": <float in [-1, 1]> }`. With the loader, optional **`correlations_auto_mode: random`** is expanded into extra list entries (deterministic from **`seed`** and channel names) before validation; the spend stage still sees a plain list. Those `rho` values feed the **log-space multivariate normal** in `generate_spend` when the list is non-empty. **`loader.load_config`** validates that each pair names two real channels after merge. The **CLI** prints `print_correlation_report`. The Streamlit **Correlations** tab edits manual pairs; **Random append** under **Simulation settings** sets **`correlations_auto_mode`**. The **results** correlation tab (see below) shows heatmaps, badges for **observed spend ρ**, gray parenthetical **configured log-copula ρ** when a row exists, rolling charts, and drift labels (see captions in `app/ui_results.py`).

---

## Streamlit: merge, YAML, cache

**Entry:** `streamlit_app.py`. First session: `sim_config` loads from **`app/ui_yaml_io.load_example_text()`** (same idea as `example.yaml`). **`inject_theme_css`** (`theme.py`) applies night mode styling.

**Layout:** **Simulation settings** on the main page (**Week range**, **Run Name** / run identifier, **Random seed**, **Random append** for **`budget_shifts_auto_mode`** / **`correlations_auto_mode`** — same semantics as the section **Streamlit: seeded random `budget_shifts` and `correlations` (YAML intent + loader expansion)** earlier in this file). **Settings** appear in the **sidebar** and again inside a main-page **popover** (duplicate widgets use prefixed keys so Streamlit does not collide). **Tabs:** **Channels**, **Correlations**, **Budget shifts**, **Advanced**.

- **`st.session_state.sim_config`** is the canonical dict. Top-level widgets sync into it on change via **`yaml_sync_from_form`** (marks that the YAML textarea should track the form).

**Channels tab:** Add channels with a name field and **Add channel** (unique names, **`default_channel_dict()`** template). Each channel is an expander driven by **`ui_schema.yaml`** plus custom sections (availability, saturation, adstock reference expanders from **`ui_help_markdown.py`**, **trend slope**, **seasonality** radio: none, repeating cycle, sin, random Fourier, comma pattern; cycle mode uses **`st.data_editor`** and a Plotly preview). **`merge_ui_into_config`** in **`ui_config_merge.py`** deep-copies `sim_config`, collects overrides from all widget keys (including **`sea_*`** seasonality, **`corr_*`** correlation rows, **`bs_*`** budget-shift rows, and the seed-append selectboxes), applies **`apply_overrides`**, merges renamed channels, runs **`merge_channel_toggles_into_config`**, and returns **`(merged, warnings)`** including top-level **`budget_shifts_auto_mode`** / **`correlations_auto_mode`**. **`ensure_seasonality_widgets_warmed`** runs inside merge so the YAML preview is not empty for seasonality before expanders render.

**Correlations tab:** Edits manual **`correlations`** rows (pairs and ρ) into session state for merge. **`ensure_corr_rows_initialized`** seeds rows from YAML. Random extra pairs are controlled from **Simulation settings**, not this tab alone.

**Budget shifts tab:** Manual **`budget_shifts`** rules only; seed-based extras use **`budget_shifts_auto_mode`** on the main page.

**Advanced tab:** **`st.text_area`** bound to **`advanced_yaml`**. If the user has **not** marked manual YAML edit, each rerun performs **`merge_ui_into_config(..., silent=True)`** and overwrites the textarea with **`yaml_dump(merged)`** so the panel tracks the form. Editing the textarea calls **`yaml_mark_dirty`** so the silent overwrite stops until the user applies or resets. **Apply YAML to form** parses YAML, assigns **`sim_config`**, calls **`_resync_form_from_sim_config`** (clears channel widget keys, reapplies correlation and budget-shift rows from **`sim_config`**, **`sync_seed_extra_modes_from_cfg`**, syncs week range / seed / run id, queues a fresh **`yaml_dump`**), then **`st.rerun`**.

**Run simulation:** **`merge_ui_into_config(schema)`** (non-silent warnings shown), require at least one channel, then **`run_with_cache(merged, run_pipeline)`**. On success: store **`last_df`**, run id, cache hit flag, config hash, **`last_corr_results`**, copy merged config back to **`sim_config`**, queue **`pending_yaml_dump`**, set **`config_collapsed`** true, **`rerun`**. On failure: **`last_error`** string, **`last_df`** cleared, config panel stays open.

**`app/cache.py`:** Hash = SHA-256 of sorted JSON including **`CACHE_VERSION`** and the merged config dict. Cache hit returns the CSV from **`app/.cache/runs/`**; miss runs **`run_pipeline`**, then saves CSV + small JSON metadata if the dataframe passes **`cached_dataframe_schema_ok`** (requires each `*_impressions` channel column to have a matching `*_revenue` column). **`corr_results` is `None` on cache hits**; the results UI rebuilds correlation structures from the cached CSV plus current **`sim_config`** when possible.

**`app/pipeline_runner.py`:** **`run_pipeline(user_data)`** → **`load_config_from_dict`** → **`run_simulation`** (same stack as tests using **`load_config`**).

---

## Results view (after a run)

**Code:** `app/ui_results.py`, triggered from `streamlit_app.py` when **`last_df`** is set.

- If **`config_collapsed`** is true and a dataframe exists, the app shows a **compact results-only** view: title **Results**, **Edit configuration** (sets flags to resync widgets from **`sim_config`** and reruns), **Download CSV** (filename uses **`last_run_id`**), caption showing run id / cache hit / config hash tail, then the same results content as below.
- When the config panel is still visible but a previous **`last_df`** exists, a **Latest results** block appears under the form with the same controls.

**Inside the results block:** expander **Configuration (YAML snapshot)** shows **`yaml_dump(sim_config)`** — the same shape as the Advanced panel after a run: manual **`budget_shifts`** / **`correlations`** plus **`budget_shifts_auto_mode`** / **`correlations_auto_mode`** (expansion into concrete rules happens inside **`load_config_from_dict`**, not in this snapshot). Three inner tabs:

1. **Chart view:** select **totals** or a single channel; optional **Overlay series** (min-max normalized revenue, spend, impressions on one Plotly chart). Respects **night mode** and **colorblind-safe** palette from session state.
2. **Correlation analysis:** heatmap of **Pearson ρ on weekly spend levels**, pairwise badges, configured **log-copula ρ** in muted text, rolling ρ chart for a chosen pair, multicollinearity-style bars. Rebuilt from CSV + **`sim_config`** when needed so cache hits stay consistent.
3. **Data preview:** first 25 rows, rounded floats.

If an **old cached CSV** lacks per-channel revenue columns, the UI explains clearing cache or re-running.

---

## Channel availability and effect toggles

**Code:** `scripts/synth_input_classes/channel.py`, `input_configurations.py`, spend / impressions / revenue generators.

Channels can be fully disabled, partially off by inclusive **1-indexed** week ranges, or subject to optional **sticky** random pauses inside windows (`start_probability`, `continue_probability`). Per-channel `adstock_enabled` and `saturation_enabled` combine with global `adstock.global` and `saturation.global`. Defaults are fail-open (everything on) so older YAML keeps working.

### YAML schema (toggles)

Each entry under `channel_list` may include optional toggle keys. Omit any of them to stay fully on.

```yaml
channel_list:
  - channel:
      channel_name: TikTok
      # ... usual fields (cpm, spend_range, saturation_config, etc.) ...

      # Option 1: simple on/off for the whole run.
      # enabled: false

      # Option 2: always on, except during the listed inclusive week ranges.
      enabled:
        default: true
        off_ranges:
          - start_week: 10
            end_week: 12
          - start_week: 30
            end_week: 35

      # Optional: sticky (Markov) random pauses (spend/impressions only; Policy A echo unchanged).
      # sticky_pause_ranges:
      #   - start_week: 10
      #     end_week: 40
      #     start_probability: 0.15
      #     continue_probability: 0.85

      adstock_enabled: true
      saturation_enabled: true

adstock:
  global: true      # false disables adstock for every channel
saturation:
  global: true      # false disables saturation for every channel
```

Rules:

- Weeks are **1-indexed**; `off_ranges` entries are **inclusive** (`start_week <= end_week`).
- `enabled` may be a boolean or a `{ default, off_ranges }` mapping.
- Fully disabled channels (`enabled: false`) contribute **zero** revenue for the full run (no spend, impressions, adstock echo, baseline, or noise).
- Globals override per-channel flags when set to false.

### Off-week semantics (Policy A)

On weeks inside an `off_ranges` window for an otherwise enabled channel:

- **Spend** is zeroed in `generate_spend`.
- **Impressions** are zeroed in `generate_impressions`.
- **Revenue** is not forced to zero: adstock carry-over from earlier active weeks still flows, and baseline plus revenue noise still apply. A row can show non-zero `{channel}_revenue` with zero spend and impressions; that reflects decay from prior weeks.

### Sticky random pauses

Optional `sticky_pause_ranges` lists objects with `start_week`, `end_week`, `start_probability`, `continue_probability` in `[0, 1]`. Hard off weeks are applied first; on a deterministic off week the sticky chain does not advance. Sticky draws use an RNG branch from `(seed, channel_index)` so masks match between spend and impressions. Multiple windows OR together for pauses.

### Where masking is applied

| Stage | Behavior |
|-------|----------|
| `generate_spend` | Zeros spend for fully disabled channels; zeros cells where deterministic or sticky rules disallow spend. |
| `generate_impressions` | Inherits zeros from spend; applies the same spend-allowed mask. |
| `generate_revenue` | Short-circuits fully disabled channels to zero. Honors adstock/saturation gates. Keeps adstock echo on off weeks. |
| `construct_csv` | No extra masking; matrices already reflect rules. |

### Minimal toggle example

```yaml
channel_list:
  - channel:
      channel_name: TikTok
      enabled:
        default: true
        off_ranges:
          - start_week: 10
            end_week: 12
      cpm: 25
  - channel:
      channel_name: LegacyRadio
      enabled: false
      cpm: 8
```

**Tests:** `tests/test_channel_toggles.py`, `tests/test_ui_channel_toggles.py`.

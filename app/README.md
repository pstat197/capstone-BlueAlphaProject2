# Streamlit UI

This file is intentionally **UI-operations focused**.  
The root [`README.md`](../README.md) is the single source of truth for:

- end-to-end pipeline math
- YAML/schema semantics (`budget_shifts`, correlations, toggles, seasonality)
- CLI behavior and config loading details
- correlation interpretation and theory

Use this file for how to run/debug the app layer only.

---

## Run

From repo root:

```bash
pip install -e .
streamlit run app/streamlit_app.py
```

For tests:

```bash
python -m pytest tests/ -q
```

---

## UI flow (quick)

- `streamlit_app.py` bootstraps `st.session_state.sim_config` from `example.yaml`.
- Form tabs write widget state; `merge_ui_into_config(...)` builds the run payload.
- **Run simulation** calls `run_with_cache(..., run_pipeline)`.
- Results render via `ui_results.py` (charts + correlations + data preview).

---

## Session-state keys you’ll touch most

- `sim_config`: canonical merged config dict.
- `advanced_yaml`: editable YAML panel text.
- `yaml_manual_edit`: prevents silent YAML overwrite when user edits manually.
- `last_df`, `last_corr_results`, `last_run_id`, `last_cache_hit`, `last_hash`, `last_error`: latest run outputs.
- `config_collapsed`: switches between config editor and compact results mode.

---

## Caching notes

- Cache files live under `app/.cache/runs/`.
- Hashing includes `CACHE_VERSION`, normalized config payload, and runtime signature.
- Cache fingerprint preview is non-destructive (does not delete rows while checking).
- Correlation payloads:
  - non-cache runs keep in-memory run correlations
  - cache-hit runs rebuild correlation structures from cached CSV + `sim_config`

---

## Key UI modules

| File | Responsibility |
|------|----------------|
| `streamlit_app.py` | App entrypoint, layout, run/preview flow, reset/cache controls |
| `ui_config_merge.py` | Merge widget state into run config |
| `ui_channel_form.py` | Per-channel editors + curve previews |
| `ui_channel_toggles.py` | Availability/off-range/sticky pause UI + merge |
| `ui_correlations.py` | Manual correlation pair editor |
| `ui_budget_shifts.py` | Manual budget-shift editor |
| `ui_seed_extras.py` | Auto append mode controls above tabs |
| `ui_results.py` | Result charts, correlation panel, YAML snapshot |
| `cache.py` | Disk cache read/write/hash helpers |

For semantics of these controls, see root [`README.md`](../README.md).

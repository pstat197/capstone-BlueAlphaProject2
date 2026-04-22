# Streamlit UI

Modular interface for the marketing simulator. Core logic lives under `scripts/`; this folder only contains the app layer.

## Setup

From the repository root (install the repo in editable mode so imports work without `PYTHONPATH`):

```bash
pip install -e .
```

Alternatively: `pip install -r requirements.txt` and set `PYTHONPATH` to the repo root when running tests or the CLI.

## Run

From the repository root:

```bash
streamlit run app/streamlit_app.py
```

The app also inserts the repo root on `sys.path` for `scripts` imports when you run the file directly; editable install is still recommended for tests and `python -m scripts.main`.

## Behavior

- **Settings** (sidebar): **Night mode** (toolbar stays a light strip so Deploy/menu stay visible), **Colorblind-safe chart colors** (orange / blue / green lines), **Reset to example.yaml**, **Clear simulation cache**.
- **Simulation settings**: one row with **Week range** (no fixed max—matches the simulator), **Run identifier**, **Random seed**.
- **Channels**: add/remove; each channel is an expander with **Remove** on the right of the name row. Numeric fields use text inputs—leave blank to keep the current default (placeholder shows it); **Apply YAML to form** loads the advanced editor into the structured form.
- **Channel availability** (per-channel): each expander has an **Availability** section with an **Enabled** checkbox, toggles for **Apply adstock decay** and **Apply saturation**, and a single **Pause rules** table: each row has a **type** (fixed off every week in range, or sticky Markov random), **start/end** weeks, and for sticky rows **Start P** / **Continue P**. Fixed rules emit as `enabled` / `off_ranges`; sticky rows as `sticky_pause_ranges`. When **Enabled** is off, spend/impressions/revenue for that channel are forced to zero; pauses zero spend/impressions but preserve adstock echo (Policy A). Sticky draws are reproducible from the run seed and channel index.
- **Global effect switches**: below the channel list, two checkboxes control the **global** adstock/saturation gates (written to YAML as `adstock.global` / `saturation.global`). A channel's effect only runs when both its per-channel toggle and the matching global switch are on.
- After **Run simulation**, the form hides; **Edit configuration** and **Download CSV** sit on one row. Optional **overlay** plots normalize the three series onto one chart.
- **Cache:** identical configs load from `app/.cache/runs/`; clear via sidebar when needed.

Bump `CACHE_VERSION` in `app/cache.py` when simulation outputs change for the same config.

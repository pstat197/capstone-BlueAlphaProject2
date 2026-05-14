# capstone-BlueAlphaProject2

See also: [Documentation of Code](https://docs.google.com/document/d/1glQWezaB3eBH13Mxp2eR0Y7SM1zAaVAaS1qHFy2uu-o/edit?usp=sharing)

---

## Setup

From the project root, create a virtual environment (recommended) and install the project in **editable** mode so `scripts` and `app` import without setting `PYTHONPATH`:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -e .
```

This installs dependencies from `pyproject.toml` (same set as `requirements.txt`). If you prefer not to use editable install, you can instead run `pip install -r requirements.txt` and keep the repository root on `PYTHONPATH` when running tests or the CLI.

### Streamlit UI

After `pip install -e .`, from the project root:

```bash
streamlit run app/streamlit_app.py
```

Use the **same Python environment** where you installed dependencies. If the **Bayesian MMM (Meridian)** tab says Meridian is missing, you are almost certainly launching Streamlit with a different interpreter than the one where you installed the optional stack.

**Recommended (uses `.venv` in this repo):**

```bash
pip install -e ".[mmm]"
./scripts/run_streamlit.sh
```

Or explicitly:

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

See [app/README.md](app/README.md) for UI behavior and options.

Requirements (declared in `pyproject.toml` / `requirements.txt`): `numpy`, `pandas`, `matplotlib`, `PyYAML`, `pytest`, `streamlit`, `plotly`.

**Bayesian MMM tab (optional):** TensorFlow + Google Meridian — `pip install -r requirements-meridian.txt` or `pip install -e ".[mmm]"` (Python 3.11–3.13 recommended), then **restart Streamlit from that same environment** (`./scripts/run_streamlit.sh` or `.venv/bin/streamlit`).

---

## Table of contents

- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Streamlit UI](#streamlit-ui)
- [Running tests](#running-tests)
- [Pipeline overview](#pipeline-overview)
- [1. Config & loading](#1-config--loading)
- [2. Spend generation](#2-spend-generation)
- [3. Impressions simulation](#3-impressions-simulation)
- [4. Revenue simulation](#4-revenue-simulation)
- [5. CSV output & full pipeline](#5-csv-output--full-pipeline)

---

## Running the pipeline

From the project root, with a YAML config (e.g. `example.yaml`):

```bash
# Run as a module (recommended)
python -m scripts.main example.yaml

# Or, with explicit flag
python -m scripts.main -c path/to/config.yaml
```

Output CSV is written under `output/` and includes one row per week with per-channel impressions, spend, revenue, and totals.

---

## Running tests

From the project root, run all tests:

```bash
python test.py
```

Run a single test suite:

```bash
python -m tests.test_config
python -m tests.test_spend_generation
python -m tests.test_impressions_simulation
python -m tests.test_revenue_simulation
python -m tests.test_pipeline
```

Test modules live under `tests/` and mirror the pipeline: config/loading, spend generation, impressions simulation, revenue simulation, and full-pipeline tests.

---

## Pipeline overview

The pipeline takes a user YAML config, merges it with defaults, then runs four steps in order. Data flows as follows:

```
YAML config  →  load_config  →  InputConfigurations
                                      ↓
                              generate_spend  →  spend_matrix (weeks × channels)
                                      ↓
                              generate_impressions(config, spend_matrix)  →  impressions_matrix (weeks × channels)
                                      ↓
                              generate_revenue(config, impressions_matrix)  →  revenue_matrix (weeks × channels)
                                      ↓
                              construct_csv  →  DataFrame  →  CSV file in output/
```

- **Input:** Path to a YAML config file (e.g. `example.yaml`).
- **Output:** A CSV with one row per week: `week`, `revenue` (sum across channels), per-channel `{channel}_impressions`, `{channel}_spend`, `{channel}_revenue`, and `total_impressions`, `total_spend`.

Entry point: `scripts.main` (see [Running the pipeline](#running-the-pipeline)).

---

## 1. Config & loading

**Code:** `scripts/config/loader.py`, `scripts/config/default.yaml`, `scripts/config/defaults.py`, `scripts/synth_input_classes/`

- **What it does:** Reads the user-provided YAML config file and deep-merges it with the project default configuration in `scripts/config/default.yaml` (which provides all required fields and sensible defaults). The loader fills any missing per-channel fields from the default channel template, then passes that template into `InputConfigurations.from_yaml_dict(merged, default_channel_template=...)` so the builder can fill any remaining missing config keys without depending on the config package. The merged config yields an `InputConfigurations` object containing metadata (run id), `week_range`, `channel_list`, and an optional `seed`. If the user specifies fewer channels than `number_of_channels`, additional channels are auto-generated based on the template from `default.yaml`.
- **Key behavior:**
  - If a `seed` is specified in the YAML, the global random number generator (RNG) is initialized accordingly, ensuring reproducible results for all downstream stochastic steps.
  - The `number_of_channels` key allows for the dynamic creation of placeholder channels (named "Generated Channel 1", etc.) modeled from `default.yaml`'s template.
  - Each channel is defined by: `channel_name`, `spend_sampling_gamma_params` (shape, scale), `spend_range`, `true_roi`, `baseline_revenue`, `saturation_config`, `adstock_decay_config`, and `noise_variance`. All of these config dicts are filled from `default.yaml` when missing; noise is applied only to non-config fields (e.g. `true_roi`, `spend_range`) when generating extra channels.
- **default.yaml:** Single source of truth for default channel values. The loader and `scripts/config/defaults.py` (via `get_default_channel_template()`) read it; change this file once to affect all default behavior.
- **Output:** A validated `InputConfigurations` object is produced and provided as input to all subsequent pipeline steps.

---

## 2. Spend generation

**Code:** `scripts/spend_simulation/spend_generation.py`

- **What it does:** For each channel and each week, samples one spend value from that channel’s gamma distribution (using `spend_sampling_gamma_params`), then clips to the channel’s `spend_range`. Uses the config RNG (so the config `seed` controls reproducibility).
- **Input:** `InputConfigurations`.
- **Output:** 2D `np.ndarray` of shape `(num_weeks, num_channels)`; rows = weeks, columns = channels, entries = spend amounts.

---

## 3. Impressions simulation

**Code:** `scripts/impressions_simulation/impressions_generation.py`

- **What it does:** Converts weekly spend per channel into impressions per channel per week, using each channel’s **CPM** and impression noise variance:
  - Base impressions: $\text{impressions} = \frac{\text{spend}}{\text{CPM}} \times 1000$.
  - Adds Gaussian noise with variance proportional to the base impressions and the channel’s `noise_variance["impression"]`.
  - Clips results at 0 so impressions are non‑negative.
- **Input:** `InputConfigurations`, `spend_matrix` (weeks × channels).
- **Output:** 2D `np.ndarray` of shape `(num_weeks, num_channels)` (impressions per week per channel).

---

## 4. Revenue simulation

**Code:** `scripts/revenue_simulation/revenue_generation.py`

- **What it does:** Maps impressions to total weekly revenue across all channels:
  - Applies the channel’s `saturation_config` (e.g. linear = `slope` × impressions with default slope 1, hill, diminishing_returns) to impressions.
  - Applies the channel’s `adstock_decay_config` (geometric/exponential, weighted, or linear = uniform moving average over `lag`+1 weeks when `lag` > 0) to model carry‑over of effects across weeks.
  - Scales by `true_roi` and adds `baseline_revenue`.
  - Adds Gaussian revenue noise controlled by `noise_variance["revenue"]` for each channel.
- **Input:** `InputConfigurations`, `impressions_matrix` (weeks × channels).
- **Output:** 1D `np.ndarray` of length `num_weeks` (one total revenue value per week).

---

## 5. CSV output & full pipeline

**Code:** `scripts/main.py` (`construct_csv`, `main`)

- **What it does:** Builds a pandas DataFrame with columns: `week`, total `revenue`, then for **each channel** (in order) `{channel}_impressions`, `{channel}_spend`, and `{channel}_revenue`, then `total_impressions` and `total_spend`. Saves to `output/{run_identifier}_{timestamp}.csv`.
- **Full pipeline:** Load config → generate spend → generate impressions → generate revenue → construct CSV → write file.


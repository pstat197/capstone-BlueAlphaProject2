# capstone-BlueAlphaProject2

See also: [Documentation of Code](https://docs.google.com/document/d/1glQWezaB3eBH13Mxp2eR0Y7SM1zAaVAaS1qHFy2uu-o/edit?usp=sharing)

---

## Setup

From the project root, create a virtual environment (recommended) and install dependencies:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

Requirements: `numpy`, `pandas`, `PyYAML`.

---

## Table of contents

- [Setup](#setup)
- [Pipeline overview](#pipeline-overview)
- [1. Config & loading](#1-config--loading)
- [2. Spend generation](#2-spend-generation)
- [3. Impressions simulation](#3-impressions-simulation)
- [4. Revenue simulation](#4-revenue-simulation)
- [5. CSV output & full pipeline](#5-csv-output--full-pipeline)
- [Running the pipeline](#running-the-pipeline)
- [Running tests](#running-tests)

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
                              generate_revenue(config, impressions_matrix)  →  revenue_vector (weeks,)
                                      ↓
                              construct_csv  →  DataFrame  →  CSV file in output/
```

- **Input:** Path to a YAML config file (e.g. `example.yaml`).
- **Output:** A CSV with one row per week: `week`, `revenue`, per-channel `{channel}_impressions` and `{channel}_spend`, and `total_impressions`, `total_spend`.

Entry point: `scripts/main.py` (see [Running the pipeline](#running-the-pipeline)).

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

- **What it does:** *(To be implemented.)* Should map spend to impressions per channel per week, using channel-specific logic (e.g. saturation, noise).
- **Input:** `InputConfigurations`, `spend_matrix` (weeks × channels).
- **Output:** 2D `np.ndarray` of shape `(num_weeks, num_channels)` (impressions per week per channel).

---

## 4. Revenue simulation

**Code:** `scripts/revenue_simulation/revenue_generation.py`

- **What it does:** *(To be implemented.)* Should map impressions (and possibly config such as baseline, ROI, saturation) to total revenue per week.
- **Input:** `InputConfigurations`, `impressions_matrix` (weeks × channels).
- **Output:** 1D `np.ndarray` of length `num_weeks` (one total revenue value per week).

---

## 5. CSV output & full pipeline

**Code:** `scripts/main.py` (`construct_csv`, `main`)

- **What it does:** Builds a pandas DataFrame with columns: `week`, `revenue`, then for each channel `{channel}_impressions` and `{channel}_spend`, then `total_impressions` and `total_spend`. Saves to `output/{run_identifier}_{timestamp}.csv`.
- **Full pipeline:** Load config → generate spend → generate impressions → generate revenue → construct CSV → write file.

---

## Running the pipeline

From the project root, with a YAML config (e.g. `example.yaml`):

```bash
python scripts/main.py example.yaml
# or
python scripts/main.py -c path/to/config.yaml
```

Output CSV is written under `output/`.

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

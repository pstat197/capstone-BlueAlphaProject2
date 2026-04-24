"""Long-form markdown shown in channel configuration expanders."""

SATURATION_TYPES_GUIDE_MD = """
**Order in the simulator:** saturation runs **first** (impressions → effective media), then adstock, then ROI scaling.

- **`linear`** — Pick for a **simple proportional** link: response grows in line with impressions (good baseline or teaching runs).  
  **Formula:** `slope × impressions`. `slope = 1` is “no curve”; higher amplifies, lower dampens. There is **no** built-in ceiling.

- **`hill`** — Pick when you want a **clear ceiling** and an S-shaped / saturating curve at high volume.  
  **Formula:** `x^slope / (x^slope + K^slope)` (then multiplied by ROI later). **K** sets the **impression scale** where the curve bends (larger K → need more weekly impressions before flattening). **Slope** controls how **sharp** the transition is.

- **`diminishing_returns`** — Pick for a smooth **concave** curve with one main shape parameter (no separate K like Hill).  
  **Formula:** `impressions / (1 + β × impressions)`. Larger **β** → stronger diminishing returns sooner; **β = 0** → no saturation (raw impressions through).
"""

ADSTOCK_TYPES_GUIDE_MD = """
**Order in the simulator:** adstock runs **after** saturation; it **spreads** each week’s effective response over neighboring weeks (carry-over).

- **`linear`** — Pick when memory should be a **flat** average over a fixed window.  
  **Uniform** moving average over **lag + 1** weeks (this week + **lag** prior). **Lag 0** = no carry-over (this week only).

- **`geometric`** and **`exponential`** — Same behavior in this app: weights `1, λ, λ², …` through **lag**, then convolved with the series.  
  Pick for **exponential decay** of ad effects. **λ** closer to **1** → longer memory; closer to **0** → mostly immediate.

- **`weighted`** — Pick when you need a **custom** lag shape (e.g. delayed peak).  
  Enter **comma-separated** weights from **oldest** lag → **newest**; the simulator convolves that kernel with the saturated series.
"""

NOISE_PARAMETERS_GUIDE_MD = """
Both values are **non-negative**. **0** turns that noise off.

**Impression noise** (applied right after spend → CPM → **base impressions** each week)

- For each week, `base_impressions = (spend / CPM) × 1000`.
- Random noise is Gaussian with **standard deviation = √(impression noise) × base_impressions** for that week.
- So the number you enter controls how large random swings are **relative to that week’s expected impressions** — busy weeks get proportionally wider noise.
- Impressions are clipped at **0** after noise.

**Revenue noise** (applied **last** for the channel: after saturation, adstock, ROI, and baseline)

- Let `R` be that week’s channel revenue **before** this noise step.
- Random noise is Gaussian with **standard deviation = √(revenue noise) × |R|**.
- So noise scales with **how big the channel’s contribution already is** that week; small `R` → small absolute noise, large `R` → larger absolute noise (same relative spread).
"""

SEASONALITY_OVERVIEW_MD = """
**How it fits in:** Seasonality multiplies **baseline revenue** each week (after optional linear **trend**). It does **not** change spend or impressions.
"""

SEASONALITY_TYPES_GUIDE_MD = """
- **`none`** — No seasonal multiplier (baseline stays trend-only).

- **`Repeating cycle`** — You set a **cycle length** and a **table** of baseline multipliers (one row per week in the cycle). A line chart previews the shape. The merged YAML stores a **fitted deterministic Fourier** (smooth, reproducible), not the raw table. Edit multipliers in the table (native point-drag on the chart is not available in Streamlit).

- **`sin`** — Single smooth wave (intuitive knobs). When merged, this becomes an equivalent **deterministic Fourier** (one harmonic) so runs stay reproducible.

- **`fourier`** — Either:
  - **Random Fourier** (no saved `coefficients`): `period`, `K`, `scale` (+ optional `seed`) — smooth seasonal deviation drawn each run.
  - **Fitted** (deterministic): `period`, `K`, `intercept`, `coefficients` — least-squares seasonal shape (what **repeating cycle** and **comma pattern** save as).

- **`categorical` (comma pattern)** — You enter repeating multipliers as **comma-separated** values; the app **fits a smoothed Fourier** and stores **only** that `fourier` block. Shorter patterns use fewer harmonics for a smoother curve.
"""

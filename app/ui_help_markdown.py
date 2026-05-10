"""Long-form markdown shown in channel configuration expanders."""

SATURATION_TYPES_GUIDE_MD = """
**Order in the simulator (default):** adstock runs **first** (lag / carry-over on raw impressions), then saturation, then ROI scaling. You can switch to saturation-first under **Advanced → Media response order** in the app or set top-level YAML ``media_transform_order: saturation_first``.

- **`linear`** — Pick for a **simple proportional** link: response grows in line with impressions (good baseline or teaching runs).  
  **Formula:** `slope × impressions`. `slope = 1` is “no curve”; higher amplifies, lower dampens. There is **no** built-in ceiling.

- **`hill`** — Pick when you want a **clear ceiling** and an S-shaped / saturating curve at high volume.  
  **Formula:** `x^slope / (x^slope + K^slope)` (Meridian-style Hill; then multiplied by ROI later). **`K` must use the same units as weekly impressions** in this simulator: impressions = `(weekly spend) / CPM × 1000`. At `slope = 1`, response is **half** of its ceiling when weekly impressions equal **`K`**. If **`K` is far below** typical weekly impressions, the curve stays **≈ 1** (saturation looks like a constant). Example: spend **50,000** with **CPM 10** → about **5,000,000** impressions/week — choose **`K`** on that order (e.g. millions), not **50,000**. Use **Suggest K** in the Channels tab (Hill selected) to fill **`K`** from midpoint spend and CPM. **Slope** controls how **sharp** the transition is.

- **`diminishing_returns`** — Pick for a smooth **concave** curve with one main shape parameter (no separate K like Hill).  
  **Formula:** `impressions / (1 + β × impressions)`. Larger **β** → stronger diminishing returns sooner; **β = 0** → no saturation (raw impressions through).
"""

ADSTOCK_TYPES_GUIDE_MD = """
**Order in the simulator (default):** adstock runs **before** saturation; it **spreads** each week’s impressions over neighboring weeks (carry-over), then saturation maps that to effective response. Reversible via **Advanced → Media response order** / ``media_transform_order`` in YAML.

- **`linear`** — Pick when memory should be a **flat** average over a fixed window.  
  **Uniform** moving average over **lag + 1** weeks (this week + **lag** prior). **Lag 0** = no carry-over (this week only).

- **`geometric`** and **`exponential`** — Same behavior in this app: weights `1, λ, λ², …` through **lag**, then convolved with the series.  
  Pick for **exponential decay** of ad effects. **λ** closer to **1** → longer memory; closer to **0** → mostly immediate.

- **`weighted`** — Pick when you need a **custom** lag shape (e.g. delayed peak).  
  Enter **comma-separated** weights from **oldest** lag → **newest**; the simulator convolves that kernel with the weekly impression series.
"""

NOISE_PARAMETERS_GUIDE_MD = """
The value is **non-negative**. **0** turns outcome revenue noise off.

**Revenue noise (outcome-level)** — applied **once** to **total** weekly revenue after forming `media sum + (baseline + trend) × outcome seasonality` (additive structure per [Meridian model spec](https://developers.google.com/meridian/docs/advanced-modeling/model-spec)).

- Each week, one draw **ε ~ N(0, σ²)** is added to that week’s total, with **σ = √(revenue noise)** in the **same units as revenue** (homoskedastic; σ does **not** scale with the level of `R`).
- The field is the shock **variance** σ² (squared KPI units).

Set **`outcome_revenue.noise_variance.revenue`** in YAML (or omit the whole ``outcome_revenue`` block to inherit the **first channel’s** revenue noise for the outcome step). Per-channel `noise_variance.revenue` is the fallback source for that single outcome draw, not a separate shock per channel.
"""

SEASONALITY_OVERVIEW_MD = """
**How it fits in:** Meridian’s **μ_t** is a knot-interpolated time-varying intercept (see [Model specification — μ_t parameters](https://developers.google.com/meridian/docs/advanced-modeling/model-spec)). Here, **μ_t^sim** = `(baseline + trend)` × your outcome seasonality multiplier **σ_t**; total mean revenue is **μ_t^sim + sum of media contributions** (+ noise). It does **not** change spend or impressions.
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

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

SUBSCRIPTION_PARAMETERS_GUIDE_MD = """
Subscriptions use the **same** saturation and adstock transforms as revenue (the media effect shape is a channel property), but apply a different scale:

**Pipeline:** impressions → saturation → adstock → × conversion_rate + baseline_subscriptions + noise → round to integer

- **Conversion rate**: fraction of effective impressions that become subscribers (e.g., 0.001 = 1 per 1000).
- **Baseline subscriptions**: organic weekly sign-ups independent of media.
- **Subscription noise**: random variation, same mechanism as revenue noise but applied to subscription counts.
- Output is **clipped to ≥ 0** and **rounded to the nearest integer**.
"""

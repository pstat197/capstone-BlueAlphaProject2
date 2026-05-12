import type {
  BudgetShift,
  ChannelDef,
  ChannelEntry,
  CorrelationEntry,
  SimConfig,
} from "@/types/api";

export const CHART_COLORS = {
  primary: "#1d63ed",
  secondary: "#f39c59",
  tertiary: "#10a37f",
} as const;

export const CHART_PAL_CVD = ["#e69f00", "#0072b2", "#009e73"] as const;

export function defaultChannel(name = "New channel"): ChannelEntry {
  return {
    channel: {
      channel_name: name,
      true_roi: 2.5,
      baseline_revenue: 5000,
      trend_slope: 0.0,
      cpm: 10.0,
      spend_range: [1000, 50000],
      noise_variance: { revenue: 1_000_000.0 },
      saturation_config: { type: "linear", slope: 1.0, K: 2_550_000.0, beta: 0.5 },
      adstock_decay_config: { type: "geometric", lambda: 0.5, lag: 10, weights: [1.0] },
      spend_sampling_gamma_params: { shape: 2.5, scale: 1000 },
    },
  };
}

export function getChannel(entry: ChannelEntry): ChannelDef {
  return entry?.channel ?? ({ channel_name: "" } as ChannelDef);
}

export function setChannel(entry: ChannelEntry, ch: ChannelDef): ChannelEntry {
  return { ...entry, channel: ch };
}

export function blankConfig(): SimConfig {
  return {
    run_identifier: "Demo run",
    week_range: 26,
    seed: 0,
    channel_list: [],
    correlations: [],
  };
}

/** Replace channel at index with a partially-updated channel def. */
export function updateChannelAt(
  config: SimConfig,
  index: number,
  patch: Partial<ChannelDef>,
): SimConfig {
  const list = [...(config.channel_list ?? [])];
  if (index < 0 || index >= list.length) return config;
  const current = getChannel(list[index]!);
  list[index] = setChannel(list[index]!, { ...current, ...patch });
  return { ...config, channel_list: list };
}

/** Insert a new default channel at the end. */
export function appendChannel(config: SimConfig, channel?: ChannelEntry): SimConfig {
  const list = [...(config.channel_list ?? []), channel ?? defaultChannel()];
  return { ...config, channel_list: list };
}

/** Remove channel at index. Returns the new config. */
export function removeChannelAt(config: SimConfig, index: number): SimConfig {
  const list = [...(config.channel_list ?? [])];
  if (index < 0 || index >= list.length) return config;
  list.splice(index, 1);
  return { ...config, channel_list: list };
}

/**
 * Deep-clone channel at `index`, append "(copy)" to its name, and insert
 * right after the original. Returns the new config and the index of the
 * inserted copy so the caller can navigate to it.
 *
 * We `JSON.parse(JSON.stringify(...))` because every nested object
 * (saturation_config, adstock_decay_config, noise_variance, etc.) must be
 * copied independently — otherwise edits to the duplicate would mutate
 * the original.
 */
export function duplicateChannelAt(
  config: SimConfig,
  index: number,
): { config: SimConfig; newIndex: number } {
  const list = [...(config.channel_list ?? [])];
  if (index < 0 || index >= list.length) return { config, newIndex: index };
  const original = list[index]!;
  const clone = JSON.parse(JSON.stringify(original)) as ChannelEntry;
  const ch = getChannel(clone);
  if (ch.channel_name) {
    ch.channel_name = `${ch.channel_name} (copy)`;
  }
  clone.channel = ch;
  const newIndex = index + 1;
  list.splice(newIndex, 0, clone);
  return { config: { ...config, channel_list: list }, newIndex };
}

/**
 * Suggest a Hill K from a channel's spend midpoint and CPM. Matches the
 * Streamlit "Suggest K" button intent: K ≈ typical weekly impressions, which
 * is roughly (mean spend / CPM) × 1000. Returns `null` if inputs are missing
 * or non-positive (UI hides the button in that case).
 */
export function suggestHillK(ch: ChannelDef): number | null {
  const spend = ch.spend_range ?? [0, 0];
  const meanSpend = ((spend[0] ?? 0) + (spend[1] ?? 0)) / 2;
  const cpm = ch.cpm;
  if (!cpm || cpm <= 0 || meanSpend <= 0) return null;
  const k = (meanSpend / cpm) * 1000;
  if (!Number.isFinite(k) || k <= 0) return null;
  // Round to a nice human-friendly step.
  const step = k > 1e6 ? 50000 : k > 1e5 ? 5000 : 500;
  return Math.round(k / step) * step;
}

/** Append an empty correlation pair. */
export function appendCorrelation(config: SimConfig, pair?: CorrelationEntry): SimConfig {
  const next: CorrelationEntry =
    pair ?? ({ channels: ["", ""], rho: 0.0 } as CorrelationEntry);
  const list = [...(config.correlations ?? []), next];
  return { ...config, correlations: list };
}

export function updateCorrelationAt(
  config: SimConfig,
  index: number,
  patch: Partial<CorrelationEntry>,
): SimConfig {
  const list = [...(config.correlations ?? [])];
  if (index < 0 || index >= list.length) return config;
  list[index] = { ...list[index]!, ...patch } as CorrelationEntry;
  return { ...config, correlations: list };
}

export function removeCorrelationAt(config: SimConfig, index: number): SimConfig {
  const list = [...(config.correlations ?? [])];
  if (index < 0 || index >= list.length) return config;
  list.splice(index, 1);
  return { ...config, correlations: list };
}

/** Append a new budget-shift rule (defaults to `scale 1.0 from week 1 to week_range`). */
export function appendBudgetShift(config: SimConfig, rule?: BudgetShift): SimConfig {
  const wr =
    typeof config.week_range === "number" && config.week_range > 0 ? config.week_range : 52;
  const next: BudgetShift =
    rule ?? ({ type: "scale", start_week: 1, end_week: wr, factor: 1.1 } as BudgetShift);
  const list = [...(config.budget_shifts ?? []), next];
  return { ...config, budget_shifts: list };
}

export function updateBudgetShiftAt(
  config: SimConfig,
  index: number,
  patch: Partial<BudgetShift>,
): SimConfig {
  const list = [...(config.budget_shifts ?? [])];
  if (index < 0 || index >= list.length) return config;
  list[index] = { ...list[index]!, ...patch } as BudgetShift;
  return { ...config, budget_shifts: list };
}

export function removeBudgetShiftAt(config: SimConfig, index: number): SimConfig {
  const list = [...(config.budget_shifts ?? [])];
  if (index < 0 || index >= list.length) return config;
  list.splice(index, 1);
  return { ...config, budget_shifts: list };
}

export function totalSeriesSafe(values: Array<number | null | undefined>): number {
  let acc = 0;
  for (const v of values) {
    if (v == null || Number.isNaN(v) || !Number.isFinite(v)) continue;
    acc += v;
  }
  return acc;
}

export function formatCurrency(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v) || !Number.isFinite(v)) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(v);
}

export function formatNumber(v: number | null | undefined, digits = 0): string {
  if (v == null || Number.isNaN(v) || !Number.isFinite(v)) return "—";
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
  }).format(v);
}

export function formatHash(h: string | null | undefined): string {
  if (!h) return "";
  return `…${h.slice(-8)}`;
}

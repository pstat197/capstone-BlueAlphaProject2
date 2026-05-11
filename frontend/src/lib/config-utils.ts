import type { ChannelDef, ChannelEntry, SimConfig } from "@/types/api";

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
      true_roi: 2.0,
      baseline_revenue: 5000,
      cpm: 15,
      spend_range: [1500, 30000],
      noise_variance: { impression: 0.05, revenue: 0.1 },
      saturation_config: { type: "linear", slope: 1.0, K: 50000, beta: 0.5 },
      adstock_decay_config: { type: "linear", lambda: 0.5, lag: 4, weights: [1.0] },
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

/* JSON shapes returned by the FastAPI server in `server/main.py`. Keep in sync. */

export type SimConfig = Record<string, unknown> & {
  run_identifier?: string;
  week_range?: number;
  seed?: number;
  channel_list?: Array<ChannelEntry>;
  correlations?: Array<CorrelationEntry>;
};

export interface ChannelEntry {
  channel: ChannelDef;
}

export interface ChannelDef {
  channel_name: string;
  true_roi?: number;
  baseline_revenue?: number;
  cpm?: number;
  spend_range?: [number, number];
  noise_variance?: { impression?: number; revenue?: number };
  saturation_config?: {
    type?: "linear" | "hill" | "diminishing_returns";
    slope?: number;
    K?: number;
    beta?: number;
  };
  adstock_decay_config?: {
    type?: "linear" | "geometric" | "exponential" | "weighted";
    lambda?: number;
    lag?: number;
    weights?: number[];
  };
  spend_sampling_gamma_params?: { shape?: number; scale?: number };
}

export interface CorrelationEntry {
  channels: [string, string];
  rho: number;
}

export interface ExampleConfigResponse {
  config: SimConfig;
  yaml_text: string;
}

export interface RunChannelSeries {
  name: string;
  revenue: number[];
  spend: number[];
  impressions: number[];
}

export interface RunPreview {
  columns: string[];
  rows: Array<Record<string, number | string | null>>;
}

export interface PairwiseSummary {
  pair: [string, string];
  configured_rho: number;
  observed_rho: number;
  drift: number;
  drift_label: string;
}

export interface CorrelationResult {
  channel_names: string[];
  static_corr: number[][] | null;
  rolling_corr: number[][][] | null;
  drift: number[][] | null;
  avg_abs_corr: Record<string, number | null>;
  most_correlated_channel: string;
  pairwise_summary: PairwiseSummary[];
  window: number;
}

export interface RunResponse {
  run_id: string;
  config_hash: string;
  cache_hit: boolean;
  config: SimConfig;
  weeks: number[];
  totals: { revenue: number[]; spend: number[]; impressions: number[] };
  channels: RunChannelSeries[];
  preview: RunPreview;
  correlation: CorrelationResult | null;
}

export interface RunListItem {
  config_hash: string;
  run_identifier: string;
  created_at: string | null;
  last_seen_at: string | null;
  last_was_cache_hit: boolean;
}

export interface MeridianStatus {
  installed: boolean;
  ui_status: string;
  message: string;
}

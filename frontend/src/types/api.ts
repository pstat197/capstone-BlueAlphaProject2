/* JSON shapes returned by the FastAPI server in `server/main.py`. Keep in sync. */

export type SimConfig = Record<string, unknown> & {
  run_identifier?: string;
  week_range?: number;
  seed?: number;
  number_of_channels?: number;
  media_transform_order?: "adstock_first" | "saturation_first";
  outcome_revenue?: OutcomeRevenue;
  adstock?: GlobalKillSwitch;
  saturation?: GlobalKillSwitch;
  channel_list?: Array<ChannelEntry>;
  correlations?: Array<CorrelationEntry>;
  correlations_auto_mode?: CorrelationsAutoMode;
  budget_shifts?: Array<BudgetShift>;
  budget_shifts_auto_mode?: BudgetShiftsAutoMode;
};

export interface OutcomeRevenue {
  baseline_revenue?: number;
  trend_slope?: number;
  seasonality_config?: SeasonalityConfig;
  noise_variance?: { revenue?: number };
}

/** Optional global kill-switch (e.g. `adstock: { global: false }`). */
export interface GlobalKillSwitch {
  global?: boolean;
}

export interface ChannelEntry {
  channel: ChannelDef;
}

/** Per-channel availability:
 *   bool        → channel fully on/off for the whole run
 *   object      → on by default, off during inclusive `off_ranges`
 *   undefined   → fully on (default)
 */
export type ChannelEnabled =
  | boolean
  | {
      default?: boolean;
      off_ranges?: Array<{ start_week?: number; end_week?: number }>;
    };

export interface StickyPauseRange {
  start_week?: number;
  end_week?: number;
  /** Probability the pause starts on each week within the window. */
  start_probability?: number;
  /** Probability the pause continues given it started. */
  continue_probability?: number;
}

export interface ChannelDef {
  channel_name: string;
  true_roi?: number;
  baseline_revenue?: number;
  trend_slope?: number;
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
    type?: "linear" | "geometric" | "exponential" | "binomial" | "weighted";
    lambda?: number;
    lag?: number;
    weights?: number[];
  };
  spend_sampling_gamma_params?: { shape?: number; scale?: number };
  seasonality_config?: SeasonalityConfig;
  enabled?: ChannelEnabled;
  sticky_pause_ranges?: StickyPauseRange[];
  adstock_enabled?: boolean;
  saturation_enabled?: boolean;
}

/**
 * Seasonality is intentionally loose-typed at the React layer — the
 * backend supports multiple modes (sin → normalized to fourier, basic
 * cycles, raw deterministic fourier with coefficients, and arbitrary
 * patterns). The structured Streamlit UI passes them through verbatim.
 * We expose a minimal schema-aware shape and let unknown keys ride
 * along so the YAML pane can edit anything the backend accepts.
 */
export interface SeasonalityConfig {
  type?: "none" | "basic" | "sin" | "fourier";
  amplitude?: number;
  phase?: number;
  cycles?: Array<{ period_weeks?: number; amplitude?: number; phase?: number }>;
  [key: string]: unknown;
}

/** Budget-shift rule. The backend tolerates omission of `end_week`
 *  to mean "from start_week through the end of the run". */
export type BudgetShift =
  | {
      type: "scale";
      start_week?: number;
      end_week?: number;
      factor?: number;
    }
  | {
      type: "scale_channel";
      start_week?: number;
      end_week?: number;
      channel_name?: string;
      factor?: number;
    }
  | {
      type: "reallocate";
      start_week?: number;
      end_week?: number;
      from_channel?: string;
      to_channel?: string;
      fraction?: number;
    };

export type BudgetShiftsAutoMode = "none" | "global" | "global_and_channel";
export type CorrelationsAutoMode = "none" | "random";

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

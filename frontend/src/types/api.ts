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

export interface GroundTruthChannel {
  channel_name: string;
  enabled?: boolean;
  off_ranges?: Array<[number, number]>;
  sticky_pause_ranges?: Array<{
    start_week: number;
    end_week: number;
    start_probability: number;
    continue_probability: number;
  }>;
  true_roi?: number;
  baseline_revenue?: number;
  trend_slope?: number;
  seasonality_config?: Record<string, unknown> | null;
  saturation_enabled?: boolean;
  saturation_config?: Record<string, unknown> | null;
  adstock_enabled?: boolean;
  adstock_decay_config?: Record<string, unknown> | null;
  spend_range?: [number, number];
  cpm?: number;
  spend_sampling_gamma_params?: Record<string, unknown> | null;
  noise_variance?: Record<string, unknown> | null;
}

/** Snapshot of the generative parameters the simulator was given.
 *  Mirrors `scripts/ground_truth_io.extract_ground_truth`. */
export interface GroundTruth {
  ground_truth_version: number;
  generated_at: string;
  run_identifier: string;
  week_range: number;
  seed?: number | null;
  outcome_revenue: {
    baseline_revenue: number;
    trend_slope: number;
    seasonality_config?: Record<string, unknown> | null;
    noise_variance?: Record<string, unknown> | null;
    total_revenue_mechanism?: { description?: string };
  };
  global_toggles: {
    adstock_global: boolean;
    saturation_global: boolean;
    media_transform_order: string;
  };
  correlations?: unknown;
  budget_shifts?: unknown;
  channels: GroundTruthChannel[];
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
  ground_truth: GroundTruth | null;
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
  /** "available" once Meridian imports cleanly, "unavailable" otherwise. */
  ui_status: "available" | "unavailable" | string;
  message: string;
  /** Import error text when `installed` is false. */
  error?: string | null;
}

/* -----------------------------------------------------------------------
 * Bayesian MMM (Meridian) — see server/mmm.py for the source-of-truth shapes.
 * --------------------------------------------------------------------- */

export type MmmProfile = "fast" | "balanced" | "slow" | "custom";

export type MmmJobStatus = "queued" | "running" | "succeeded" | "failed";

export type MmmJobStage =
  | "queued"
  | "preparing"
  | "sampling"
  | "diagnostics"
  | "serializing"
  | "done"
  | "error";

export interface MmmFitRequest {
  config_hash: string;
  profile: MmmProfile;
  n_chains?: number;
  n_adapt?: number;
  n_burnin?: number;
  n_keep?: number;
  n_prior?: number;
  seed?: number;
  enable_aks?: boolean;
  knots?: number[] | null;
  channel_roi_mus?: number[] | null;
  channel_roi_sigmas?: number[] | null;
}

export interface MmmJob {
  job_id: string;
  status: MmmJobStatus;
  stage: MmmJobStage;
  config_hash: string;
  cache_key: string;
  profile: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  note: string | null;
  cache_hit: boolean;
  n_channels: number;
  n_weeks: number;
  channels: string[];
}

/** Single row in the Meridian "predictive accuracy" table (R², MAPE, wMAPE, ...). */
export type MmmFitMetricRow = Record<string, number | string | boolean | null>;

export interface MmmRoiForestRow {
  channel: string;
  /** Posterior mean ROI ($ revenue per $1 spend). */
  mean: number;
  /** Lower bound of the 50% credible interval. */
  ci_low: number;
  /** Upper bound of the 50% credible interval. */
  ci_high: number;
  /** True ROI from the simulator config when available. */
  true_roi?: number | null;
}

export interface MmmRoiForest {
  rows: MmmRoiForestRow[];
  rhat_by_channel: Record<string, number | null>;
  error?: string | null;
}

export interface MmmBudgetRow {
  channel: string;
  spend_baseline_total: number;
  spend_optimized_total: number;
  spend_baseline_weekly: number;
  spend_optimized_weekly: number;
  delta_weekly: number;
  change_pct: number;
}

export interface MmmBudgetPieSlice {
  channel: string;
  value: number;
  share: number;
}

export interface MmmBudgetOptimization {
  rows?: MmmBudgetRow[];
  pies?: { current: MmmBudgetPieSlice[]; optimized: MmmBudgetPieSlice[] };
  total_spend_baseline?: number;
  total_spend_optimized?: number;
  n_weeks?: number;
  /** Set when Meridian's BudgetOptimizer raised; the rest of the payload is empty. */
  error?: string | null;
}

export interface MmmFitResults {
  summary: {
    rhat_max: number | null;
    note: string | null;
    ok: boolean;
  };
  channels: string[];
  n_weeks: number;
  fit_metrics: MmmFitMetricRow[];
  fit_metrics_error?: string | null;
  roi_forest: MmmRoiForest;
  budget_optimization: MmmBudgetOptimization;
}

export interface MmmFitResultsResponse {
  job: MmmJob;
  results: MmmFitResults | null;
}

/** Path is a JSON-Pointer-ish array (e.g. ["channel_list", 0, "channel", "true_roi"]).
 *  Empty for global issues. */
export interface ConfigIssue {
  path: Array<string | number>;
  message: string;
  severity: "error" | "warning";
  /** Coarse rollup so the UI can light up section badges. */
  section: "general" | "outcome" | "channels" | "correlations" | "budget_shifts" | null;
}

export interface ValidateConfigResponse {
  ok: boolean;
  issues: ConfigIssue[];
}

/* Saturation + adstock math ported from
 * `scripts/revenue_simulation/revenue_generation.py`. Used by the channel
 * curve preview so users can see the shape of their current settings live.
 *
 * Keep these formulas in sync with the Python implementation. The point of
 * the preview is curve shape, not absolute response magnitude — we don't
 * apply ROI here, just the per-week transform that feeds it.
 */

import type { ChannelDef } from "@/types/api";

export type SaturationConfig = NonNullable<ChannelDef["saturation_config"]>;
export type AdstockConfig = NonNullable<ChannelDef["adstock_decay_config"]>;

/**
 * Apply the configured saturation transform to a single impressions value.
 * Returns the value untouched when `type` is unknown so the chart doesn't
 * blow up while the user is mid-typing.
 */
export function applySaturation(x: number, cfg: SaturationConfig | undefined): number {
  if (!cfg || !cfg.type) {
    return cfg && typeof cfg.slope === "number" ? cfg.slope * x : x;
  }
  if (cfg.type === "linear") {
    const slope = typeof cfg.slope === "number" ? cfg.slope : 1.0;
    return slope * x;
  }
  if (cfg.type === "hill") {
    const slope = typeof cfg.slope === "number" ? cfg.slope : 1.0;
    const K = typeof cfg.K === "number" && cfg.K > 0 ? cfg.K : 1.0;
    if (x <= 0) return 0;
    const xs = Math.pow(x, slope);
    const ks = Math.pow(K, slope);
    return xs / (xs + ks);
  }
  if (cfg.type === "diminishing_returns") {
    const beta = typeof cfg.beta === "number" ? cfg.beta : 0;
    return x / (1.0 + beta * x);
  }
  return x;
}

/**
 * Compute the per-lag adstock kernel for the configured type. Returns
 * weights w[0..L] summing to 1, in the same order as the Python convolution
 * (w[0] = current week, w[1] = previous week, ...). Defaults to a
 * single 1.0 spike on unknown types.
 */
export function adstockKernel(cfg: AdstockConfig | undefined): number[] {
  if (!cfg) return [1.0];
  const type = cfg.type ?? "geometric";
  const lag = clampInt(cfg.lag, 0, 200);

  if (type === "linear") {
    if (lag === 0) return [1.0];
    const n = lag + 1;
    return new Array<number>(n).fill(1.0 / n);
  }

  if (type === "geometric" || type === "exponential") {
    const alpha = clamp01(cfg.lambda ?? 0.5);
    if (alpha <= 0) {
      const arr = new Array<number>(lag + 1).fill(0);
      arr[0] = 1.0;
      return arr;
    }
    const raw = Array.from({ length: lag + 1 }, (_, s) => Math.pow(alpha, s));
    return normalize(raw);
  }

  if (type === "binomial") {
    const alpha = clamp01(cfg.lambda ?? 0.5);
    const L = lag;
    if (alpha <= 0) {
      const arr = new Array<number>(L + 1).fill(0);
      arr[0] = 1.0;
      return arr;
    }
    if (alpha >= 1) {
      const n = L + 1;
      return new Array<number>(n).fill(1.0 / n);
    }
    const alphaStar = 1.0 / alpha - 1.0;
    const raw = Array.from({ length: L + 1 }, (_, s) => {
      const base = Math.max(0, 1 - s / (1 + L));
      return Math.pow(base, alphaStar);
    });
    return normalize(raw);
  }

  if (type === "weighted") {
    const weights = Array.isArray(cfg.weights) && cfg.weights.length > 0 ? cfg.weights : [1.0];
    return normalize(weights);
  }

  return [1.0];
}

/**
 * Pick a sensible x-axis upper bound for the saturation preview based on
 * the channel's spend range and CPM (so we cover the realistic operating
 * range). Falls back to a generic large value if those aren't set yet.
 */
export function saturationXMax(channel: ChannelDef): number {
  const spend = channel.spend_range ?? [0, 0];
  const max = Math.max(spend[0] ?? 0, spend[1] ?? 0);
  const cpm = channel.cpm ?? 0;
  if (max <= 0 || cpm <= 0) {
    // Hill K is a useful fallback when the user has set it directly.
    const k = channel.saturation_config?.K;
    if (typeof k === "number" && k > 0) return k * 2;
    return 5_000_000;
  }
  return (max / cpm) * 1000 * 1.5; // 1.5× max-spend impressions
}

export function saturationCurvePoints(channel: ChannelDef, samples = 60): Array<{ x: number; y: number }> {
  const xMax = saturationXMax(channel);
  const cfg = channel.saturation_config;
  const out: Array<{ x: number; y: number }> = [];
  for (let i = 0; i <= samples; i += 1) {
    const x = (xMax * i) / samples;
    out.push({ x, y: applySaturation(x, cfg) });
  }
  return out;
}

function clamp01(v: number | undefined): number {
  if (v === undefined || !Number.isFinite(v)) return 0.5;
  return Math.max(0, Math.min(1, v));
}

function clampInt(v: number | undefined, lo: number, hi: number): number {
  if (v === undefined || !Number.isFinite(v)) return lo;
  return Math.max(lo, Math.min(hi, Math.floor(v)));
}

function normalize(arr: number[]): number[] {
  const sum = arr.reduce((a, b) => a + b, 0);
  if (sum <= 0) {
    const out = new Array<number>(arr.length).fill(0);
    if (out.length > 0) out[0] = 1;
    return out;
  }
  return arr.map((v) => v / sum);
}

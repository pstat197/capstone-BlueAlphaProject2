/**
 * Pure-TS port of the deterministic Fourier helpers in
 * `scripts/revenue_simulation/seasonality_fit.py`. Used so the UI can
 * render an *exactly* equivalent preview without round-tripping to the
 * backend on every keystroke.
 *
 * The model is:
 *   m(t) = 1 + intercept + Σ_{k=1..K} a_k * sin(2πkt/P) + b_k * cos(2πkt/P)
 */

export interface FourierConfig {
  type?: "fourier";
  period?: number;
  K?: number;
  intercept?: number;
  /** Array of [a_k, b_k] pairs, length should equal K. */
  coefficients?: Array<[number, number]>;
}

export function evaluateDeterministicFourier(
  t: number[] | Float64Array,
  cfg: FourierConfig,
): Float64Array {
  const period = Math.max(1, Math.trunc(Number(cfg.period ?? 52)));
  const intercept = Number.isFinite(Number(cfg.intercept)) ? Number(cfg.intercept) : 0;
  const coeffs = cfg.coefficients ?? [];
  const out = new Float64Array(t.length);
  for (let i = 0; i < t.length; i += 1) {
    const ti = Number(t[i]);
    let s = intercept;
    for (let k = 1; k <= coeffs.length; k += 1) {
      const pair = coeffs[k - 1];
      if (!pair) continue;
      const ak = Number(pair[0]);
      const bk = Number(pair[1]);
      if (!Number.isFinite(ak) && !Number.isFinite(bk)) continue;
      const ang = (2 * Math.PI * k * ti) / period;
      s += (Number.isFinite(ak) ? ak : 0) * Math.sin(ang);
      s += (Number.isFinite(bk) ? bk : 0) * Math.cos(ang);
    }
    out[i] = 1 + s;
  }
  return out;
}

/** Equivalent to `sin_to_deterministic_fourier` in Python. */
export function sinToFourier(input: {
  amplitude?: number;
  period?: number;
  phase?: number;
}): Required<Pick<FourierConfig, "type" | "period" | "K" | "intercept" | "coefficients">> {
  const amplitude = Number(input.amplitude ?? 0.2);
  let period = Math.trunc(Number(input.period ?? 52));
  if (!Number.isFinite(period) || period < 1) period = 52;
  const phase = Number(input.phase ?? 0);
  const a1 = amplitude * Math.cos((2 * Math.PI * phase) / period);
  const b1 = amplitude * Math.sin((2 * Math.PI * phase) / period);
  return {
    type: "fourier",
    period,
    K: 1,
    intercept: 0,
    coefficients: [[a1, b1]],
  };
}

/** Build the (t, multiplier) points for plotting. Default window 2× period. */
export function fourierPreviewPoints(
  cfg: FourierConfig,
  options: { weeks?: number } = {},
): Array<{ week: number; multiplier: number }> {
  const period = Math.max(1, Math.trunc(Number(cfg.period ?? 52)));
  const weeks = Math.max(period, Math.trunc(options.weeks ?? period * 2));
  const t = new Float64Array(weeks + 1);
  for (let i = 0; i <= weeks; i += 1) t[i] = i;
  const m = evaluateDeterministicFourier(t, cfg);
  const out: Array<{ week: number; multiplier: number }> = [];
  for (let i = 0; i <= weeks; i += 1) out.push({ week: i, multiplier: m[i] });
  return out;
}

/** Force the `coefficients` array length to match K (pad with zeros / truncate). */
export function reshapeCoefficients(
  coeffs: Array<[number, number]> | undefined,
  K: number,
): Array<[number, number]> {
  const safeK = Math.max(1, Math.trunc(K));
  const src = coeffs ?? [];
  const out: Array<[number, number]> = [];
  for (let i = 0; i < safeK; i += 1) {
    const pair = src[i];
    out.push([Number(pair?.[0] ?? 0), Number(pair?.[1] ?? 0)]);
  }
  return out;
}

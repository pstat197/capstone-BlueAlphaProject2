import { Info, Plus, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tooltip } from "@/components/ui/tooltip";
import { getChannel, setChannel } from "@/lib/config-utils";
import {
  fourierPreviewPoints,
  reshapeCoefficients,
  sinToFourier,
  type FourierConfig,
} from "@/lib/fourier";
import { useConfig } from "@/state/config-store";
import type {
  OutcomeRevenue,
  SeasonalityConfig,
  SimConfig,
} from "@/types/api";

type SeasonalityMode = "none" | "sin" | "fourier" | "advanced";

const OUTCOME_TARGET = "__outcome__";

function detectMode(cfg: SeasonalityConfig | undefined): SeasonalityMode {
  if (!cfg || Object.keys(cfg).length === 0) return "none";
  if (cfg.type === "sin") return "sin";
  if (cfg.type === "fourier") {
    const hasCoeffs = Array.isArray(
      (cfg as { coefficients?: unknown }).coefficients,
    );
    return hasCoeffs ? "fourier" : "advanced";
  }
  return "advanced";
}

function readNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return undefined;
}

function NumInput({
  id,
  value,
  onChange,
  step,
  min,
  max,
  asInt = false,
}: {
  id: string;
  value: number | undefined;
  onChange: (v: number | undefined) => void;
  step: number;
  min?: number;
  max?: number;
  asInt?: boolean;
}) {
  return (
    <Input
      id={id}
      type="number"
      step={step}
      min={min}
      max={max}
      value={value ?? ""}
      onChange={(e) => {
        const raw = e.target.value;
        if (raw === "") return onChange(undefined);
        const n = asInt ? parseInt(raw, 10) : parseFloat(raw);
        onChange(Number.isFinite(n) ? n : undefined);
      }}
    />
  );
}

/**
 * Pick a "target" for the seasonality config: outcome path or any channel.
 * Editing per-channel seasonality previously required jumping into the YAML
 * pane; this picker pipes the same editor to either slot.
 */
interface TargetSpec {
  kind: "outcome" | "channel";
  channelIndex?: number;
  label: string;
}

function buildTargets(config: SimConfig): TargetSpec[] {
  const list = config.channel_list ?? [];
  return [
    { kind: "outcome", label: "Outcome (total revenue)" },
    ...list.map((entry, idx) => ({
      kind: "channel" as const,
      channelIndex: idx,
      label: getChannel(entry).channel_name || `Channel ${idx + 1}`,
    })),
  ];
}

function readSeasonality(config: SimConfig, target: TargetSpec): SeasonalityConfig {
  if (target.kind === "outcome") {
    const outcome = (config.outcome_revenue ?? {}) as OutcomeRevenue;
    return (outcome.seasonality_config ?? {}) as SeasonalityConfig;
  }
  const entry = config.channel_list?.[target.channelIndex ?? -1];
  if (!entry) return {};
  return (getChannel(entry).seasonality_config ?? {}) as SeasonalityConfig;
}

/**
 * Full seasonality editor.
 *
 * Modes:
 *   none      – no seasonality at all
 *   sin       – single harmonic (amplitude / period / phase)
 *   fourier   – arbitrary deterministic Fourier: period, K, intercept,
 *               and a list of (a_k, b_k) coefficient pairs. Live preview
 *               chart re-evaluates the same formula the Python pipeline
 *               uses (`evaluate_deterministic_fourier`).
 *   advanced  – categorical patterns / random Fourier (no coefficients);
 *               read-only banner pointing to the YAML pane.
 *
 * Targets:
 *   outcome   – `outcome_revenue.seasonality_config`
 *   channel N – `channel_list[N].channel.seasonality_config`
 */
export function SeasonalityPanel() {
  const { config, patchConfig, setConfig } = useConfig();
  const targets = useMemo(() => buildTargets(config), [config]);
  const [target, setTarget] = useTargetState(targets);

  const sea = readSeasonality(config, target);
  const mode = detectMode(sea);

  const writeSeasonality = (next: SeasonalityConfig) => {
    if (target.kind === "outcome") {
      const outcome = { ...((config.outcome_revenue ?? {}) as OutcomeRevenue) };
      if (Object.keys(next).length === 0) {
        delete outcome.seasonality_config;
      } else {
        outcome.seasonality_config = next;
      }
      const empty = Object.keys(outcome).length === 0;
      patchConfig({ outcome_revenue: empty ? undefined : outcome });
      return;
    }
    // Channel target.
    const idx = target.channelIndex ?? -1;
    const entry = config.channel_list?.[idx];
    if (!entry) return;
    const current = getChannel(entry);
    const nextChannel = { ...current };
    if (Object.keys(next).length === 0) {
      delete nextChannel.seasonality_config;
    } else {
      nextChannel.seasonality_config = next;
    }
    // We bypass updateChannelAt's "partial patch" wrapping so we can also
    // delete a key — `setChannel` rebuilds the channel from a full record.
    const list = [...(config.channel_list ?? [])];
    list[idx] = setChannel(entry, nextChannel);
    setConfig({ ...config, channel_list: list });
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor="sea_target">Target</Label>
          <Select
            value={
              target.kind === "outcome"
                ? OUTCOME_TARGET
                : `ch_${target.channelIndex}`
            }
            onValueChange={(v) => {
              if (v === OUTCOME_TARGET)
                setTarget({ kind: "outcome", label: "Outcome (total revenue)" });
              else {
                const idx = parseInt(v.replace("ch_", ""), 10);
                const t = targets.find(
                  (x) => x.kind === "channel" && x.channelIndex === idx,
                );
                if (t) setTarget(t);
              }
            }}
          >
            <SelectTrigger id="sea_target">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {targets.map((t) =>
                t.kind === "outcome" ? (
                  <SelectItem key="outcome" value={OUTCOME_TARGET}>
                    {t.label}
                  </SelectItem>
                ) : (
                  <SelectItem
                    key={`ch_${t.channelIndex}`}
                    value={`ch_${t.channelIndex}`}
                  >
                    {t.label}
                  </SelectItem>
                ),
              )}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="sea_mode">Mode</Label>
          <Select
            value={mode === "advanced" ? "advanced" : mode}
            onValueChange={(v) => {
              const next = v as SeasonalityMode;
              if (next === "none") return writeSeasonality({});
              if (next === "sin") {
                return writeSeasonality({
                  type: "sin",
                  amplitude: readNumber(sea.amplitude) ?? 0.2,
                  period: readNumber((sea as { period?: number }).period) ?? 52,
                  phase: readNumber(sea.phase) ?? 0,
                });
              }
              if (next === "fourier") {
                // If we were on sin, convert exactly via the Python-equivalent.
                if (mode === "sin") {
                  return writeSeasonality(sinToFourier(sea));
                }
                // Otherwise seed a sensible blank: K=1, period=52.
                return writeSeasonality({
                  type: "fourier",
                  period: readNumber((sea as { period?: number }).period) ?? 52,
                  K: 1,
                  intercept: 0,
                  coefficients: [[0.2, 0]],
                });
              }
              // advanced is read-only in this form.
            }}
          >
            <SelectTrigger id="sea_mode">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">None (flat)</SelectItem>
              <SelectItem value="sin">Sinusoidal (single harmonic)</SelectItem>
              <SelectItem value="fourier">
                Fourier (multiple harmonics)
              </SelectItem>
              {mode === "advanced" && (
                <SelectItem value="advanced" disabled>
                  Advanced (pattern / random fourier — edit via YAML)
                </SelectItem>
              )}
            </SelectContent>
          </Select>
        </div>
      </div>

      {mode === "advanced" && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50/60 px-3 py-2 text-xs text-amber-800">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <p>
            This config uses a categorical pattern or random-draw Fourier (no
            coefficients) that this form can't edit safely. Switch to the
            Advanced YAML pane to tweak it directly.
          </p>
        </div>
      )}

      {mode === "sin" && (
        <SinEditor cfg={sea} write={writeSeasonality} />
      )}

      {mode === "fourier" && (
        <FourierEditor cfg={sea as FourierConfig} write={writeSeasonality} />
      )}
    </div>
  );
}

// --------------------------------------------------------------------------
// Sub-editors
// --------------------------------------------------------------------------

function SinEditor({
  cfg,
  write,
}: {
  cfg: SeasonalityConfig;
  write: (next: SeasonalityConfig) => void;
}) {
  const amplitude = readNumber(cfg.amplitude);
  const period = readNumber((cfg as { period?: number }).period);
  const phase = readNumber(cfg.phase);

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <Label htmlFor="sea_amp">Amplitude</Label>
          <Tooltip content="Peak deviation as a fraction of the baseline (e.g. 0.2 = ±20%).">
            <span className="cursor-help text-[10px] text-slate-400" aria-hidden>
              ⓘ
            </span>
          </Tooltip>
        </div>
        <NumInput
          id="sea_amp"
          value={amplitude}
          onChange={(v) => write({ type: "sin", amplitude: v, period, phase })}
          step={0.05}
          min={0}
          max={1}
        />
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <Label htmlFor="sea_per">Period (weeks)</Label>
          <Tooltip content="Length of one full cycle. 52 = yearly with weekly data.">
            <span className="cursor-help text-[10px] text-slate-400" aria-hidden>
              ⓘ
            </span>
          </Tooltip>
        </div>
        <NumInput
          id="sea_per"
          value={period}
          onChange={(v) =>
            write({ type: "sin", amplitude, period: v, phase })
          }
          step={1}
          min={1}
          asInt
        />
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <Label htmlFor="sea_phase">Phase (weeks)</Label>
          <Tooltip content="Offset in weeks. 0 = peak at week 0.">
            <span className="cursor-help text-[10px] text-slate-400" aria-hidden>
              ⓘ
            </span>
          </Tooltip>
        </div>
        <NumInput
          id="sea_phase"
          value={phase}
          onChange={(v) =>
            write({ type: "sin", amplitude, period, phase: v })
          }
          step={0.5}
        />
      </div>
    </div>
  );
}

function FourierEditor({
  cfg,
  write,
}: {
  cfg: FourierConfig;
  write: (next: SeasonalityConfig) => void;
}) {
  const period = readNumber(cfg.period) ?? 52;
  const K = Math.max(1, Math.trunc(readNumber(cfg.K) ?? (cfg.coefficients?.length ?? 1)));
  const intercept = readNumber(cfg.intercept) ?? 0;
  const coefficients = useMemo(
    () => reshapeCoefficients(cfg.coefficients, K),
    [cfg.coefficients, K],
  );

  const previewCfg: FourierConfig = useMemo(
    () => ({
      type: "fourier",
      period,
      K: coefficients.length,
      intercept,
      coefficients,
    }),
    [period, coefficients, intercept],
  );

  const preview = useMemo(
    () => fourierPreviewPoints(previewCfg, { weeks: Math.max(period * 2, 52) }),
    [previewCfg, period],
  );

  const [previewMin, previewMax] = useMemo(() => {
    let mn = Infinity;
    let mx = -Infinity;
    for (const pt of preview) {
      if (pt.multiplier < mn) mn = pt.multiplier;
      if (pt.multiplier > mx) mx = pt.multiplier;
    }
    if (!Number.isFinite(mn) || !Number.isFinite(mx)) return [0.5, 1.5];
    const pad = Math.max(0.05, (mx - mn) * 0.15);
    return [mn - pad, mx + pad];
  }, [preview]);

  const writeFull = (next: FourierConfig) =>
    write({
      type: "fourier",
      period: next.period,
      K: (next.coefficients ?? []).length,
      intercept: next.intercept,
      coefficients: next.coefficients,
    } as SeasonalityConfig);

  const updateCoeff = (idx: number, which: 0 | 1, v: number | undefined) => {
    const nextCoeffs = coefficients.map((pair, i) => {
      if (i !== idx) return pair;
      const copy: [number, number] = [pair[0], pair[1]];
      copy[which] = Number.isFinite(Number(v)) ? Number(v) : 0;
      return copy;
    });
    writeFull({ ...previewCfg, coefficients: nextCoeffs, K: nextCoeffs.length });
  };

  const addHarmonic = () => {
    const next = [...coefficients, [0, 0] as [number, number]];
    writeFull({ ...previewCfg, coefficients: next, K: next.length });
  };

  const removeHarmonic = (idx: number) => {
    if (coefficients.length <= 1) return;
    const next = coefficients.filter((_, i) => i !== idx);
    writeFull({ ...previewCfg, coefficients: next, K: next.length });
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Label htmlFor="sea_f_period">Period (weeks)</Label>
            <Tooltip content="Cycle length used for every harmonic. 52 = yearly with weekly data.">
              <span className="cursor-help text-[10px] text-slate-400" aria-hidden>
                ⓘ
              </span>
            </Tooltip>
          </div>
          <NumInput
            id="sea_f_period"
            value={period}
            onChange={(v) =>
              writeFull({ ...previewCfg, period: v ?? 52 })
            }
            step={1}
            min={1}
            asInt
          />
        </div>
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Label htmlFor="sea_f_intercept">Intercept</Label>
            <Tooltip content="Constant level shift added to (1 + Σ harmonics).">
              <span className="cursor-help text-[10px] text-slate-400" aria-hidden>
                ⓘ
              </span>
            </Tooltip>
          </div>
          <NumInput
            id="sea_f_intercept"
            value={intercept}
            onChange={(v) =>
              writeFull({ ...previewCfg, intercept: v ?? 0 })
            }
            step={0.01}
          />
        </div>
        <div className="space-y-1.5 sm:col-span-1">
          <div className="flex items-center gap-2">
            <Label>K (number of harmonics)</Label>
            <Tooltip content="Each harmonic is one (a_k, b_k) sin/cos pair. K=1 ≡ a single sinusoid.">
              <span className="cursor-help text-[10px] text-slate-400" aria-hidden>
                ⓘ
              </span>
            </Tooltip>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-700">
            <span className="rounded-md border border-brand-border bg-white px-2 py-1.5 font-mono">
              {coefficients.length}
            </span>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={addHarmonic}
              disabled={coefficients.length >= Math.max(1, Math.floor(period / 2))}
            >
              <Plus className="h-3 w-3" />
              Add
            </Button>
          </div>
          <p className="text-[11px] text-slate-500">
            Nyquist cap: ⌊period/2⌋ = {Math.max(1, Math.floor(period / 2))}.
          </p>
        </div>
      </div>

      <div className="overflow-hidden rounded-lg border border-brand-border">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-3 py-2 font-medium">k</th>
              <th className="px-3 py-2 font-medium">a_k (sin)</th>
              <th className="px-3 py-2 font-medium">b_k (cos)</th>
              <th className="px-3 py-2" />
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {coefficients.map((pair, idx) => (
              <tr key={idx}>
                <td className="whitespace-nowrap px-3 py-2 font-mono text-slate-700">
                  {idx + 1}
                </td>
                <td className="px-2 py-1.5">
                  <Input
                    type="number"
                    step={0.01}
                    value={pair[0]}
                    onChange={(e) =>
                      updateCoeff(
                        idx,
                        0,
                        e.target.value === ""
                          ? 0
                          : parseFloat(e.target.value),
                      )
                    }
                  />
                </td>
                <td className="px-2 py-1.5">
                  <Input
                    type="number"
                    step={0.01}
                    value={pair[1]}
                    onChange={(e) =>
                      updateCoeff(
                        idx,
                        1,
                        e.target.value === ""
                          ? 0
                          : parseFloat(e.target.value),
                      )
                    }
                  />
                </td>
                <td className="px-2 py-1.5 text-right">
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    onClick={() => removeHarmonic(idx)}
                    disabled={coefficients.length <= 1}
                    aria-label={`Remove harmonic ${idx + 1}`}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <Label className="text-xs uppercase tracking-[0.14em] text-slate-500">
            Preview multiplier
          </Label>
          <span className="text-[11px] text-slate-500">
            m(t) = 1 + intercept + Σₖ aₖ·sin(2πkt/P) + bₖ·cos(2πkt/P)
          </span>
        </div>
        <div className="h-44 w-full rounded-lg border border-brand-border bg-white px-2 py-3">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={preview} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="week"
                fontSize={11}
                tick={{ fill: "#64748b" }}
                label={{
                  value: "week",
                  position: "insideBottomRight",
                  offset: -2,
                  fill: "#94a3b8",
                  fontSize: 11,
                }}
              />
              <YAxis
                fontSize={11}
                tick={{ fill: "#64748b" }}
                domain={[previewMin, previewMax]}
                tickFormatter={(v) => Number(v).toFixed(2)}
              />
              <RechartsTooltip
                formatter={(v: unknown) => Number(v).toFixed(3)}
                labelFormatter={(label) => `week ${label}`}
              />
              <Line
                type="monotone"
                dataKey="multiplier"
                stroke="#1d4ed8"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------
// Target state hook
// --------------------------------------------------------------------------

/**
 * We store only a stable string key (e.g. "outcome" or "ch_2") rather than the
 * whole TargetSpec. That lets us resolve to the *current* targets list at
 * render time and silently fall back to outcome if the user deleted a channel
 * they had selected — without firing a setState-in-effect.
 */
function useTargetState(targets: TargetSpec[]): [TargetSpec, (t: TargetSpec) => void] {
  const [selectedKey, setSelectedKey] = useState<string>(OUTCOME_TARGET);

  const resolved = useMemo<TargetSpec>(() => {
    if (selectedKey === OUTCOME_TARGET) {
      return targets[0] ?? { kind: "outcome", label: "Outcome (total revenue)" };
    }
    const idx = parseInt(selectedKey.replace("ch_", ""), 10);
    const found = targets.find(
      (t) => t.kind === "channel" && t.channelIndex === idx,
    );
    return (
      found ??
      targets[0] ?? { kind: "outcome", label: "Outcome (total revenue)" }
    );
  }, [targets, selectedKey]);

  const setTarget = (t: TargetSpec) => {
    if (t.kind === "outcome") setSelectedKey(OUTCOME_TARGET);
    else setSelectedKey(`ch_${t.channelIndex}`);
  };

  return [resolved, setTarget];
}

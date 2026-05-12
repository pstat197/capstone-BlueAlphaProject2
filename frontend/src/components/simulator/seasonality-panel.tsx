import { Info } from "lucide-react";

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
import { useConfig } from "@/state/config-store";
import type { OutcomeRevenue, SeasonalityConfig } from "@/types/api";

type SeasonalityMode = "none" | "sin" | "advanced";

/**
 * Detect the high-level mode of a `seasonality_config` so the form can
 * collapse into a simple sin editor when possible and otherwise route
 * users to the Advanced YAML view.
 *
 * - `{}` / undefined  → "none"
 * - `{type: "sin", amplitude, period, phase}` → "sin"
 * - anything else (pattern, fourier, etc.) → "advanced"
 */
function detectMode(cfg: SeasonalityConfig | undefined): SeasonalityMode {
  if (!cfg || Object.keys(cfg).length === 0) return "none";
  if (cfg.type === "sin") return "sin";
  return "advanced";
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
 * Outcome-level seasonality editor — covers the common "sinusoidal yearly
 * cycle" form inline. For pattern arrays / Fourier coefficient editing we
 * defer to the Advanced YAML pane to avoid baking a full data editor into
 * the React UI; the backend loader normalizes both forms identically.
 */
export function SeasonalityPanel() {
  const { config, patchConfig } = useConfig();
  const outcome: OutcomeRevenue = (config.outcome_revenue ?? {}) as OutcomeRevenue;
  const sea = outcome.seasonality_config ?? {};
  const mode = detectMode(sea);

  const amplitude = typeof sea.amplitude === "number" ? sea.amplitude : undefined;
  const period =
    typeof (sea as { period?: number }).period === "number"
      ? ((sea as { period?: number }).period as number)
      : undefined;
  const phase = typeof sea.phase === "number" ? sea.phase : undefined;

  const writeSeasonality = (next: SeasonalityConfig) => {
    const cleaned = { ...outcome };
    if (Object.keys(next).length === 0) {
      delete cleaned.seasonality_config;
    } else {
      cleaned.seasonality_config = next;
    }
    const empty = Object.keys(cleaned).length === 0;
    patchConfig({ outcome_revenue: empty ? undefined : cleaned });
  };

  const setMode = (next: SeasonalityMode) => {
    if (next === "none") return writeSeasonality({});
    if (next === "sin") {
      return writeSeasonality({
        type: "sin",
        amplitude: amplitude ?? 0.2,
        period: period ?? 52,
        phase: phase ?? 0.0,
      });
    }
    // Advanced is read-only here — the user must use the YAML pane to edit.
  };

  return (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <Label htmlFor="sea_mode">Mode (outcome path)</Label>
        <Select value={mode === "advanced" ? "advanced" : mode} onValueChange={(v) => setMode(v as SeasonalityMode)}>
          <SelectTrigger id="sea_mode">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">None (flat baseline)</SelectItem>
            <SelectItem value="sin">Sinusoidal (single harmonic)</SelectItem>
            {mode === "advanced" && (
              <SelectItem value="advanced" disabled>
                Advanced (Fourier / pattern — edit via YAML)
              </SelectItem>
            )}
          </SelectContent>
        </Select>
        <p className="text-[11px] text-slate-500">
          Applies to the outcome revenue path. Per-channel seasonality is not edited inline yet —
          use the Advanced YAML pane on the right.
        </p>
      </div>

      {mode === "advanced" && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50/60 px-3 py-2 text-xs text-amber-800">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <p>
            This config uses a Fourier or pattern form not editable here. Switch to the Advanced
            YAML view to tweak coefficients or fitted patterns.
          </p>
        </div>
      )}

      {mode === "sin" && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <Label htmlFor="sea_amp">Amplitude</Label>
              <Tooltip content="Peak deviation as a fraction of the baseline (e.g. 0.2 = ±20%).">
                <span className="text-[10px] cursor-help text-slate-400" aria-hidden>
                  ⓘ
                </span>
              </Tooltip>
            </div>
            <NumInput
              id="sea_amp"
              value={amplitude}
              onChange={(v) => writeSeasonality({ type: "sin", amplitude: v, period, phase })}
              step={0.05}
              min={0}
              max={1}
            />
          </div>
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <Label htmlFor="sea_per">Period (weeks)</Label>
              <Tooltip content="Length of one full cycle. 52 = yearly with weekly data.">
                <span className="text-[10px] cursor-help text-slate-400" aria-hidden>
                  ⓘ
                </span>
              </Tooltip>
            </div>
            <NumInput
              id="sea_per"
              value={period}
              onChange={(v) =>
                writeSeasonality({ type: "sin", amplitude, period: v, phase })
              }
              step={1}
              min={1}
              asInt
            />
          </div>
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <Label htmlFor="sea_phase">Phase (weeks)</Label>
              <Tooltip content="Offset in weeks. 0 = peak at week 0; phase = period/4 ⇒ peak at quarter cycle.">
                <span className="text-[10px] cursor-help text-slate-400" aria-hidden>
                  ⓘ
                </span>
              </Tooltip>
            </div>
            <NumInput
              id="sea_phase"
              value={phase}
              onChange={(v) =>
                writeSeasonality({ type: "sin", amplitude, period, phase: v })
              }
              step={0.5}
            />
          </div>
        </div>
      )}
    </div>
  );
}

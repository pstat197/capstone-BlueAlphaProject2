import { Activity } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import type { MmmProfile } from "@/types/api";

export interface McmcSettings {
  profile: MmmProfile;
  n_chains: number;
  n_adapt: number;
  n_burnin: number;
  n_keep: number;
  n_prior: number;
  seed: number;
  enable_aks: boolean;
  knots: string;
}

export const DEFAULT_MCMC: McmcSettings = {
  profile: "balanced",
  n_chains: 4,
  n_adapt: 1000,
  n_burnin: 500,
  n_keep: 500,
  n_prior: 500,
  seed: 0,
  enable_aks: false,
  knots: "",
};

/* Mirrors server/mmm.py MCMC_PRESETS so the UI shows the resolved numbers
 * even though the backend re-computes them from the profile name. */
const PRESET_NUMBERS: Record<Exclude<MmmProfile, "custom">, [number, number, number, number, number]> = {
  fast: [2, 500, 200, 200, 200],
  balanced: [4, 1000, 500, 500, 500],
  slow: [4, 2000, 500, 1000, 1000],
};

const PROFILE_LABELS: Record<MmmProfile, string> = {
  fast: "Fast (smaller, quicker)",
  balanced: "Balanced (recommended)",
  slow: "Slower (more reliable)",
  custom: "Custom (set every number)",
};

interface Props {
  value: McmcSettings;
  onChange: (next: McmcSettings) => void;
  disabled?: boolean;
}

/** Parse the comma-separated knots field. Invalid tokens are dropped silently
 *  (matches Streamlit's behavior of `.lstrip("-").isdigit()`). */
export function parseKnots(s: string): number[] | null {
  const trimmed = (s || "").trim();
  if (!trimmed) return null;
  const out: number[] = [];
  for (const tok of trimmed.split(",")) {
    const t = tok.trim();
    const n = Number(t);
    if (Number.isInteger(n) && n >= 0) out.push(n);
  }
  return out.length ? out : null;
}

export function MmmMcmcCard({ value, onChange, disabled }: Props) {
  const isPreset = value.profile !== "custom";
  const presetNumbers = isPreset ? PRESET_NUMBERS[value.profile as Exclude<MmmProfile, "custom">] : null;

  const setProfile = (next: MmmProfile) => {
    if (next === "custom") {
      onChange({ ...value, profile: "custom" });
      return;
    }
    const [nc, na, nb, nk, npr] = PRESET_NUMBERS[next];
    onChange({
      ...value,
      profile: next,
      n_chains: nc,
      n_adapt: na,
      n_burnin: nb,
      n_keep: nk,
      n_prior: npr,
    });
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-brand-500" />
          <CardTitle>MCMC (NUTS) sampling</CardTitle>
        </div>
        <CardDescription>
          Pick a preset to fill chains / adapt / burn-in / keep / prior draws automatically.{" "}
          <strong className="font-semibold">Fast</strong> for a quick sanity check;{" "}
          <strong className="font-semibold">Balanced</strong> is a sensible default;{" "}
          <strong className="font-semibold">Slower</strong> tightens credible intervals at the cost
          of wall time. Or pick <strong className="font-semibold">Custom</strong> to drive every
          number yourself.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-[260px_minmax(0,1fr)]">
          <div className="space-y-1">
            <Label className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
              Profile
            </Label>
            <Select value={value.profile} onValueChange={(v) => setProfile(v as MmmProfile)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(Object.keys(PROFILE_LABELS) as MmmProfile[]).map((p) => (
                  <SelectItem key={p} value={p}>
                    {PROFILE_LABELS[p]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {presetNumbers && (
              <p className="text-[11px] text-slate-500">
                chains={presetNumbers[0]}, n_adapt={presetNumbers[1]}, n_burnin={presetNumbers[2]},
                n_keep={presetNumbers[3]}, prior draws={presetNumbers[4]}
              </p>
            )}
          </div>
          <div className="grid gap-3 sm:grid-cols-4">
            <NumberField
              label="Chains"
              value={value.n_chains}
              min={1}
              max={32}
              step={1}
              disabled={disabled || isPreset}
              onChange={(n) => onChange({ ...value, n_chains: n })}
            />
            <NumberField
              label="n_adapt"
              value={value.n_adapt}
              min={100}
              max={20000}
              step={100}
              disabled={disabled || isPreset}
              onChange={(n) => onChange({ ...value, n_adapt: n })}
            />
            <NumberField
              label="n_burnin"
              value={value.n_burnin}
              min={0}
              max={20000}
              step={50}
              disabled={disabled || isPreset}
              onChange={(n) => onChange({ ...value, n_burnin: n })}
            />
            <NumberField
              label="n_keep"
              value={value.n_keep}
              min={50}
              max={10000}
              step={50}
              disabled={disabled || isPreset}
              onChange={(n) => onChange({ ...value, n_keep: n })}
            />
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          <NumberField
            label="Prior draws (sample_prior)"
            value={value.n_prior}
            min={50}
            max={5000}
            step={50}
            disabled={disabled || isPreset}
            onChange={(n) => onChange({ ...value, n_prior: n })}
          />
          <NumberField
            label="MCMC seed"
            value={value.seed}
            min={0}
            max={2147483647}
            step={1}
            disabled={disabled}
            onChange={(n) => onChange({ ...value, seed: n })}
          />
          <div className="space-y-1">
            <Label className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
              Automatic Knot Selection
            </Label>
            <div className="flex h-9 items-center gap-2 rounded-md border border-brand-border bg-white px-3 shadow-[inset_0_1px_2px_rgba(15,23,42,0.04)]">
              <Switch
                checked={value.enable_aks}
                onCheckedChange={(checked) => onChange({ ...value, enable_aks: !!checked })}
                disabled={disabled}
              />
              <span className="text-xs text-slate-700">enable_aks (slower; GPU recommended)</span>
            </div>
          </div>
        </div>

        <div className="space-y-1">
          <Label
            htmlFor="mmm-knots"
            className="text-[11px] font-medium uppercase tracking-wide text-slate-500"
          >
            Knots (comma-separated 0-based time indices)
          </Label>
          <Input
            id="mmm-knots"
            placeholder="e.g. 0, 13, 26, 51 (blank = Meridian default)"
            value={value.knots}
            onChange={(e) => onChange({ ...value, knots: e.target.value })}
            disabled={disabled || value.enable_aks}
          />
          <p className="text-[11px] text-slate-500">
            Time indices where the trend spline may change slope. Cannot be combined with
            enable_aks. National model defaults to a single (flat-trend) knot.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
}: {
  label: string;
  value: number;
  onChange: (next: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}) {
  return (
    <div className="space-y-1">
      <Label className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
        {label}
      </Label>
      <Input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => {
          const v = parseFloat(e.target.value || "0");
          onChange(Number.isFinite(v) ? v : 0);
        }}
        disabled={disabled}
      />
    </div>
  );
}

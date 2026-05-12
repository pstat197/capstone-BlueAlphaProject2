import { Copy, Wand2 } from "lucide-react";
import { useMemo } from "react";

import { ChannelAvailabilityCard } from "@/components/simulator/channel-availability";
import { CurvePreview } from "@/components/simulator/curve-preview";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tooltip } from "@/components/ui/tooltip";
import {
  duplicateChannelAt,
  getChannel,
  suggestHillK,
  updateChannelAt,
} from "@/lib/config-utils";
import { useConfig } from "@/state/config-store";
import type { ChannelDef, SimConfig } from "@/types/api";

interface ChannelDetailProps {
  index: number;
  onIndexChange?: (next: number) => void;
}

type SaturationType = NonNullable<NonNullable<ChannelDef["saturation_config"]>["type"]>;
type AdstockType = NonNullable<NonNullable<ChannelDef["adstock_decay_config"]>["type"]>;

const SATURATION_OPTIONS: SaturationType[] = ["linear", "hill", "diminishing_returns"];
const ADSTOCK_OPTIONS: AdstockType[] = [
  "geometric",
  "binomial",
  "exponential",
  "linear",
  "weighted",
];

/**
 * Which adstock types take the scalar `lambda` decay parameter. `linear` uses a
 * flat moving average (no lambda), `weighted` reads `weights` directly.
 */
const LAMBDA_ADSTOCK_TYPES: AdstockType[] = ["geometric", "exponential", "binomial"];

function FieldRow({
  children,
  hint,
}: {
  children: React.ReactNode;
  hint?: string;
}) {
  return (
    <div className="space-y-1.5">
      {children}
      {hint && <p className="text-[11px] leading-snug text-slate-500">{hint}</p>}
    </div>
  );
}

function NumInput({
  id,
  value,
  onChange,
  step = 0.1,
  min,
  max,
  asInt = false,
  className,
}: {
  id?: string;
  value: number | undefined;
  onChange: (v: number | undefined) => void;
  step?: number;
  min?: number;
  max?: number;
  asInt?: boolean;
  className?: string;
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
      className={className}
    />
  );
}

export function ChannelDetail({ index, onIndexChange }: ChannelDetailProps) {
  const { config, setConfig } = useConfig();
  const channel = useMemo(() => {
    const list = config.channel_list ?? [];
    return list[index] ? getChannel(list[index]!) : null;
  }, [config, index]);

  if (!channel) {
    return (
      <Card>
        <CardContent className="px-6 py-12 text-center text-slate-500">
          No channel selected.
        </CardContent>
      </Card>
    );
  }

  const patch = (partial: Partial<ChannelDef>) => {
    setConfig(updateChannelAt(config, index, partial) as SimConfig, {
      resetYamlDirty: true,
    });
  };

  const handleDuplicate = () => {
    const { config: nextConfig, newIndex } = duplicateChannelAt(config, index);
    setConfig(nextConfig, { resetYamlDirty: true });
    onIndexChange?.(newIndex);
  };

  const sat = channel.saturation_config ?? { type: "linear" };
  const ad = channel.adstock_decay_config ?? { type: "linear" };
  const noise = channel.noise_variance ?? {};
  const spend = channel.spend_range ?? [0, 0];

  /* `undefined` means "use default = enabled". `false` is the only way to disable. */
  const satEnabled = channel.saturation_enabled !== false;
  const adEnabled = channel.adstock_enabled !== false;

  const suggestedK = suggestHillK(channel);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <CardTitle>Channel · {channel.channel_name || `#${index + 1}`}</CardTitle>
              <CardDescription>
                All fields here flow into the YAML on the right-hand sidebar entry.
              </CardDescription>
            </div>
            <Tooltip content="Duplicate this channel with the same settings">
              <Button size="sm" variant="secondary" onClick={handleDuplicate}>
                <Copy className="h-3.5 w-3.5" />
                Duplicate
              </Button>
            </Tooltip>
          </div>
        </CardHeader>
        <CardContent className="space-y-5">
          <FieldRow hint="Used everywhere as the channel label and column prefix in the output CSV.">
            <Label htmlFor="ch_name">Name</Label>
            <Input
              id="ch_name"
              value={channel.channel_name}
              onChange={(e) => patch({ channel_name: e.target.value })}
            />
          </FieldRow>

          <div className="grid grid-cols-2 gap-4">
            <FieldRow hint="Multiplies saturated, adstocked effective media into revenue contribution.">
              <Label htmlFor="ch_roi">True ROI</Label>
              <NumInput
                id="ch_roi"
                value={channel.true_roi}
                onChange={(v) => patch({ true_roi: v })}
                step={0.1}
                min={0}
              />
            </FieldRow>
            <FieldRow hint="Per-channel baseline (used when this channel feeds the outcome path).">
              <Label htmlFor="ch_baseline">Baseline revenue</Label>
              <NumInput
                id="ch_baseline"
                value={channel.baseline_revenue}
                onChange={(v) => patch({ baseline_revenue: v })}
                step={100}
                min={0}
              />
            </FieldRow>
            <FieldRow hint="Cost per thousand impressions; links spend to simulated impressions.">
              <Label htmlFor="ch_cpm">CPM</Label>
              <NumInput
                id="ch_cpm"
                value={channel.cpm}
                onChange={(v) => patch({ cpm: v })}
                step={0.5}
                min={0}
              />
            </FieldRow>
            <FieldRow hint="Lower / upper bounds of weekly spend sampling (gamma distribution).">
              <Label>Spend range</Label>
              <div className="flex items-center gap-2">
                <NumInput
                  value={typeof spend[0] === "number" ? spend[0] : undefined}
                  onChange={(v) => patch({ spend_range: [v ?? 0, spend[1] ?? 0] })}
                  step={500}
                  min={0}
                />
                <span className="text-xs text-slate-400">–</span>
                <NumInput
                  value={typeof spend[1] === "number" ? spend[1] : undefined}
                  onChange={(v) => patch({ spend_range: [spend[0] ?? 0, v ?? 0] })}
                  step={500}
                  min={0}
                />
              </div>
            </FieldRow>
          </div>
        </CardContent>
      </Card>

      <ChannelAvailabilityCard index={index} />

      <Card>
        <CardHeader className="pb-3">
          <CardTitle>Revenue noise</CardTitle>
          <CardDescription>
            Variance σ² of a weekly N(0, σ²) shock on revenue (squared KPI units). Used for the
            outcome path when this channel feeds it; ignored otherwise.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <FieldRow hint="Set to 0 to disable revenue noise. Typical values are large (e.g. 1e6 for 1k σ).">
            <Label htmlFor="ch_noise_rev">Variance (σ²)</Label>
            <NumInput
              id="ch_noise_rev"
              value={noise.revenue}
              onChange={(v) => patch({ noise_variance: { ...noise, revenue: v } })}
              step={1000}
              min={0}
            />
          </FieldRow>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <CardTitle>Saturation</CardTitle>
              <CardDescription>
                How impressions translate into effective media. Toggle off to skip the saturation
                step for this channel.
              </CardDescription>
            </div>
            <div className="flex items-center gap-2 pt-1">
              <span className="text-xs text-slate-500">Enabled</span>
              <Switch
                checked={satEnabled}
                onCheckedChange={(checked) =>
                  patch({ saturation_enabled: checked ? undefined : false })
                }
              />
            </div>
          </div>
        </CardHeader>
        <CardContent
          className={`space-y-4 ${satEnabled ? "" : "pointer-events-none opacity-50"}`}
        >
          <FieldRow>
            <Label htmlFor="ch_sat_type">Type</Label>
            <Select
              value={sat.type ?? "linear"}
              onValueChange={(v) =>
                patch({ saturation_config: { ...sat, type: v as SaturationType } })
              }
            >
              <SelectTrigger id="ch_sat_type">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SATURATION_OPTIONS.map((opt) => (
                  <SelectItem key={opt} value={opt}>
                    {opt}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </FieldRow>
          <div className="grid grid-cols-2 gap-4">
            {(sat.type === "linear" || sat.type === "hill" || !sat.type) && (
              <FieldRow hint="Linear: multiplier on impressions. Hill: power in x^slope/(x^slope+K^slope).">
                <Label htmlFor="ch_sat_slope">Slope</Label>
                <NumInput
                  id="ch_sat_slope"
                  value={sat.slope}
                  onChange={(v) => patch({ saturation_config: { ...sat, slope: v } })}
                  step={0.1}
                  min={0.1}
                  max={5}
                />
              </FieldRow>
            )}
            {sat.type === "hill" && (
              <FieldRow hint="Half-saturation scale in weekly impression units (≈ mean spend / CPM × 1000).">
                <div className="flex items-end justify-between gap-2">
                  <Label htmlFor="ch_sat_K">K</Label>
                  {suggestedK !== null && (
                    <Tooltip
                      content={`Suggest K ≈ ${suggestedK.toLocaleString()} (mean spend / CPM × 1000)`}
                    >
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() =>
                          patch({ saturation_config: { ...sat, K: suggestedK } })
                        }
                      >
                        <Wand2 className="h-3.5 w-3.5" />
                        Suggest
                      </Button>
                    </Tooltip>
                  )}
                </div>
                <NumInput
                  id="ch_sat_K"
                  value={sat.K}
                  onChange={(v) => patch({ saturation_config: { ...sat, K: v } })}
                  step={1000}
                  min={1}
                />
              </FieldRow>
            )}
            {sat.type === "diminishing_returns" && (
              <FieldRow hint="Diminishing returns: x / (1 + beta·x). Larger beta = stronger saturation.">
                <Label htmlFor="ch_sat_beta">Beta</Label>
                <NumInput
                  id="ch_sat_beta"
                  value={sat.beta}
                  onChange={(v) => patch({ saturation_config: { ...sat, beta: v } })}
                  step={0.05}
                  min={0}
                  max={2}
                />
              </FieldRow>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <CardTitle>Adstock (carry-over)</CardTitle>
              <CardDescription>
                How past spend keeps contributing in subsequent weeks. Toggle off to skip the
                adstock step for this channel.
              </CardDescription>
            </div>
            <div className="flex items-center gap-2 pt-1">
              <span className="text-xs text-slate-500">Enabled</span>
              <Switch
                checked={adEnabled}
                onCheckedChange={(checked) =>
                  patch({ adstock_enabled: checked ? undefined : false })
                }
              />
            </div>
          </div>
        </CardHeader>
        <CardContent
          className={`space-y-4 ${adEnabled ? "" : "pointer-events-none opacity-50"}`}
        >
          <FieldRow>
            <Label htmlFor="ch_ad_type">Type</Label>
            <Select
              value={ad.type ?? "geometric"}
              onValueChange={(v) =>
                patch({ adstock_decay_config: { ...ad, type: v as AdstockType } })
              }
            >
              <SelectTrigger id="ch_ad_type">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {ADSTOCK_OPTIONS.map((opt) => (
                  <SelectItem key={opt} value={opt}>
                    {opt}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </FieldRow>
          <div className="grid grid-cols-2 gap-4">
            {LAMBDA_ADSTOCK_TYPES.includes((ad.type ?? "geometric") as AdstockType) && (
              <FieldRow
                hint={
                  ad.type === "binomial"
                    ? "Maps to Meridian's α* = 1/α − 1 over s ∈ [0, L]."
                    : "Closer to 1 = longer memory; closer to 0 = fast fade."
                }
              >
                <Label htmlFor="ch_ad_lambda">Lambda (α)</Label>
                <NumInput
                  id="ch_ad_lambda"
                  value={ad.lambda}
                  onChange={(v) => patch({ adstock_decay_config: { ...ad, lambda: v } })}
                  step={0.05}
                  min={0}
                  max={1}
                />
              </FieldRow>
            )}
            <FieldRow hint="Linear = window length. Geometric/exp/binomial = number of decay terms.">
              <Label htmlFor="ch_ad_lag">Lag</Label>
              <NumInput
                id="ch_ad_lag"
                value={ad.lag}
                onChange={(v) => patch({ adstock_decay_config: { ...ad, lag: v } })}
                step={1}
                min={0}
                max={52}
                asInt
              />
            </FieldRow>
            {ad.type === "weighted" && (
              <FieldRow hint="Comma-separated weights, e.g. 0.5, 0.3, 0.2.">
                <Label htmlFor="ch_ad_weights">Weights</Label>
                <Tooltip content="Order is t, t-1, t-2, …">
                  <Input
                    id="ch_ad_weights"
                    value={(ad.weights ?? []).join(", ")}
                    onChange={(e) => {
                      const parts = e.target.value
                        .split(",")
                        .map((s) => parseFloat(s.trim()))
                        .filter((n) => Number.isFinite(n));
                      patch({ adstock_decay_config: { ...ad, weights: parts } });
                    }}
                    placeholder="0.5, 0.3, 0.2"
                  />
                </Tooltip>
              </FieldRow>
            )}
          </div>
        </CardContent>
      </Card>

      <CurvePreview channel={channel} />
    </div>
  );
}

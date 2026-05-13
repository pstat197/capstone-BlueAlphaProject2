import { useMemo, useState } from "react";

import { IssueCountBadge } from "@/components/simulator/issue-count-badge";
import { Button } from "@/components/ui/button";
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
import { Tooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/cn";
import { configPathAttr, countIssues } from "@/lib/use-config-validation";
import { useConfig } from "@/state/config-store";
import type { ConfigIssue, GlobalKillSwitch, OutcomeRevenue } from "@/types/api";

interface AdvancedSettingsCardProps {
  issues?: ConfigIssue[];
  /** Optional controlled open state. When provided, the card defers to the
   *  parent for visibility — used so the validation navigator can expand it. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

function FieldHelp({ text }: { text: string }) {
  return (
    <Tooltip content={text}>
      <span className="text-[10px] cursor-help text-slate-400" aria-hidden>
        ⓘ
      </span>
    </Tooltip>
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
  placeholder,
  pathAttr,
}: {
  id?: string;
  value: number | undefined;
  onChange: (v: number | undefined) => void;
  step?: number;
  min?: number;
  max?: number;
  asInt?: boolean;
  placeholder?: string;
  pathAttr?: { "data-config-path": string };
}) {
  return (
    <Input
      id={id}
      type="number"
      step={step}
      min={min}
      max={max}
      placeholder={placeholder}
      value={value ?? ""}
      onChange={(e) => {
        const raw = e.target.value;
        if (raw === "") return onChange(undefined);
        const n = asInt ? parseInt(raw, 10) : parseFloat(raw);
        onChange(Number.isFinite(n) ? n : undefined);
      }}
      {...(pathAttr ?? {})}
    />
  );
}

/**
 * Settings that aren't tied to any single channel: the outcome-revenue path,
 * pipeline-level pre/post-transform order, and the two global kill-switches.
 *
 * Hidden behind an "Expand" toggle by default so the simulator page stays
 * scannable — most users tweak channels, not these.
 */
export function AdvancedSettingsCard({
  issues,
  open: controlledOpen,
  onOpenChange,
}: AdvancedSettingsCardProps) {
  const { config, patchConfig } = useConfig();
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const open = controlledOpen ?? uncontrolledOpen;
  const setOpen = (next: boolean | ((prev: boolean) => boolean)) => {
    const resolved = typeof next === "function" ? next(open) : next;
    setUncontrolledOpen(resolved);
    onOpenChange?.(resolved);
  };

  const outcome: OutcomeRevenue = (config.outcome_revenue ?? {}) as OutcomeRevenue;
  const outcomeNoise = outcome.noise_variance ?? {};
  const adstockGlobal: GlobalKillSwitch = (config.adstock ?? {}) as GlobalKillSwitch;
  const saturationGlobal: GlobalKillSwitch = (config.saturation ?? {}) as GlobalKillSwitch;
  const adstockOn = adstockGlobal.global !== false;
  const saturationOn = saturationGlobal.global !== false;

  const patchOutcome = (partial: Partial<OutcomeRevenue>) => {
    patchConfig({ outcome_revenue: { ...outcome, ...partial } });
  };

  const patchAdstockGlobal = (next: boolean) => {
    if (next) {
      const { global: _drop, ...rest } = adstockGlobal;
      // Remove the key entirely when re-enabling so YAML stays clean.
      void _drop;
      const isEmpty = Object.keys(rest).length === 0;
      patchConfig({ adstock: isEmpty ? undefined : rest });
    } else {
      patchConfig({ adstock: { ...adstockGlobal, global: false } });
    }
  };

  const patchSaturationGlobal = (next: boolean) => {
    if (next) {
      const { global: _drop, ...rest } = saturationGlobal;
      void _drop;
      const isEmpty = Object.keys(rest).length === 0;
      patchConfig({ saturation: isEmpty ? undefined : rest });
    } else {
      patchConfig({ saturation: { ...saturationGlobal, global: false } });
    }
  };

  /** Outcome-section issues drive a badge on this collapsed card so users see
   *  problems without having to expand it first. Seasonality issues are
   *  excluded — they live under the Scenarios card. */
  const outcomeIssues = useMemo(
    () =>
      countIssues(
        issues,
        (issue) =>
          issue.section === "outcome" &&
          !issue.path.some((p) => p === "seasonality_config"),
      ),
    [issues],
  );

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              Outcome path &amp; advanced
              <IssueCountBadge
                errors={outcomeIssues.errors}
                warnings={outcomeIssues.warnings}
                label="outcome"
              />
            </CardTitle>
            <CardDescription>
              Outcome-level baseline / trend / noise, pipeline transform order, and global
              kill-switches.
            </CardDescription>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setOpen((p) => !p)}
            aria-expanded={open}
          >
            {open ? "Collapse" : "Expand"}
          </Button>
        </div>
      </CardHeader>

      {open && (
        <CardContent className="space-y-6">
          <section className="space-y-3">
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Outcome revenue
            </div>
            <p className="text-[11px] text-slate-500">
              When set, this overrides the first-channel fallback for the total weekly revenue path
              (baseline + trend × week + seasonality + N(0, σ²) shock).
            </p>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <div className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <Label htmlFor="oc_baseline">Baseline revenue</Label>
                  <FieldHelp text="Weekly intercept of the outcome series, in KPI units." />
                </div>
                <NumInput
                  id="oc_baseline"
                  value={outcome.baseline_revenue}
                  onChange={(v) => patchOutcome({ baseline_revenue: v })}
                  step={100}
                  min={0}
                  pathAttr={configPathAttr(["outcome_revenue", "baseline_revenue"])}
                />
              </div>
              <div className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <Label htmlFor="oc_trend">Trend / week</Label>
                  <FieldHelp text="Linear slope applied to the outcome (KPI units per week)." />
                </div>
                <NumInput
                  id="oc_trend"
                  value={outcome.trend_slope}
                  onChange={(v) => patchOutcome({ trend_slope: v })}
                  step={10}
                />
              </div>
              <div className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <Label htmlFor="oc_noise">Revenue noise σ²</Label>
                  <FieldHelp text="Variance of weekly N(0, σ²) shock on total revenue (squared KPI units; 0 disables)." />
                </div>
                <NumInput
                  id="oc_noise"
                  value={outcomeNoise.revenue}
                  onChange={(v) =>
                    patchOutcome({
                      noise_variance: { ...outcomeNoise, revenue: v },
                    })
                  }
                  step={1000}
                  min={0}
                  pathAttr={configPathAttr([
                    "outcome_revenue",
                    "noise_variance",
                    "revenue",
                  ])}
                />
              </div>
            </div>
          </section>

          <section className="space-y-3">
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Pipeline &amp; globals
            </div>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <Label htmlFor="mto">Transform order</Label>
                  <FieldHelp text="Order of weekly media transforms before ROI. adstock_first matches Meridian's default." />
                </div>
                <Select
                  value={
                    (config.media_transform_order as string | undefined) ?? "adstock_first"
                  }
                  onValueChange={(v) =>
                    patchConfig({
                      media_transform_order: v as "adstock_first" | "saturation_first",
                    })
                  }
                >
                  <SelectTrigger id="mto" {...configPathAttr(["media_transform_order"])}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="adstock_first">adstock then saturation</SelectItem>
                    <SelectItem value="saturation_first">saturation then adstock</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center gap-2">
                  <Label htmlFor="noc">Auto-generate channel count</Label>
                  <FieldHelp text="If set, the loader grows channel_list to this many channels by jittering the default template." />
                </div>
                <NumInput
                  id="noc"
                  value={
                    typeof config.number_of_channels === "number"
                      ? config.number_of_channels
                      : undefined
                  }
                  onChange={(v) => patchConfig({ number_of_channels: v })}
                  min={0}
                  step={1}
                  asInt
                  placeholder="leave blank to use channel_list as-is"
                  pathAttr={configPathAttr(["number_of_channels"])}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div
                className={cn(
                  "flex items-center justify-between rounded-lg border border-brand-border bg-slate-50/50 px-3 py-2",
                  !adstockOn && "border-amber-300 bg-amber-50/60",
                )}
              >
                <div>
                  <Label className="text-sm font-medium">Global adstock</Label>
                  <p className="text-[11px] text-slate-500">Off = adstock disabled for every channel.</p>
                </div>
                <Switch checked={adstockOn} onCheckedChange={patchAdstockGlobal} />
              </div>
              <div
                className={cn(
                  "flex items-center justify-between rounded-lg border border-brand-border bg-slate-50/50 px-3 py-2",
                  !saturationOn && "border-amber-300 bg-amber-50/60",
                )}
              >
                <div>
                  <Label className="text-sm font-medium">Global saturation</Label>
                  <p className="text-[11px] text-slate-500">
                    Off = saturation disabled for every channel.
                  </p>
                </div>
                <Switch checked={saturationOn} onCheckedChange={patchSaturationGlobal} />
              </div>
            </div>
          </section>
        </CardContent>
      )}
    </Card>
  );
}

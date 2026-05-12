import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  adstockKernel,
  saturationCurvePoints,
  saturationXMax,
} from "@/lib/curves";
import type { ChannelDef } from "@/types/api";

interface CurvePreviewProps {
  channel: ChannelDef;
}

function formatImpressions(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (Math.abs(v) >= 1_000) return `${(v / 1_000).toFixed(0)}k`;
  return v.toFixed(0);
}

/**
 * Two small line charts that mirror the math in
 * `scripts/revenue_simulation/revenue_generation.py`:
 *
 *   1. Saturation: impressions → effective media (curve shape only — ROI is
 *      applied later in the pipeline).
 *   2. Adstock kernel: weight per lag week. Bar-like via Recharts `stepAfter`
 *      gives a clear visual of the decay shape.
 *
 * The previews intentionally cover the channel's realistic operating range
 * derived from `spend_range / CPM × 1000`, so users see how their settings
 * behave at the magnitudes they'll actually run.
 */
export function CurvePreview({ channel }: CurvePreviewProps) {
  const saturationData = useMemo(() => saturationCurvePoints(channel, 60), [channel]);
  const adstockKernelData = useMemo(() => {
    const weights = adstockKernel(channel.adstock_decay_config);
    return weights.map((w, lag) => ({ lag, weight: w }));
  }, [channel]);

  const xMax = saturationXMax(channel);
  const satType = channel.saturation_config?.type ?? "linear";
  const adType = channel.adstock_decay_config?.type ?? "geometric";

  const satEnabled = channel.saturation_enabled !== false;
  const adEnabled = channel.adstock_enabled !== false;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle>Curve preview</CardTitle>
        <CardDescription>
          Live preview of the saturation transform and adstock kernel for the current settings.
          Curve shape only — ROI is applied later.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Saturation · {satType}
            </span>
            {!satEnabled && (
              <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-700">
                disabled
              </span>
            )}
          </div>
          <div className={`h-44 ${satEnabled ? "" : "opacity-50"}`}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={saturationData} margin={{ top: 5, right: 8, bottom: 5, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(15, 23, 42, 0.08)" />
                <XAxis
                  dataKey="x"
                  tickFormatter={formatImpressions}
                  type="number"
                  domain={[0, xMax]}
                  tick={{ fontSize: 11, fill: "#64748b" }}
                  stroke="#cbd5e1"
                />
                <YAxis
                  tickFormatter={formatImpressions}
                  tick={{ fontSize: 11, fill: "#64748b" }}
                  stroke="#cbd5e1"
                />
                <RechartsTooltip
                  contentStyle={{ fontSize: 12 }}
                  labelFormatter={(v) => `Impressions: ${formatImpressions(Number(v))}`}
                  formatter={(v) => formatImpressions(Number(v))}
                />
                <Line
                  type="monotone"
                  dataKey="y"
                  stroke="#1d63ed"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                  name="effective"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-slate-500">
            x = weekly impressions (auto-ranged from spend × CPM). y = effective media after
            saturation.
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Adstock kernel · {adType}
            </span>
            {!adEnabled && (
              <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-amber-700">
                disabled
              </span>
            )}
          </div>
          <div className={`h-44 ${adEnabled ? "" : "opacity-50"}`}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={adstockKernelData}
                margin={{ top: 5, right: 8, bottom: 5, left: 8 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(15, 23, 42, 0.08)" />
                <XAxis
                  dataKey="lag"
                  type="number"
                  tick={{ fontSize: 11, fill: "#64748b" }}
                  stroke="#cbd5e1"
                  allowDecimals={false}
                />
                <YAxis
                  tickFormatter={(v) => Number(v).toFixed(2)}
                  tick={{ fontSize: 11, fill: "#64748b" }}
                  stroke="#cbd5e1"
                />
                <RechartsTooltip
                  contentStyle={{ fontSize: 12 }}
                  labelFormatter={(v) => `Lag week ${v}`}
                  formatter={(v) => Number(v).toFixed(3)}
                />
                <Line
                  type="stepAfter"
                  dataKey="weight"
                  stroke="#f39c59"
                  strokeWidth={2}
                  dot
                  isAnimationActive={false}
                  name="weight"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-slate-500">
            Lag 0 = current week. Weights sum to 1; longer tails carry past spend forward.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

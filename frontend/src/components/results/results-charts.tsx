import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { CHART_COLORS, CHART_PAL_CVD, formatCurrency, formatNumber } from "@/lib/config-utils";
import { useSettings } from "@/state/settings-store";
import type { RunChannelSeries, RunResponse } from "@/types/api";

interface ResultsChartsProps {
  run: RunResponse;
}

type SeriesKey = "revenue" | "spend" | "impressions";

const SERIES_TITLES: Record<SeriesKey, string> = {
  revenue: "Revenue",
  spend: "Total spend",
  impressions: "Total impressions",
};

const SERIES_FORMATTERS: Record<SeriesKey, (v: number | null | undefined) => string> = {
  revenue: formatCurrency,
  spend: formatCurrency,
  impressions: (v) => formatNumber(v, 0),
};

function pickColors(colorblind: boolean): { revenue: string; spend: string; impressions: string } {
  if (colorblind) {
    return {
      revenue: CHART_PAL_CVD[0],
      spend: CHART_PAL_CVD[1],
      impressions: CHART_PAL_CVD[2],
    };
  }
  return {
    revenue: CHART_COLORS.primary,
    spend: CHART_COLORS.secondary,
    impressions: CHART_COLORS.tertiary,
  };
}

function normalize(values: number[]): number[] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) {
    return values.map(() => 0.5);
  }
  return values.map((v) => (Number.isFinite(v) ? (v - lo) / (hi - lo) : 0));
}

function buildSingleSeries(
  weeks: number[],
  values: number[],
): Array<{ week: number; value: number }> {
  return weeks.map((week, i) => ({ week, value: values[i] ?? 0 }));
}

function buildOverlaySeries(
  weeks: number[],
  revenue: number[],
  spend: number[],
  impressions: number[],
) {
  const r = normalize(revenue);
  const s = normalize(spend);
  const im = normalize(impressions);
  return weeks.map((week, i) => ({
    week,
    revenue: r[i] ?? 0,
    spend: s[i] ?? 0,
    impressions: im[i] ?? 0,
  }));
}

function getScopeSeries(
  run: RunResponse,
  scope: string,
): { revenue: number[]; spend: number[]; impressions: number[]; subtitle: string } {
  if (scope === "__totals__") {
    return {
      revenue: run.totals.revenue.map((v) => v ?? 0),
      spend: run.totals.spend.map((v) => v ?? 0),
      impressions: run.totals.impressions.map((v) => v ?? 0),
      subtitle: "All channels (totals)",
    };
  }
  const channel: RunChannelSeries | undefined = run.channels.find((c) => c.name === scope);
  if (!channel) {
    return {
      revenue: run.totals.revenue.map((v) => v ?? 0),
      spend: run.totals.spend.map((v) => v ?? 0),
      impressions: run.totals.impressions.map((v) => v ?? 0),
      subtitle: "All channels (totals)",
    };
  }
  return {
    revenue: channel.revenue.map((v) => v ?? 0),
    spend: channel.spend.map((v) => v ?? 0),
    impressions: channel.impressions.map((v) => v ?? 0),
    subtitle: channel.name,
  };
}

export function ResultsCharts({ run }: ResultsChartsProps) {
  const { colorblindCharts, overlayCharts, setOverlayCharts } = useSettings();
  const [scope, setScope] = useState<string>("__totals__");
  const series = useMemo(() => getScopeSeries(run, scope), [run, scope]);
  const colors = useMemo(() => pickColors(colorblindCharts), [colorblindCharts]);

  return (
    <Card>
      <CardHeader className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <CardTitle>Series</CardTitle>
          <CardDescription>{series.subtitle}</CardDescription>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <div className="space-y-1">
            <Label htmlFor="series-scope">Series scope</Label>
            <Select value={scope} onValueChange={setScope}>
              <SelectTrigger id="series-scope" className="w-56">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__totals__">All channels (totals)</SelectItem>
                {run.channels.map((c) => (
                  <SelectItem key={c.name} value={c.name}>
                    {c.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <label className="flex items-center gap-2 pb-1.5 text-sm text-slate-700">
            <Switch checked={overlayCharts} onCheckedChange={setOverlayCharts} />
            <span>Overlay (normalized)</span>
          </label>
        </div>
      </CardHeader>
      <CardContent>
        {overlayCharts ? (
          <div className="h-[420px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={buildOverlaySeries(run.weeks, series.revenue, series.spend, series.impressions)}
                margin={{ top: 16, right: 24, left: 12, bottom: 8 }}
              >
                <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
                <XAxis dataKey="week" stroke="#94a3b8" />
                <YAxis
                  domain={[0, 1]}
                  stroke="#94a3b8"
                  tickFormatter={(v) => v.toFixed(2)}
                />
                <RechartsTooltip
                  formatter={(value, name) => [Number(value).toFixed(3), String(name)]}
                  contentStyle={{
                    borderRadius: 12,
                    border: "1px solid #e3eaf5",
                    boxShadow: "0 8px 32px rgba(20,63,160,0.12)",
                  }}
                />
                <Legend wrapperStyle={{ paddingTop: 10 }} />
                <Line
                  type="monotone"
                  dataKey="revenue"
                  name="Revenue (norm)"
                  stroke={colors.revenue}
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="spend"
                  name="Spend (norm)"
                  stroke={colors.spend}
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="impressions"
                  name="Impressions (norm)"
                  stroke={colors.impressions}
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="space-y-6">
            {(["revenue", "spend", "impressions"] as SeriesKey[]).map((key) => {
              const data = buildSingleSeries(run.weeks, series[key]);
              const color = colors[key];
              const fmt = SERIES_FORMATTERS[key];
              return (
                <div key={key} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-semibold text-slate-700">
                      {SERIES_TITLES[key]}
                    </h4>
                  </div>
                  <div className="h-44">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={data} margin={{ top: 6, right: 24, left: 12, bottom: 6 }}>
                        <CartesianGrid stroke="#eef2f7" strokeDasharray="3 3" />
                        <XAxis dataKey="week" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                        <YAxis
                          stroke="#94a3b8"
                          tick={{ fontSize: 11 }}
                          tickFormatter={(v) =>
                            key === "impressions"
                              ? formatNumber(v, 0)
                              : new Intl.NumberFormat("en-US", {
                                  notation: "compact",
                                }).format(v)
                          }
                        />
                        <RechartsTooltip
                          formatter={(value) => [fmt(Number(value)), SERIES_TITLES[key]]}
                          labelFormatter={(label) => `Week ${label}`}
                          contentStyle={{
                            borderRadius: 12,
                            border: "1px solid #e3eaf5",
                            boxShadow: "0 8px 32px rgba(20,63,160,0.12)",
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="value"
                          name={SERIES_TITLES[key]}
                          stroke={color}
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

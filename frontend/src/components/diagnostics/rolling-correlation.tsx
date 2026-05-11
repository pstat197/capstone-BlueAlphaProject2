import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { CorrelationResult, PairwiseSummary } from "@/types/api";

interface RollingCorrelationProps {
  correlation: CorrelationResult;
}

export function RollingCorrelation({ correlation }: RollingCorrelationProps) {
  const pairs: PairwiseSummary[] = correlation.pairwise_summary;
  const [pairKey, setPairKey] = useState<string>(
    pairs.length ? `${pairs[0]!.pair[0]}__${pairs[0]!.pair[1]}` : "",
  );

  const data = useMemo(() => {
    if (!pairKey || !correlation.rolling_corr) return [];
    const [a, b] = pairKey.split("__");
    const i = correlation.channel_names.indexOf(a as string);
    const j = correlation.channel_names.indexOf(b as string);
    if (i < 0 || j < 0) return [];
    return correlation.rolling_corr.map((mat, idx) => ({
      week: idx + correlation.window,
      rho: mat[i]?.[j] ?? 0,
    }));
  }, [pairKey, correlation]);

  if (!correlation.rolling_corr || pairs.length === 0) {
    return null;
  }

  return (
    <Card>
      <CardHeader className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <CardTitle>Rolling correlation</CardTitle>
          <CardDescription>
            Window of {correlation.window} weeks. Track when a pair drifts apart or together.
          </CardDescription>
        </div>
        <div className="space-y-1">
          <Label htmlFor="pair-select">Channel pair</Label>
          <Select value={pairKey} onValueChange={setPairKey}>
            <SelectTrigger id="pair-select" className="w-56">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {pairs.map((p) => {
                const key = `${p.pair[0]}__${p.pair[1]}`;
                return (
                  <SelectItem key={key} value={key}>
                    {p.pair[0]} / {p.pair[1]}
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 8, right: 24, left: 12, bottom: 8 }}>
              <defs>
                <linearGradient id="rhoFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#1d63ed" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#1d63ed" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef2f7" />
              <XAxis dataKey="week" stroke="#94a3b8" />
              <YAxis domain={[-1, 1]} stroke="#94a3b8" />
              <RechartsTooltip
                contentStyle={{
                  borderRadius: 12,
                  border: "1px solid #e3eaf5",
                  boxShadow: "0 8px 32px rgba(20,63,160,0.12)",
                }}
                formatter={(v) => [Number(v).toFixed(3), "rho"]}
                labelFormatter={(label) => `Week ${label}`}
              />
              <Area
                type="monotone"
                dataKey="rho"
                stroke="#1d63ed"
                fill="url(#rhoFill)"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}

import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/cn";
import { formatCurrency, formatNumber } from "@/lib/config-utils";
import type { RunResponse } from "@/types/api";

interface RunKpisProps {
  run: RunResponse;
}

function sumNullable(values: Array<number | null | undefined>): number {
  let acc = 0;
  for (const v of values) {
    if (v == null || !Number.isFinite(v)) continue;
    acc += v;
  }
  return acc;
}

interface ChannelTotal {
  name: string;
  total: number;
}

function topChannel(run: RunResponse): { entry: ChannelTotal; share: number } | null {
  if (run.channels.length === 0) return null;
  const totals: ChannelTotal[] = run.channels.map((c) => ({
    name: c.name,
    total: sumNullable(c.revenue),
  }));
  totals.sort((a, b) => b.total - a.total);
  const top = totals[0]!;
  const totalRevenue = sumNullable(run.totals.revenue);
  return { entry: top, share: totalRevenue > 0 ? top.total / totalRevenue : 0 };
}

interface KpiProps {
  label: string;
  value: string;
  hint?: string;
  accent?: "primary" | "secondary" | "tertiary" | "neutral";
}

const ACCENT_BG: Record<NonNullable<KpiProps["accent"]>, string> = {
  primary: "bg-gradient-to-br from-brand-50 to-white border-brand-100",
  secondary: "bg-gradient-to-br from-accent-50 to-white border-accent-100",
  tertiary: "bg-gradient-to-br from-emerald-50 to-white border-emerald-100",
  neutral: "bg-white border-brand-border",
};

function Kpi({ label, value, hint, accent = "primary" }: KpiProps) {
  return (
    <Card className={cn("border", ACCENT_BG[accent])}>
      <CardContent className="px-4 py-3.5">
        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
          {label}
        </p>
        <p className="mt-1.5 text-[1.65rem] leading-tight font-semibold tracking-tight text-slate-900 tabular-nums">
          {value}
        </p>
        {hint && <p className="mt-0.5 text-xs text-slate-500">{hint}</p>}
      </CardContent>
    </Card>
  );
}

export function RunKpis({ run }: RunKpisProps) {
  const totalRevenue = sumNullable(run.totals.revenue);
  const totalSpend = sumNullable(run.totals.spend);
  const totalImpressions = sumNullable(run.totals.impressions);
  const roas = totalSpend > 0 ? totalRevenue / totalSpend : null;
  const avgCpm = totalImpressions > 0 ? (totalSpend / totalImpressions) * 1000 : null;
  const top = topChannel(run);

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
      <Kpi
        label="Total revenue"
        value={formatCurrency(totalRevenue)}
        hint={`${run.weeks.length} weeks`}
        accent="primary"
      />
      <Kpi label="Total spend" value={formatCurrency(totalSpend)} accent="secondary" />
      <Kpi
        label="Implied ROAS"
        value={roas == null ? "—" : `${roas.toFixed(2)}×`}
        hint="Revenue ÷ spend"
        accent="tertiary"
      />
      <Kpi
        label="Avg CPM"
        value={avgCpm == null ? "—" : formatCurrency(avgCpm)}
        hint={`${formatNumber(totalImpressions, 0)} impressions`}
        accent="neutral"
      />
      <Kpi
        label="Top channel"
        value={top ? top.entry.name : "—"}
        hint={top ? `${(top.share * 100).toFixed(0)}% of revenue` : undefined}
        accent="secondary"
      />
    </div>
  );
}

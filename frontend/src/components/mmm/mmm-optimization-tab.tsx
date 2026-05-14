import { useMemo } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

import { formatCurrency } from "@/lib/config-utils";
import type { MmmBudgetOptimization, MmmBudgetPieSlice, MmmBudgetRow } from "@/types/api";

/* Brand-friendly palette (cycled if there are more channels than colors).
 * Picked for AA contrast on white and to read distinctly in pies. */
const PIE_PALETTE = [
  "#1d63ed",
  "#f39c59",
  "#10a37f",
  "#7c3aed",
  "#0ea5e9",
  "#dc2626",
  "#facc15",
  "#0f766e",
  "#9333ea",
  "#ea580c",
];

interface Props {
  data: MmmBudgetOptimization;
}

export function MmmOptimizationTab({ data }: Props) {
  if (data.error) {
    return (
      <p className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
        Budget optimization: {data.error}
      </p>
    );
  }
  const rows = data.rows ?? [];
  const pies = data.pies;
  if (!pies || rows.length === 0) {
    return <p className="text-sm text-slate-500">No optimization output available.</p>;
  }

  return (
    <div className="space-y-6">
      <p className="text-xs text-slate-600">
        Under Meridian's default <strong>fixed-budget</strong> scenario, total spend is held constant
        and reallocated to maximize incremental revenue (within channel bounds). The pies show each
        channel's <strong>share of spend</strong> before vs after optimization.
      </p>

      <div className="grid gap-6 lg:grid-cols-2">
        <PieCard
          title="Current budget allocation"
          slices={pies.current}
          totalSpend={data.total_spend_baseline ?? 0}
        />
        <PieCard
          title="Optimized budget allocation"
          slices={pies.optimized}
          totalSpend={data.total_spend_optimized ?? 0}
        />
      </div>

      <ReallocationList rows={rows} />
    </div>
  );
}

function PieCard({
  title,
  slices,
  totalSpend,
}: {
  title: string;
  slices: MmmBudgetPieSlice[];
  totalSpend: number;
}) {
  const data = useMemo(
    () =>
      slices.map((s) => ({
        name: s.channel,
        value: Math.max(0, s.value),
        share: s.share,
      })),
    [slices],
  );
  return (
    <div className="rounded-2xl border border-brand-border bg-white p-4">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <h4 className="text-sm font-semibold text-slate-900">{title}</h4>
        <span className="text-xs text-slate-500">total {formatCurrency(totalSpend)}</span>
      </div>
      {/*
       * Recharts ResponsiveContainer can't measure inside a Radix tab that
       * starts off as `display: none`, so it logs 'width(-1) and height(-1)'
       * warnings and the pie renders empty until the next resize. `aspect`
       * sidesteps this — it only depends on width, which is always defined
       * once the tab is shown.
       */}
      <ResponsiveContainer width="100%" aspect={1.1} debounce={50}>
        <PieChart>
          <Tooltip
            contentStyle={{
              borderRadius: 8,
              border: "1px solid #dbeafe",
              fontSize: 12,
            }}
            formatter={(value: number, _name: string, props: { payload?: { share?: number } }) => [
              `${formatCurrency(value)} (${(props.payload?.share ?? 0).toFixed(1)}%)`,
              "spend",
            ]}
          />
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            outerRadius="78%"
            innerRadius="42%"
            paddingAngle={1}
            labelLine={false}
            label={(entry: { name?: string; share?: number }) =>
              entry?.share != null ? `${entry.name} ${entry.share.toFixed(0)}%` : ""
            }
          >
            {data.map((_d, i) => (
              <Cell key={i} fill={PIE_PALETTE[i % PIE_PALETTE.length]} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

function ReallocationList({ rows }: { rows: MmmBudgetRow[] }) {
  return (
    <div className="space-y-2">
      <h4 className="text-sm font-semibold text-slate-900">
        Recommended reallocation (weekly)
      </h4>
      <ul className="space-y-1.5">
        {rows.map((r) => (
          <li key={r.channel}>
            <ReallocationLine row={r} />
          </li>
        ))}
      </ul>
    </div>
  );
}

function ReallocationLine({ row }: { row: MmmBudgetRow }) {
  const delta = row.delta_weekly;
  const cur = row.spend_baseline_weekly;
  const opt = row.spend_optimized_weekly;
  const pct = row.change_pct;
  const isHold = Math.abs(delta) < 0.5;
  const isUp = !isHold && delta > 0;
  const klass = isHold
    ? "text-slate-600"
    : isUp
      ? "text-emerald-700"
      : "text-rose-700";
  return (
    <p className={`text-sm ${klass}`}>
      <strong>{row.channel}</strong>:{" "}
      {isHold ? (
        <>
          keep weekly spend at about <strong>{formatCurrency(opt)}</strong> (within $1 of current).
        </>
      ) : isUp ? (
        <>
          increase weekly spend by <strong>{formatCurrency(delta)}</strong>{" "}
          (<strong>+{pct.toFixed(1)}%</strong>) — target <strong>{formatCurrency(opt)}</strong>/wk
          vs <strong>{formatCurrency(cur)}</strong>/wk now.
        </>
      ) : (
        <>
          decrease weekly spend by <strong>{formatCurrency(-delta)}</strong>{" "}
          (<strong>{pct.toFixed(1)}%</strong>) — target <strong>{formatCurrency(opt)}</strong>/wk
          vs <strong>{formatCurrency(cur)}</strong>/wk now.
        </>
      )}
    </p>
  );
}

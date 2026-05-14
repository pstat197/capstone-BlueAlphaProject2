import { Activity } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import type { MmmFitResults } from "@/types/api";

const METRIC_HELP: Record<string, string> = {
  R_Squared: "Coefficient of determination on in-sample fit. 1 = perfect, 0 = mean baseline.",
  R_squared: "Coefficient of determination on in-sample fit. 1 = perfect, 0 = mean baseline.",
  MAPE: "Mean Absolute Percentage Error |actual − predicted| / |actual|, averaged over weeks.",
  wMAPE: "Weighted MAPE — larger-revenue weeks count more.",
};

interface Props {
  results: MmmFitResults;
}

function rhatBadge(rhat: number | null) {
  if (rhat == null || !Number.isFinite(rhat)) return null;
  let color: "success" | "warn" | "muted" = "success";
  if (rhat > 1.1) color = "warn";
  else if (rhat > 1.05) color = "muted";
  return (
    <Badge variant={color} className="font-mono">
      R̂ {rhat.toFixed(4)}
    </Badge>
  );
}

export function MmmFitTab({ results }: Props) {
  const { summary, fit_metrics, fit_metrics_error } = results;
  const rhat = summary.rhat_max;

  /* Determine columns from the first row so we render whatever Meridian
   * surfaced (predictive_accuracy_table can include geo_granularity, etc.). */
  const columns = fit_metrics.length > 0 ? Object.keys(fit_metrics[0]!) : [];

  return (
    <div className="space-y-6">
      <section className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-xl border border-brand-border bg-white p-4 shadow-[inset_0_1px_2px_rgba(15,23,42,0.04)]">
          <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-500">
            <Activity className="h-3.5 w-3.5" />
            Max R̂ (posterior)
          </div>
          <div className="mt-2 flex items-end gap-2">
            <span className="text-2xl font-semibold text-slate-900">
              {rhat == null ? "—" : rhat.toFixed(4)}
            </span>
            {rhatBadge(rhat)}
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
            Largest Gelman–Rubin R̂ across parameters. Near 1 means chains mixed well; above ~1.1 is
            a red flag. Not a measure of business "accuracy" — see R² / MAPE for that.
          </p>
        </div>
      </section>

      <section className="space-y-2">
        <h3 className="text-sm font-semibold text-slate-900">In-sample predictive accuracy</h3>
        <p className="text-xs text-slate-600">
          Posterior-mean predicted revenue vs realized revenue. <strong>MAPE</strong> averages
          |actual − predicted| / |actual| (by week); <strong>wMAPE</strong> weights larger-revenue
          weeks more. These are <em>in-sample</em>: they don't measure holdout or business forecast
          accuracy on their own.
        </p>

        {fit_metrics_error ? (
          <p className="rounded-md border border-rose-200 bg-rose-50 p-3 text-xs text-rose-700">
            Fit metrics: {fit_metrics_error}
          </p>
        ) : fit_metrics.length === 0 ? (
          <p className="text-xs text-slate-500">No predictive accuracy table returned.</p>
        ) : (
          <div className="overflow-x-auto rounded-xl border border-brand-border bg-white">
            <table className="min-w-full divide-y divide-brand-border text-xs">
              <thead className="bg-slate-50/60 text-slate-700">
                <tr>
                  {columns.map((col) => (
                    <th
                      key={col}
                      title={METRIC_HELP[col]}
                      className="px-3 py-2 text-left font-semibold uppercase tracking-wide"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-brand-border/60 text-slate-700">
                {fit_metrics.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-50/40">
                    {columns.map((col) => (
                      <td key={col} className="px-3 py-2 font-mono">
                        {formatCell(row[col])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function formatCell(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "—";
    if (Math.abs(v) > 0 && (Math.abs(v) >= 1000 || Math.abs(v) < 0.01)) {
      return v.toExponential(3);
    }
    return v.toFixed(4);
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}

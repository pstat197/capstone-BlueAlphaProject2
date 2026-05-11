import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { PairwiseSummary } from "@/types/api";

interface PairwiseSummaryListProps {
  pairs: PairwiseSummary[];
}

function rhoColor(rho: number): string {
  if (rho < 0) return "#e74c3c";
  if (rho >= 0.8) return "#27ae60";
  if (rho >= 0.5) return "#3498db";
  if (rho >= 0.2) return "#e67e22";
  return "#95a5a6";
}

function driftColor(label: string): string {
  if (label === "stable") return "#27ae60";
  if (label.startsWith("-")) return "#e74c3c";
  return "#e67e22";
}

export function PairwiseSummaryList({ pairs }: PairwiseSummaryListProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Pairwise summary + drift</CardTitle>
        <CardDescription>
          Drift = change in rolling rho between the first and last 5 windows.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {pairs.length === 0 ? (
          <p className="text-sm text-slate-500">No correlation pairs configured.</p>
        ) : (
          <ul className="space-y-2">
            {pairs.map((p) => (
              <li
                key={`${p.pair[0]}__${p.pair[1]}`}
                className="flex items-center justify-between gap-3 rounded-lg border border-brand-border bg-white px-3 py-2"
              >
                <div className="min-w-0">
                  <p className="text-sm font-medium text-slate-800 truncate">
                    {p.pair[0]} <span className="text-slate-400">/</span> {p.pair[1]}
                  </p>
                  <p className="text-xs text-slate-500">
                    Configured rho:{" "}
                    <span className="font-mono">{p.configured_rho.toFixed(2)}</span>
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span
                    className="rounded-md px-2 py-0.5 text-xs font-semibold text-white"
                    style={{ backgroundColor: rhoColor(p.observed_rho) }}
                  >
                    rho {p.observed_rho.toFixed(2)}
                  </span>
                  <span
                    className="rounded-md px-2 py-0.5 text-xs font-semibold text-white"
                    style={{ backgroundColor: driftColor(p.drift_label) }}
                  >
                    {p.drift_label}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

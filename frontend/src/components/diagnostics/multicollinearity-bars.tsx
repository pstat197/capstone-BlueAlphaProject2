import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { CorrelationResult } from "@/types/api";

interface MulticollinearityBarsProps {
  correlation: CorrelationResult;
}

function barColor(v: number): string {
  if (v < 0) return "#e74c3c";
  if (v >= 0.8) return "#27ae60";
  if (v >= 0.5) return "#3498db";
  if (v >= 0.2) return "#e67e22";
  return "#95a5a6";
}

export function MulticollinearityBars({ correlation }: MulticollinearityBarsProps) {
  const entries = Object.entries(correlation.avg_abs_corr ?? {})
    .map(([name, val]) => ({ name, value: typeof val === "number" ? val : 0 }))
    .sort((a, b) => b.value - a.value);

  if (entries.length === 0) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Per-channel avg |rho|</CardTitle>
        <CardDescription>
          Higher = more redundant with other channels (multicollinearity risk).
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2">
          {entries.map((entry) => {
            const pct = Math.min(100, Math.round(entry.value * 100));
            const color = barColor(entry.value);
            return (
              <li key={entry.name} className="flex items-center gap-3">
                <span className="w-32 truncate text-sm text-slate-700">{entry.name}</span>
                <div className="flex-1 rounded-full bg-slate-100">
                  <div
                    className="h-2.5 rounded-full"
                    style={{ width: `${pct}%`, backgroundColor: color }}
                  />
                </div>
                <span className="w-12 text-right text-xs font-mono tabular-nums text-slate-600">
                  {entry.value.toFixed(3)}
                </span>
              </li>
            );
          })}
        </ul>
      </CardContent>
    </Card>
  );
}

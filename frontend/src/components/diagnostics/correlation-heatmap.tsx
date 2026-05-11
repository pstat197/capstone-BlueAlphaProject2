import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface CorrelationHeatmapProps {
  names: string[];
  matrix: number[][];
}

function corrCellColor(rho: number): string {
  if (rho < 0) return "#e74c3c";
  if (rho >= 0.8) return "#27ae60";
  if (rho >= 0.5) return "#3498db";
  if (rho >= 0.2) return "#e67e22";
  return "#95a5a6";
}

export function CorrelationHeatmap({ names, matrix }: CorrelationHeatmapProps) {
  if (!matrix.length) {
    return null;
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>Static correlation matrix</CardTitle>
        <CardDescription>Pearson rho across all weeks for each channel pair.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="border-separate border-spacing-1 text-xs">
            <thead>
              <tr>
                <th aria-hidden />
                {names.map((n) => (
                  <th key={n} className="px-2 py-1 text-left font-medium text-slate-600">
                    {n}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {names.map((row, i) => (
                <tr key={row}>
                  <th className="pr-2 text-right font-medium text-slate-600">{row}</th>
                  {names.map((_col, j) => {
                    const rho = matrix[i]?.[j] ?? 0;
                    const bg = corrCellColor(rho);
                    return (
                      <td key={j}>
                        <div
                          className="flex h-12 w-16 items-center justify-center rounded-md text-xs font-semibold text-white shadow-sm"
                          style={{ backgroundColor: bg }}
                        >
                          {rho.toFixed(2)}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-slate-500">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-3 w-3 rounded" style={{ backgroundColor: "#e74c3c" }} /> negative
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-3 w-3 rounded" style={{ backgroundColor: "#95a5a6" }} /> &lt; 0.2
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-3 w-3 rounded" style={{ backgroundColor: "#e67e22" }} /> 0.2 – 0.5
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-3 w-3 rounded" style={{ backgroundColor: "#3498db" }} /> 0.5 – 0.8
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-3 w-3 rounded" style={{ backgroundColor: "#27ae60" }} /> ≥ 0.8
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

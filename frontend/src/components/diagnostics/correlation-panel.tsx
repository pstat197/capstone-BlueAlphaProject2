import { CorrelationHeatmap } from "@/components/diagnostics/correlation-heatmap";
import { MulticollinearityBars } from "@/components/diagnostics/multicollinearity-bars";
import { PairwiseSummaryList } from "@/components/diagnostics/pairwise-summary";
import { RollingCorrelation } from "@/components/diagnostics/rolling-correlation";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import type { CorrelationResult } from "@/types/api";

interface Props {
  correlation: CorrelationResult | null | undefined;
  /** Show the summary chip row at the top. Off when the host already shows
   *  the same metadata in a parent header (e.g. Results page's RunSummary). */
  showHeader?: boolean;
}

/**
 * Channel-spend correlation diagnostics, factored out so it can render both
 * inside the Results page (as a tab) and on the standalone /diagnostics route.
 */
export function CorrelationPanel({ correlation, showHeader = false }: Props) {
  if (!correlation) {
    return (
      <Card>
        <CardContent className="px-6 py-10 text-center text-sm text-slate-600">
          No correlation diagnostics available for this run.
        </CardContent>
      </Card>
    );
  }

  const corr = correlation;
  const avgRho =
    Object.values(corr.avg_abs_corr ?? {}).reduce<number>((acc, v) => acc + (v ?? 0), 0) /
    Math.max(Object.keys(corr.avg_abs_corr ?? {}).length, 1);

  return (
    <div className="flex flex-col gap-6">
      {showHeader && (
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <Badge variant="outline">avg |rho| {avgRho.toFixed(2)}</Badge>
          <Badge variant="outline">
            most correlated · {corr.most_correlated_channel || "—"}
          </Badge>
          <Badge variant="muted">window {corr.window} wks</Badge>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {corr.static_corr && (
          <CorrelationHeatmap names={corr.channel_names} matrix={corr.static_corr} />
        )}
        <PairwiseSummaryList pairs={corr.pairwise_summary} />
      </div>

      <RollingCorrelation correlation={corr} />
      <MulticollinearityBars correlation={corr} />
    </div>
  );
}

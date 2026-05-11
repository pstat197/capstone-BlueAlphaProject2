import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import { CorrelationHeatmap } from "@/components/diagnostics/correlation-heatmap";
import { MulticollinearityBars } from "@/components/diagnostics/multicollinearity-bars";
import { PairwiseSummaryList } from "@/components/diagnostics/pairwise-summary";
import { RollingCorrelation } from "@/components/diagnostics/rolling-correlation";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { formatHash } from "@/lib/config-utils";

export default function DiagnosticsRoute() {
  const { runId } = useParams<{ runId: string }>();

  const runQuery = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRun(runId as string),
    enabled: Boolean(runId),
  });

  if (!runId) return null;

  if (runQuery.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-12 w-72" />
        <Skeleton className="h-72 w-full" />
      </div>
    );
  }

  if (runQuery.isError) {
    return (
      <Card>
        <CardContent className="space-y-3 px-6 py-10 text-center">
          <p className="text-sm text-rose-700">
            Could not load this run: {(runQuery.error as Error).message}
          </p>
          <Button variant="ghost" size="sm" asChild>
            <Link to="/simulator">
              <ArrowLeft className="h-3.5 w-3.5" />
              Back to simulator
            </Link>
          </Button>
        </CardContent>
      </Card>
    );
  }

  const run = runQuery.data;
  if (!run || !run.correlation) {
    return (
      <Card>
        <CardContent className="space-y-3 px-6 py-10 text-center">
          <p className="text-sm text-slate-600">
            No correlation diagnostics available for this run.
          </p>
          <Button variant="ghost" size="sm" asChild>
            <Link to={`/results/${runId}`}>
              <ArrowLeft className="h-3.5 w-3.5" />
              Back to results
            </Link>
          </Button>
        </CardContent>
      </Card>
    );
  }

  const corr = run.correlation;
  const avgRho =
    Object.values(corr.avg_abs_corr ?? {}).reduce<number>((acc, v) => acc + (v ?? 0), 0) /
    Math.max(Object.keys(corr.avg_abs_corr ?? {}).length, 1);

  return (
    <div className="flex flex-1 flex-col gap-6">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-2">
          <p className="text-xs font-medium uppercase tracking-[0.18em] text-brand-600">
            Diagnostics
          </p>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
            Channel spend correlation · {run.run_id}
          </h1>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <Badge variant="outline">avg |rho| {avgRho.toFixed(2)}</Badge>
            <Badge variant="outline">most correlated · {corr.most_correlated_channel || "—"}</Badge>
            <Badge variant="muted">window {corr.window} wks</Badge>
            <span className="font-mono text-xs text-slate-500">{formatHash(run.config_hash)}</span>
          </div>
        </div>
        <Button variant="ghost" size="sm" asChild>
          <Link to={`/results/${runId}`}>
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to results
          </Link>
        </Button>
      </header>

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

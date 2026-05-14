import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import { CorrelationPanel } from "@/components/diagnostics/correlation-panel";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { formatHash } from "@/lib/config-utils";

/**
 * Standalone correlation diagnostics page. The same content is now also
 * rendered inside the Results page under the "Correlation" tab; this route
 * is preserved so existing bookmarks / deep links keep working.
 */
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
  if (!run) return null;

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
            <span className="font-mono text-xs text-slate-500">{formatHash(run.config_hash)}</span>
            <Badge variant="muted">
              Tip: the same diagnostics live as a tab on the Results page.
            </Badge>
          </div>
        </div>
        <Button variant="ghost" size="sm" asChild>
          <Link to={`/results/${runId}`}>
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to results
          </Link>
        </Button>
      </header>

      <CorrelationPanel correlation={run.correlation} showHeader />
    </div>
  );
}

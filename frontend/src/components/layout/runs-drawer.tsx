import { useQuery } from "@tanstack/react-query";
import { History, ArrowRight } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { formatHash } from "@/lib/config-utils";

function formatRelative(iso: string | null | undefined): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const diff = Date.now() - date.getTime();
  const sec = Math.round(diff / 1000);
  if (sec < 60) return "just now";
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h ago`;
  const day = Math.round(hr / 24);
  if (day < 30) return `${day}d ago`;
  return date.toLocaleDateString();
}

export function RunsDrawer() {
  const [open, setOpen] = useState(false);
  const runsQuery = useQuery({
    queryKey: ["runs"],
    queryFn: () => api.listRuns(),
    enabled: open,
  });

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="sm" className="text-slate-600 hover:bg-brand-50">
          <History className="h-4 w-4" />
          History
        </Button>
      </SheetTrigger>
      <SheetContent
        side="right"
        title="Run history"
        description="Past simulations cached on disk. Click any to re-open results."
      >
        {runsQuery.isLoading && (
          <div className="space-y-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-16 w-full rounded-xl" />
            ))}
          </div>
        )}

        {runsQuery.isError && (
          <p className="text-sm text-rose-600">
            Could not load runs: {(runsQuery.error as Error).message}
          </p>
        )}

        {runsQuery.data && runsQuery.data.runs.length === 0 && (
          <div className="rounded-xl border border-dashed border-brand-border p-6 text-center text-sm text-slate-500">
            No runs yet. Run your first simulation and it will appear here.
          </div>
        )}

        {runsQuery.data && runsQuery.data.runs.length > 0 && (
          <ul className="space-y-2">
            {runsQuery.data.runs.map((run) => (
              <li key={run.config_hash}>
                <Link
                  to={`/results/${run.config_hash}`}
                  onClick={() => setOpen(false)}
                  className="group flex items-center justify-between gap-3 rounded-xl border border-brand-border bg-white p-3 transition hover:border-brand-300 hover:shadow-[0_4px_12px_rgba(29,99,237,0.08)]"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-semibold text-slate-900">
                        {run.run_identifier || "run"}
                      </span>
                      {run.last_was_cache_hit ? (
                        <Badge variant="muted">cached</Badge>
                      ) : (
                        <Badge variant="success">fresh</Badge>
                      )}
                    </div>
                    <p className="text-xs text-slate-500 truncate">
                      <span className="font-mono">{formatHash(run.config_hash)}</span>
                      {" · "}
                      {formatRelative(run.last_seen_at)}
                    </p>
                  </div>
                  <ArrowRight className="h-4 w-4 shrink-0 text-slate-300 transition group-hover:translate-x-0.5 group-hover:text-brand-500" />
                </Link>
              </li>
            ))}
          </ul>
        )}
      </SheetContent>
    </Sheet>
  );
}

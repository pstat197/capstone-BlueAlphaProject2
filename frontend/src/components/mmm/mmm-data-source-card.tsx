import { useQuery } from "@tanstack/react-query";
import { ChevronsUpDown, Database } from "lucide-react";
import { Link } from "react-router-dom";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { formatHash } from "@/lib/config-utils";

interface Props {
  selectedHash: string | null;
  onChange: (hash: string) => void;
  /** Channel names + week count of the currently-selected run, for the small
   *  badge row. Loaded by the parent (we share the run query). */
  loadedChannels?: string[];
  loadedWeeks?: number;
  loading?: boolean;
}

export function MmmDataSourceCard({
  selectedHash,
  onChange,
  loadedChannels,
  loadedWeeks,
  loading,
}: Props) {
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: () => api.listRuns() });
  const runs = runsQuery.data?.runs ?? [];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-brand-500" />
          <CardTitle>Synthetic data source</CardTitle>
        </div>
        <CardDescription>
          Bayesian MMM fits one of your cached simulator runs. Pick a run below — the most recent
          one is selected by default.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {runsQuery.isLoading && <Skeleton className="h-9 w-full" />}

        {!runsQuery.isLoading && runs.length === 0 && (
          <div className="rounded-lg border border-dashed border-brand-border bg-brand-50/40 p-4 text-sm text-slate-700">
            <p className="font-medium text-slate-900">No simulator runs yet.</p>
            <p className="mt-1 text-slate-600">
              The MMM tab fits Meridian on data produced by the Simulator. Run a simulation first;
              it will be cached and become available here.
            </p>
            <Button asChild size="sm" className="mt-3" variant="default">
              <Link to="/simulator">Open Simulator</Link>
            </Button>
          </div>
        )}

        {runs.length > 0 && (
          <>
            <div className="grid grid-cols-[1fr_auto] items-center gap-3">
              <Select value={selectedHash ?? undefined} onValueChange={onChange}>
                <SelectTrigger>
                  <span className="flex items-center gap-2 truncate">
                    <ChevronsUpDown className="h-3.5 w-3.5 text-slate-400" />
                    <SelectValue placeholder="Select a cached run…" />
                  </span>
                </SelectTrigger>
                <SelectContent>
                  {runs.map((r) => (
                    <SelectItem key={r.config_hash} value={r.config_hash}>
                      <span className="flex items-center gap-2">
                        <span className="truncate font-medium">{r.run_identifier || "run"}</span>
                        <span className="font-mono text-[11px] text-slate-500">
                          {formatHash(r.config_hash)}
                        </span>
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedHash && (
                <Button asChild size="sm" variant="ghost">
                  <Link to={`/results/${selectedHash}`}>Open results</Link>
                </Button>
              )}
            </div>

            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
              {loading && <Skeleton className="h-5 w-32" />}
              {!loading && loadedWeeks != null && (
                <Badge variant="muted">{loadedWeeks} weeks</Badge>
              )}
              {!loading && loadedChannels && loadedChannels.length > 0 && (
                <>
                  <Badge variant="muted">
                    {loadedChannels.length} channel{loadedChannels.length === 1 ? "" : "s"}
                  </Badge>
                  <span className="truncate">{loadedChannels.join(", ")}</span>
                </>
              )}
            </div>

            {loadedWeeks != null && loadedWeeks < 20 && (
              <p className="text-xs text-amber-700">
                Short series ({loadedWeeks} weeks): expect weak in-sample R² and wide credible
                intervals even when R̂ looks fine.
              </p>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

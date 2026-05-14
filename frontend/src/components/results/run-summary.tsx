import { ArrowLeftRight, Download } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { formatHash } from "@/lib/config-utils";
import type { RunResponse } from "@/types/api";

interface RunSummaryProps {
  run: RunResponse;
  onEditConfiguration: () => void;
}

export function RunSummary({ run, onEditConfiguration }: RunSummaryProps) {
  const channelCount = run.channels.length;
  const weeks = run.weeks.length;
  return (
    <header className="flex flex-wrap items-start justify-between gap-4">
      <div className="space-y-2">
        <p className="text-xs font-medium uppercase tracking-[0.18em] text-brand-600">Results</p>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
          {run.run_id || "Untitled run"}
        </h1>
        <div className="flex flex-wrap items-center gap-2 text-sm text-slate-600">
          {run.cache_hit ? (
            <Badge variant="muted">served from cache</Badge>
          ) : (
            <Badge variant="success">freshly computed</Badge>
          )}
          <Badge variant="outline">{weeks} weeks</Badge>
          <Badge variant="outline">
            {channelCount} channel{channelCount === 1 ? "" : "s"}
          </Badge>
          <span className="font-mono text-xs text-slate-500">{formatHash(run.config_hash)}</span>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="ghost" size="sm" onClick={onEditConfiguration}>
          <ArrowLeftRight className="h-3.5 w-3.5" />
          Edit configuration
        </Button>
        <Button size="sm" asChild>
          <a href={api.csvUrl(run.config_hash)} download>
            <Download className="h-3.5 w-3.5" />
            Download CSV
          </a>
        </Button>
      </div>
    </header>
  );
}

import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Play,
  RotateCcw,
  Settings,
  TerminalSquare,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { MmmDataSourceCard } from "@/components/mmm/mmm-data-source-card";
import {
  DEFAULT_MCMC,
  MmmMcmcCard,
  parseKnots,
  type McmcSettings,
} from "@/components/mmm/mmm-mcmc-card";
import { MmmPriorsCard, makeDefaultPriors } from "@/components/mmm/mmm-priors-card";
import { MmmResultsTabs } from "@/components/mmm/mmm-results-tabs";
import { api } from "@/lib/api";
import { useMmmFit } from "@/lib/use-mmm-fit";
import type { MmmFitRequest, MmmJobStage, RunResponse } from "@/types/api";

const STAGE_LABEL: Record<MmmJobStage, string> = {
  queued: "Queued",
  preparing: "Preparing inputs",
  sampling: "Sampling (NUTS)…",
  diagnostics: "Computing diagnostics",
  serializing: "Packaging results",
  done: "Done",
  error: "Failed",
};

export default function MmmRoute() {
  const statusQuery = useQuery({
    queryKey: ["meridian-status"],
    queryFn: () => api.meridianStatus(),
  });
  const meridianInstalled = statusQuery.data?.installed ?? false;

  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: () => api.listRuns() });
  const runs = runsQuery.data?.runs ?? [];

  const [selectedHash, setSelectedHash] = useState<string | null>(null);

  /* Default to the most recently seen run as soon as the runs list loads. */
  useEffect(() => {
    if (selectedHash == null && runs.length > 0) {
      setSelectedHash(runs[0]!.config_hash);
    }
  }, [runs, selectedHash]);

  /* Pulling the full run gives us channel names + week count for priors UI.
   * The payload is already cached on disk by the simulator, so this is fast. */
  const runQuery = useQuery({
    queryKey: ["run", selectedHash],
    queryFn: () => api.getRun(selectedHash as string),
    enabled: selectedHash !== null,
  });
  const run: RunResponse | undefined = runQuery.data;
  const channels = useMemo(() => (run?.channels ?? []).map((c) => c.name), [run]);
  const nWeeks = run?.weeks?.length ?? 0;

  const [priors, setPriors] = useState<{ mus: number[]; sigmas: number[] }>(
    () => ({ mus: [], sigmas: [] }),
  );
  /* Reset priors whenever the channel list changes (different run loaded). */
  useEffect(() => {
    setPriors(makeDefaultPriors(channels));
  }, [channels]);

  const [mcmc, setMcmc] = useState<McmcSettings>(DEFAULT_MCMC);
  const [settingsOpen, setSettingsOpen] = useState(true);

  const fit = useMmmFit();

  /* Once a fit succeeds, collapse the configuration panel so results take
   * centre stage. The user can re-open it with "Edit settings & re-run". */
  useEffect(() => {
    if (fit.status === "succeeded") setSettingsOpen(false);
  }, [fit.status]);

  const channelCountMismatch =
    channels.length > 0 &&
    (priors.mus.length !== channels.length || priors.sigmas.length !== channels.length);

  const canRun =
    meridianInstalled &&
    selectedHash !== null &&
    channels.length > 0 &&
    !channelCountMismatch &&
    !fit.loading;

  const handleRun = () => {
    if (!selectedHash) return;
    const knots = mcmc.enable_aks ? null : parseKnots(mcmc.knots);
    const body: MmmFitRequest = {
      config_hash: selectedHash,
      profile: mcmc.profile,
      n_chains: mcmc.n_chains,
      n_adapt: mcmc.n_adapt,
      n_burnin: mcmc.n_burnin,
      n_keep: mcmc.n_keep,
      n_prior: mcmc.n_prior,
      seed: mcmc.seed,
      enable_aks: mcmc.enable_aks,
      knots,
      channel_roi_mus: priors.mus,
      channel_roi_sigmas: priors.sigmas,
    };
    fit.start(body);
  };

  return (
    <div className="flex flex-1 flex-col gap-4">
      <header className="space-y-2">
        <p className="text-xs font-medium uppercase tracking-[0.18em] text-brand-600">
          Bayesian MMM
        </p>
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
              Meridian-powered MMM
            </h1>
            <p className="mt-1 text-sm text-slate-500">
              Fit Google Meridian on a cached simulator run, then explore the recovered ROI and
              budget optimization results.
            </p>
          </div>
          <MeridianBadge installed={meridianInstalled} />
        </div>
      </header>

      {!statusQuery.isLoading && !meridianInstalled && (
        <MeridianMissingPanel error={statusQuery.data?.error ?? null} />
      )}

      <MmmDataSourceCard
        selectedHash={selectedHash}
        onChange={(hash) => {
          setSelectedHash(hash);
          fit.reset();
        }}
        loadedChannels={channels}
        loadedWeeks={nWeeks || undefined}
        loading={runQuery.isLoading}
      />

      {selectedHash !== null && (
        <>
          {(settingsOpen || fit.status !== "succeeded") && (
            <>
              <MmmPriorsCard
                channels={channels}
                mus={priors.mus}
                sigmas={priors.sigmas}
                onChange={setPriors}
                disabled={fit.loading}
              />
              <MmmMcmcCard value={mcmc} onChange={setMcmc} disabled={fit.loading} />

              <Card>
                <CardContent className="flex flex-wrap items-center justify-between gap-3 py-4">
                  <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
                    {fit.error && (
                      <span className="flex items-center gap-1.5 rounded-full bg-rose-50 px-3 py-1 font-medium text-rose-700">
                        <AlertTriangle className="h-3.5 w-3.5" />
                        {fit.error}
                      </span>
                    )}
                    {fit.status === "queued" || fit.status === "running" ? (
                      <FitProgress
                        stage={fit.job?.stage ?? "queued"}
                        startedAt={fit.job?.started_at ?? null}
                      />
                    ) : null}
                    {fit.job?.cache_hit && fit.status === "succeeded" && (
                      <Badge variant="muted">cache hit — instant</Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {fit.status === "succeeded" && (
                      <Button variant="ghost" size="sm" onClick={() => fit.reset()}>
                        <RotateCcw className="h-3.5 w-3.5" />
                        Clear results
                      </Button>
                    )}
                    <Button
                      size="default"
                      onClick={handleRun}
                      disabled={!canRun}
                      title={
                        !meridianInstalled
                          ? "Meridian is not installed in the API venv"
                          : channelCountMismatch
                            ? "Per-channel priors length does not match the loaded data"
                            : undefined
                      }
                    >
                      {fit.loading ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin" />
                          {fit.job?.cache_hit ? "Loading…" : "Running…"}
                        </>
                      ) : (
                        <>
                          <Play className="h-4 w-4" />
                          Run model
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </>
          )}

          {fit.status === "succeeded" && !settingsOpen && (
            <Card>
              <CardContent className="flex flex-wrap items-center justify-between gap-3 py-4">
                <p className="text-xs text-slate-600">
                  Showing the most recent fit. Settings are collapsed; re-open to change priors or
                  MCMC and run again.
                </p>
                <Button size="sm" variant="secondary" onClick={() => setSettingsOpen(true)}>
                  <Settings className="h-3.5 w-3.5" />
                  Edit settings & re-run
                </Button>
              </CardContent>
            </Card>
          )}

          {fit.results && fit.status === "succeeded" && (
            <>
              {fit.job?.note && (
                <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                  <strong className="font-semibold">Heads up:</strong> {fit.job.note}
                </div>
              )}
              <MmmResultsTabs results={fit.results} />
            </>
          )}

          {fit.loading && fit.status === "running" && !fit.results && (
            <Card>
              <CardContent className="space-y-3 py-6">
                <Skeleton className="h-6 w-1/3" />
                <Skeleton className="h-3 w-2/3" />
                <Skeleton className="h-72 w-full" />
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function MeridianBadge({ installed }: { installed: boolean }) {
  return installed ? (
    <Badge variant="success">
      <CheckCircle2 className="mr-1 h-3 w-3" /> meridian installed
    </Badge>
  ) : (
    <Badge variant="warn">
      <AlertTriangle className="mr-1 h-3 w-3" /> meridian unavailable
    </Badge>
  );
}

function MeridianMissingPanel({ error }: { error: string | null }) {
  return (
    <Card className="border-amber-200 bg-amber-50/40">
      <CardHeader>
        <div className="flex items-center gap-2">
          <TerminalSquare className="h-4 w-4 text-amber-600" />
          <CardTitle>Meridian is not importable in the API venv</CardTitle>
        </div>
        <CardDescription>
          The MMM tab needs the optional Meridian + TensorFlow stack. Install it into the same venv
          that runs the FastAPI server, then restart it.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <pre className="overflow-x-auto rounded-md bg-slate-900 p-3 text-xs text-slate-100">
{`pip install -r requirements.txt
pip install -r requirements-meridian.txt
# or one line:
pip install -e ".[mmm]"`}
        </pre>
        {error && (
          <details className="text-xs text-amber-800">
            <summary className="cursor-pointer font-medium">Import error details</summary>
            <pre className="mt-1 overflow-x-auto rounded bg-white/60 p-2 font-mono">{error}</pre>
          </details>
        )}
      </CardContent>
    </Card>
  );
}

function FitProgress({ stage, startedAt }: { stage: MmmJobStage; startedAt: string | null }) {
  const elapsed = useElapsed(startedAt);
  return (
    <span className="flex items-center gap-2 rounded-full border border-brand-border bg-white/80 px-3 py-1 font-medium text-slate-700">
      <Loader2 className="h-3.5 w-3.5 animate-spin text-brand-500" />
      {STAGE_LABEL[stage] ?? stage}
      {elapsed && <span className="font-mono text-[11px] text-slate-500">{elapsed}</span>}
    </span>
  );
}

function useElapsed(startedAt: string | null): string {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!startedAt) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [startedAt]);
  if (!startedAt) return "";
  const t = new Date(startedAt).getTime();
  if (!Number.isFinite(t)) return "";
  const sec = Math.max(0, Math.round((now - t) / 1000));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}m${s.toString().padStart(2, "0")}s`;
}

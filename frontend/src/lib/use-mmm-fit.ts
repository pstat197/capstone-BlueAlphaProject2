import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";

import { api } from "@/lib/api";
import type {
  MmmFitRequest,
  MmmFitResults,
  MmmJob,
  MmmJobStatus,
} from "@/types/api";

const POLL_MS = 2000;

export interface MmmFitController {
  job: MmmJob | null;
  results: MmmFitResults | null;
  status: MmmJobStatus | "idle";
  /** True while a request is in-flight (start or polling-then-fetch-results). */
  loading: boolean;
  /** Last error from start / polling / results fetch. */
  error: string | null;
  start: (req: MmmFitRequest) => void;
  reset: () => void;
}

/**
 * Manages a single Bayesian MMM fit lifecycle:
 *   1. POST /api/mmm/fits   → returns a queued job (or a synchronously-finished cached one)
 *   2. GET  /api/mmm/fits/:id (every {POLL_MS}ms while queued/running)
 *   3. GET  /api/mmm/fits/:id/results once status flips to "succeeded"
 *
 * The hook intentionally keeps job state in memory only — it does NOT survive
 * a hard refresh. The on-disk fit cache (server side, keyed by simulator
 * config + MCMC settings) is what makes "re-run with same settings" snap back
 * to results instantly: the start mutation immediately resolves with a cached
 * "succeeded" job in that case.
 */
export function useMmmFit(): MmmFitController {
  const queryClient = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reset = useCallback(() => {
    setJobId(null);
    setError(null);
  }, []);

  const startMutation = useMutation({
    mutationFn: (req: MmmFitRequest) => api.startMmmFit(req),
    onSuccess: (data: MmmJob) => {
      setError(null);
      setJobId(data.job_id);
      queryClient.setQueryData(["mmm-job", data.job_id], data);
    },
    onError: (e: Error) => setError(e.message),
  });

  const statusQuery = useQuery({
    queryKey: ["mmm-job", jobId],
    queryFn: () => api.getMmmFitStatus(jobId as string),
    enabled: jobId !== null,
    /* React Query 5 lets `refetchInterval` be a function returning false to stop. */
    refetchInterval: (query) => {
      const data = query.state.data as MmmJob | undefined;
      if (!data) return POLL_MS;
      return data.status === "queued" || data.status === "running" ? POLL_MS : false;
    },
  });

  const job = statusQuery.data ?? startMutation.data ?? null;

  const succeeded = job?.status === "succeeded";

  const resultsQuery = useQuery({
    queryKey: ["mmm-results", jobId],
    queryFn: () => api.getMmmFitResults(jobId as string),
    enabled: jobId !== null && succeeded,
    staleTime: Infinity,
  });

  /* Surface backend errors uniformly through the controller's `error` slot. */
  useEffect(() => {
    if (job?.status === "failed" && job.error) setError(job.error);
  }, [job?.status, job?.error]);

  useEffect(() => {
    if (statusQuery.error) setError((statusQuery.error as Error).message);
  }, [statusQuery.error]);

  useEffect(() => {
    if (resultsQuery.error) setError((resultsQuery.error as Error).message);
  }, [resultsQuery.error]);

  const status: MmmJobStatus | "idle" = job?.status ?? "idle";
  const loading =
    startMutation.isPending ||
    status === "queued" ||
    status === "running" ||
    (status === "succeeded" && resultsQuery.isLoading);

  return {
    job,
    results: (resultsQuery.data?.results as MmmFitResults | null) ?? null,
    status,
    loading,
    error,
    start: (req) => startMutation.mutate(req),
    reset,
  };
}

import type {
  ExampleConfigResponse,
  MeridianStatus,
  MmmFitRequest,
  MmmFitResultsResponse,
  MmmJob,
  RunListItem,
  RunResponse,
  SeasonalityConfig,
  SimConfig,
  ValidateConfigResponse,
} from "@/types/api";

/* API base URL.
 *
 * In local dev BASE stays empty so Vite's proxy in vite.config.ts forwards
 * /api/* to 127.0.0.1:8000. In a hosted build (e.g. Vercel) the frontend and
 * backend live on different origins, so we read the backend URL from a
 * Vite-time env var:
 *
 *     # .env.production (or Vercel env)
 *     VITE_API_BASE_URL=https://<your-hf-user>-bluealpha-api.hf.space
 *
 * Trailing slashes are stripped so paths always look like `${BASE}/api/...`.
 */
const BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      /* ignore non-JSON error bodies */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ ok: boolean; name: string }>("/api/health"),
  schema: () => request<Record<string, unknown>>("/api/schema"),
  exampleConfig: () => request<ExampleConfigResponse>("/api/example-config"),

  validateYaml: (yamlText: string) =>
    request<{ ok: boolean; error?: string; parsed?: SimConfig }>("/api/yaml/validate", {
      method: "POST",
      body: JSON.stringify({ yaml_text: yamlText }),
    }),
  dumpYaml: (config: SimConfig) =>
    request<{ yaml_text: string }>("/api/yaml/dump", {
      method: "POST",
      body: JSON.stringify({ config }),
    }),

  createRun: (config: SimConfig) =>
    request<RunResponse>("/api/runs", {
      method: "POST",
      body: JSON.stringify({ config }),
    }),
  getRun: (configHash: string) => request<RunResponse>(`/api/runs/${configHash}`),
  listRuns: () => request<{ runs: RunListItem[] }>("/api/runs"),
  csvUrl: (configHash: string) => `${BASE}/api/runs/${configHash}/csv`,
  hashConfig: (config: SimConfig) =>
    request<{ config_hash: string }>("/api/config/hash", {
      method: "POST",
      body: JSON.stringify({ config }),
    }),
  cacheStatus: (configHash: string) =>
    request<{
      config_hash: string;
      cached: boolean;
      run_identifier: string | null;
      last_seen_at: string | null;
    }>(`/api/cache/${configHash}`),

  validateConfig: (config: SimConfig) =>
    request<ValidateConfigResponse>("/api/config/validate", {
      method: "POST",
      body: JSON.stringify({ config }),
    }),

  fitSeasonalityPattern: (pattern: number[], K?: number) =>
    request<{
      type: "fourier";
      period: number;
      K: number;
      intercept: number;
      coefficients: Array<[number, number]>;
    }>("/api/seasonality/fit-pattern", {
      method: "POST",
      body: JSON.stringify({ pattern, K }),
    }),

  evaluateSeasonality: (
    config: SeasonalityConfig | Record<string, unknown>,
    weeks: number,
    seed?: number | null,
  ) =>
    request<{ multipliers: number[]; weeks: number }>(
      "/api/seasonality/evaluate",
      {
        method: "POST",
        body: JSON.stringify({ config, weeks, seed: seed ?? null }),
      },
    ),

  clearCache: () =>
    request<{ removed: number }>("/api/cache/clear", { method: "POST" }),

  meridianStatus: () => request<MeridianStatus>("/api/meridian/status"),

  startMmmFit: (body: MmmFitRequest) =>
    request<MmmJob>("/api/mmm/fits", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getMmmFitStatus: (jobId: string) => request<MmmJob>(`/api/mmm/fits/${jobId}`),
  getMmmFitResults: (jobId: string) =>
    request<MmmFitResultsResponse>(`/api/mmm/fits/${jobId}/results`),
};

import type {
  ExampleConfigResponse,
  MeridianStatus,
  RunListItem,
  RunResponse,
  SimConfig,
} from "@/types/api";

const BASE = ""; // Vite dev proxies /api -> 127.0.0.1:8000

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
  csvUrl: (configHash: string) => `/api/runs/${configHash}/csv`,
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

  clearCache: () =>
    request<{ removed: number }>("/api/cache/clear", { method: "POST" }),

  meridianStatus: () => request<MeridianStatus>("/api/meridian/status"),
};

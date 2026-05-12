import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { SimConfig } from "@/types/api";

/**
 * Snapshot of the React UI's "pre-run cache hint" for the current config.
 *
 * `hash`        canonical config hash (matches what POST /api/runs would produce).
 *               `null` while we're computing or before the first debounce fires.
 * `cached`      true when the backend has a cached result for that hash.
 * `runId`       the user-supplied run identifier (if any) attached to the cache
 *               entry — handy for the "re-open" affordance.
 * `loading`     true while either the hash or cache-status request is in flight.
 */
export interface PrerunCache {
  hash: string | null;
  cached: boolean;
  runId: string | null;
  loading: boolean;
}

/**
 * Debounce-aware hook that watches the live config, computes its hash, and
 * tells the caller whether that hash is cached server-side. Both steps go
 * through TanStack Query so React's identity churn doesn't refetch when the
 * actual content hasn't changed.
 *
 * The debounce is deliberately short (350ms) — long enough that typing into a
 * number input doesn't fire a hash request per keystroke, short enough that
 * the badge feels live.
 */
export function usePrerunCache(config: SimConfig, debounceMs = 350): PrerunCache {
  /*
   * Use the JSON representation of the config as the query key. The config
   * object identity changes on every reducer pass even when nothing material
   * changed, which would otherwise refetch constantly.
   */
  const configKey = JSON.stringify(config);
  const [debouncedKey, setDebouncedKey] = useState(configKey);

  useEffect(() => {
    const t = setTimeout(() => setDebouncedKey(configKey), debounceMs);
    return () => clearTimeout(t);
  }, [configKey, debounceMs]);

  const ready = (config.channel_list ?? []).length > 0;

  const hashQuery = useQuery({
    queryKey: ["config-hash", debouncedKey],
    queryFn: () => api.hashConfig(config),
    enabled: ready,
    staleTime: 60_000,
  });

  const hash = hashQuery.data?.config_hash ?? null;

  const cacheQuery = useQuery({
    queryKey: ["cache-status", hash],
    queryFn: () => (hash ? api.cacheStatus(hash) : Promise.resolve(null)),
    enabled: Boolean(hash),
    staleTime: 30_000,
  });

  return {
    hash,
    cached: cacheQuery.data?.cached ?? false,
    runId: cacheQuery.data?.run_identifier ?? null,
    loading: hashQuery.isFetching || cacheQuery.isFetching,
  };
}

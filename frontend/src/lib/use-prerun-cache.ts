import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";
import { useCanonicalHash } from "@/lib/use-canonical-hash";
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
 * The hash step is factored out into {@link useCanonicalHash} so the same
 * request is shared with `useConfigValidation` (one network call, two
 * consumers).
 */
export function usePrerunCache(config: SimConfig, debounceMs = 350): PrerunCache {
  const { hash, isFetching: hashFetching } = useCanonicalHash(config, debounceMs);

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
    loading: hashFetching || cacheQuery.isFetching,
  };
}

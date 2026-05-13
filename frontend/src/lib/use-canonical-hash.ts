import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";
import type { SimConfig } from "@/types/api";

/**
 * Snapshot of the server-computed canonical hash for the current config.
 *
 * `hash`           SHA-256 hex of `canonical_config_hash(config)`. `null` until
 *                  the first debounce fires.
 * `debouncedConfig` the config payload that actually produced `hash`, so
 *                  downstream queries don't drift from the key they're using.
 * `isFetching`     true while the hash request is in flight.
 */
export interface CanonicalHash {
  hash: string | null;
  debouncedConfig: SimConfig;
  isFetching: boolean;
}

/**
 * Debounce-aware hook that hashes the live config through `/api/config/hash`.
 *
 * Both {@link usePrerunCache} and {@link useConfigValidation} call this so that
 * the hash request is shared (TanStack dedupes by `queryKey`), and so any
 * downstream query that's keyed by *hash* (instead of by JSON.stringify) keeps
 * a much smaller cache map and naturally cache-hits across structurally
 * identical configs.
 *
 * The debounce is intentionally short (350 ms): long enough that typing into a
 * number input doesn't fire a hash per keystroke, short enough that downstream
 * badges/banners feel live.
 */
export function useCanonicalHash(config: SimConfig, debounceMs = 350): CanonicalHash {
  /*
   * React rebuilds the config object on every reducer pass, so its identity
   * changes even when nothing material did. Serializing pins the key to actual
   * content and lets TanStack dedupe the request.
   */
  const serialized = useMemo(() => JSON.stringify(config), [config]);
  const [debouncedKey, setDebouncedKey] = useState(serialized);

  useEffect(() => {
    const t = setTimeout(() => setDebouncedKey(serialized), debounceMs);
    return () => clearTimeout(t);
  }, [serialized, debounceMs]);

  // Re-parse on demand so consumers always get a fresh deep clone they can
  // hand to validateConfig or any other endpoint without aliasing.
  const debouncedConfig = useMemo(
    () => JSON.parse(debouncedKey) as SimConfig,
    [debouncedKey],
  );

  const ready = (debouncedConfig.channel_list ?? []).length > 0;

  const query = useQuery({
    queryKey: ["config-hash", debouncedKey],
    queryFn: () => api.hashConfig(debouncedConfig),
    enabled: ready,
    // Hash is deterministic for a given config + backend version, so we can
    // keep it warm for a while. 1 minute matches usePrerunCache's prior value.
    staleTime: 60_000,
    gcTime: 5 * 60_000,
  });

  return {
    hash: query.data?.config_hash ?? null,
    debouncedConfig,
    isFetching: query.isFetching,
  };
}

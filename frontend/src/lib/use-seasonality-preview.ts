import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";

/** Loose shape: any object the simulator's `_seasonality()` accepts.
 *  Typed as `object` (rather than `Record<string, unknown>`) so callers can
 *  pass structural shapes like `FourierConfig` without an index signature. */
type SeasonalityConfigLike = object | undefined | null;

/**
 * Debounced "let the Python pipeline tell us what this config evaluates to"
 * hook for the seasonality preview chart.
 *
 * The Fourier editor renders a TypeScript port of the deterministic Fourier
 * math for an instant preview as the user drags coefficients. This hook
 * fetches the same series from the actual simulator (`/api/seasonality/evaluate`)
 * and returns it so the chart can overlay both. The two lines should sit on
 * top of each other for deterministic configs — that overlap is exactly the
 * point: it gives users visual confidence that the preview matches what the
 * simulator will produce, and surfaces real values for random-Fourier configs
 * the TS port can't render.
 *
 * The query is keyed by the serialized config + week count, debounced at
 * 250 ms so a single keystroke doesn't queue a request.
 */
export function useSeasonalityPreview(
  cfg: SeasonalityConfigLike,
  weeks: number,
  options: { debounceMs?: number; enabled?: boolean; seed?: number | null } = {},
) {
  const { debounceMs = 250, enabled = true, seed = 0 } = options;

  const serialized = useMemo(() => JSON.stringify(cfg ?? {}), [cfg]);
  const [debounced, setDebounced] = useState(serialized);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(serialized), debounceMs);
    return () => clearTimeout(t);
  }, [serialized, debounceMs]);

  const empty = !debounced || debounced === "{}" || debounced === "null";

  const query = useQuery({
    queryKey: ["seasonality-eval", debounced, weeks, seed],
    queryFn: async () => {
      const parsed = JSON.parse(debounced) as Record<string, unknown>;
      return api.evaluateSeasonality(parsed, weeks, seed);
    },
    enabled: enabled && !empty && weeks > 0,
    staleTime: 60_000,
    gcTime: 5 * 60_000,
  });

  return {
    multipliers: query.data?.multipliers ?? null,
    weeks: query.data?.weeks ?? weeks,
    isFetching: query.isFetching,
    error: query.error?.message ?? null,
  };
}

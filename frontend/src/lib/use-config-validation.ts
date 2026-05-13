import { keepPreviousData, useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";
import { useCanonicalHash } from "@/lib/use-canonical-hash";
import type { ConfigIssue, SimConfig, ValidateConfigResponse } from "@/types/api";

const DEBOUNCE_MS = 350;

interface ValidationState {
  loading: boolean;
  data: ValidateConfigResponse | null;
  error: string | null;
}

/**
 * Validate the current config via the FastAPI server.
 *
 * Keyed off the **canonical config hash** rather than `JSON.stringify(config)`
 * so that React Query's cache stays small even for big configs, and so we
 * share a single hash request with {@link usePrerunCache}. Two structurally
 * identical configs (e.g. visited via undo/redo) hash the same, hit the
 * validation cache, and never trigger a network round-trip.
 *
 * The last good result is kept visible via `placeholderData: keepPreviousData`
 * so the banner doesn't flash empty between keystrokes.
 */
export function useConfigValidation(config: SimConfig): ValidationState {
  const { hash, debouncedConfig, isFetching: hashing } = useCanonicalHash(
    config,
    DEBOUNCE_MS,
  );

  const query = useQuery<ValidateConfigResponse, Error>({
    queryKey: ["config-validate", hash],
    // queryFn captures the same debounced snapshot the hash was computed
    // from, so the validation result and the cache key can never drift.
    queryFn: () => api.validateConfig(debouncedConfig),
    enabled: Boolean(hash),
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
    placeholderData: keepPreviousData,
  });

  return {
    loading: hashing || query.isFetching,
    data: query.data ?? null,
    error: query.error?.message ?? null,
  };
}

// -- Helpers --------------------------------------------------------------

export function issuesBySection(
  issues: ConfigIssue[] | undefined,
): Record<string, ConfigIssue[]> {
  const out: Record<string, ConfigIssue[]> = {};
  for (const issue of issues ?? []) {
    const key = issue.section ?? "general";
    (out[key] ||= []).push(issue);
  }
  return out;
}

/** Count error-severity issues for a given channel index. */
export function channelErrorCount(
  issues: ConfigIssue[] | undefined,
  channelIndex: number,
): number {
  let n = 0;
  for (const issue of issues ?? []) {
    if (issue.severity !== "error") continue;
    if (issue.path[0] === "channel_list" && issue.path[1] === channelIndex) n += 1;
  }
  return n;
}

export function hasBlockingErrors(issues: ConfigIssue[] | undefined): boolean {
  return (issues ?? []).some((i) => i.severity === "error");
}

/** Count issues matching an arbitrary predicate, split by severity. */
export function countIssues(
  issues: ConfigIssue[] | undefined,
  predicate: (issue: ConfigIssue) => boolean,
): { errors: number; warnings: number } {
  let errors = 0;
  let warnings = 0;
  for (const issue of issues ?? []) {
    if (!predicate(issue)) continue;
    if (issue.severity === "error") errors += 1;
    else if (issue.severity === "warning") warnings += 1;
  }
  return { errors, warnings };
}

/** Convenience: count issues whose JSON-Pointer path starts with the given prefix. */
export function countIssuesAtPath(
  issues: ConfigIssue[] | undefined,
  prefix: ReadonlyArray<string | number>,
): { errors: number; warnings: number } {
  return countIssues(issues, (issue) => {
    if (issue.path.length < prefix.length) return false;
    for (let i = 0; i < prefix.length; i += 1) {
      if (issue.path[i] !== prefix[i]) return false;
    }
    return true;
  });
}

/** Stringify path like ["channel_list", 0, "channel", "true_roi"] →
 *  `channel_list[0].channel.true_roi`. */
export function formatPath(path: Array<string | number>): string {
  if (!path.length) return "(global)";
  let out = "";
  for (const token of path) {
    if (typeof token === "number") out += `[${token}]`;
    else if (out === "") out = token;
    else out += `.${token}`;
  }
  return out;
}

/** Spread onto a form control to make it discoverable by the issue navigator.
 *
 * @example
 *   <Input id="week_range" {...configPathAttr(["week_range"])} ... />
 */
export function configPathAttr(
  path: ReadonlyArray<string | number>,
): { "data-config-path": string } {
  return { "data-config-path": formatPath([...path]) };
}

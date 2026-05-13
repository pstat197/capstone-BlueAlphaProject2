import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";
import type { ConfigIssue, SimConfig, ValidateConfigResponse } from "@/types/api";

const DEBOUNCE_MS = 350;

interface ValidationState {
  loading: boolean;
  data: ValidateConfigResponse | null;
  error: string | null;
}

/** Debounces the serialized config so that identical shapes only key one
 *  query in the React Query cache, even when the user is mid-typing. */
function useDebouncedSerialized(config: SimConfig, ms: number): string {
  const serialized = useMemo(() => JSON.stringify(config), [config]);
  const [debounced, setDebounced] = useState(serialized);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(serialized), ms);
    return () => clearTimeout(timer);
  }, [serialized, ms]);
  return debounced;
}

/**
 * Validate the current config via the FastAPI server.
 *
 * Backed by React Query so the result survives route changes (e.g.
 * Simulator → Results → back) without re-hitting the server, and so
 * identical configs visited again are served from cache. The last good
 * result is kept visible while the next request is in flight via
 * `placeholderData: keepPreviousData`, which prevents the banner from
 * flashing empty between keystrokes.
 */
export function useConfigValidation(config: SimConfig): ValidationState {
  const debounced = useDebouncedSerialized(config, DEBOUNCE_MS);
  const query = useQuery<ValidateConfigResponse, Error>({
    queryKey: ["config-validate", debounced],
    queryFn: () => api.validateConfig(JSON.parse(debounced) as SimConfig),
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
    placeholderData: keepPreviousData,
  });

  return {
    loading: query.isFetching,
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

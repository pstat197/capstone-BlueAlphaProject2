import { useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";
import type { ConfigIssue, SimConfig, ValidateConfigResponse } from "@/types/api";

const DEBOUNCE_MS = 350;

interface ValidationState {
  loading: boolean;
  data: ValidateConfigResponse | null;
  error: string | null;
}

/**
 * Debounced POST /api/config/validate. The hook is intentionally tolerant:
 * a network failure leaves the last successful result in place so the UI
 * doesn't flash empty errors as the user types, and an in-flight request
 * is superseded by the next config change (cheap throwaway).
 */
export function useConfigValidation(config: SimConfig): ValidationState {
  const [state, setState] = useState<ValidationState>({
    loading: false,
    data: null,
    error: null,
  });

  const serialized = useMemo(() => JSON.stringify(config), [config]);

  useEffect(() => {
    let cancelled = false;
    const timer = setTimeout(() => {
      setState((s) => ({ ...s, loading: true }));
      api
        .validateConfig(config)
        .then((data) => {
          if (cancelled) return;
          setState({ loading: false, data, error: null });
        })
        .catch((err: unknown) => {
          if (cancelled) return;
          setState((s) => ({
            ...s,
            loading: false,
            error: err instanceof Error ? err.message : String(err),
          }));
        });
    }, DEBOUNCE_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // We deliberately key off the serialized config so deep edits trigger
    // a re-validate but identical shapes are deduped.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serialized]);

  return state;
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

import { AlertCircle, AlertTriangle, CheckCircle2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { formatPath } from "@/lib/use-config-validation";
import type { ConfigIssue } from "@/types/api";

interface ValidationBannerProps {
  issues: ConfigIssue[] | undefined;
  loading?: boolean;
  /** When true, the banner is rendered in a compact "clean" success state. */
  showClean?: boolean;
  /** Click handler for "jump to the offending field". When omitted, the
   *  issue path renders as static text. */
  onNavigate?: (path: Array<string | number>) => void;
}

const sectionLabels: Record<string, string> = {
  general: "General",
  outcome: "Outcome",
  channels: "Channels",
  correlations: "Correlations",
  budget_shifts: "Budget shifts",
};

/**
 * Single source of truth for surfacing schema-tied validation errors.
 *
 * - Suppressed entirely when the config validates clean (unless `showClean`
 *   is set, in which case a tiny green pill is rendered).
 * - When there are issues, shows a categorized list with field paths in
 *   monospace so users can correlate to the YAML editor / form controls.
 * - Warnings (e.g. Fourier K above Nyquist) are rendered amber but do not
 *   block the Run button — see `hasBlockingErrors`.
 */
export function ValidationBanner({
  issues,
  loading,
  showClean,
  onNavigate,
}: ValidationBannerProps) {
  const errors = (issues ?? []).filter((i) => i.severity === "error");
  const warnings = (issues ?? []).filter((i) => i.severity === "warning");

  if (errors.length === 0 && warnings.length === 0) {
    if (!showClean) return null;
    return (
      <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50/60 px-3 py-2 text-xs text-emerald-800">
        <CheckCircle2 className="h-3.5 w-3.5" />
        Config validates against the simulator schema.
        {loading && <span className="text-emerald-700/70">(rechecking…)</span>}
      </div>
    );
  }

  const hasErrors = errors.length > 0;
  return (
    <div
      className={
        hasErrors
          ? "rounded-lg border border-rose-200 bg-rose-50/70 px-4 py-3 text-sm text-rose-900"
          : "rounded-lg border border-amber-200 bg-amber-50/70 px-4 py-3 text-sm text-amber-900"
      }
    >
      <div className="flex items-start gap-3">
        {hasErrors ? (
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-600" />
        ) : (
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
        )}
        <div className="flex-1 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium">
              {hasErrors
                ? `${errors.length} configuration ${errors.length === 1 ? "error" : "errors"}`
                : `${warnings.length} ${warnings.length === 1 ? "warning" : "warnings"}`}
            </span>
            {hasErrors && warnings.length > 0 && (
              <Badge variant="warn">{warnings.length} warnings</Badge>
            )}
            {loading && <span className="text-xs opacity-70">rechecking…</span>}
          </div>
          <ul className="space-y-1 text-xs leading-relaxed">
            {errors.map((issue, idx) => (
              <IssueRow
                key={`err_${idx}`}
                issue={issue}
                tone="error"
                onNavigate={onNavigate}
              />
            ))}
            {warnings.map((issue, idx) => (
              <IssueRow
                key={`warn_${idx}`}
                issue={issue}
                tone="warn"
                onNavigate={onNavigate}
              />
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function IssueRow({
  issue,
  tone,
  onNavigate,
}: {
  issue: ConfigIssue;
  tone: "error" | "warn";
  onNavigate?: (path: Array<string | number>) => void;
}) {
  const sectionLabel = sectionLabels[issue.section ?? "general"] ?? "General";
  const pathText = formatPath(issue.path);
  const pathColor =
    tone === "error" ? "text-rose-700/90" : "text-amber-700/90";
  const navigable = Boolean(onNavigate) && issue.path.length > 0;

  return (
    <li className="flex flex-wrap items-baseline gap-2">
      <Badge variant="outline" className="bg-white text-[10px]">
        {sectionLabel}
      </Badge>
      {navigable ? (
        <button
          type="button"
          onClick={() => onNavigate?.(issue.path)}
          className={`font-mono text-[11px] ${pathColor} underline-offset-2 hover:underline focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-current rounded`}
          title="Jump to this field"
        >
          {pathText}
        </button>
      ) : (
        <span className={`font-mono text-[11px] ${pathColor}`}>{pathText}</span>
      )}
      <span>{issue.message}</span>
    </li>
  );
}

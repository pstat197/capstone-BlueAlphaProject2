import { AlertCircle, AlertTriangle } from "lucide-react";

import { Tooltip } from "@/components/ui/tooltip";

interface IssueCountBadgeProps {
  errors: number;
  warnings?: number;
  label?: string;
}

/**
 * Compact dot+count chip used in tab triggers / card headers to surface
 * validation results without taking up real estate. Errors take precedence
 * visually; warnings only render their own pill when there are zero errors.
 */
export function IssueCountBadge({ errors, warnings = 0, label }: IssueCountBadgeProps) {
  if (errors === 0 && warnings === 0) return null;
  const tooltip = (() => {
    const parts: string[] = [];
    if (errors > 0) parts.push(`${errors} error${errors === 1 ? "" : "s"}`);
    if (warnings > 0) parts.push(`${warnings} warning${warnings === 1 ? "" : "s"}`);
    const suffix = label ? ` in ${label}` : "";
    return parts.join(" · ") + suffix;
  })();
  return (
    <Tooltip content={tooltip}>
      <span
        className={
          errors > 0
            ? "inline-flex items-center gap-1 rounded-full bg-rose-100 px-1.5 py-0.5 text-[10px] font-semibold text-rose-700"
            : "inline-flex items-center gap-1 rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold text-amber-700"
        }
      >
        {errors > 0 ? (
          <AlertCircle className="h-3 w-3" />
        ) : (
          <AlertTriangle className="h-3 w-3" />
        )}
        {errors > 0 ? errors : warnings}
      </span>
    </Tooltip>
  );
}

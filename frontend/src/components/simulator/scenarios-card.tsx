import { useMemo } from "react";

import { BudgetShiftsPanel } from "@/components/simulator/budget-shifts-panel";
import { CorrelationsPanel } from "@/components/simulator/correlations-panel";
import { IssueCountBadge } from "@/components/simulator/issue-count-badge";
import { SeasonalityPanel } from "@/components/simulator/seasonality-panel";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { countIssues, countIssuesAtPath } from "@/lib/use-config-validation";
import { useConfig } from "@/state/config-store";
import type { ConfigIssue } from "@/types/api";

export type ScenariosTab = "seasonality" | "budget" | "correlations";

interface ScenariosCardProps {
  issues?: ConfigIssue[];
  /** Controlled active tab. Falls back to "seasonality" when undefined. */
  tab?: ScenariosTab;
  onTabChange?: (tab: ScenariosTab) => void;
}

/**
 * Single card that groups the three scenario knobs that don't belong to any
 * single channel: seasonality (outcome path), budget shifts (post-draw spend
 * rules), and spend correlations.
 *
 * Each tab gets a small count/state badge so the user can see at a glance
 * which scenarios are active without opening every tab.
 */
export function ScenariosCard({ issues, tab, onTabChange }: ScenariosCardProps) {
  const { config } = useConfig();

  const summary = useMemo(() => {
    const bsManual = (config.budget_shifts ?? []).length;
    const bsAuto =
      ((config.budget_shifts_auto_mode as string | undefined) ?? "none") !== "none";
    const corrManual = (config.correlations ?? []).length;
    const corrAuto =
      ((config.correlations_auto_mode as string | undefined) ?? "none") !== "none";
    const outcome = config.outcome_revenue as { seasonality_config?: Record<string, unknown> } | undefined;
    const seaCount = outcome?.seasonality_config && Object.keys(outcome.seasonality_config).length > 0 ? 1 : 0;
    return {
      seasonalityActive: seaCount > 0,
      budgetShifts: { manual: bsManual, auto: bsAuto },
      correlations: { manual: corrManual, auto: corrAuto },
    };
  }, [config]);

  /** Seasonality issues can land on the outcome OR any channel — match by
   *  the substring "seasonality_config" anywhere in the path. */
  const seasonalityIssues = useMemo(
    () =>
      countIssues(issues, (issue) =>
        issue.path.some((p) => p === "seasonality_config"),
      ),
    [issues],
  );
  const budgetIssues = useMemo(
    () => countIssuesAtPath(issues, ["budget_shifts"]),
    [issues],
  );
  const correlationIssues = useMemo(
    () => countIssuesAtPath(issues, ["correlations"]),
    [issues],
  );
  const totalIssues = {
    errors:
      seasonalityIssues.errors + budgetIssues.errors + correlationIssues.errors,
    warnings:
      seasonalityIssues.warnings +
      budgetIssues.warnings +
      correlationIssues.warnings,
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              Scenarios
              <IssueCountBadge
                errors={totalIssues.errors}
                warnings={totalIssues.warnings}
                label="scenarios"
              />
            </CardTitle>
            <CardDescription>
              Time-based knobs that aren't tied to a single channel. Seasonality shapes the
              outcome path; budget shifts override post-draw spend; correlations couple weekly
              spend across channels.
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs
          value={tab ?? "seasonality"}
          onValueChange={(v) => onTabChange?.(v as ScenariosTab)}
        >
          <TabsList className="flex flex-wrap">
            <TabsTrigger value="seasonality" className="flex items-center gap-2">
              Seasonality
              {summary.seasonalityActive && (
                <Badge variant="success" className="text-[9px] uppercase">on</Badge>
              )}
              <IssueCountBadge
                errors={seasonalityIssues.errors}
                warnings={seasonalityIssues.warnings}
                label="seasonality"
              />
            </TabsTrigger>
            <TabsTrigger value="budget" className="flex items-center gap-2">
              Budget shifts
              {summary.budgetShifts.manual > 0 && (
                <Badge variant="default" className="text-[9px]">{summary.budgetShifts.manual}</Badge>
              )}
              {summary.budgetShifts.auto && (
                <Badge variant="muted" className="text-[9px] uppercase">auto</Badge>
              )}
              <IssueCountBadge
                errors={budgetIssues.errors}
                warnings={budgetIssues.warnings}
                label="budget shifts"
              />
            </TabsTrigger>
            <TabsTrigger value="correlations" className="flex items-center gap-2">
              Correlations
              {summary.correlations.manual > 0 && (
                <Badge variant="default" className="text-[9px]">{summary.correlations.manual}</Badge>
              )}
              {summary.correlations.auto && (
                <Badge variant="muted" className="text-[9px] uppercase">auto</Badge>
              )}
              <IssueCountBadge
                errors={correlationIssues.errors}
                warnings={correlationIssues.warnings}
                label="correlations"
              />
            </TabsTrigger>
          </TabsList>

          <TabsContent value="seasonality" className="pt-4">
            <SeasonalityPanel />
          </TabsContent>
          <TabsContent value="budget" className="pt-4">
            <BudgetShiftsPanel />
          </TabsContent>
          <TabsContent value="correlations" className="pt-4">
            <CorrelationsPanel />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

import { useMemo } from "react";

import { BudgetShiftsPanel } from "@/components/simulator/budget-shifts-panel";
import { CorrelationsPanel } from "@/components/simulator/correlations-panel";
import { SeasonalityPanel } from "@/components/simulator/seasonality-panel";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useConfig } from "@/state/config-store";

/**
 * Single card that groups the three scenario knobs that don't belong to any
 * single channel: seasonality (outcome path), budget shifts (post-draw spend
 * rules), and spend correlations.
 *
 * Each tab gets a small count/state badge so the user can see at a glance
 * which scenarios are active without opening every tab.
 */
export function ScenariosCard() {
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

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle>Scenarios</CardTitle>
        <CardDescription>
          Time-based knobs that aren't tied to a single channel. Seasonality shapes the outcome
          path; budget shifts override post-draw spend; correlations couple weekly spend across
          channels.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="seasonality">
          <TabsList className="flex flex-wrap">
            <TabsTrigger value="seasonality" className="flex items-center gap-2">
              Seasonality
              {summary.seasonalityActive && (
                <Badge variant="success" className="text-[9px] uppercase">on</Badge>
              )}
            </TabsTrigger>
            <TabsTrigger value="budget" className="flex items-center gap-2">
              Budget shifts
              {summary.budgetShifts.manual > 0 && (
                <Badge variant="default" className="text-[9px]">{summary.budgetShifts.manual}</Badge>
              )}
              {summary.budgetShifts.auto && (
                <Badge variant="muted" className="text-[9px] uppercase">auto</Badge>
              )}
            </TabsTrigger>
            <TabsTrigger value="correlations" className="flex items-center gap-2">
              Correlations
              {summary.correlations.manual > 0 && (
                <Badge variant="default" className="text-[9px]">{summary.correlations.manual}</Badge>
              )}
              {summary.correlations.auto && (
                <Badge variant="muted" className="text-[9px] uppercase">auto</Badge>
              )}
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

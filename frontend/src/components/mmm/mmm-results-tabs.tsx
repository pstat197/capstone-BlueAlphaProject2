import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MmmFitTab } from "@/components/mmm/mmm-fit-tab";
import { MmmOptimizationTab } from "@/components/mmm/mmm-optimization-tab";
import { MmmRoiForest } from "@/components/mmm/mmm-roi-forest";
import type { MmmFitResults } from "@/types/api";

interface Props {
  results: MmmFitResults;
}

export function MmmResultsTabs({ results }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Fit results</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="roi">
          <TabsList>
            <TabsTrigger value="roi">Recovered media ROI</TabsTrigger>
            <TabsTrigger value="fit">Model fit & diagnostics</TabsTrigger>
            <TabsTrigger value="opt">Budget optimization</TabsTrigger>
          </TabsList>
          <TabsContent value="roi">
            <p className="mb-3 text-xs text-slate-600">
              Per-channel posterior <strong>mean</strong> ROI with the 50% credible interval. The
              red dashed line, when present, is the <strong>true</strong> ROI from your simulator
              YAML. Hover a row for details.
            </p>
            <MmmRoiForest data={results.roi_forest} />
          </TabsContent>
          <TabsContent value="fit">
            <MmmFitTab results={results} />
          </TabsContent>
          <TabsContent value="opt">
            <MmmOptimizationTab data={results.budget_optimization} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

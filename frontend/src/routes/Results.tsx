import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { useEffect } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ConfigSnapshot } from "@/components/results/config-snapshot";
import { PreviewTable } from "@/components/results/preview-table";
import { ResultsCharts } from "@/components/results/results-charts";
import { RunSummary } from "@/components/results/run-summary";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/lib/api";
import { useConfig } from "@/state/config-store";

export default function ResultsRoute() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const { setConfig, setLastHash } = useConfig();

  const runQuery = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRun(runId as string),
    enabled: Boolean(runId),
  });

  useEffect(() => {
    if (runQuery.data) {
      setLastHash(runQuery.data.config_hash);
    }
  }, [runQuery.data, setLastHash]);

  if (!runId) {
    return (
      <Card>
        <CardContent className="px-6 py-10 text-center text-sm text-slate-500">
          Missing run id.
        </CardContent>
      </Card>
    );
  }

  if (runQuery.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-12 w-72" />
        <Skeleton className="h-[420px] w-full" />
      </div>
    );
  }

  if (runQuery.isError) {
    return (
      <Card>
        <CardContent className="space-y-3 px-6 py-10 text-center">
          <p className="text-sm text-rose-700">
            Could not load this run: {(runQuery.error as Error).message}
          </p>
          <Button variant="ghost" size="sm" asChild>
            <Link to="/simulator">
              <ArrowLeft className="h-3.5 w-3.5" />
              Back to simulator
            </Link>
          </Button>
        </CardContent>
      </Card>
    );
  }

  const run = runQuery.data;
  if (!run) return null;

  const handleEditConfiguration = () => {
    setConfig(run.config, { resetYamlDirty: true });
    setLastHash(run.config_hash);
    navigate("/simulator");
  };

  return (
    <div className="flex flex-1 flex-col gap-6">
      <RunSummary run={run} onEditConfiguration={handleEditConfiguration} />

      <Tabs defaultValue="charts">
        <TabsList>
          <TabsTrigger value="charts">Chart view</TabsTrigger>
          <TabsTrigger value="preview">Data preview</TabsTrigger>
          <TabsTrigger value="config">Configuration</TabsTrigger>
        </TabsList>

        <TabsContent value="charts">
          <ResultsCharts run={run} />
        </TabsContent>
        <TabsContent value="preview">
          <PreviewTable preview={run.preview} />
        </TabsContent>
        <TabsContent value="config">
          <ConfigSnapshot config={run.config} />
        </TabsContent>
      </Tabs>
    </div>
  );
}

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ExternalLink, Loader2, Play, RotateCcw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

import { AdvancedSettingsCard } from "@/components/simulator/advanced-settings-card";
import { ChannelDetail } from "@/components/simulator/channel-detail";
import { ChannelList, type SimulatorPane } from "@/components/simulator/channel-list";
import { RunSettingsCard } from "@/components/simulator/run-settings-card";
import { ScenariosCard, type ScenariosTab } from "@/components/simulator/scenarios-card";
import { ValidationBanner } from "@/components/simulator/validation-banner";
import { YamlEditorCard } from "@/components/simulator/yaml-editor-card";
import { StickyActionBar } from "@/components/layout/sticky-action-bar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { formatHash } from "@/lib/config-utils";
import { useConfigPathNavigator } from "@/lib/use-config-path-navigator";
import { hasBlockingErrors, useConfigValidation } from "@/lib/use-config-validation";
import { usePrerunCache } from "@/lib/use-prerun-cache";
import { useConfig } from "@/state/config-store";
import type { RunResponse } from "@/types/api";

export default function SimulatorRoute() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { config, setConfig, setLastHash, lastHash } = useConfig();
  const [selected, setSelected] = useState<SimulatorPane>({ kind: "channel", index: 0 });
  const [scenariosTab, setScenariosTab] = useState<ScenariosTab>("seasonality");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const seededRef = useRef(false);
  const [error, setError] = useState<string | null>(null);

  const exampleQuery = useQuery({
    queryKey: ["example-config"],
    queryFn: () => api.exampleConfig(),
    staleTime: Infinity,
  });

  /*
   * Seed the in-flight config from example.yaml exactly once when the user lands
   * with a blank config. This is the canonical "sync external data into local state"
   * case the React docs sanction; the ref makes it idempotent so it doesn't cascade.
   */
  useEffect(() => {
    if (seededRef.current) return;
    if (!exampleQuery.data) return;
    if ((config.channel_list?.length ?? 0) === 0) {
      setConfig(exampleQuery.data.config, { resetYamlDirty: true });
    }
    seededRef.current = true;
  }, [exampleQuery.data, config.channel_list, setConfig]);

  /* Derive a valid selected pane every render so we never need an effect to fix it. */
  const channelCount = config.channel_list?.length ?? 0;
  const effectiveSelected = useMemo<SimulatorPane>(() => {
    if (selected.kind === "channel") {
      if (channelCount === 0) return { kind: "yaml" };
      if (selected.index >= channelCount) {
        return { kind: "channel", index: channelCount - 1 };
      }
    }
    return selected;
  }, [selected, channelCount]);

  const prerun = usePrerunCache(config);
  const validation = useConfigValidation(config);
  const blocked = hasBlockingErrors(validation.data?.issues);
  const navigateToPath = useConfigPathNavigator({
    setSelected,
    setScenariosTab,
    setAdvancedOpen,
  });

  const runMutation = useMutation({
    mutationFn: () => api.createRun(config),
    onSuccess: (data: RunResponse) => {
      queryClient.setQueryData(["run", data.config_hash], data);
      void queryClient.invalidateQueries({ queryKey: ["runs"] });
      setLastHash(data.config_hash);
      setError(null);
      navigate(`/results/${data.config_hash}`);
    },
    onError: (e: Error) => {
      setError(e.message);
    },
  });

  const handleResetExample = () => {
    if (!exampleQuery.data) return;
    setConfig(exampleQuery.data.config, { resetYamlDirty: true });
    setLastHash(null);
    setSelected({ kind: "channel", index: 0 });
    setError(null);
  };

  return (
    <div className="flex flex-1 flex-col gap-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Simulator</h1>
          <p className="mt-1 text-sm text-slate-500">
            Configure the synthetic marketing-mix simulation and run it. Results land in a dedicated
            view with a re-open link.
          </p>
        </div>
        {lastHash && (
          <Badge variant="muted" className="font-mono">
            last config {formatHash(lastHash)}
          </Badge>
        )}
      </header>

      <ValidationBanner
        issues={validation.data?.issues}
        loading={validation.loading}
        onNavigate={navigateToPath}
      />

      <RunSettingsCard issues={validation.data?.issues} />
      <AdvancedSettingsCard
        issues={validation.data?.issues}
        open={advancedOpen}
        onOpenChange={setAdvancedOpen}
      />
      <ScenariosCard
        issues={validation.data?.issues}
        tab={scenariosTab}
        onTabChange={setScenariosTab}
      />

      <div className="grid flex-1 gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
        <div className="lg:max-h-[calc(100vh-260px)]">
          <ChannelList
            selected={effectiveSelected}
            onSelect={setSelected}
            issues={validation.data?.issues}
          />
        </div>
        <div className="min-w-0">
          {effectiveSelected.kind === "channel" && channelCount > 0 ? (
            <ChannelDetail
              index={effectiveSelected.index}
              onIndexChange={(next) => setSelected({ kind: "channel", index: next })}
            />
          ) : (
            <YamlEditorCard />
          )}
        </div>
      </div>

      <StickyActionBar
        left={
          <>
            <span>
              <strong className="text-slate-900">{channelCount}</strong>{" "}
              channel{channelCount === 1 ? "" : "s"} ·{" "}
              <strong className="text-slate-900">
                {(config.week_range as number | undefined) ?? "?"}
              </strong>{" "}
              weeks
            </span>
            {prerun.hash && (
              <Badge
                variant={prerun.cached ? "success" : "muted"}
                className="font-mono text-[10px]"
              >
                {prerun.cached ? "cached" : "new"} {formatHash(prerun.hash)}
              </Badge>
            )}
            {prerun.cached && prerun.hash && (
              <Link
                to={`/results/${prerun.hash}`}
                className="inline-flex items-center gap-1 text-xs font-medium text-brand-700 underline-offset-2 hover:underline"
              >
                <ExternalLink className="h-3 w-3" />
                Open cached results
              </Link>
            )}
            {error && (
              <span className="rounded-full bg-rose-50 px-3 py-1 text-xs font-medium text-rose-700">
                {error}
              </span>
            )}
          </>
        }
        right={
          <>
            <Button variant="ghost" size="sm" onClick={handleResetExample} disabled={!exampleQuery.data}>
              <RotateCcw className="h-3.5 w-3.5" />
              Reset to example
            </Button>
            <Button
              size="default"
              onClick={() => runMutation.mutate()}
              disabled={runMutation.isPending || channelCount === 0 || blocked}
              variant={prerun.cached ? "secondary" : "default"}
              title={blocked ? "Fix configuration errors to run" : undefined}
            >
              {runMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Running…
                </>
              ) : blocked ? (
                <>
                  <Play className="h-4 w-4" />
                  Fix errors to run
                </>
              ) : prerun.cached ? (
                <>
                  <Play className="h-4 w-4" />
                  Re-run from cache
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Run simulation
                </>
              )}
            </Button>
          </>
        }
      />
    </div>
  );
}

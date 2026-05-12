import { yaml as yamlLang } from "@codemirror/lang-yaml";
import CodeMirror from "@uiw/react-codemirror";
import { ChevronDown, Download } from "lucide-react";
import { useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { dumpYaml } from "@/lib/yaml";
import type { GroundTruth } from "@/types/api";

interface GroundTruthCardProps {
  groundTruth: GroundTruth | null;
}

function fmtNumber(value: number | undefined | null, fractionDigits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString(undefined, {
    maximumFractionDigits: fractionDigits,
  });
}

function downloadJson(filename: string, payload: unknown): void {
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function GroundTruthCard({ groundTruth }: GroundTruthCardProps) {
  const [open, setOpen] = useState(false);

  const yamlText = useMemo(
    () => (groundTruth ? dumpYaml(groundTruth as unknown as Record<string, unknown>) : ""),
    [groundTruth],
  );

  if (!groundTruth) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Ground truth</CardTitle>
          <CardDescription>
            The simulator could not derive a ground-truth snapshot for this run.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const outcome = groundTruth.outcome_revenue;
  const toggles = groundTruth.global_toggles;
  const channels = groundTruth.channels ?? [];
  const filename = `${groundTruth.run_identifier || "run"}_ground_truth.json`;

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle className="flex items-center gap-2">
              Ground truth
              <Badge variant="outline">v{groundTruth.ground_truth_version}</Badge>
            </CardTitle>
            <CardDescription>
              The exact generative parameters used to synthesize this run — the &ldquo;true&rdquo;
              answer to compare any downstream model against.
            </CardDescription>
          </div>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => downloadJson(filename, groundTruth)}
          >
            <Download className="h-3.5 w-3.5" />
            Download JSON
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="Baseline revenue" value={fmtNumber(outcome.baseline_revenue)} />
          <Stat label="Trend slope" value={fmtNumber(outcome.trend_slope, 4)} />
          <Stat label="Seed" value={groundTruth.seed ?? "—"} />
          <Stat label="Weeks" value={groundTruth.week_range} />
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          <Badge variant={toggles.adstock_global ? "muted" : "warn"}>
            adstock {toggles.adstock_global ? "on" : "off"}
          </Badge>
          <Badge variant={toggles.saturation_global ? "muted" : "warn"}>
            saturation {toggles.saturation_global ? "on" : "off"}
          </Badge>
          <Badge variant="outline">order: {toggles.media_transform_order}</Badge>
        </div>

        <div className="overflow-hidden rounded-lg border border-brand-border">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-4 py-2 font-medium">Channel</th>
                <th className="px-4 py-2 font-medium">True ROI</th>
                <th className="px-4 py-2 font-medium">CPM</th>
                <th className="px-4 py-2 font-medium">Spend range</th>
                <th className="px-4 py-2 font-medium">Adstock</th>
                <th className="px-4 py-2 font-medium">Saturation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {channels.map((ch) => {
                const sat = (ch.saturation_config ?? {}) as Record<string, unknown>;
                const ads = (ch.adstock_decay_config ?? {}) as Record<string, unknown>;
                const satType = (sat.type as string | undefined) ?? "—";
                const adsType = (ads.type as string | undefined) ?? "—";
                const spendRange = ch.spend_range
                  ? `${fmtNumber(ch.spend_range[0])} – ${fmtNumber(ch.spend_range[1])}`
                  : "—";
                return (
                  <tr key={ch.channel_name}>
                    <td className="whitespace-nowrap px-4 py-2 font-medium text-slate-900">
                      {ch.channel_name}
                      {!ch.enabled && (
                        <Badge variant="outline" className="ml-2 text-[10px]">
                          off
                        </Badge>
                      )}
                    </td>
                    <td className="px-4 py-2 text-slate-700">{fmtNumber(ch.true_roi, 3)}</td>
                    <td className="px-4 py-2 text-slate-700">{fmtNumber(ch.cpm)}</td>
                    <td className="px-4 py-2 text-slate-700">{spendRange}</td>
                    <td className="px-4 py-2 text-slate-700">
                      {ch.adstock_enabled === false ? (
                        <Badge variant="warn" className="text-[10px]">
                          disabled
                        </Badge>
                      ) : (
                        adsType
                      )}
                    </td>
                    <td className="px-4 py-2 text-slate-700">
                      {ch.saturation_enabled === false ? (
                        <Badge variant="warn" className="text-[10px]">
                          disabled
                        </Badge>
                      ) : (
                        satType
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="space-y-2">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="flex items-center gap-1 text-xs font-medium text-slate-600 hover:text-slate-900"
          >
            <ChevronDown
              className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-180" : ""}`}
            />
            {open ? "Hide raw ground-truth YAML" : "Show raw ground-truth YAML"}
          </button>
          {open && (
            <div className="overflow-hidden rounded-lg border border-brand-border">
              <CodeMirror
                value={yamlText}
                height="280px"
                editable={false}
                extensions={[yamlLang()]}
                basicSetup={{
                  lineNumbers: true,
                  highlightActiveLine: false,
                  foldGutter: true,
                }}
              />
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-brand-border bg-white px-3 py-2.5">
      <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-slate-500">{label}</p>
      <p className="mt-1 text-sm font-semibold text-slate-900">{value}</p>
    </div>
  );
}

import { Plus, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tooltip } from "@/components/ui/tooltip";
import { getChannel, updateChannelAt } from "@/lib/config-utils";
import { useConfig } from "@/state/config-store";
import type { ChannelDef, ChannelEnabled, StickyPauseRange } from "@/types/api";

interface ChannelAvailabilityCardProps {
  index: number;
}

/**
 * Read the current channel's `enabled` value as a normalized object form,
 * so the UI can edit `default` and `off_ranges` without having to special-case
 * the "single boolean" representation everywhere.
 *
 * Backend tolerates `enabled: true | false | { default, off_ranges }`. We
 * collapse back to the boolean form on save when there are no off_ranges
 * (keeps YAML output clean).
 */
function readEnabled(enabled: ChannelEnabled | undefined): {
  active: boolean;
  off_ranges: Array<{ start_week?: number; end_week?: number }>;
} {
  if (enabled === undefined) return { active: true, off_ranges: [] };
  if (typeof enabled === "boolean") return { active: enabled, off_ranges: [] };
  return {
    active: enabled.default ?? true,
    off_ranges: Array.isArray(enabled.off_ranges) ? [...enabled.off_ranges] : [],
  };
}

/** Convert the normalized form back into the minimal YAML representation. */
function writeEnabled(active: boolean, off_ranges: Array<{ start_week?: number; end_week?: number }>): ChannelEnabled | undefined {
  if (active && off_ranges.length === 0) return undefined;
  if (!active && off_ranges.length === 0) return false;
  return { default: active, off_ranges };
}

function NumInput({
  value,
  onChange,
  min = 1,
  step = 1,
  placeholder,
}: {
  value: number | undefined;
  onChange: (v: number | undefined) => void;
  min?: number;
  step?: number;
  placeholder?: string;
}) {
  return (
    <Input
      type="number"
      min={min}
      step={step}
      placeholder={placeholder}
      value={value ?? ""}
      onChange={(e) => {
        const raw = e.target.value;
        if (raw === "") return onChange(undefined);
        const n = parseInt(raw, 10);
        onChange(Number.isFinite(n) ? n : undefined);
      }}
      className="w-20"
    />
  );
}

function FloatInput({
  value,
  onChange,
  step = 0.05,
  min = 0,
  max = 1,
}: {
  value: number | undefined;
  onChange: (v: number | undefined) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <Input
      type="number"
      step={step}
      min={min}
      max={max}
      value={value ?? ""}
      onChange={(e) => {
        const raw = e.target.value;
        if (raw === "") return onChange(undefined);
        const n = parseFloat(raw);
        onChange(Number.isFinite(n) ? n : undefined);
      }}
      className="w-24"
    />
  );
}

export function ChannelAvailabilityCard({ index }: ChannelAvailabilityCardProps) {
  const { config, setConfig } = useConfig();
  const list = config.channel_list ?? [];
  const entry = list[index];
  if (!entry) return null;
  const channel = getChannel(entry);

  const { active, off_ranges } = readEnabled(channel.enabled);
  const stickyRanges: StickyPauseRange[] = channel.sticky_pause_ranges ?? [];

  const patch = (partial: Partial<ChannelDef>) => {
    setConfig(updateChannelAt(config, index, partial) as typeof config, {
      resetYamlDirty: true,
    });
  };

  const setEnabled = (
    nextActive: boolean,
    nextRanges: Array<{ start_week?: number; end_week?: number }>,
  ) => {
    patch({ enabled: writeEnabled(nextActive, nextRanges) });
  };

  const addOffRange = () => {
    const wr =
      typeof config.week_range === "number" && config.week_range > 0 ? config.week_range : 52;
    const last = off_ranges[off_ranges.length - 1];
    const startGuess = last?.end_week ? Math.min(last.end_week + 1, wr) : 1;
    setEnabled(active, [
      ...off_ranges,
      { start_week: startGuess, end_week: Math.min(startGuess + 1, wr) },
    ]);
  };

  const updateOffRange = (
    i: number,
    field: "start_week" | "end_week",
    v: number | undefined,
  ) => {
    const next = off_ranges.map((r, j) => (i === j ? { ...r, [field]: v } : r));
    setEnabled(active, next);
  };

  const removeOffRange = (i: number) => {
    setEnabled(
      active,
      off_ranges.filter((_, j) => j !== i),
    );
  };

  const setStickyRanges = (next: StickyPauseRange[]) => {
    patch({ sticky_pause_ranges: next.length === 0 ? undefined : next });
  };

  const addStickyRange = () => {
    const wr =
      typeof config.week_range === "number" && config.week_range > 0 ? config.week_range : 52;
    setStickyRanges([
      ...stickyRanges,
      { start_week: 1, end_week: wr, start_probability: 0.1, continue_probability: 0.8 },
    ]);
  };

  const updateStickyRange = (i: number, patchRow: Partial<StickyPauseRange>) => {
    setStickyRanges(stickyRanges.map((r, j) => (i === j ? { ...r, ...patchRow } : r)));
  };

  const removeStickyRange = (i: number) => {
    setStickyRanges(stickyRanges.filter((_, j) => j !== i));
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle>Availability</CardTitle>
        <CardDescription>
          Pause this channel for parts of the run. Pauses zero out spend, impressions, and revenue
          contribution for the affected weeks.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="flex items-center justify-between rounded-lg border border-brand-border bg-slate-50/50 px-3 py-2">
          <div>
            <Label className="text-sm font-medium">Channel active</Label>
            <p className="text-[11px] text-slate-500">
              When off, the channel is fully disabled for the entire run.
            </p>
          </div>
          <Switch
            checked={active}
            onCheckedChange={(checked) => setEnabled(checked, off_ranges)}
          />
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-sm font-medium">Off-week ranges</Label>
              <p className="text-[11px] text-slate-500">
                Inclusive ranges where this channel is paused. Week numbers are 1-based and clamped
                to the run length.
              </p>
            </div>
            <Tooltip content={active ? "Add an off-week range" : "Channel is fully off — off-ranges have no extra effect"}>
              <Button
                size="sm"
                variant="secondary"
                onClick={addOffRange}
                disabled={!active}
              >
                <Plus className="h-3.5 w-3.5" />
                Add range
              </Button>
            </Tooltip>
          </div>

          {off_ranges.length === 0 ? (
            <p className="rounded-lg border border-dashed border-slate-200 bg-white px-3 py-3 text-center text-xs text-slate-500">
              No off-week ranges. Channel runs every week.
            </p>
          ) : (
            <ul className="space-y-1.5">
              {off_ranges.map((r, i) => (
                <li
                  key={i}
                  className="flex items-center gap-2 rounded-lg border border-brand-border bg-white px-3 py-2"
                >
                  <span className="text-[11px] uppercase tracking-wide text-slate-500">Weeks</span>
                  <NumInput
                    value={r.start_week}
                    onChange={(v) => updateOffRange(i, "start_week", v)}
                    placeholder="start"
                  />
                  <span className="text-xs text-slate-400">–</span>
                  <NumInput
                    value={r.end_week}
                    onChange={(v) => updateOffRange(i, "end_week", v)}
                    placeholder="end"
                  />
                  <div className="flex-1" />
                  <Tooltip content="Remove this range">
                    <button
                      type="button"
                      onClick={() => removeOffRange(i)}
                      className="rounded-full p-1.5 text-slate-400 transition hover:bg-rose-50 hover:text-rose-600"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </Tooltip>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div>
              <Label className="text-sm font-medium">Sticky random pauses</Label>
              <p className="text-[11px] text-slate-500">
                Markov-style stochastic pauses. Each week within the window may start a pause with
                probability <em>start</em>; if active, it continues with probability{" "}
                <em>continue</em>.
              </p>
            </div>
            <Button size="sm" variant="secondary" onClick={addStickyRange} disabled={!active}>
              <Plus className="h-3.5 w-3.5" />
              Add window
            </Button>
          </div>

          {stickyRanges.length === 0 ? (
            <p className="rounded-lg border border-dashed border-slate-200 bg-white px-3 py-3 text-center text-xs text-slate-500">
              No sticky pause windows.
            </p>
          ) : (
            <ul className="space-y-1.5">
              {stickyRanges.map((r, i) => (
                <li
                  key={i}
                  className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-lg border border-brand-border bg-white px-3 py-2"
                >
                  <span className="text-[11px] uppercase tracking-wide text-slate-500">Weeks</span>
                  <NumInput
                    value={r.start_week}
                    onChange={(v) => updateStickyRange(i, { start_week: v })}
                    placeholder="start"
                  />
                  <span className="text-xs text-slate-400">–</span>
                  <NumInput
                    value={r.end_week}
                    onChange={(v) => updateStickyRange(i, { end_week: v })}
                    placeholder="end"
                  />
                  <span className="ml-3 text-[11px] uppercase tracking-wide text-slate-500">
                    p(start)
                  </span>
                  <FloatInput
                    value={r.start_probability}
                    onChange={(v) => updateStickyRange(i, { start_probability: v })}
                    step={0.05}
                  />
                  <span className="ml-2 text-[11px] uppercase tracking-wide text-slate-500">
                    p(continue)
                  </span>
                  <FloatInput
                    value={r.continue_probability}
                    onChange={(v) => updateStickyRange(i, { continue_probability: v })}
                    step={0.05}
                  />
                  <div className="flex-1" />
                  <Tooltip content="Remove this window">
                    <button
                      type="button"
                      onClick={() => removeStickyRange(i)}
                      className="rounded-full p-1.5 text-slate-400 transition hover:bg-rose-50 hover:text-rose-600"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </Tooltip>
                </li>
              ))}
            </ul>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

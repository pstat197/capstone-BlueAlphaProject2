import { AlertCircle, Code2, Copy, Plus, Trash2 } from "lucide-react";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tooltip } from "@/components/ui/tooltip";
import {
  appendChannel,
  defaultChannel,
  duplicateChannelAt,
  getChannel,
  removeChannelAt,
} from "@/lib/config-utils";
import { cn } from "@/lib/cn";
import { channelErrorCount } from "@/lib/use-config-validation";
import { useConfig } from "@/state/config-store";
import type { ChannelEnabled, ConfigIssue } from "@/types/api";

/**
 * Pre-compute the visual status of a channel for the row badges so the
 * heavy logic (parsing `enabled`, checking flags) doesn't get inlined into
 * JSX. Returns the labels to render — empty array means "fully on".
 */
function channelStatusLabels(enabled: ChannelEnabled | undefined, adOn: boolean, satOn: boolean): string[] {
  const out: string[] = [];
  if (enabled === false) {
    out.push("off");
  } else if (enabled && typeof enabled === "object") {
    const hasRanges = Array.isArray(enabled.off_ranges) && enabled.off_ranges.length > 0;
    if (enabled.default === false) out.push("off");
    else if (hasRanges) out.push("paused");
  }
  if (!adOn) out.push("no adstock");
  if (!satOn) out.push("no saturation");
  return out;
}

export type SimulatorPane =
  | { kind: "channel"; index: number }
  | { kind: "yaml" };

interface ChannelListProps {
  selected: SimulatorPane;
  onSelect: (pane: SimulatorPane) => void;
  /** Validation issues from /api/config/validate, used to badge bad rows. */
  issues?: ConfigIssue[];
}

export function ChannelList({ selected, onSelect, issues }: ChannelListProps) {
  const { config, setConfig, setYamlDirty } = useConfig();
  const [newName, setNewName] = useState("");

  const channels = config.channel_list ?? [];

  const handleAdd = () => {
    const name = newName.trim() || `Channel ${channels.length + 1}`;
    const next = appendChannel(config, defaultChannel(name));
    setConfig(next, { resetYamlDirty: true });
    setYamlDirty(false);
    setNewName("");
    onSelect({ kind: "channel", index: (next.channel_list?.length ?? 1) - 1 });
  };

  const handleRemove = (index: number) => {
    const next = removeChannelAt(config, index);
    setConfig(next, { resetYamlDirty: true });
    setYamlDirty(false);
    if (selected.kind === "channel") {
      const last = (next.channel_list?.length ?? 1) - 1;
      const newIndex = Math.min(selected.index, Math.max(last, 0));
      if ((next.channel_list?.length ?? 0) === 0) {
        onSelect({ kind: "yaml" });
      } else {
        onSelect({ kind: "channel", index: newIndex });
      }
    }
  };

  const handleDuplicate = (index: number) => {
    const { config: nextConfig, newIndex } = duplicateChannelAt(config, index);
    setConfig(nextConfig, { resetYamlDirty: true });
    setYamlDirty(false);
    onSelect({ kind: "channel", index: newIndex });
  };

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="space-y-2 rounded-xl border border-brand-border bg-white p-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
          Add channel
        </div>
        <div className="flex gap-2">
          <Input
            placeholder="e.g. TikTok"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleAdd();
            }}
          />
          <Button size="sm" onClick={handleAdd}>
            <Plus className="h-3.5 w-3.5" />
            Add
          </Button>
        </div>
      </div>

      <div className="flex-1 space-y-2 overflow-y-auto rounded-xl border border-brand-border bg-white p-2">
        {channels.length === 0 && (
          <p className="px-3 py-6 text-center text-sm text-slate-500">
            No channels yet. Add one above to start.
          </p>
        )}
        <ul className="space-y-1">
          {channels.map((entry, index) => {
            const ch = getChannel(entry);
            const isActive = selected.kind === "channel" && selected.index === index;
            const statusLabels = channelStatusLabels(
              ch.enabled,
              ch.adstock_enabled !== false,
              ch.saturation_enabled !== false,
            );
            const errCount = channelErrorCount(issues, index);
            return (
              <li key={index}>
                <div
                  className={cn(
                    "group flex w-full items-center gap-2 rounded-lg px-3 py-2 transition-colors",
                    isActive
                      ? "bg-brand-50 text-brand-700 ring-1 ring-brand-200"
                      : errCount > 0
                        ? "text-slate-700 hover:bg-slate-100/70 ring-1 ring-rose-200/70"
                        : "text-slate-700 hover:bg-slate-100/70",
                  )}
                >
                  <button
                    type="button"
                    onClick={() => onSelect({ kind: "channel", index })}
                    className="min-w-0 flex-1 text-left"
                  >
                    <div className="flex flex-wrap items-center gap-1.5">
                      {errCount > 0 && (
                        <Tooltip
                          content={`${errCount} configuration error${errCount === 1 ? "" : "s"}`}
                        >
                          <span className="inline-flex h-4 w-4 items-center justify-center rounded-full bg-rose-50 text-rose-600">
                            <AlertCircle className="h-3 w-3" />
                          </span>
                        </Tooltip>
                      )}
                      <span className="truncate text-sm font-medium">
                        {ch.channel_name || `Channel ${index + 1}`}
                      </span>
                      {ch.saturation_config?.type && (
                        <Badge variant="muted" className="text-[10px]">
                          {ch.saturation_config.type}
                        </Badge>
                      )}
                      {statusLabels.map((label) => (
                        <Badge
                          key={label}
                          variant="warn"
                          className="text-[10px] uppercase tracking-wide"
                        >
                          {label}
                        </Badge>
                      ))}
                    </div>
                    <p className="truncate text-[11px] text-slate-500">
                      ROI {ch.true_roi ?? "—"} · CPM {ch.cpm ?? "—"}
                    </p>
                  </button>
                  <Tooltip content="Duplicate channel">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDuplicate(index);
                      }}
                      className="rounded-full p-1.5 text-slate-400 transition hover:bg-brand-50 hover:text-brand-700"
                    >
                      <Copy className="h-3.5 w-3.5" />
                    </button>
                  </Tooltip>
                  <Tooltip content="Remove channel">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRemove(index);
                      }}
                      className="rounded-full p-1.5 text-slate-400 transition hover:bg-rose-50 hover:text-rose-600"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </Tooltip>
                </div>
              </li>
            );
          })}
        </ul>
      </div>

      <button
        type="button"
        onClick={() => onSelect({ kind: "yaml" })}
        className={cn(
          "flex items-center gap-2 rounded-xl border px-4 py-3 text-left text-sm transition-colors",
          selected.kind === "yaml"
            ? "border-brand-300 bg-brand-50 text-brand-700"
            : "border-brand-border bg-white text-slate-700 hover:border-brand-200 hover:bg-brand-50/50",
        )}
      >
        <Code2 className="h-4 w-4" />
        <span className="flex-1 font-medium">Advanced · edit YAML</span>
      </button>
    </div>
  );
}

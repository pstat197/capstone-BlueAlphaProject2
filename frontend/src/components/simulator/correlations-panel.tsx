import { Plus, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tooltip } from "@/components/ui/tooltip";
import {
  appendCorrelation,
  getChannel,
  removeCorrelationAt,
  updateCorrelationAt,
} from "@/lib/config-utils";
import { useConfig } from "@/state/config-store";
import type { CorrelationsAutoMode, SimConfig } from "@/types/api";

const AUTO_MODES: Array<{ value: CorrelationsAutoMode; label: string; hint: string }> = [
  { value: "none", label: "None", hint: "Use only the manual pairs below." },
  {
    value: "random",
    label: "Random copula pairs",
    hint: "Loader seeds extra copula pairs deterministically from the run seed.",
  },
];

function channelNames(config: SimConfig): string[] {
  const out: string[] = [];
  for (const entry of config.channel_list ?? []) {
    const name = getChannel(entry).channel_name?.trim();
    if (name) out.push(name);
  }
  return out;
}

function RhoInput({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <Input
      type="number"
      min={-1}
      max={1}
      step={0.05}
      value={value}
      onChange={(e) => {
        const raw = e.target.value;
        if (raw === "") return;
        const n = parseFloat(raw);
        if (Number.isFinite(n)) onChange(Math.max(-1, Math.min(1, n)));
      }}
      className="w-24"
    />
  );
}

export function CorrelationsPanel() {
  const { config, setConfig, patchConfig } = useConfig();
  const pairs = config.correlations ?? [];
  const channelOptions = channelNames(config);
  const autoMode: CorrelationsAutoMode =
    (config.correlations_auto_mode as CorrelationsAutoMode | undefined) ?? "none";

  const handleAdd = () => {
    const first = channelOptions[0] ?? "";
    const second = channelOptions[1] ?? "";
    setConfig(appendCorrelation(config, { channels: [first, second], rho: 0.3 }), {
      resetYamlDirty: true,
    });
  };

  const handleUpdatePair = (i: number, j: 0 | 1, name: string) => {
    const current = pairs[i]?.channels;
    if (!current) return;
    const next: [string, string] = [current[0] ?? "", current[1] ?? ""];
    next[j] = name;
    setConfig(updateCorrelationAt(config, i, { channels: next }), { resetYamlDirty: true });
  };

  const handleUpdateRho = (i: number, rho: number) => {
    setConfig(updateCorrelationAt(config, i, { rho }), { resetYamlDirty: true });
  };

  const handleRemove = (i: number) => {
    setConfig(removeCorrelationAt(config, i), { resetYamlDirty: true });
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
        <div className="space-y-1.5">
          <Label htmlFor="corr_auto">Auto-mode</Label>
          <Select
            value={autoMode}
            onValueChange={(v) =>
              patchConfig({ correlations_auto_mode: v as CorrelationsAutoMode })
            }
          >
            <SelectTrigger id="corr_auto">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {AUTO_MODES.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-[11px] text-slate-500">
            {AUTO_MODES.find((m) => m.value === autoMode)?.hint}
          </p>
        </div>
        <Tooltip
          content={
            channelOptions.length < 2
              ? "Need at least two named channels to add a correlation pair"
              : "Add a manual correlation pair"
          }
        >
          <Button
            size="sm"
            variant="secondary"
            onClick={handleAdd}
            disabled={channelOptions.length < 2}
          >
            <Plus className="h-3.5 w-3.5" />
            Add manual pair
          </Button>
        </Tooltip>
      </div>

      {pairs.length === 0 ? (
        <p className="rounded-lg border border-dashed border-slate-200 bg-white px-3 py-6 text-center text-xs text-slate-500">
          No manual correlation pairs. The loader will emit any pairs produced by auto-mode at
          run time.
        </p>
      ) : (
        <ul className="space-y-2">
          {pairs.map((pair, i) => (
            <li
              key={i}
              className="flex flex-wrap items-center gap-2 rounded-lg border border-brand-border bg-white px-3 py-2"
            >
              <Select
                value={pair.channels[0] ?? ""}
                onValueChange={(v) => handleUpdatePair(i, 0, v)}
              >
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="channel A" />
                </SelectTrigger>
                <SelectContent>
                  {channelOptions.map((name) => (
                    <SelectItem key={name} value={name}>
                      {name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <span className="text-xs text-slate-400">↔</span>
              <Select
                value={pair.channels[1] ?? ""}
                onValueChange={(v) => handleUpdatePair(i, 1, v)}
              >
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="channel B" />
                </SelectTrigger>
                <SelectContent>
                  {channelOptions.map((name) => (
                    <SelectItem key={name} value={name}>
                      {name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <span className="ml-3 text-[11px] uppercase tracking-wide text-slate-500">ρ</span>
              <RhoInput value={pair.rho} onChange={(v) => handleUpdateRho(i, v)} />
              <div className="flex-1" />
              <Tooltip content="Remove this pair">
                <button
                  type="button"
                  onClick={() => handleRemove(i)}
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
  );
}

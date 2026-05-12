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
  appendBudgetShift,
  getChannel,
  removeBudgetShiftAt,
  updateBudgetShiftAt,
} from "@/lib/config-utils";
import { useConfig } from "@/state/config-store";
import type { BudgetShift, BudgetShiftsAutoMode, SimConfig } from "@/types/api";

const AUTO_MODES: Array<{ value: BudgetShiftsAutoMode; label: string; hint: string }> = [
  { value: "none", label: "None", hint: "Use only the manual rules below." },
  {
    value: "global",
    label: "Global only",
    hint: "Loader seeds whole-market scale shifts deterministically from the run seed.",
  },
  {
    value: "global_and_channel",
    label: "Global + per-channel",
    hint: "Loader seeds both global scales and per-channel reallocations from the run seed.",
  },
];

const SHIFT_TYPES: Array<{ value: BudgetShift["type"]; label: string; hint: string }> = [
  { value: "scale", label: "Scale all channels", hint: "Multiply every channel's weekly spend in the window." },
  { value: "scale_channel", label: "Scale one channel", hint: "Multiply a single channel's spend in the window." },
  {
    value: "reallocate",
    label: "Reallocate budget",
    hint: "Move a fraction of one channel's weekly spend to another for every week in the window.",
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

function NumCell({
  value,
  onChange,
  step = 1,
  min = 0,
  asInt = false,
  placeholder,
}: {
  value: number | undefined;
  onChange: (v: number | undefined) => void;
  step?: number;
  min?: number;
  asInt?: boolean;
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
        const n = asInt ? parseInt(raw, 10) : parseFloat(raw);
        onChange(Number.isFinite(n) ? n : undefined);
      }}
      className="w-24"
    />
  );
}

interface RowProps {
  rule: BudgetShift;
  channelOptions: string[];
  onChange: (patch: Partial<BudgetShift>) => void;
  onRemove: () => void;
}

function ShiftRow({ rule, channelOptions, onChange, onRemove }: RowProps) {
  const isReallocate = rule.type === "reallocate";
  const isScaleChannel = rule.type === "scale_channel";

  return (
    <li className="rounded-lg border border-brand-border bg-white px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <Select value={rule.type} onValueChange={(v) => onChange({ type: v as BudgetShift["type"] })}>
          <SelectTrigger className="w-44">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {SHIFT_TYPES.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <span className="text-[11px] uppercase tracking-wide text-slate-500">Weeks</span>
        <NumCell
          value={rule.start_week}
          onChange={(v) => onChange({ start_week: v } as Partial<BudgetShift>)}
          asInt
          min={1}
          placeholder="start"
        />
        <span className="text-xs text-slate-400">–</span>
        <NumCell
          value={rule.end_week}
          onChange={(v) => onChange({ end_week: v } as Partial<BudgetShift>)}
          asInt
          min={1}
          placeholder="end"
        />

        {!isReallocate && (
          <>
            {isScaleChannel && (
              <Select
                value={(rule as { channel_name?: string }).channel_name ?? ""}
                onValueChange={(v) => onChange({ channel_name: v } as Partial<BudgetShift>)}
              >
                <SelectTrigger className="w-36">
                  <SelectValue placeholder="channel" />
                </SelectTrigger>
                <SelectContent>
                  {channelOptions.map((name) => (
                    <SelectItem key={name} value={name}>
                      {name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            <span className="text-[11px] uppercase tracking-wide text-slate-500">Factor</span>
            <NumCell
              value={(rule as { factor?: number }).factor}
              onChange={(v) => onChange({ factor: v } as Partial<BudgetShift>)}
              step={0.05}
            />
          </>
        )}

        {isReallocate && (
          <>
            <span className="text-[11px] uppercase tracking-wide text-slate-500">From</span>
            <Select
              value={(rule as { from_channel?: string }).from_channel ?? ""}
              onValueChange={(v) => onChange({ from_channel: v } as Partial<BudgetShift>)}
            >
              <SelectTrigger className="w-32">
                <SelectValue placeholder="source" />
              </SelectTrigger>
              <SelectContent>
                {channelOptions.map((name) => (
                  <SelectItem key={name} value={name}>
                    {name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <span className="text-[11px] uppercase tracking-wide text-slate-500">To</span>
            <Select
              value={(rule as { to_channel?: string }).to_channel ?? ""}
              onValueChange={(v) => onChange({ to_channel: v } as Partial<BudgetShift>)}
            >
              <SelectTrigger className="w-32">
                <SelectValue placeholder="target" />
              </SelectTrigger>
              <SelectContent>
                {channelOptions.map((name) => (
                  <SelectItem key={name} value={name}>
                    {name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <span className="text-[11px] uppercase tracking-wide text-slate-500">Fraction</span>
            <NumCell
              value={(rule as { fraction?: number }).fraction}
              onChange={(v) => onChange({ fraction: v } as Partial<BudgetShift>)}
              step={0.05}
              min={0}
            />
          </>
        )}

        <div className="flex-1" />
        <Tooltip content="Remove this rule">
          <button
            type="button"
            onClick={onRemove}
            className="rounded-full p-1.5 text-slate-400 transition hover:bg-rose-50 hover:text-rose-600"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </Tooltip>
      </div>
    </li>
  );
}

export function BudgetShiftsPanel() {
  const { config, setConfig, patchConfig } = useConfig();
  const rules = config.budget_shifts ?? [];
  const channelOptions = channelNames(config);
  const autoMode: BudgetShiftsAutoMode =
    (config.budget_shifts_auto_mode as BudgetShiftsAutoMode | undefined) ?? "none";

  const handleAdd = () => {
    setConfig(appendBudgetShift(config), { resetYamlDirty: true });
  };

  const handleUpdate = (i: number, patch: Partial<BudgetShift>) => {
    setConfig(updateBudgetShiftAt(config, i, patch), { resetYamlDirty: true });
  };

  const handleRemove = (i: number) => {
    setConfig(removeBudgetShiftAt(config, i), { resetYamlDirty: true });
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
        <div className="space-y-1.5">
          <Label htmlFor="bs_auto">Auto-mode</Label>
          <Select
            value={autoMode}
            onValueChange={(v) =>
              patchConfig({ budget_shifts_auto_mode: v as BudgetShiftsAutoMode })
            }
          >
            <SelectTrigger id="bs_auto">
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
        <Button size="sm" variant="secondary" onClick={handleAdd}>
          <Plus className="h-3.5 w-3.5" />
          Add manual rule
        </Button>
      </div>

      {rules.length === 0 ? (
        <p className="rounded-lg border border-dashed border-slate-200 bg-white px-3 py-6 text-center text-xs text-slate-500">
          No manual budget-shift rules. The loader will emit any rows produced by auto-mode at
          run time.
        </p>
      ) : (
        <ul className="space-y-2">
          {rules.map((rule, i) => (
            <ShiftRow
              key={i}
              rule={rule}
              channelOptions={channelOptions}
              onChange={(patch) => handleUpdate(i, patch)}
              onRemove={() => handleRemove(i)}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

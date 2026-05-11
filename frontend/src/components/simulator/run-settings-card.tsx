import { Calendar, FileText, Sparkles } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tooltip } from "@/components/ui/tooltip";
import { useConfig } from "@/state/config-store";

function NumField({
  id,
  label,
  value,
  onChange,
  min,
  step = 1,
  helper,
  icon,
  asInt = false,
}: {
  id: string;
  label: string;
  value: number | undefined;
  onChange: (next: number | undefined) => void;
  min?: number;
  step?: number;
  helper?: string;
  icon?: React.ReactNode;
  asInt?: boolean;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2">
        <Label htmlFor={id}>{label}</Label>
        {helper && (
          <Tooltip content={helper}>
            <span className="text-[10px] cursor-help text-slate-400" aria-hidden>
              ⓘ
            </span>
          </Tooltip>
        )}
      </div>
      <div className="relative">
        {icon && (
          <span className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-slate-400">
            {icon}
          </span>
        )}
        <Input
          id={id}
          type="number"
          value={value ?? ""}
          min={min}
          step={step}
          onChange={(e) => {
            const raw = e.target.value;
            if (raw === "") return onChange(undefined);
            const num = asInt ? parseInt(raw, 10) : parseFloat(raw);
            onChange(Number.isFinite(num) ? num : undefined);
          }}
          className={icon ? "pl-9" : undefined}
        />
      </div>
    </div>
  );
}

export function RunSettingsCard() {
  const { config, patchConfig } = useConfig();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Simulation settings</CardTitle>
        <CardDescription>
          How long to run, what to call it, and how to fix randomness so runs reproduce.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <NumField
          id="week_range"
          label="Week range"
          value={
            typeof config.week_range === "number" ? config.week_range : undefined
          }
          onChange={(v) => patchConfig({ week_range: v })}
          min={1}
          step={1}
          asInt
          helper="Simulation length in weeks. Larger = more series points."
          icon={<Calendar className="h-3.5 w-3.5" />}
        />
        <div className="space-y-1.5">
          <Label htmlFor="run_identifier">Run name</Label>
          <div className="relative">
            <span className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-slate-400">
              <FileText className="h-3.5 w-3.5" />
            </span>
            <Input
              id="run_identifier"
              value={(config.run_identifier as string | undefined) ?? ""}
              onChange={(e) => patchConfig({ run_identifier: e.target.value })}
              placeholder="e.g. Example Alpha"
              className="pl-9"
            />
          </div>
        </div>
        <NumField
          id="seed"
          label="Random seed"
          value={typeof config.seed === "number" ? config.seed : undefined}
          onChange={(v) => patchConfig({ seed: v })}
          min={0}
          step={1}
          asInt
          helper="Fixes randomness so the same settings reproduce the same series and cache key."
          icon={<Sparkles className="h-3.5 w-3.5" />}
        />
      </CardContent>
    </Card>
  );
}

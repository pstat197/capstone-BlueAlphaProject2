import { Sliders } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface Props {
  channels: string[];
  mus: number[];
  sigmas: number[];
  onChange: (next: { mus: number[]; sigmas: number[] }) => void;
  disabled?: boolean;
}

const DEFAULT_MU = 0.2;
const DEFAULT_SIGMA = 0.9;

export function MmmPriorsCard({ channels, mus, sigmas, onChange, disabled }: Props) {
  const updateAt = (i: number, key: "mu" | "sig", value: number) => {
    const nextMus = [...mus];
    const nextSigs = [...sigmas];
    while (nextMus.length < channels.length) nextMus.push(DEFAULT_MU);
    while (nextSigs.length < channels.length) nextSigs.push(DEFAULT_SIGMA);
    if (key === "mu") nextMus[i] = value;
    else nextSigs[i] = value;
    onChange({ mus: nextMus.slice(0, channels.length), sigmas: nextSigs.slice(0, channels.length) });
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Sliders className="h-4 w-4 text-brand-500" />
          <CardTitle>ROI prior (lognormal on roi_m)</CardTitle>
        </div>
        <CardDescription>
          Lognormal <span className="font-mono">μ, σ</span> parameterize the prior on each channel’s
          media ROI. Per-channel values let you encode beliefs (e.g. brand vs performance). Defaults
          (μ = 0.2, σ = 0.9) are broad and roughly uninformative.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {channels.length === 0 ? (
          <p className="text-sm text-slate-500">Load a simulator run to configure priors.</p>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2">
            {channels.map((ch, i) => (
              <div
                key={ch}
                className="rounded-lg border border-brand-border bg-white px-4 py-3 shadow-[inset_0_1px_2px_rgba(15,23,42,0.04)]"
              >
                <p className="truncate text-sm font-semibold text-slate-900">{ch}</p>
                <div className="mt-2 grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <Label
                      htmlFor={`mu-${i}`}
                      className="text-[11px] font-medium uppercase tracking-wide text-slate-500"
                    >
                      μ (mu)
                    </Label>
                    <Input
                      id={`mu-${i}`}
                      type="number"
                      step={0.1}
                      value={mus[i] ?? DEFAULT_MU}
                      onChange={(e) => updateAt(i, "mu", parseFloat(e.target.value || "0"))}
                      disabled={disabled}
                    />
                  </div>
                  <div className="space-y-1">
                    <Label
                      htmlFor={`sig-${i}`}
                      className="text-[11px] font-medium uppercase tracking-wide text-slate-500"
                    >
                      σ (sigma)
                    </Label>
                    <Input
                      id={`sig-${i}`}
                      type="number"
                      step={0.1}
                      min={0.01}
                      value={sigmas[i] ?? DEFAULT_SIGMA}
                      onChange={(e) => updateAt(i, "sig", parseFloat(e.target.value || "0"))}
                      disabled={disabled}
                    />
                  </div>
                </div>
                <p className="mt-2 text-[11px] text-slate-500">
                  Implied prior median ROI ≈ <span className="font-mono">exp(μ)</span> ={" "}
                  <span className="font-mono">{Math.exp(mus[i] ?? DEFAULT_MU).toFixed(2)}</span>.
                </p>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function makeDefaultPriors(channels: string[]): { mus: number[]; sigmas: number[] } {
  return {
    mus: channels.map(() => DEFAULT_MU),
    sigmas: channels.map(() => DEFAULT_SIGMA),
  };
}

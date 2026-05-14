import { useMemo, useState } from "react";

import { cn } from "@/lib/cn";
import type { MmmRoiForest, MmmRoiForestRow } from "@/types/api";

const CI_BLUE = "#378ADD";
const CI_BAND = "rgba(55, 138, 221, 0.18)";
const TRUE_RED = "#E24B4A";

interface Props {
  data: MmmRoiForest;
}

interface PlotRow {
  channel: string;
  mean: number;
  ci_low: number;
  ci_high: number;
  true_roi: number | null;
  rhat: number | null;
}

function rhatTone(rh: number | null): { color: string; label: string } | null {
  if (rh == null || !Number.isFinite(rh)) return null;
  let color = "#15803d";
  if (rh > 1.1) color = "#dc2626";
  else if (rh > 1.05) color = "#b45309";
  return { color, label: `R̂ ${rh.toFixed(3)}${rh > 1.1 ? " ⚠" : ""}` };
}

export function MmmRoiForest({ data }: Props) {
  const [hover, setHover] = useState<{ row: PlotRow; x: number; y: number } | null>(null);

  const rows: PlotRow[] = useMemo(() => {
    return (data.rows || []).map((r: MmmRoiForestRow) => ({
      channel: String(r.channel),
      mean: Number(r.mean),
      ci_low: Number(r.ci_low),
      ci_high: Number(r.ci_high),
      true_roi: r.true_roi == null || !Number.isFinite(Number(r.true_roi)) ? null : Number(r.true_roi),
      rhat:
        data.rhat_by_channel[r.channel] == null
          ? null
          : Number(data.rhat_by_channel[r.channel]),
    }));
  }, [data]);

  const { xMin, xMax } = useMemo(() => {
    if (rows.length === 0) return { xMin: 0, xMax: 1 };
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (const r of rows) {
      lo = Math.min(lo, r.ci_low);
      hi = Math.max(hi, r.ci_high);
      if (r.true_roi != null) {
        lo = Math.min(lo, r.true_roi);
        hi = Math.max(hi, r.true_roi);
      }
    }
    if (!Number.isFinite(lo)) lo = 0;
    if (!Number.isFinite(hi)) hi = 1;
    if (hi - lo < 1e-6) {
      hi += 0.5;
      lo -= 0.5;
    }
    const pad = (hi - lo) * 0.08;
    return { xMin: lo - pad, xMax: hi + pad };
  }, [rows]);

  if (data.error) {
    return (
      <p className="rounded-md border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700">
        ROI forest plot: {data.error}
      </p>
    );
  }
  if (rows.length === 0) {
    return <p className="text-sm text-slate-500">No channels to plot.</p>;
  }

  /* SVG layout: fixed height per channel band; left margin for channel names,
   * right margin for R̂ chip when present. */
  const chartLeft = 160;
  const chartRight = 110;
  const chartTop = 12;
  const rowHeight = 56;
  const totalHeight = chartTop + rows.length * rowHeight + 56;
  const totalWidth = 880;
  const innerWidth = totalWidth - chartLeft - chartRight;
  const xScale = (v: number) => chartLeft + ((v - xMin) / (xMax - xMin)) * innerWidth;
  const yFor = (i: number) => chartTop + i * rowHeight + rowHeight / 2;

  const ticks = niceTicks(xMin, xMax, 6);
  const hasTrue = rows.some((r) => r.true_roi != null);

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${totalWidth} ${totalHeight}`}
        className="w-full text-slate-700"
        role="img"
        aria-label="Recovered media ROI per channel — posterior mean with 50% credible interval"
      >
        <line
          x1={chartLeft}
          x2={chartLeft + innerWidth}
          y1={totalHeight - 36}
          y2={totalHeight - 36}
          stroke="#cbd5e1"
        />
        {ticks.map((t) => (
          <g key={t} transform={`translate(${xScale(t)}, 0)`}>
            <line
              y1={chartTop}
              y2={totalHeight - 36}
              stroke="#e2e8f0"
              strokeDasharray="3 3"
            />
            <text
              y={totalHeight - 20}
              textAnchor="middle"
              className="fill-slate-500"
              fontSize="11"
            >
              {formatTick(t)}
            </text>
          </g>
        ))}
        <text
          x={chartLeft + innerWidth / 2}
          y={totalHeight - 4}
          textAnchor="middle"
          fontSize="12"
          className="fill-slate-600"
        >
          ROI ($ returned per $1 spent)
        </text>

        {rows.map((r, i) => {
          const y = yFor(i);
          const xMean = xScale(r.mean);
          const xLo = xScale(r.ci_low);
          const xHi = xScale(r.ci_high);
          const xTrue = r.true_roi != null ? xScale(r.true_roi) : null;
          const rh = rhatTone(r.rhat);
          return (
            <g
              key={r.channel}
              onMouseEnter={(e) =>
                setHover({ row: r, x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY })
              }
              onMouseMove={(e) =>
                setHover({ row: r, x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY })
              }
              onMouseLeave={() => setHover(null)}
              className="cursor-default"
            >
              <text
                x={chartLeft - 12}
                y={y + 4}
                textAnchor="end"
                fontSize="13"
                fontWeight="500"
                className="fill-slate-900"
              >
                {r.channel}
              </text>
              <rect
                x={xLo}
                y={y - 7}
                width={Math.max(1, xHi - xLo)}
                height={14}
                fill={CI_BAND}
              />
              <line
                x1={xLo}
                x2={xHi}
                y1={y}
                y2={y}
                stroke={CI_BLUE}
                strokeWidth={2.5}
                strokeLinecap="round"
              />
              <line x1={xLo} x2={xLo} y1={y - 7} y2={y + 7} stroke={CI_BLUE} strokeWidth={2} />
              <line x1={xHi} x2={xHi} y1={y - 7} y2={y + 7} stroke={CI_BLUE} strokeWidth={2} />
              {xTrue != null && (
                <line
                  x1={xTrue}
                  x2={xTrue}
                  y1={y - 22}
                  y2={y + 22}
                  stroke={TRUE_RED}
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                />
              )}
              <circle
                cx={xMean}
                cy={y}
                r={6}
                fill={CI_BLUE}
                stroke="white"
                strokeWidth={1.5}
              />
              {rh && (
                <text
                  x={chartLeft + innerWidth + 8}
                  y={y + 4}
                  textAnchor="start"
                  fontSize="11"
                  fill={rh.color}
                  className="font-mono"
                >
                  {rh.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      <div className="mt-2 flex flex-wrap items-center gap-4 px-2 text-xs text-slate-600">
        <LegendDot color={CI_BLUE} label="Posterior mean" />
        <LegendBand color={CI_BLUE} label="50% credible interval" />
        {hasTrue && <LegendDash color={TRUE_RED} label="True synthetic ROI" />}
      </div>

      {hover && (
        <div
          className={cn(
            "pointer-events-none absolute z-20 max-w-xs rounded-lg border border-brand-border",
            "bg-white px-3 py-2 text-xs text-slate-700 shadow-[0_8px_28px_rgba(15,23,42,0.12)]",
          )}
          style={{
            left: Math.min(hover.x + 16, 600),
            top: hover.y + 12,
          }}
        >
          <p className="font-semibold text-slate-900">{hover.row.channel}</p>
          <p>
            Mean ROI: <span className="font-mono">{hover.row.mean.toFixed(3)}</span>
          </p>
          <p>
            50% CI:{" "}
            <span className="font-mono">
              [{hover.row.ci_low.toFixed(3)}, {hover.row.ci_high.toFixed(3)}]
            </span>
          </p>
          {hover.row.true_roi != null && (
            <p>
              True (YAML):{" "}
              <span className="font-mono">{hover.row.true_roi.toFixed(3)}</span>
            </p>
          )}
          {hover.row.rhat != null && (
            <p>
              R̂ (roi_m): <span className="font-mono">{hover.row.rhat.toFixed(4)}</span>
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-block h-2.5 w-2.5 rounded-full"
        style={{ backgroundColor: color }}
      />
      {label}
    </span>
  );
}

function LegendBand({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-block h-2.5 w-6 rounded-sm"
        style={{ backgroundColor: color, opacity: 0.35 }}
      />
      {label}
    </span>
  );
}

function LegendDash({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <svg width={28} height={6} className="overflow-visible">
        <line
          x1={0}
          x2={28}
          y1={3}
          y2={3}
          stroke={color}
          strokeWidth={1.5}
          strokeDasharray="4 3"
        />
      </svg>
      {label}
    </span>
  );
}

/** Friendly ROI tick labels: integer when range is wide, two decimals when narrow. */
function formatTick(v: number): string {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

/** Pleasant ~N tick locations, similar to d3-scale's nice ticks. */
function niceTicks(min: number, max: number, count: number): number[] {
  const span = max - min;
  if (span <= 0) return [min];
  const step = niceStep(span / Math.max(1, count));
  const start = Math.ceil(min / step) * step;
  const out: number[] = [];
  for (let v = start; v <= max + step / 2; v += step) {
    out.push(Number(v.toFixed(8)));
  }
  return out;
}

function niceStep(raw: number): number {
  const exp = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / exp;
  let nice: number;
  if (f < 1.5) nice = 1;
  else if (f < 3) nice = 2;
  else if (f < 7) nice = 5;
  else nice = 10;
  return nice * exp;
}

# Blue Alpha · Simulator (React UI)

Vite + React + TypeScript + Tailwind v4 SPA that talks to the FastAPI server in `../server` (which wraps the same `scripts/*` simulator the Streamlit app uses).

## Stack

- **Vite 8** + **React 19** + **TypeScript**
- **Tailwind CSS v4** (CSS-first theme via `src/index.css`, `@tailwindcss/vite` plugin)
- **shadcn-style primitives** built on Radix UI (`src/components/ui/`)
- **React Router v6** for `/simulator`, `/results/:runId`, `/results/:runId/diagnostics`, `/mmm`
- **TanStack Query** for server state (config, runs, history)
- **Recharts** for charts (line, area, heatmap-as-grid)
- **CodeMirror 6** + `@codemirror/lang-yaml` for the Advanced YAML editor
- **lucide-react** for icons, **clsx + tailwind-merge + cva** for class composition

## Run alongside the FastAPI server

From the repo root:

```bash
./scripts/run_dev.sh
```

This boots **uvicorn `server.main:app` on 8000** and **Vite on 5173**. Vite proxies `/api/*` to the backend, so the frontend can use relative URLs.

If you want them independently:

```bash
# backend
.venv/bin/uvicorn server.main:app --reload --port 8000

# frontend
npm install            # first time
npm run dev            # http://localhost:5173
```

## Brand theme

Tailwind v4 reads CSS variables from `@theme` in `src/index.css`. The brand scale (`brand-50`…`brand-900`) is anchored on Blue Alpha's primary blue `#1D63ED`; the accent (`accent-500 = #F39C59`) is reserved for the "spend" series and callouts so totals/revenue keep the brand color. Edit those variables to retheme without touching components.

## Project layout

```
src/
  components/
    ui/                # Button, Card, Tabs, Sheet, Popover, Select, Switch, …
    layout/            # AppShell, TopBar, BrandMark, SettingsPopover, RunsDrawer
    simulator/         # ChannelList, ChannelDetail, RunSettingsCard, YamlEditorCard
    results/           # RunSummary, ResultsCharts, PreviewTable, ConfigSnapshot
    diagnostics/       # CorrelationHeatmap, RollingCorrelation, …
  routes/              # Simulator, Results, Diagnostics, MmmComingSoon
  state/               # ConfigProvider, SettingsProvider (React Context)
  lib/                 # api, yaml, cn, config helpers
  types/               # API + config types matching server/main.py
```

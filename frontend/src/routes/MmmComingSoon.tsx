import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Sparkles, Terminal } from "lucide-react";
import { Link } from "react-router-dom";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { api } from "@/lib/api";

export default function MmmComingSoonRoute() {
  const statusQuery = useQuery({
    queryKey: ["meridian-status"],
    queryFn: () => api.meridianStatus(),
  });

  const installed = statusQuery.data?.installed ?? false;

  return (
    <div className="flex flex-1 flex-col gap-6">
      <header className="space-y-2">
        <p className="text-xs font-medium uppercase tracking-[0.18em] text-brand-600">
          Bayesian MMM
        </p>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
          Meridian-powered MMM
        </h1>
        <p className="text-sm text-slate-500">
          Full Meridian + ROI random-forest workflow is coming to the React UI in a follow-up. The
          Streamlit version remains fully functional in the meantime.
        </p>
      </header>

      <Card className="overflow-hidden">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-brand-500" />
            <CardTitle>Coming soon</CardTitle>
            <Badge variant={installed ? "success" : "warn"}>
              meridian {installed ? "installed" : "not installed"}
            </Badge>
          </div>
          <CardDescription>
            We rebuilt the simulator first to validate the new design. The MMM tab will follow with
            the same shell, theme, and run-history flow.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <ul className="grid gap-2 text-sm text-slate-700 sm:grid-cols-2">
            <li className="rounded-lg border border-brand-border bg-brand-50/40 px-3 py-2">
              Meridian inference, posteriors, and ROI plots
            </li>
            <li className="rounded-lg border border-brand-border bg-brand-50/40 px-3 py-2">
              ROI random forest with feature importance
            </li>
            <li className="rounded-lg border border-brand-border bg-brand-50/40 px-3 py-2">
              Hand-off from a simulator run into Meridian fit
            </li>
            <li className="rounded-lg border border-brand-border bg-brand-50/40 px-3 py-2">
              Cached fits stored alongside simulation runs
            </li>
          </ul>

          <Separator />

          <div className="flex flex-wrap items-center gap-3 rounded-lg bg-slate-50 px-4 py-3">
            <Terminal className="h-4 w-4 text-slate-500" />
            <span className="text-xs text-slate-600">
              Need MMM today? Start the existing Streamlit app:
            </span>
            <code className="rounded bg-white px-2 py-1 font-mono text-xs text-slate-700 shadow-sm">
              ./scripts/run_streamlit.sh
            </code>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button asChild variant="secondary" size="sm">
          <Link to="/simulator">
            Back to simulator
            <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </Button>
      </div>
    </div>
  );
}

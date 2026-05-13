import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "@/components/layout/app-shell";
import { Skeleton } from "@/components/ui/skeleton";

/* Each route is its own chunk so the initial bundle doesn't pull in
 * Recharts / CodeMirror / etc. unless the user actually navigates there. */
const SimulatorRoute = lazy(() => import("@/routes/Simulator"));
const ResultsRoute = lazy(() => import("@/routes/Results"));
const DiagnosticsRoute = lazy(() => import("@/routes/Diagnostics"));
const MmmComingSoonRoute = lazy(() => import("@/routes/MmmComingSoon"));

function RouteFallback() {
  return (
    <div className="flex flex-1 flex-col gap-4" aria-busy="true">
      <Skeleton className="h-9 w-64" />
      <Skeleton className="h-4 w-96" />
      <div className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
        <Skeleton className="h-[420px] w-full" />
        <Skeleton className="h-[420px] w-full" />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<Navigate to="/simulator" replace />} />
        <Route
          path="/simulator"
          element={
            <Suspense fallback={<RouteFallback />}>
              <SimulatorRoute />
            </Suspense>
          }
        />
        <Route
          path="/results/:runId"
          element={
            <Suspense fallback={<RouteFallback />}>
              <ResultsRoute />
            </Suspense>
          }
        />
        <Route
          path="/results/:runId/diagnostics"
          element={
            <Suspense fallback={<RouteFallback />}>
              <DiagnosticsRoute />
            </Suspense>
          }
        />
        <Route
          path="/mmm"
          element={
            <Suspense fallback={<RouteFallback />}>
              <MmmComingSoonRoute />
            </Suspense>
          }
        />
        <Route path="*" element={<Navigate to="/simulator" replace />} />
      </Route>
    </Routes>
  );
}

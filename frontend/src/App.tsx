import { Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "@/components/layout/app-shell";
import DiagnosticsRoute from "@/routes/Diagnostics";
import MmmComingSoonRoute from "@/routes/MmmComingSoon";
import ResultsRoute from "@/routes/Results";
import SimulatorRoute from "@/routes/Simulator";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<Navigate to="/simulator" replace />} />
        <Route path="/simulator" element={<SimulatorRoute />} />
        <Route path="/results/:runId" element={<ResultsRoute />} />
        <Route path="/results/:runId/diagnostics" element={<DiagnosticsRoute />} />
        <Route path="/mmm" element={<MmmComingSoonRoute />} />
        <Route path="*" element={<Navigate to="/simulator" replace />} />
      </Route>
    </Routes>
  );
}

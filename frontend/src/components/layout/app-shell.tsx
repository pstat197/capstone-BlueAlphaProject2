import { Outlet } from "react-router-dom";

import { TopBar } from "@/components/layout/top-bar";

export function AppShell() {
  return (
    <div className="flex min-h-screen flex-col">
      <TopBar />
      <main className="mx-auto flex w-full max-w-[1400px] flex-1 flex-col px-6 pb-6 pt-6">
        <Outlet />
      </main>
    </div>
  );
}

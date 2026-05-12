import { Outlet } from "react-router-dom";

import { TopBar } from "@/components/layout/top-bar";

export function AppShell() {
  return (
    <div className="flex min-h-screen flex-col">
      <TopBar />
      {/*
       * Spacer that reserves the height of the (fixed) top bar so route
       * content starts below it. Matches the unstuck nav height (h-14).
       */}
      <div aria-hidden className="h-14 shrink-0" />
      {/*
       * `min-h-[120vh]` makes every route at least taller than the viewport so
       * the user can always scroll, which is required for the top-bar pill
       * morph (and for the sticky action bar) to engage. Without it, short
       * pages like the MMM coming-soon stub never trigger sticky.
       */}
      <main className="mx-auto flex min-h-[120vh] w-full max-w-[1400px] flex-1 flex-col px-6 pb-6 pt-6">
        <Outlet />
      </main>
    </div>
  );
}

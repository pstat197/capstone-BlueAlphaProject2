import { NavLink } from "react-router-dom";

import { BrandMark } from "@/components/layout/brand-mark";
import { RunsDrawer } from "@/components/layout/runs-drawer";
import { SettingsPopover } from "@/components/layout/settings-popover";
import { cn } from "@/lib/cn";
import { useStuck } from "@/lib/use-scrolled";

const NAV_ITEMS = [
  { to: "/simulator", label: "Simulator" },
  { to: "/mmm", label: "Bayesian MMM" },
] as const;

export function TopBar() {
  const [stuck, sentinelRef] = useStuck();

  return (
    <>
      {/*
       * 1px sentinel just above the sticky header. When it scrolls out of view
       * the header is considered "stuck" and `stuck` flips true.
       * To verify in DevTools: inspect the inner div below — its `data-stuck`
       * attribute should toggle between "false" and "true" as you scroll.
       */}
      <div ref={sentinelRef} aria-hidden className="h-px w-full" />
      {/*
       * `position: fixed` (not sticky) so the bar is rock-solid at the top
       * regardless of scroll position. Sticky inside a flex column can jump
       * by a pixel at the very end of its containing block on some browsers;
       * fixed avoids that entirely. AppShell renders an h-14 spacer to keep
       * page content from sliding underneath this fixed header.
       */}
      <header className="fixed inset-x-0 top-0 z-40">
        <div
          data-stuck={stuck}
          className={cn(
            "mx-auto flex items-center justify-between gap-4 backdrop-blur border",
            "transition-all duration-300 ease-out",
            stuck
              ? cn(
                  /* Stuck: modestly narrower pill, lifted, big shadow. */
                  "mt-3 h-12 max-w-4xl rounded-full px-5",
                  "border-brand-border bg-white/95",
                  "shadow-[0_12px_40px_-8px_rgba(20,63,160,0.35)]",
                )
              : cn(
                  /* Default: full-width bar with subtle bottom rule. */
                  "mt-0 h-14 max-w-[1400px] rounded-none px-6",
                  "border-transparent border-b-brand-border/70 bg-white/80",
                  "supports-[backdrop-filter]:bg-white/65",
                  "shadow-none",
                ),
          )}
        >
          <div className="flex items-center gap-6">
            <BrandMark />
            <nav className="hidden items-center gap-1 md:flex">
              {NAV_ITEMS.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    cn(
                      "rounded-full px-3 py-1.5 text-sm font-medium transition-colors",
                      isActive
                        ? "bg-brand-50 text-brand-700"
                        : "text-slate-600 hover:bg-slate-100/70 hover:text-slate-900",
                    )
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </nav>
          </div>
          <div className="flex items-center gap-1">
            <RunsDrawer />
            <SettingsPopover />
          </div>
        </div>
      </header>
    </>
  );
}

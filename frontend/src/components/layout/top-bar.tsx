import { NavLink } from "react-router-dom";

import { BrandMark } from "@/components/layout/brand-mark";
import { RunsDrawer } from "@/components/layout/runs-drawer";
import { SettingsPopover } from "@/components/layout/settings-popover";
import { cn } from "@/lib/cn";
import { useScrolled } from "@/lib/use-scrolled";

const NAV_ITEMS = [
  { to: "/simulator", label: "Simulator" },
  { to: "/mmm", label: "Bayesian MMM" },
] as const;

export function TopBar() {
  const scrolled = useScrolled(8);

  return (
    /*
     * The outer <header> is just the sticky positioner — no visible chrome.
     * The inner div carries all the visuals and morphs between two states:
     *   - default: full-width bar with bottom rule
     *   - scrolled: centered floating pill with shadow
     * Both states share the same DOM nodes so CSS transitions handle the morph.
     */
    <header className="sticky top-0 z-40">
      <div
        className={cn(
          "mx-auto flex h-14 items-center justify-between gap-4 backdrop-blur",
          "transition-[max-width,border-radius,margin,box-shadow,background-color,padding] duration-300 ease-out",
          scrolled
            ? cn(
                "mt-3 max-w-3xl rounded-full px-4",
                "border border-brand-border bg-white/95",
                "shadow-[0_8px_32px_rgba(20,63,160,0.18)]",
              )
            : cn(
                "mt-0 max-w-[1400px] rounded-none px-6",
                "border-x-0 border-t-0 border-b border-brand-border/70 bg-white/80",
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
  );
}

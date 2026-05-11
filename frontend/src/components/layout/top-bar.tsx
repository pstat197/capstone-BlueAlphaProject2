import { NavLink } from "react-router-dom";

import { BrandMark } from "@/components/layout/brand-mark";
import { RunsDrawer } from "@/components/layout/runs-drawer";
import { SettingsPopover } from "@/components/layout/settings-popover";
import { cn } from "@/lib/cn";

const NAV_ITEMS = [
  { to: "/simulator", label: "Simulator" },
  { to: "/mmm", label: "Bayesian MMM" },
] as const;

export function TopBar() {
  return (
    <header className="sticky top-0 z-30 border-b border-brand-border/70 bg-white/80 backdrop-blur supports-[backdrop-filter]:bg-white/65">
      <div className="mx-auto flex h-14 w-full max-w-[1400px] items-center justify-between gap-4 px-6">
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

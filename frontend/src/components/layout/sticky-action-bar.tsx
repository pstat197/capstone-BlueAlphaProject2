import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

interface StickyActionBarProps {
  left?: ReactNode;
  right: ReactNode;
  className?: string;
}

export function StickyActionBar({ left, right, className }: StickyActionBarProps) {
  return (
    <div
      className={cn(
        "sticky bottom-4 z-20 mx-auto mt-6 flex max-w-[1400px] items-center justify-between gap-3 rounded-2xl border border-brand-border bg-white/95 px-4 py-3 shadow-[0_8px_32px_rgba(20,63,160,0.16)] backdrop-blur",
        className,
      )}
    >
      <div className="flex min-w-0 flex-1 items-center gap-3 text-sm text-slate-600">
        {left}
      </div>
      <div className="flex items-center gap-2">{right}</div>
    </div>
  );
}

import { Link } from "react-router-dom";

import { cn } from "@/lib/cn";

interface BrandMarkProps {
  className?: string;
  to?: string;
}

export function BrandMark({ className, to = "/simulator" }: BrandMarkProps) {
  return (
    <Link
      to={to}
      className={cn(
        "group flex items-center gap-2.5 rounded-full px-1.5 py-1 -ml-1.5 outline-none focus-visible:ring-2 focus-visible:ring-brand-500/40",
        className,
      )}
    >
      <span
        aria-hidden
        className="inline-flex h-8 w-8 items-center justify-center rounded-xl bg-brand-500 text-white shadow-[0_4px_12px_rgba(29,99,237,0.35)] transition-transform group-hover:scale-105"
      >
        <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4">
          <path
            d="M5 19V5l7 7 7-7v14"
            stroke="currentColor"
            strokeWidth="2.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </span>
      <span className="flex flex-col leading-none">
        <span className="text-sm font-semibold tracking-tight text-slate-900">Blue Alpha</span>
        <span className="text-[11px] font-medium uppercase tracking-[0.14em] text-slate-500">
          Simulator
        </span>
      </span>
    </Link>
  );
}

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
      {/*
       * BlueAlpha brand mark. The source PNG already has the blue rounded
       * square + white 'a' glyph, so we render it as an <img> rather than
       * wrapping it in a coloured span. Width/height attributes are 2x the
       * displayed size so the image stays crisp on retina displays.
       */}
      <img
        src="/bluealpha-logo.png"
        alt=""
        aria-hidden
        width={64}
        height={64}
        className="h-8 w-8 rounded-xl shadow-[0_4px_12px_rgba(29,99,237,0.35)] transition-transform group-hover:scale-105"
      />
      <span className="flex flex-col leading-none">
        <span className="text-sm font-semibold tracking-tight text-slate-900">BlueAlpha</span>
        <span className="text-[11px] font-medium uppercase tracking-[0.14em] text-slate-500">
          Simulator
        </span>
      </span>
    </Link>
  );
}

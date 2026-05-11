import { cva, type VariantProps } from "class-variance-authority";
import { forwardRef, type HTMLAttributes } from "react";

import { cn } from "@/lib/cn";

const badgeVariants = cva(
  "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
  {
    variants: {
      variant: {
        default: "bg-brand-50 text-brand-700 border border-brand-100",
        outline: "border border-slate-200 text-slate-600",
        success: "bg-emerald-50 text-emerald-700 border border-emerald-100",
        warn: "bg-amber-50 text-amber-700 border border-amber-100",
        muted: "bg-slate-100 text-slate-600",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(function Badge(
  { className, variant, ...props },
  ref,
) {
  return (
    <span ref={ref} className={cn(badgeVariants({ variant, className }))} {...props} />
  );
});

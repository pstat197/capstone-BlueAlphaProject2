import { forwardRef, type InputHTMLAttributes } from "react";

import { cn } from "@/lib/cn";

export type InputProps = InputHTMLAttributes<HTMLInputElement>;

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { className, type = "text", ...props },
  ref,
) {
  return (
    <input
      ref={ref}
      type={type}
      className={cn(
        "flex h-9 w-full rounded-md border border-brand-border bg-white px-3 py-1 text-sm text-slate-900",
        "shadow-[inset_0_1px_2px_rgba(15,23,42,0.04)]",
        "transition-colors placeholder:text-slate-400",
        "focus:outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-500/20",
        "disabled:cursor-not-allowed disabled:opacity-60",
        className,
      )}
      {...props}
    />
  );
});

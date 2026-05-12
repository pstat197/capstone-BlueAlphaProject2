import { useEffect, useRef, useState, type RefObject } from "react";

/**
 * Returns `[stuck, sentinelRef]`. Place the ref on a 1px sentinel positioned
 * just above any element you want to morph when the user scrolls past it.
 *
 * Implementation uses IntersectionObserver against the viewport, which is
 * more robust than `window.scrollY` (works regardless of whether the
 * scrollable container is the window or a wrapper, no scroll-throttle math,
 * and no extra render on every scroll tick).
 */
export function useStuck(): [boolean, RefObject<HTMLDivElement | null>] {
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  const [stuck, setStuck] = useState(false);

  useEffect(() => {
    const el = sentinelRef.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (!entry) return;
        setStuck(!entry.isIntersecting);
      },
      { threshold: 0, rootMargin: "0px 0px 0px 0px" },
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return [stuck, sentinelRef];
}

/* Backwards-compatible scroll-position helper if anything else needs it. */
export function useScrolled(threshold = 16): boolean {
  const [scrolled, setScrolled] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.scrollY > threshold;
  });

  useEffect(() => {
    const onScroll = () => {
      const next = window.scrollY > threshold;
      setScrolled((prev) => (prev === next ? prev : next));
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [threshold]);

  return scrolled;
}

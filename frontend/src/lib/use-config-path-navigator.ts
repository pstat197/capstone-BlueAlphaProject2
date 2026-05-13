import { useCallback } from "react";

import { formatPath } from "@/lib/use-config-validation";
import type { ScenariosTab } from "@/components/simulator/scenarios-card";
import type { SimulatorPane } from "@/components/simulator/channel-list";

type Path = ReadonlyArray<string | number>;

interface NavigateContext {
  setSelected: (pane: SimulatorPane) => void;
  setScenariosTab: (tab: ScenariosTab) => void;
  setAdvancedOpen: (open: boolean) => void;
}

/**
 * Translate a validation path into a navigation hint: which channel to
 * select (if any), which Scenarios tab to open (if any), and whether to
 * expand the Advanced/Outcome card.
 *
 * This is a pure function so it can be unit-reasoned about without React;
 * `useConfigPathNavigator` below applies the result to the real setters.
 */
export function planNavigation(path: Path): {
  pane?: SimulatorPane;
  scenariosTab?: ScenariosTab;
  expandAdvanced?: boolean;
} {
  if (path.length === 0) return {};

  const [head, second] = path;

  // Per-channel issues route to that channel's detail pane. Seasonality
  // edits are still done via the SeasonalityPanel target picker, but the
  // detail pane is the closest contextual location.
  if (head === "channel_list" && typeof second === "number") {
    const inSeasonality = path.some((p) => p === "seasonality_config");
    return {
      pane: { kind: "channel", index: second },
      scenariosTab: inSeasonality ? "seasonality" : undefined,
    };
  }

  if (head === "outcome_revenue") {
    const inSeasonality = path.some((p) => p === "seasonality_config");
    if (inSeasonality) return { scenariosTab: "seasonality" };
    return { expandAdvanced: true };
  }

  if (head === "correlations") return { scenariosTab: "correlations" };
  if (head === "budget_shifts") return { scenariosTab: "budget" };

  if (head === "media_transform_order" || head === "number_of_channels" || head === "adstock" || head === "saturation") {
    return { expandAdvanced: true };
  }
  return {};
}

/**
 * Best-effort focus+scroll on the field flagged by a validation issue.
 *
 * We retry across a few animation frames so React has time to mount the
 * newly-selected channel pane or expand the advanced card before the
 * data-attribute lookup. Missing elements degrade to scrolling the issue
 * banner itself.
 */
function scrollAndFocus(path: Path): void {
  const selector = `[data-config-path="${cssEscape(formatPath([...path]))}"]`;
  let tries = 0;
  const maxTries = 12; // ~200ms at 60fps
  const tick = () => {
    tries += 1;
    const el = document.querySelector<HTMLElement>(selector);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      // Inputs / buttons are focusable; for non-focusable elements `focus()`
      // is a safe no-op, so we always attempt it.
      try {
        el.focus({ preventScroll: true });
      } catch {
        /* ignore */
      }
      // Brief outline so the user sees what changed even without focus styles.
      el.classList.add("ring-2", "ring-brand-400", "ring-offset-2");
      setTimeout(() => {
        el.classList.remove("ring-2", "ring-brand-400", "ring-offset-2");
      }, 1500);
      return;
    }
    if (tries < maxTries) {
      requestAnimationFrame(tick);
    }
  };
  requestAnimationFrame(tick);
}

/** Minimal CSS.escape polyfill — only the bits we need for our paths. */
function cssEscape(value: string): string {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
    return CSS.escape(value);
  }
  return value.replace(/(["\\])/g, "\\$1");
}

export function useConfigPathNavigator(ctx: NavigateContext) {
  const { setSelected, setScenariosTab, setAdvancedOpen } = ctx;
  return useCallback(
    (path: Path) => {
      if (!path || path.length === 0) return;
      const plan = planNavigation(path);
      if (plan.pane) setSelected(plan.pane);
      if (plan.scenariosTab) setScenariosTab(plan.scenariosTab);
      if (plan.expandAdvanced) setAdvancedOpen(true);
      scrollAndFocus(path);
    },
    [setSelected, setScenariosTab, setAdvancedOpen],
  );
}

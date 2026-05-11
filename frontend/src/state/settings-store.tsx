import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

type Settings = {
  colorblindCharts: boolean;
  overlayCharts: boolean;
};

type SettingsContextValue = Settings & {
  setColorblindCharts: (v: boolean) => void;
  setOverlayCharts: (v: boolean) => void;
};

const STORAGE_KEY = "ba.settings.v1";
const SettingsContext = createContext<SettingsContextValue | null>(null);

function readInitial(): Settings {
  if (typeof window === "undefined") {
    return { colorblindCharts: true, overlayCharts: false };
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<Settings>;
      return {
        colorblindCharts: parsed.colorblindCharts ?? true,
        overlayCharts: parsed.overlayCharts ?? false,
      };
    }
  } catch {
    /* ignore corrupt storage */
  }
  return { colorblindCharts: true, overlayCharts: false };
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(readInitial);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      /* storage may be unavailable in private mode */
    }
  }, [settings]);

  const setColorblindCharts = useCallback(
    (v: boolean) => setSettings((s) => ({ ...s, colorblindCharts: v })),
    [],
  );
  const setOverlayCharts = useCallback(
    (v: boolean) => setSettings((s) => ({ ...s, overlayCharts: v })),
    [],
  );

  const value = useMemo(
    () => ({ ...settings, setColorblindCharts, setOverlayCharts }),
    [settings, setColorblindCharts, setOverlayCharts],
  );

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings must be used within SettingsProvider");
  return ctx;
}

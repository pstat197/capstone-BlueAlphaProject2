import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useReducer,
  type ReactNode,
} from "react";

import { blankConfig } from "@/lib/config-utils";
import type { SimConfig } from "@/types/api";

type State = {
  config: SimConfig;
  /** True when user has hand-edited the YAML pane and form sync is paused. */
  yamlDirty: boolean;
  /** Last known config_hash for the current `config`, if known. */
  lastHash: string | null;
};

type Action =
  | { type: "set"; config: SimConfig; resetYamlDirty?: boolean }
  | { type: "patch"; patch: Partial<SimConfig> }
  | { type: "set-yaml-dirty"; dirty: boolean }
  | { type: "set-hash"; hash: string | null }
  | { type: "reset"; config?: SimConfig };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "set":
      return {
        ...state,
        config: action.config,
        yamlDirty: action.resetYamlDirty ? false : state.yamlDirty,
      };
    case "patch":
      return { ...state, config: { ...state.config, ...action.patch } };
    case "set-yaml-dirty":
      return { ...state, yamlDirty: action.dirty };
    case "set-hash":
      return { ...state, lastHash: action.hash };
    case "reset":
      return {
        config: action.config ?? blankConfig(),
        yamlDirty: false,
        lastHash: null,
      };
  }
}

type ConfigContextValue = State & {
  setConfig: (config: SimConfig, opts?: { resetYamlDirty?: boolean }) => void;
  patchConfig: (patch: Partial<SimConfig>) => void;
  setYamlDirty: (dirty: boolean) => void;
  setLastHash: (hash: string | null) => void;
  reset: (config?: SimConfig) => void;
};

const ConfigContext = createContext<ConfigContextValue | null>(null);

export function ConfigProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, undefined, () => ({
    config: blankConfig(),
    yamlDirty: false,
    lastHash: null,
  }));

  const setConfig = useCallback(
    (config: SimConfig, opts?: { resetYamlDirty?: boolean }) =>
      dispatch({ type: "set", config, resetYamlDirty: opts?.resetYamlDirty }),
    [],
  );
  const patchConfig = useCallback(
    (patch: Partial<SimConfig>) => dispatch({ type: "patch", patch }),
    [],
  );
  const setYamlDirty = useCallback(
    (dirty: boolean) => dispatch({ type: "set-yaml-dirty", dirty }),
    [],
  );
  const setLastHash = useCallback(
    (hash: string | null) => dispatch({ type: "set-hash", hash }),
    [],
  );
  const reset = useCallback(
    (config?: SimConfig) => dispatch({ type: "reset", config }),
    [],
  );

  const value = useMemo(
    () => ({ ...state, setConfig, patchConfig, setYamlDirty, setLastHash, reset }),
    [state, setConfig, patchConfig, setYamlDirty, setLastHash, reset],
  );

  return <ConfigContext.Provider value={value}>{children}</ConfigContext.Provider>;
}

export function useConfig(): ConfigContextValue {
  const ctx = useContext(ConfigContext);
  if (!ctx) throw new Error("useConfig must be used within ConfigProvider");
  return ctx;
}

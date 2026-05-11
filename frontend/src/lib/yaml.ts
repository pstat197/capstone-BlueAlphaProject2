import { parse, stringify } from "yaml";

import type { SimConfig } from "@/types/api";

export function parseYaml(text: string): { ok: true; data: SimConfig } | { ok: false; error: string } {
  try {
    const parsed = parse(text ?? "");
    if (parsed == null) return { ok: true, data: {} as SimConfig };
    if (typeof parsed !== "object" || Array.isArray(parsed)) {
      return { ok: false, error: "YAML must parse to a mapping (dict)." };
    }
    return { ok: true, data: parsed as SimConfig };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}

export function dumpYaml(config: SimConfig): string {
  return stringify(config, { sortMapEntries: false, lineWidth: 0, indent: 2 });
}

import { yaml as yamlLang } from "@codemirror/lang-yaml";
import CodeMirror from "@uiw/react-codemirror";
import { ChevronDown } from "lucide-react";
import { useMemo, useState } from "react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { dumpYaml } from "@/lib/yaml";
import type { SimConfig } from "@/types/api";

interface ConfigSnapshotProps {
  config: SimConfig;
}

export function ConfigSnapshot({ config }: ConfigSnapshotProps) {
  const [open, setOpen] = useState(false);
  const text = useMemo(() => dumpYaml(config ?? ({} as SimConfig)), [config]);

  return (
    <Card>
      <CardHeader className="cursor-pointer" onClick={() => setOpen((v) => !v)}>
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle>Configuration snapshot</CardTitle>
            <CardDescription>
              The merged YAML used for this run. Click to {open ? "collapse" : "expand"}.
            </CardDescription>
          </div>
          <ChevronDown
            className={`h-4 w-4 text-slate-400 transition-transform ${open ? "rotate-180" : ""}`}
          />
        </div>
      </CardHeader>
      {open && (
        <CardContent>
          <div className="overflow-hidden rounded-lg border border-brand-border">
            <CodeMirror
              value={text}
              height="320px"
              editable={false}
              extensions={[yamlLang()]}
              basicSetup={{ lineNumbers: true, highlightActiveLine: false, foldGutter: true }}
            />
          </div>
        </CardContent>
      )}
    </Card>
  );
}

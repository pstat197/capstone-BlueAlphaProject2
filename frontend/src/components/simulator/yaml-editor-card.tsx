import { yaml as yamlLang } from "@codemirror/lang-yaml";
import CodeMirror from "@uiw/react-codemirror";
import { CheckCircle2, RotateCcw, AlertCircle } from "lucide-react";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { dumpYaml, parseYaml } from "@/lib/yaml";
import { useConfig } from "@/state/config-store";

export function YamlEditorCard() {
  const { config, setConfig, yamlDirty, setYamlDirty } = useConfig();
  /* When clean: derive text from the live config. When dirty: show the user's edits. */
  const formText = useMemo(() => dumpYaml(config), [config]);
  const [editedText, setEditedText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const text = yamlDirty && editedText != null ? editedText : formText;

  const handleApply = () => {
    const parsed = parseYaml(text);
    if (!parsed.ok) {
      setError(parsed.error);
      return;
    }
    setError(null);
    setConfig(parsed.data, { resetYamlDirty: true });
    setEditedText(null);
    setYamlDirty(false);
  };

  const handleReset = () => {
    setEditedText(null);
    setError(null);
    setYamlDirty(false);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Advanced · YAML editor</CardTitle>
        <CardDescription>
          Stays in sync with the form. When you edit here, click <strong>Apply YAML</strong> to push
          changes back to the form. Otherwise the form drives this view.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="overflow-hidden rounded-lg border border-brand-border">
          <CodeMirror
            value={text}
            height="540px"
            extensions={[yamlLang()]}
            onChange={(value) => {
              setEditedText(value);
              setYamlDirty(true);
              if (error) setError(null);
            }}
            basicSetup={{
              lineNumbers: true,
              highlightActiveLine: true,
              foldGutter: true,
            }}
          />
        </div>

        {error ? (
          <div className="flex items-start gap-2 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        ) : yamlDirty ? (
          <div className="flex items-center justify-between gap-3 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-800">
            <span>YAML edited — apply to push into the form, or reset to discard.</span>
          </div>
        ) : (
          <div className="flex items-center gap-2 rounded-lg bg-emerald-50 px-3 py-2 text-sm text-emerald-700">
            <CheckCircle2 className="h-4 w-4" />
            In sync with the form.
          </div>
        )}

        <Separator />
        <div className="flex justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={handleReset} disabled={!yamlDirty}>
            <RotateCcw className="h-3.5 w-3.5" />
            Reset to form
          </Button>
          <Button size="sm" onClick={handleApply} disabled={!yamlDirty}>
            Apply YAML to form
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Settings2, Trash2 } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { api } from "@/lib/api";
import { useSettings } from "@/state/settings-store";

export function SettingsPopover() {
  const { colorblindCharts, overlayCharts, setColorblindCharts, setOverlayCharts } =
    useSettings();
  const [open, setOpen] = useState(false);
  const queryClient = useQueryClient();

  const clearCacheMutation = useMutation({
    mutationFn: () => api.clearCache(),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["runs"] });
    },
  });

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Open settings"
          className="hover:bg-brand-50 text-slate-600"
        >
          <Settings2 className="h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80">
        <div className="space-y-4">
          <div>
            <h4 className="text-sm font-semibold text-slate-900">Settings</h4>
            <p className="text-xs text-slate-500">
              Saved locally on this browser.
            </p>
          </div>

          <Separator />

          <div className="space-y-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <Label className="text-[11px] text-slate-700 normal-case tracking-normal font-semibold">
                  Colorblind-safe palette
                </Label>
                <p className="text-xs text-slate-500 mt-0.5">
                  Wong-style orange / blue / green so series stay distinguishable.
                </p>
              </div>
              <Switch
                checked={colorblindCharts}
                onCheckedChange={setColorblindCharts}
                aria-label="Colorblind-safe palette"
              />
            </div>

            <div className="flex items-start justify-between gap-3">
              <div>
                <Label className="text-[11px] text-slate-700 normal-case tracking-normal font-semibold">
                  Default to overlay charts
                </Label>
                <p className="text-xs text-slate-500 mt-0.5">
                  Pre-checks the min-max overlay toggle on results.
                </p>
              </div>
              <Switch
                checked={overlayCharts}
                onCheckedChange={setOverlayCharts}
                aria-label="Overlay charts by default"
              />
            </div>
          </div>

          <Separator />

          <div className="space-y-2">
            <Button
              variant="outline"
              size="sm"
              className="w-full justify-start"
              onClick={() => clearCacheMutation.mutate()}
              disabled={clearCacheMutation.isPending}
            >
              <Trash2 className="h-3.5 w-3.5" />
              {clearCacheMutation.isPending
                ? "Clearing…"
                : clearCacheMutation.data
                  ? `Cleared ${clearCacheMutation.data.removed} file(s)`
                  : "Clear simulation cache"}
            </Button>
            <p className="text-[11px] text-slate-500">
              Removes cached run CSVs on disk. Runs re-compute next time.
            </p>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

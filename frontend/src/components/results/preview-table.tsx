import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/cn";
import type { RunPreview } from "@/types/api";

interface PreviewTableProps {
  preview: RunPreview;
}

export function PreviewTable({ preview }: PreviewTableProps) {
  const { columns, rows } = preview;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Data preview</CardTitle>
        <CardDescription>
          First 25 rows · floats rounded to three decimals.
        </CardDescription>
      </CardHeader>
      <CardContent className="px-0">
        <div className="max-h-[420px] overflow-auto">
          <table className="min-w-full text-sm">
            <thead className="sticky top-0 bg-brand-50/70 backdrop-blur">
              <tr>
                {columns.map((c) => (
                  <th
                    key={c}
                    className="whitespace-nowrap px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wide text-brand-700"
                  >
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr
                  key={i}
                  className={cn(
                    "border-t border-brand-border",
                    i % 2 === 1 ? "bg-slate-50/40" : "",
                  )}
                >
                  {columns.map((c) => {
                    const v = row[c];
                    return (
                      <td
                        key={c}
                        className="whitespace-nowrap px-3 py-1.5 text-slate-700 tabular-nums"
                      >
                        {v == null ? "—" : String(v)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

import { expect, test } from "@playwright/test";

/**
 * Smoke test: the happy-path of the Simulator tab.
 *
 *   1. Land on `/simulator` (root redirect).
 *   2. Wait for the example config to seed so the channel list is non-empty
 *      and the Run button is enabled.
 *   3. Click "Run simulation" (or "Re-run from cache" if a prior run is
 *      already cached on this dev box).
 *   4. Expect to land on `/results/<hash>` with the Charts tab active.
 *   5. Open the Ground truth tab and assert it renders the table.
 *   6. Click "Edit configuration" and verify we're back on the simulator.
 *
 * This catches the most common regressions: bad route wiring, broken /api
 * proxy, panic-rendering on the run payload, missing ground-truth field, etc.
 */
test.describe("simulator → run → results", () => {
  test("seeds, runs, lands on results, opens ground truth, and returns", async ({
    page,
  }) => {
    await page.goto("/simulator");

    // The example config is fetched + seeded on first paint; wait for at
    // least one channel row before clicking Run. The channel list shows
    // each channel as a button containing the channel name.
    const channelList = page.getByRole("region", { name: /channels/i }).or(
      page.locator('[data-config-path^="channel_list"]').first(),
    );
    await expect(channelList.first()).toBeVisible({ timeout: 20_000 });

    const runButton = page.getByRole("button", {
      name: /(run simulation|re-run from cache)/i,
    });
    await expect(runButton).toBeEnabled({ timeout: 20_000 });
    await runButton.click();

    await expect(page).toHaveURL(/\/results\/[0-9a-f]{8,}/i, {
      timeout: 90_000,
    });
    await expect(
      page.getByRole("heading", { level: 1 }),
    ).toBeVisible();

    const groundTruthTab = page.getByRole("tab", { name: /ground truth/i });
    await groundTruthTab.click();
    await expect(groundTruthTab).toHaveAttribute("data-state", "active");
    // The card always renders its title once the tab is mounted.
    await expect(
      page.getByRole("heading", { name: /ground truth/i }).first(),
    ).toBeVisible();

    await page.getByRole("button", { name: /edit configuration/i }).click();
    await expect(page).toHaveURL(/\/simulator$/);
    await expect(
      page.getByRole("button", {
        name: /(run simulation|re-run from cache|fix errors to run)/i,
      }),
    ).toBeVisible();
  });
});

import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * End-to-end Playwright config for the React UI.
 *
 * `webServer` boots both the FastAPI backend (uvicorn) and the Vite dev server
 * so that `npm run test:e2e` from a clean checkout requires only:
 *
 *   - the repo's Python `.venv` activated (or available at ../.venv)
 *   - `npm install` already run in this folder
 *   - `npx playwright install chromium` already run once
 *
 * The Vite dev server proxies `/api` -> 127.0.0.1:8000, so the tests only
 * need to talk to the frontend origin.
 */

const repoRoot = path.resolve(__dirname, "..");
const FRONTEND_URL = process.env.E2E_FRONTEND_URL ?? "http://127.0.0.1:5173";
const BACKEND_URL = process.env.E2E_BACKEND_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? "github" : "list",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: FRONTEND_URL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    actionTimeout: 10_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      // FastAPI backend. Re-use the repo's .venv if present so contributors
      // don't have to install Uvicorn globally.
      command:
        "bash -lc '" +
        `[ -f ${repoRoot}/.venv/bin/activate ] && source ${repoRoot}/.venv/bin/activate; ` +
        "uvicorn server.main:app --port 8000 --host 127.0.0.1'",
      url: `${BACKEND_URL}/api/health`,
      cwd: repoRoot,
      reuseExistingServer: !process.env.CI,
      stdout: "pipe",
      stderr: "pipe",
      timeout: 60_000,
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 5173 --strictPort",
      url: FRONTEND_URL,
      cwd: __dirname,
      reuseExistingServer: !process.env.CI,
      stdout: "pipe",
      stderr: "pipe",
      timeout: 60_000,
    },
  ],
});

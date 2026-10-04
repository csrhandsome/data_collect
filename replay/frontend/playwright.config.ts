import { defineConfig, devices } from '@playwright/test'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'

const repoRoot = fileURLToPath(new URL('../../', import.meta.url))

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  expect: { timeout: 10_000 },
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:5173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    ...devices['Desktop Chrome'],
    channel: process.env.REPLAY_BROWSER_CHANNEL || 'chrome',
    launchOptions: { args: ['--no-sandbox'] },
  },
  webServer: [
    {
      command:
        'env -u PYTHONPATH uv run python -m replay.scripts.generate_demo && env -u PYTHONPATH uv run python -m replay.scripts.generate_image_demo && env -u PYTHONPATH uv run uvicorn replay.backend.main:app --host 127.0.0.1 --port 8000',
      cwd: repoRoot,
      url: 'http://127.0.0.1:8000/api/health',
      env: { REPLAY_DATA_ROOT: join(repoRoot, 'replay', 'demo_data') },
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
    {
      command: 'pnpm --dir replay/frontend dev --host 127.0.0.1 --port 5173 --strictPort',
      cwd: repoRoot,
      url: 'http://127.0.0.1:5173',
      env: { REPLAY_API_TARGET: 'http://127.0.0.1:8000' },
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
  ],
})

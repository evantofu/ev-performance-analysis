import { defineConfig } from '@playwright/test'
export default defineConfig({
  testDir: './tests/e2e',
  use: { baseURL: 'http://127.0.0.1:8000', viewport: { width: 1440, height: 1000 }, trace: 'retain-on-failure', launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {} },
  webServer: { command: 'python -m uvicorn main:app --app-dir ../backend --host 127.0.0.1 --port 8000',
    url: 'http://127.0.0.1:8000/ready', reuseExistingServer: !process.env.CI, timeout: 60000 },
})

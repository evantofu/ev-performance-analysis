import { chromium } from '@playwright/test'
import { fileURLToPath } from 'node:url'

const browser = await chromium.launch(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
  ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH } : {})
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 }, deviceScaleFactor: 1 })
  await page.goto(`${process.env.PREVIEW_BASE_URL || 'http://127.0.0.1:8000'}/compare`)
  await page.locator('.filter-panel select').first().selectOption('Tesla')
  await page.locator('.filter-panel select').nth(1).selectOption('2025')
  await page.getByRole('button', { name: 'Expand all' }).click()
  const choices = page.locator('[role="button"]').filter({ hasText: /MPGe/ })
  await choices.nth(0).click()
  await choices.nth(1).click()
  await page.getByRole('heading', { name: 'Head-to-head comparison' }).waitFor()
  await page.evaluate(() => document.fonts.ready)
  await page.mouse.move(5, 5)
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }))
  await page.screenshot({ path: fileURLToPath(new URL('../../../docs/ev-explorer.png', import.meta.url)) })
} finally {
  await browser.close()
}

import { test, expect } from '@playwright/test'

test('home renders snapshot data and navigation survives a direct deep link', async ({ page }) => {
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/')
  await expect(page.getByText('BEV model-year / trim records')).toBeVisible()
  await expect(page.getByText('1,274', { exact: true })).toBeVisible()
  await page.goto('/compare')
  await expect(page.getByRole('heading', { name: 'Filters', exact: true })).toBeVisible()
  await expect(page.locator('.js-plotly-plot')).toBeVisible()
  await expect(page.locator('.xtitle')).toHaveText('Efficiency (MPGe)')
  await expect(page.locator('.ytitle')).toHaveText('Range (miles)')
  expect(errors).toEqual([])
})

test('vehicle selection supports keyboard comparison and operating costs', async ({ page }) => {
  await page.goto('/compare')
  await expect(page.getByRole('button', { name: 'Expand all' })).toBeVisible()
  await page.getByRole('button', { name: 'Expand all' }).click()
  const choices = page.locator('[role="button"]').filter({ hasText: /MPGe/ })
  await expect(choices.first()).toBeVisible()
  await choices.nth(0).focus()
  await page.keyboard.press('Enter')
  await expect(page.getByRole('heading', { name: 'Head-to-head comparison' })).toBeVisible()
  await choices.nth(1).focus()
  await page.keyboard.press('Enter')
  await expect(page.getByText('Annual fuel cost', { exact: true })).toBeVisible()
  await expect(page.locator('.operating-cost').first()).toBeVisible()
})

test('map pans request geographic bounds and filtering changes returned records', async ({ page }) => {
  await page.goto('/chargers')
  const initial = await page.waitForResponse(r => r.url().includes('/api/stations?') && r.status() === 200)
  const before = new URL(initial.url())
  await expect(page.locator('.leaflet-container')).toBeVisible()
  const changed = page.waitForResponse(r => r.url().includes('/api/stations?') && r.status() === 200 &&
    new URL(r.url()).searchParams.get('lat_min') !== before.searchParams.get('lat_min'))
  await page.getByRole('button', { name: 'Zoom in' }).click()
  const response = await changed
  const bounds = new URL(response.url()).searchParams
  const result = await response.json()
  expect(result.results.length).toBeGreaterThan(0)
  for (const s of result.results) {
    expect(s.latitude).toBeGreaterThanOrEqual(Number(bounds.get('lat_min')))
    expect(s.latitude).toBeLessThanOrEqual(Number(bounds.get('lat_max')))
  }
})

test('segments and gated trends render without runtime errors', async ({ page }) => {
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/segments')
  await expect(page.getByText('Lower Efficiency', { exact: true })).toBeVisible()
  await expect(page.locator('.js-plotly-plot').first()).toBeVisible()
  await page.goto('/trends')
  await expect(page.getByRole('heading', { name: 'Manufacturer Efficiency Trends' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Top 5', exact: true })).toBeVisible()
  expect(errors).toEqual([])
})

test('API errors are visible instead of looking like an empty successful result', async ({ page }) => {
  await page.route('**/api/vehicles?**', route => route.fulfill({ status: 503, body: '{}' }))
  await page.goto('/compare')
  await expect(page.getByRole('alert')).toContainText('Request failed (503)')
})

test('unknown API routes remain JSON 404 responses', async ({ request }) => {
  const response = await request.get('/api/does-not-exist')
  expect(response.status()).toBe(404)
  expect(response.headers()['content-type']).toContain('application/json')
})

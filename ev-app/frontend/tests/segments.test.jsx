import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { annualElectricityCost } from '../src/components/OperatingCost'
import { useStore } from '../src/store'

const fixtures = vi.hoisted(() => ({
  vehicles: { results: [
    { id: 1, model: 'Unassigned', make: 'Test', year: 2021, cluster: null },
    { id: 2, model: 'Assigned', make: 'Test', base_model: 'assigned', year: 2025, cluster: 0, combined_mpge: 120, range_miles: 300 },
  ] },
  makes: [],
  segments: [{ cluster_id: 0, avg_mpge: 120, avg_range: 300, top_makes: [] }],
}))
vi.mock('../src/api', () => ({ api: {
  vehicles: vi.fn().mockResolvedValue(fixtures.vehicles),
  makes: vi.fn().mockResolvedValue(fixtures.makes),
  segments: vi.fn().mockResolvedValue(fixtures.segments),
} }))
vi.mock('../src/components/PlotlyChart', () => ({ PlotlyChart: ({ data }) => <div data-testid="chart" data-traces={JSON.stringify(data)} /> }))
import { SegmentsPage } from '../src/pages/SegmentsPage'
import { ComparePage } from '../src/pages/ComparePage'

afterEach(cleanup)

it('does not count null segment labels as High Efficiency', async () => {
  render(<SegmentsPage />)
  expect(await screen.findByText('1 vehicles · click to explore')).toBeTruthy()
  fireEvent.click(screen.getByText('High Efficiency'))
  expect(screen.getByText(/1 vehicles below/)).toBeTruthy()
})

it('computes operating costs with stated units and handles missing data', () => {
  expect(annualElectricityCost(100)).toBe(728)
  expect(annualElectricityCost(null)).toBeNull()
  expect(annualElectricityCost(0)).toBeNull()
})

it('preserves false and zero filter values', () => {
  useStore.getState().setFilter('sort_desc', false)
  useStore.getState().setFilter('min_range', 0)
  expect(useStore.getState().filters.sort_desc).toBe(false)
  expect(useStore.getState().filters.min_range).toBe(0)
  useStore.getState().resetFilters()
})

it('does not plot unassigned records as High Efficiency in comparison', async () => {
  render(<ComparePage />)
  await waitFor(() => {
    const traces = JSON.parse(screen.getByTestId('chart').getAttribute('data-traces'))
    expect(traces[0].text).toEqual(['2025 Test Assigned'])
  })
})

import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { useApi } from '../src/hooks/useApi'
import { api } from '../src/api'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

it('aborts obsolete requests and ignores their late responses', async () => {
  const pending = []
  const fetcher = vi.fn(signal => new Promise(resolve => pending.push({ signal, resolve })))
  const { result, rerender, unmount } = renderHook(({ version }) => useApi(fetcher, [version]), {
    initialProps: { version: 1 },
  })
  await waitFor(() => expect(pending).toHaveLength(1))
  rerender({ version: 2 })
  await waitFor(() => expect(pending).toHaveLength(2))
  expect(pending[0].signal.aborted).toBe(true)
  await act(async () => pending[1].resolve('new'))
  expect(result.current.data).toBe('new')
  await act(async () => pending[0].resolve('old'))
  expect(result.current.data).toBe('new')
  unmount()
  expect(pending[1].signal.aborted).toBe(true)
})

it('surfaces request failures and stops loading', async () => {
  const { result } = renderHook(() => useApi(() => Promise.reject(new Error('Offline')), []))
  await waitFor(() => expect(result.current.loading).toBe(false))
  expect(result.current.error).toBe('Offline')
})

it('passes the signal and query parameters through the API client', async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ total: 0 }) })
  vi.stubGlobal('fetch', fetch)
  const controller = new AbortController()
  await api.stations({ lat_min: 30, network: null, dc_fast_only: false }, { signal: controller.signal })
  const [url, options] = fetch.mock.calls[0]
  expect(url.searchParams.get('lat_min')).toBe('30')
  expect(url.searchParams.get('dc_fast_only')).toBe('false')
  expect(url.searchParams.has('network')).toBe(false)
  expect(options.signal).toBe(controller.signal)
})

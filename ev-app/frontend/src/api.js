const BASE = import.meta.env.VITE_API_URL || ''

export async function get(path, params = {}, { signal } = {}) {
  const url = new URL(BASE + path, window.location.origin)
  Object.entries(params).forEach(([key, value]) => value != null && url.searchParams.set(key, value))
  const res = await fetch(url, { signal })
  if (!res.ok) throw new Error(`Request failed (${res.status}). Please try again.`)
  return res.json()
}

export const api = {
  summary: (options) => get('/api/summary', {}, options),
  vehicles: (params, options) => get('/api/vehicles', params, options),
  vehicleDetail: (id, options) => get(`/api/vehicles/${id}`, {}, options),
  makes: (options) => get('/api/vehicles/makes', {}, options),
  stations: (params, options) => get('/api/stations', params, options),
  stationsNearby: (params, options) => get('/api/stations/nearby', params, options),
  networks: (options) => get('/api/stations/networks', {}, options),
  segments: (options) => get('/api/segments', {}, options),
  segmentVehicles: (options) => get('/api/segments/vehicles', {}, options),
  trends: (options) => get('/api/trends', {}, options),
}

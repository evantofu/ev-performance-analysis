const BASE = import.meta.env.VITE_API_URL || ''

async function get(path, params = {}) {
  const url = new URL(BASE + path, window.location.origin)
  Object.entries(params).forEach(([k, v]) => v != null && url.searchParams.set(k, v))
  const res = await fetch(url)
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`)
  return res.json()
}

export const api = {
  summary:        ()       => get('/api/summary'),
  vehicles:       (params) => get('/api/vehicles', params),
  vehicleDetail:  (id)     => get(`/api/vehicles/${id}`),
  makes:          ()       => get('/api/vehicles/makes'),
  stations:       (params) => get('/api/stations', params),
  stationsNearby: (params) => get('/api/stations/nearby', params),
  networks:       ()       => get('/api/stations/networks'),
  segments:       ()       => get('/api/segments'),
  segmentVehicles:()       => get('/api/segments/vehicles'),
  trends:         ()       => get('/api/trends'),
}
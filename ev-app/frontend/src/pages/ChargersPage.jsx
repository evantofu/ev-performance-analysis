import { useState, useEffect, useRef } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import { useApi } from '../hooks/useApi'
import { RequestError } from '../components/RequestError'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

const CHARGER_COLORS = { dc_fast: '#22c55e', level2: '#3b82f6', level1: '#64748b' }

function stationType(s) {
  if ((s.dc_fast_count ?? 0) > 0) return 'dc_fast'
  if ((s.level2_count  ?? 0) > 0) return 'level2'
  return 'level1'
}

async function geocode(query) {
  const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=1&countrycodes=us`
  const res  = await fetch(url, { headers: { 'Accept-Language': 'en' } })
  if (!res.ok) throw new Error('Address lookup failed')
  const data = await res.json()
  if (!data.length) throw new Error('Address not found')
  return { lat: parseFloat(data[0].lat), lon: parseFloat(data[0].lon), label: data[0].display_name }
}

export function ChargersPage() {
  const { stationFilters, setStationFilter } = useStore()
  const { data: networksData, error: networksError } = useApi(signal => api.networks({ signal }), [])

  const [bounds, setBounds] = useState(null)
  const [searchInput, setSearchInput] = useState('')
  const [searchErr,   setSearchErr]   = useState('')
  const [searching,   setSearching]   = useState(false)

  const mapRef  = useRef(null)
  const leafRef = useRef(null)
  const pinRef  = useRef(null)

  const networks = networksData?.map(n => n.network) ?? []

  const { data, loading, error } = useApi(signal => bounds
    ? api.stations({ ...bounds, ...stationFilters, limit: 2000 }, { signal })
    : Promise.resolve(null), [JSON.stringify(bounds), JSON.stringify(stationFilters)])
  const stations = data?.results ?? []
  const total = data?.total ?? 0

  useEffect(() => {
    const map = L.map(mapRef.current, { center: [39.5, -98.35], zoom: 4, worldCopyJump: true })
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '© OpenStreetMap © CARTO', maxZoom: 19,
    }).addTo(map)
    function updateBounds() {
      const b = map.getBounds()
      const wrap = value => ((value + 180) % 360 + 360) % 360 - 180
      const world = b.getEast() - b.getWest() >= 360
      setBounds({ lat_min: Math.max(-90, b.getSouth()), lat_max: Math.min(90, b.getNorth()),
        lon_min: world ? -180 : wrap(b.getWest()), lon_max: world ? 180 : wrap(b.getEast()) })
    }
    leafRef.current = map
    map.on('moveend', updateBounds)
    updateBounds()
    return () => { map.remove(); leafRef.current = null }
  }, [])

  // ── Update markers ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (!leafRef.current) return

    try {
      leafRef.current.eachLayer(l => {
        if (l instanceof L.CircleMarker && l !== pinRef.current) l.remove()
      })
    } catch { return }

    stations.forEach(s => {
      if (!s.latitude || !s.longitude) return
      const type  = stationType(s)
      const color = CHARGER_COLORS[type]
      L.circleMarker([s.latitude, s.longitude], {
        radius: 5, color, fillColor: color, fillOpacity: 0.85, weight: 0.5,
      })
        .bindPopup(() => {
          const content = document.createElement('div')
          const title = document.createElement('strong')
          title.textContent = s.station_name || 'Station'
          content.append(title, document.createElement('br'))
          content.append(document.createTextNode(`${s.network || 'Non-Networked'} · ${s.city || ''}, ${s.state || ''}`))
          content.append(document.createElement('br'), document.createTextNode(
            `L1: ${s.level1_count || 0} · L2: ${s.level2_count || 0} · DC Fast: ${s.dc_fast_count || 0}`))
          return content
        })
        .addTo(leafRef.current)
    })
  }, [stations])

  // ── Address search ─────────────────────────────────────────────────────────
  async function handleSearch(e) {
    e.preventDefault()
    if (!searchInput.trim()) return
    setSearching(true); setSearchErr('')
    try {
      const { lat, lon, label } = await geocode(searchInput)
      if (pinRef.current && leafRef.current) { pinRef.current.remove(); pinRef.current = null }
      if (leafRef.current) {
        leafRef.current.setView([lat, lon], 13)
        const L = window.L
        pinRef.current = L.circleMarker([lat, lon], {
          radius: 8, color: '#f5a623', fillColor: '#f5a623', fillOpacity: 1, weight: 2,
        }).bindPopup(() => { const el = document.createElement('strong'); el.textContent = label.split(',')[0]; return el }).addTo(leafRef.current)
      }
    } catch {
      setSearchErr('Address not found — try a city name or zip code.')
    } finally {
      setSearching(false)
    }
  }

  function handleGeolocate() {
    if (!navigator.geolocation) { setSearchErr('Geolocation not supported by your browser.'); return }
    setSearching(true); setSearchErr('')
    navigator.geolocation.getCurrentPosition(
      pos => {
        const { latitude: lat, longitude: lon } = pos.coords
        if (leafRef.current) leafRef.current.setView([lat, lon], 13)
        setSearching(false)
      },
      () => { setSearchErr('Could not get your location.'); setSearching(false) }
    )
  }

  return (
    <div className="page">
      <RequestError error={error || networksError} />
      <div className="tool-layout chargers-layout">

        {/* Sidebar */}
        <div className="filter-panel">
          <div className="section-header"><h2>Find Chargers</h2></div>

          {/* Address search */}
          <div className="filter-group">
            <label>Search by address</label>
            <form onSubmit={handleSearch} style={{ display: 'flex', gap: 4 }}>
              <input
                type="text"
                value={searchInput}
                onChange={e => setSearchInput(e.target.value)}
                placeholder="City, address, or zip…"
                style={{
                  flex: 1, background: 'var(--surface-2)',
                  border: '1px solid var(--border)', color: 'var(--text)',
                  padding: '6px 10px', borderRadius: 'var(--radius-sm)',
                  fontFamily: 'var(--font-mono)', fontSize: 12,
                }}
              />
              <button type="submit" className="btn primary"
                style={{ padding: '6px 10px', whiteSpace: 'nowrap' }}
                disabled={searching}>
                {searching ? '…' : '→'}
              </button>
            </form>
            {searchErr && (
              <div style={{ fontSize: 10, color: 'var(--red)', marginTop: 4 }}>{searchErr}</div>
            )}
          </div>

          {/* Geolocation */}
          <button className="btn" onClick={handleGeolocate} disabled={searching}
            style={{ width: '100%', justifyContent: 'center' }}>
            📍 Use my location
          </button>

          {/* Filters */}
          <div className="filter-group">
            <label>Network</label>
            <select value={stationFilters.network ?? ''}
              onChange={e => setStationFilter('network', e.target.value || null)}>
              <option value="">All networks</option>
              {networks.map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <div className="filter-group">
            <label>State</label>
            <select value={stationFilters.state ?? ''}
              onChange={e => setStationFilter('state', e.target.value || null)}>
              <option value="">All states</option>
              {['CA','TX','FL','NY','WA','OR','CO','AZ','NV','MA','IL','GA','NC','OH','MI'].map(s =>
                <option key={s} value={s}>{s}</option>
              )}
            </select>
          </div>
          <div className="filter-group">
            <label>
              <input type="checkbox"
                checked={stationFilters.dc_fast_only}
                onChange={e => setStationFilter('dc_fast_only', e.target.checked)}
                style={{ marginRight: 6 }}
              />
              DC Fast only
            </label>
          </div>

          <div className="stat-card">
            <div className="label">Stations in view</div>
            <div className="value">{total.toLocaleString()}</div>
          </div>

          {/* Legend */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {Object.entries(CHARGER_COLORS).map(([type, color]) => (
              <div key={type} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
                <div style={{ width: 9, height: 9, borderRadius: '50%', background: color }} />
                <span style={{ color: 'var(--text-dim)' }}>
                  {type === 'dc_fast' ? 'DC Fast' : type === 'level2' ? 'Level 2' : 'Level 1'}
                </span>
              </div>
            ))}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
              <div style={{ width: 9, height: 9, borderRadius: '50%', background: '#f5a623' }} />
              <span style={{ color: 'var(--text-dim)' }}>Your search</span>
            </div>
          </div>
        </div>

        {/* Map */}
        <div>
          <div className="map-container" style={{ height: 580 }}>
            <div ref={mapRef} style={{ width: '100%', height: '100%' }} />
          </div>
          {loading && (
            <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
              Loading stations…
            </div>
          )}
          {!loading && total >= 2000 && (
            <p style={{ marginTop: 8, fontSize: 10, color: 'var(--text-muted)' }}>
              Showing 2,000 of {total.toLocaleString()} stations — zoom in to see all.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
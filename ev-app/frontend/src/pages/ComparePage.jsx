import { useState, useEffect } from 'react'
import { PlotlyChart } from '../components/PlotlyChart'
import { api } from '../api'
import { useStore } from '../store'
import { MakeGrid } from '../components/MakeGrid'
import { ComparePanel } from '../components/ComparePanel'
import { useApi } from '../hooks/useApi'

const SORT_OPTIONS = [
  { value: 'combined_mpge', label: 'Efficiency' },
  { value: 'range_miles',   label: 'Range' },
  { value: 'year',          label: 'Year (newest)' },
]

const SEG_COLORS = {
  0: '#2E86AB', 1: '#A23B72', 2: '#06A77D', 3: '#C73E1D',
}
const SEG_NAMES = {
  0: 'High Efficiency', 1: 'Mainstream', 2: 'Long Range', 3: 'Performance & SUV',
}

function similarity(a, b) {
  const dMpge  = ((a.combined_mpge || 0) - (b.combined_mpge || 0)) / 50
  const dRange = ((a.range_miles   || 0) - (b.range_miles   || 0)) / 200
  return Math.sqrt(dMpge ** 2 + dRange ** 2)
}

export function ComparePage() {
  const { filters, setFilter, resetFilters, selected } = useStore()
  const { data: makesData } = useApi(() => api.makes(), [])

  // Single fetch — all vehicles matching current filters, no pagination
  const [vehicles, setVehicles] = useState([])
  const [loading,  setLoading]  = useState(true)

  // Scatter dataset — respects year filter but not other filters
  // so you always see the full market context
  const scatterParams = { limit: 2000, sort_by: 'combined_mpge',
                          year: filters.year || null }
  const { data: allData } = useApi(
    () => api.vehicles(scatterParams), [filters.year]
  )
  const allVehicles = allData?.results ?? []

  const [similar, setSimilar] = useState([])
  const makes = makesData?.map(m => m.make) ?? []

  // Fetch ALL matching vehicles on filter change (no pagination)
  useEffect(() => {
    setLoading(true)
    const params = {
      make:        filters.make        || null,
      min_range:   filters.min_range   || null,
      min_mpge:    filters.min_mpge    || null,
      year:        filters.year        || null,
      sort_by:     filters.sort_by,
      sort_desc:   filters.sort_by !== 'msrp_base',
      has_battery: filters.full_specs  ? true : null,
      min_year:    filters.current_only ? new Date().getFullYear() : null,
      limit:       2000,
    }
    api.vehicles(params)
      .then(d => setVehicles(d.results ?? []))
      .finally(() => setLoading(false))
  }, [JSON.stringify(filters)])

  // Similar vehicles when exactly 1 selected
  useEffect(() => {
    if (selected.length !== 1 || allVehicles.length === 0) { setSimilar([]); return }
    const anchor = selected[0]
    const scored = allVehicles
      .filter(v => v.id !== anchor.id)
      .map(v => ({ ...v, _dist: similarity(anchor, v) }))
      .sort((a, b) => a._dist - b._dist)
      .slice(0, 6)
    setSimilar(scored)
  }, [selected, allVehicles])

  // Scatter — all vehicles coloured by segment
  const segIds = [...new Set(allVehicles.map(v => v.cluster).filter(c => c != null))].sort()
  const scatterTraces = segIds.length > 0
    ? segIds.map(cid => {
        const sv = allVehicles.filter(v => Math.round(v.cluster) === cid)
        return {
          type: 'scatter', mode: 'markers',
          name: SEG_NAMES[cid] ?? `Seg ${cid}`,
          x: sv.map(v => v.combined_mpge),
          y: sv.map(v => v.range_miles),
          text: sv.map(v => `${v.year} ${v.make} ${v.model}`),
          marker: { color: SEG_COLORS[cid] ?? '#888', size: 6, opacity: 0.75,
                    line: { color: 'rgba(0,0,0,0.2)', width: 0.5 } },
          hovertemplate: '<b>%{text}</b><br>MPGe: %{x}<br>Range: %{y} mi<extra></extra>',
        }
      })
    : [{
        type: 'scatter', mode: 'markers',
        x: allVehicles.map(v => v.combined_mpge),
        y: allVehicles.map(v => v.range_miles),
        text: allVehicles.map(v => `${v.year} ${v.make} ${v.model}`),
        marker: { color: '#f5a623', size: 6, opacity: 0.75 },
        hovertemplate: '<b>%{text}</b><br>MPGe: %{x}<br>Range: %{y} mi<extra></extra>',
      }]

  const plotLayout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'Inter, system-ui, sans-serif', color: '#7a7670', size: 11 },
    xaxis: { title: 'Efficiency (MPGe)', gridcolor: '#ebe5de', zerolinecolor: '#ebe5de' },
    yaxis: { title: 'Range (miles)',      gridcolor: '#ebe5de', zerolinecolor: '#ebe5de' },
    legend: { bgcolor: 'rgba(255,255,255,0.95)', bordercolor: '#e0d8d0',
              borderwidth: 1, font: { size: 10 }, orientation: 'h', y: -0.32 },
    margin: { l: 48, r: 16, t: 16, b: 110 },
    hovermode: 'closest', showlegend: segIds.length > 0,
  }

  const avgRange = vehicles.length
    ? Math.round(vehicles.reduce((s, v) => s + (v.range_miles || 0), 0) / vehicles.length)
    : null
  const avgMpge = vehicles.length
    ? Math.round(vehicles.reduce((s, v) => s + (v.combined_mpge || 0), 0) / vehicles.length)
    : null

  return (
    <div className="page">
      <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr',
                    gap: 20, alignItems: 'start' }}>

        {/* Sidebar */}
        <div className="filter-panel">
          <div className="section-header"><h2>Filters</h2></div>

          <div className="filter-group">
            <label>Manufacturer</label>
            <select value={filters.make ?? ''}
              onChange={e => setFilter('make', e.target.value)}>
              <option value="">All</option>
              {makes.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>

          <div className="filter-group">
            <label>Min range</label>
            <input type="range" min={0} max={500} step={10}
              value={filters.min_range ?? 0}
              onChange={e => setFilter('min_range', +e.target.value || null)} />
            <div className="range-labels">
              <span>0</span>
              <span className="amber">{filters.min_range ?? 0} mi</span>
            </div>
          </div>

          <div className="filter-group">
            <label>Min efficiency</label>
            <input type="range" min={50} max={150} step={5}
              value={filters.min_mpge ?? 50}
              onChange={e => setFilter('min_mpge', +e.target.value <= 50 ? null : +e.target.value)} />
            <div className="range-labels">
              <span>50</span>
              <span className="amber">{filters.min_mpge ?? 'any'} MPGe</span>
            </div>
          </div>

          <div className="filter-group">
            <label>Model year</label>
            <select value={filters.year ?? ''}
              onChange={e => setFilter('year', +e.target.value || null)}>
              <option value="">All years</option>
              {[2027,2026,2025,2024,2023,2022,2021,2020,2019].map(y =>
                <option key={y} value={y}>{y}</option>)}
            </select>
          </div>

          <div className="filter-group">
            <label>Sort by</label>
            <select value={filters.sort_by}
              onChange={e => setFilter('sort_by', e.target.value)}>
              {SORT_OPTIONS.map(o =>
                <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </div>

          <div className="filter-group">
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
              <input type="checkbox"
                checked={filters.current_only ?? false}
                onChange={e => setFilter('current_only', e.target.checked || null)} />
              Current models only
            </label>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
              {new Date().getFullYear()} and newer
            </div>
          </div>

          <div className="filter-group">
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
              <input type="checkbox"
                checked={filters.full_specs ?? false}
                onChange={e => setFilter('full_specs', e.target.checked || null)} />
              Full specs only
            </label>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
              Battery &amp; charging data available
            </div>
          </div>

          <button className="btn" onClick={resetFilters}
            style={{ width: '100%', justifyContent: 'center' }}>
            Reset filters
          </button>
        </div>

        {/* Main */}
        <div>
          {/* Scatter */}
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="section-header">
              <h2>All EVs — Range vs Efficiency</h2>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                {allVehicles.length.toLocaleString()} vehicles · colours = market segments
              </span>
            </div>
            <PlotlyChart data={scatterTraces} layout={plotLayout} style={{ height: 300 }} />
          </div>

          {/* Stats */}
          <div className="grid-4" style={{ marginBottom: 20 }}>
            {[
              ['Matching',  vehicles.length.toLocaleString()],
              ['Avg range', avgRange ? `${avgRange} mi` : '—'],
              ['Avg MPGe',  avgMpge ?? '—'],
              ['Selected',  `${selected.length} / 2`],
            ].map(([label, value]) => (
              <div key={label} className="stat-card fade-up">
                <div className="label">{label}</div>
                <div className="value" style={{ fontSize: 20 }}>{value}</div>
              </div>
            ))}
          </div>

          {/* Compare panel */}
          {selected.length > 0 && <ComparePanel />}

          {/* Similar vehicles */}
          {selected.length === 1 && similar.length > 0 && (
            <div style={{ marginTop: 24, marginBottom: 8 }}>
              <div className="section-header">
                <h2>Similar to {selected[0].make} {selected[0].model}</h2>
              </div>
              <div style={{ display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(190px, 1fr))', gap: 10 }}>
                {similar.map(v => (
                  <div key={`sim-${v.id}-${v.year}`}
                    style={{ cursor: 'pointer' }}
                    onClick={() => useStore.getState().toggleSelect(v)}>
                    {/* inline mini card for similar */}
                    <div className="vehicle-card">
                      <div className="vehicle-make">{v.make}</div>
                      <div className="vehicle-model">{v.model}</div>
                      <div className="vehicle-year">{v.year}</div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
                        <span className="amber">{v.combined_mpge} MPGe</span>
                        {' · '}
                        <span className="amber">{v.range_miles} mi</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Vehicle grid — grouped by manufacturer */}
          <div className="section-header" style={{ marginTop: 24 }}>
            <h2>
              {vehicles.length.toLocaleString()} vehicles
              {filters.make && ` · ${filters.make}`}
            </h2>
          </div>

          {loading ? (
            <div className="loading">Loading vehicles…</div>
          ) : vehicles.length === 0 ? (
            <div className="empty">No vehicles match your filters.</div>
          ) : (
            <MakeGrid vehicles={vehicles} sortBy={filters.sort_by} />
          )}
        </div>
      </div>
    </div>
  )
}
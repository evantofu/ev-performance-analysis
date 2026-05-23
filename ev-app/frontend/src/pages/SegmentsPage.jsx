import { useState } from 'react'
import { PlotlyChart } from '../components/PlotlyChart'
import { api } from '../api'
import { useApi } from '../hooks/useApi'
import { VehicleCard } from '../components/VehicleCard'
import { MakeGrid } from '../components/MakeGrid'

const COLORS = { 0: '#2E86AB', 1: '#A23B72', 2: '#06A77D', 3: '#C73E1D' }

const PROFILES = {
  0: { name: 'High Efficiency', icon: '⚡',
       desc: 'Top-tier MPGe. Lowest cost per mile to run — city cars, efficient sedans, and compacts.' },
  1: { name: 'Mainstream',      icon: '🚗',
       desc: 'Well-rounded everyday EVs. Solid efficiency and range for most drivers.' },
  2: { name: 'Long Range',      icon: '🛣️',
       desc: '300+ miles per charge. Ideal for road trips and eliminating range anxiety.' },
  3: { name: 'Performance & SUV', icon: '🏎️',
       desc: 'Trucks, large SUVs, and performance EVs. Capability and power over efficiency.' },
}

export function SegmentsPage() {
  const { data: vehicleData, loading } = useApi(() => api.vehicles({ limit: 2000 }), [])
  const { data: scatterData }          = useApi(() => api.segmentVehicles(), [])
  const { data: segData }              = useApi(() => api.segments(), [])

  const [activeSeg, setActiveSeg] = useState(null)

  const vehicles       = vehicleData?.results ?? []
  const scatterVehicles = scatterData ?? []
  const segments       = (segData ?? []).sort((a, b) => a.cluster_id - b.cluster_id)

  const drillVehicles = activeSeg !== null
    ? vehicles.filter(v => Math.round(v.cluster) === activeSeg)
    : []

  // Pre-compute counts from the actual loaded vehicle list so card count
  // matches drill-down count — avoids mismatch with seg.count from API
  const segCounts = {}
  for (const v of vehicles) {
    const c = Math.round(v.cluster)
    if (!isNaN(c)) segCounts[c] = (segCounts[c] ?? 0) + 1
  }

  // Bar chart comparing avg MPGe and avg range per segment
  const segNames  = segments.map(s => PROFILES[s.cluster_id]?.name ?? `Seg ${s.cluster_id}`)
  const segColors = segments.map(s => COLORS[s.cluster_id] ?? '#888')

  const maxMpge  = Math.max(...segments.map(s => s.avg_mpge  ?? 0))
  const maxRange = Math.max(...segments.map(s => s.avg_range ?? 0))

  const barData = [
    {
      type: 'bar', name: 'Avg MPGe',
      x: segNames,
      y: segments.map(s => s.avg_mpge),
      marker: { color: segColors, opacity: 0.9 },
      text: segments.map(s => `${s.avg_mpge}`),
      textposition: 'inside',
      insidetextanchor: 'end',
      textfont: { color: '#fff', size: 12 },
      hovertemplate: '<b>%{x}</b><br>%{y} MPGe<extra></extra>',
    },
  ]

  const barLayout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'Inter, system-ui, sans-serif', color: '#7a7670', size: 11 },
    yaxis: { title: 'Average MPGe', gridcolor: '#ebe5de', zerolinecolor: '#ebe5de',
             range: [0, maxMpge * 1.18] },
    xaxis: { tickfont: { size: 12 } },
    margin: { l: 52, r: 16, t: 36, b: 48 },
    showlegend: false,
    title: { text: 'Efficiency by segment', font: { size: 13, color: '#7a7670' } },
  }

  const rangeData = [
    {
      type: 'bar', name: 'Avg Range',
      x: segNames,
      y: segments.map(s => s.avg_range),
      marker: { color: segColors, opacity: 0.9 },
      text: segments.map(s => `${Math.round(s.avg_range ?? 0)} mi`),
      textposition: 'inside',
      insidetextanchor: 'end',
      textfont: { color: '#fff', size: 12 },
      hovertemplate: '<b>%{x}</b><br>%{y:.0f} miles<extra></extra>',
    },
  ]

  const rangeLayout = {
    ...barLayout,
    yaxis: { title: 'Average Range (miles)', gridcolor: '#ebe5de', zerolinecolor: '#ebe5de',
             range: [0, maxRange * 1.18] },
    title: { text: 'Range by segment', font: { size: 13, color: '#7a7670' } },
  }

  return (
    <div className="page">
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div className="display" style={{ fontSize: 32, fontWeight: 900, marginBottom: 8 }}>
          What kind of EV buyer are you?
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: 13, maxWidth: 560, lineHeight: 1.7 }}>
          {vehicles.length.toLocaleString()} electric vehicles grouped into{' '}
          <strong style={{ color: 'var(--terra)' }}>4 buyer profiles</strong>.
          Click a profile to explore its vehicles.
        </p>
      </div>

      {/* Profile cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12, marginBottom: 28 }}>
        {Object.entries(PROFILES).map(([cid, prof]) => {
          const seg      = segments.find(s => s.cluster_id === +cid)
          const isActive = activeSeg === +cid
          const color    = COLORS[+cid]
          return (
            <div key={cid} className="card"
              style={{
                borderLeftColor: color, borderLeftWidth: 3,  cursor: 'pointer',
                opacity: activeSeg !== null && !isActive ? 0.4 : 1,
                transition: 'all 0.15s',
                transform: isActive ? 'translateY(-2px)' : 'none',
                outline: isActive ? `2px solid ${color}` : 'none',
              }}
              onClick={() => setActiveSeg(isActive ? null : +cid)}
            >
              <div style={{ fontSize: 24, marginBottom: 6 }}>{prof.icon}</div>
              <div className="display" style={{ fontSize: 14, fontWeight: 700, marginBottom: 6, color }}>
                {prof.name}
              </div>
              <p style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: 10 }}>
                {prof.desc}
              </p>
              {seg && (
                <div style={{ display: 'flex', gap: 10, fontSize: 11, marginBottom: 8 }}>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>MPGe </span>
                    <span style={{ color, fontWeight: 600 }}>{seg.avg_mpge}</span>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Range </span>
                    <span style={{ color, fontWeight: 600 }}>
                      {seg.avg_range ? `${Math.round(seg.avg_range)} mi` : '—'}
                    </span>
                  </div>
                </div>
              )}
              <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 8 }}>
                {seg?.top_makes?.map(m => (
                  <span key={m} className="badge badge-blue">{m}</span>
                ))}
              </div>
              <div style={{ fontSize: 10, color: isActive ? color : 'var(--text-muted)' }}>
                {isActive
                  ? `▼ ${drillVehicles.length} vehicles below`
                  : `${segCounts[+cid] ?? '—'} vehicles · click to explore`}
              </div>
            </div>
          )
        })}
      </div>

      {/* Bar charts — use segData loading state, not vehicleData */}
      {segments.length > 0 && (
        <div className="grid-2" style={{ marginBottom: 24 }}>
          <div className="card">
            <PlotlyChart data={barData} layout={barLayout} style={{ height: 260 }} />
          </div>
          <div className="card">
            <PlotlyChart data={rangeData} layout={rangeLayout} style={{ height: 260 }} />
          </div>
        </div>
      )}

      {/* Drill-down */}
      {activeSeg !== null && (
        <div style={{ marginBottom: 24 }}>
          <div className="section-header" style={{ marginBottom: 16 }}>
            <h2>{PROFILES[activeSeg]?.icon} {PROFILES[activeSeg]?.name}</h2>
            <button className="btn" onClick={() => setActiveSeg(null)}>✕ Close</button>
          </div>
          {drillVehicles.length === 0 ? (
            <div className="empty">No vehicles found.</div>
          ) : (
            <MakeGrid vehicles={drillVehicles} />
          )}
        </div>
      )}

      <p style={{ marginTop: 8, fontSize: 10, color: 'var(--text-muted)' }}>
        Segments computed using quantile cuts on deduplicated base model averages.
        Vehicles are grouped by their primary use case — efficiency, balance, range, or performance.
      </p>
    </div>
  )
}
import { useMemo, useState } from 'react'
import { PlotlyChart } from '../components/PlotlyChart'
import { api } from '../api'
import { useApi } from '../hooks/useApi'

const PALETTE = [
  '#f5a623','#22c55e','#3b82f6','#ef4444','#a855f7',
  '#ec4899','#14b8a6','#f97316','#06b6d4','#84cc16',
  '#8b5cf6','#f43f5e','#10b981','#0ea5e9','#d946ef',
  '#fb923c','#4ade80','#60a5fa','#c084fc','#fb7185',
]

export function TrendsPage() {
  const { data: raw, loading } = useApi(() => api.trends(), [])
  // Set of selected makes — empty set means "all visible"
  const [selected, setSelected] = useState(new Set())

  const { traces, makes } = useMemo(() => {
    if (!raw) return { traces: [], makes: [] }

    const byMake = {}
    raw.forEach(row => {
      if (!byMake[row.make]) byMake[row.make] = { years: [], mpge: [] }
      byMake[row.make].years.push(row.year)
      byMake[row.make].mpge.push(row.avg_mpge)
    })

    const sorted = Object.entries(byMake)
      .sort((a, b) => (b[1].mpge.at(-1) ?? 0) - (a[1].mpge.at(-1) ?? 0))
      .slice(0, 20)

    const makes = sorted.map(([make]) => make)
    const anySelected = selected.size > 0

    const traces = sorted.map(([make, d], i) => {
      const isActive = !anySelected || selected.has(make)
      return {
        type: 'scatter',
        mode: 'lines+markers',
        name: make,
        x: d.years,
        y: d.mpge,
        line: {
          color: PALETTE[i % PALETTE.length],
          width: isActive ? (anySelected ? 3 : 2) : 1,
        },
        marker: {
          size: isActive ? (anySelected ? 9 : 6) : 4,
          color: PALETTE[i % PALETTE.length],
        },
        opacity: isActive ? 1 : 0.12,
        hovertemplate: `<b>${make}</b><br>%{x}: %{y:.1f} MPGe<extra></extra>`,
      }
    })

    return { traces, makes }
  }, [raw, selected])

  function toggleMake(make) {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(make)) next.delete(make)
      else next.add(make)
      return next
    })
  }

  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'DM Mono, monospace', color: '#94a3b8', size: 11 },
    xaxis: { title: 'Year', gridcolor: '#1a1f2b', zerolinecolor: '#1a1f2b',
             tickmode: 'linear', dtick: 1 },
    yaxis: { title: 'Average MPGe', gridcolor: '#1a1f2b', zerolinecolor: '#1a1f2b' },
    legend: {
      bgcolor: 'rgba(18,21,28,0.9)', bordercolor: '#1a1f2b',
      borderwidth: 1, font: { size: 10 },
    },
    margin: { l: 52, r: 16, t: 16, b: 52 },
    hovermode: 'closest',
  }

  return (
    <div className="page">
      <div className="section-header">
        <h2>Manufacturer Efficiency Trends</h2>
      </div>

      {makes.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            <button
              className={`btn${selected.size === 0 ? ' primary' : ''}`}
              onClick={() => setSelected(new Set())}
            >
              All
            </button>
            {selected.size > 0 && (
              <button className="btn" onClick={() => setSelected(new Set())}>
                Clear ({selected.size})
              </button>
            )}
            <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>
              Click to select · multiple allowed
            </span>
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {makes.map((make, i) => {
              const isOn = selected.has(make)
              const color = PALETTE[i % PALETTE.length]
              return (
                <button
                  key={make}
                  onClick={() => toggleMake(make)}
                  style={{
                    padding: '4px 12px',
                    borderRadius: 20,
                    border: `1px solid ${isOn ? color : 'var(--border)'}`,
                    background: isOn ? color + '22' : 'none',
                    color: isOn ? color : 'var(--text-muted)',
                    fontSize: 11,
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                    fontFamily: 'var(--font-mono)',
                  }}
                >
                  {make}
                </button>
              )
            })}
          </div>
        </div>
      )}

      <div className="card" style={{ marginBottom: 16 }}>
        {loading ? (
          <div className="loading">Loading trends…</div>
        ) : (
          <PlotlyChart data={traces} layout={layout} style={{ height: 500 }} />
        )}
      </div>

      <p style={{ fontSize: 10, color: 'var(--text-muted)', maxWidth: 640 }}>
        BEV-only · consumer passenger vehicles · ≥4 years of data · R²≥0.3 quality gate.
        Downward trends may reflect lineup expansion into larger variants rather than
        genuine efficiency regression. Top 20 by latest average MPGe shown.
      </p>
    </div>
  )
}
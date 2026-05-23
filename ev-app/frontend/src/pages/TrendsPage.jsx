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

  // top5Mode: true  → highlight top-5 preset (no explicit set needed)
  // top5Mode: false → use `selected` set exclusively
  const [top5Mode, setTop5Mode]   = useState(true)
  const [selected, setSelected]   = useState(new Set())

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
    const top5  = new Set(makes.slice(0, 5))

    const activeSet = top5Mode ? top5 : selected

    const traces = sorted.map(([make, d], i) => {
      const isActive = activeSet.has(make)
      return {
        type: 'scatter',
        mode: 'lines+markers',
        name: make,
        x: d.years,
        y: d.mpge,
        line:   { color: PALETTE[i % PALETTE.length], width: isActive ? 2.5 : 1 },
        marker: { size: isActive ? 7 : 4, color: PALETTE[i % PALETTE.length] },
        opacity: isActive ? 1 : 0.1,
        hovertemplate: `<b>${make}</b><br>%{x}: %{y:.1f} MPGe<extra></extra>`,
      }
    })

    return { traces, makes }
  }, [raw, top5Mode, selected])

  function toggleMake(make, makeIndex) {
    if (top5Mode) {
      // Leaving top5Mode: seed the explicit set from current top-5,
      // then toggle this make in/out of that set.
      const top5 = new Set(makes.slice(0, 5))
      top5.has(make) ? top5.delete(make) : top5.add(make)
      setSelected(top5)
      setTop5Mode(false)
    } else {
      setSelected(prev => {
        const next = new Set(prev)
        next.has(make) ? next.delete(make) : next.add(make)
        return next
      })
    }
  }

  function handleTop5Button() {
    if (top5Mode) {
      // Already in top5 mode → clicking again clears everything (show all dimmed)
      setTop5Mode(false)
      setSelected(new Set())
    } else {
      // Return to top5 preset
      setTop5Mode(true)
      setSelected(new Set())
    }
  }

  const top5Set = new Set(makes.slice(0, 5))
  const activeSet = top5Mode ? top5Set : selected

  const layout = {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'DM Mono, monospace', color: '#7a7670', size: 11 },
    xaxis: { title: 'Year', gridcolor: '#ebe5de', zerolinecolor: '#ebe5de',
             tickmode: 'linear', dtick: 1 },
    yaxis: { title: 'Average MPGe', gridcolor: '#ebe5de', zerolinecolor: '#ebe5de' },
    legend: {
      bgcolor: 'rgba(255,255,255,0.95)', bordercolor: '#ebe5de',
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
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
            <button
              className={`btn${top5Mode ? ' primary' : ''}`}
              onClick={handleTop5Button}
            >
              Top 5
            </button>
            <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>
              Click manufacturers to compare · multiple allowed
            </span>
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {makes.map((make, i) => {
              const isOn  = activeSet.has(make)
              const color = PALETTE[i % PALETTE.length]
              return (
                <button
                  key={make}
                  onClick={() => toggleMake(make, i)}
                  style={{
                    padding: '5px 13px',
                    borderRadius: 20,
                    border: `1px solid ${isOn ? color : 'var(--border)'}`,
                    background: isOn ? color + '22' : 'none',
                    color: isOn ? color : 'var(--text-muted)',
                    fontSize: 12,
                    fontWeight: isOn ? 500 : 400,
                    cursor: 'pointer',
                    transition: 'all 0.12s',
                    fontFamily: 'var(--font-body)',
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
import { useStore } from '../store'

// Only include metrics we have reliable data for.
// msrp_base and fast_charge_minutes removed — not in any free structured source.
// Rows where both vehicles have null are hidden automatically.
const METRICS = [
  { key: 'range_miles',          label: 'Range',           unit: 'mi',   lowerBetter: false },
  { key: 'combined_mpge',        label: 'Efficiency',      unit: 'MPGe', lowerBetter: false },
  { key: 'battery_capacity_kwh', label: 'Battery',         unit: 'kWh',  lowerBetter: false },
  { key: 'max_dc_kw',            label: 'Max DC charge',   unit: 'kW',   lowerBetter: false },
  { key: 'max_ac_kw',            label: 'Max AC charge',   unit: 'kW',   lowerBetter: false },
  { key: 'acceleration_0_60',    label: '0–60 mph',        unit: 's',    lowerBetter: true  },
  { key: 'annual_fuel_cost_usd', label: 'Annual fuel cost',unit: '$',    lowerBetter: true  },
]

function fmt(value, unit) {
  if (value == null) return null        // null signals "hide this cell"
  if (unit === '$')  return `$${Math.round(value).toLocaleString()}`
  return `${value} ${unit}`
}

export function ComparePanel() {
  const { selected, clearSelected } = useStore()
  if (selected.length === 0) return null

  const [a, b] = selected

  // Only show metric rows where at least one vehicle has real data
  const visibleMetrics = METRICS.filter(({ key }) => {
    const va = a?.[key], vb = b?.[key]
    return va != null || vb != null
  })

  return (
    <div className="card" style={{ marginTop: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between',
                    alignItems: 'center', marginBottom: 16 }}>
        <h3 className="display" style={{ fontSize: 16, fontWeight: 600 }}>
          Head-to-head comparison
        </h3>
        <button className="btn" onClick={clearSelected}>Clear</button>
      </div>

      <div style={{ overflowX: 'auto' }}>
        {/* Header */}
        <div className="compare-row"
          style={{ gridTemplateColumns: `160px${b ? ' 1fr 1fr' : ' 1fr'}` }}>
          <div className="compare-metric" />
          <div className="compare-cell"
            style={{ fontFamily: 'var(--font-display)', fontSize: 16, fontWeight: 600 }}>
            {a.year} {a.make} {a.model}
          </div>
          {b && (
            <div className="compare-cell"
              style={{ fontFamily: 'var(--font-display)', fontSize: 16, fontWeight: 600 }}>
              {b.year} {b.make} {b.model}
            </div>
          )}
        </div>

        {visibleMetrics.map(({ key, label, unit, lowerBetter }) => {
          const va = a?.[key], vb = b?.[key]
          const aWins = va != null && vb != null && (lowerBetter ? va < vb : va > vb)
          const bWins = va != null && vb != null && (lowerBetter ? vb < va : vb > va)
          const fmtA = fmt(va, unit), fmtB = fmt(vb, unit)
          return (
            <div key={key} className="compare-row"
              style={{ gridTemplateColumns: `160px${b ? ' 1fr 1fr' : ' 1fr'}` }}>
              <div className="compare-metric">{label}</div>
              <div className={`compare-cell ${aWins ? 'winner' : bWins ? 'loser' : ''}`}>
                {fmtA ?? <span style={{ color: 'var(--text-muted)' }}>—</span>}
                {aWins ? ' ▲' : ''}
              </div>
              {b && (
                <div className={`compare-cell ${bWins ? 'winner' : aWins ? 'loser' : ''}`}>
                  {fmtB ?? <span style={{ color: 'var(--text-muted)' }}>—</span>}
                  {bWins ? ' ▲' : ''}
                </div>
              )}
            </div>
          )
        })}

        {visibleMetrics.length === 0 && (
          <div className="empty">No comparable metrics available for these vehicles.</div>
        )}
      </div>

      {/* Data coverage note */}
      {(a?.battery_capacity_kwh == null || (b && b?.battery_capacity_kwh == null)) && (
        <p style={{ marginTop: 12, fontSize: 10, color: 'var(--text-muted)' }}>
          Some specs unavailable — battery & charging data covers ~229 vehicles
          (Tesla, Hyundai, BMW, Audi, VW, Rivian have best coverage).
          Use the <strong>Full specs</strong> filter to compare only enriched vehicles.
        </p>
      )}
    </div>
  )
}
import { useStore } from '../store'

const MAX_RANGE = 530
const MAX_MPGE  = 150
const GAS_COST_PER_MILE = 0.14  // ~$3.50/gal at 25mpg
const ELEC_COST_PER_KWH = 0.16  // US avg electricity

export function VehicleCard({ vehicle }) {
  const { selected, toggleSelect } = useStore()
  const isSelected = selected.some(v => v.id === vehicle.id)

  const rangePct = Math.min(100, Math.round((vehicle.range_miles  / MAX_RANGE) * 100))
  const mpgePct  = Math.min(100, Math.round((vehicle.combined_mpge / MAX_MPGE)  * 100))

  // Est. annual fuel cost: miles/year ÷ MPGe × $/kWh × 33.7 kWh/gallon-equivalent
  const annualMiles = 13500
  const annualCost = vehicle.combined_mpge
    ? Math.round((annualMiles / vehicle.combined_mpge) * 33.7 * ELEC_COST_PER_KWH / 100) * 100
    : null
  const gasCost = Math.round(annualMiles * GAS_COST_PER_MILE / 100) * 100
  const savings = annualCost ? gasCost - annualCost : null

  return (
    <div
      className={`vehicle-card fade-up${isSelected ? ' selected' : ''}`}
      onClick={() => toggleSelect(vehicle)}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && toggleSelect(vehicle)}
    >
      <div className="vehicle-make">{vehicle.make}</div>
      <div className="vehicle-model">{vehicle.model}</div>
      <div className="vehicle-year">
        {vehicle.year}
        {vehicle.connector_type && (
          <span className="badge badge-blue" style={{ marginLeft: 8 }}>
            {vehicle.connector_type}
          </span>
        )}
      </div>

      <div className="bar-row">
        <div className="bar-label">
          <span>Range</span>
          <span style={{ color: 'var(--terra)', fontWeight: 600 }}>
            {vehicle.range_miles ? Math.round(vehicle.range_miles) : '—'} mi
          </span>
        </div>
        <div className="bar-track">
          <div className="bar-fill" style={{ width: `${rangePct}%` }} />
        </div>
      </div>

      <div className="bar-row">
        <div className="bar-label">
          <span>Efficiency</span>
          <span style={{ color: 'var(--terra)', fontWeight: 600 }}>
            {vehicle.combined_mpge ? Math.round(vehicle.combined_mpge) : '—'} MPGe
          </span>
        </div>
        <div className="bar-track">
          <div className="bar-fill" style={{ width: `${mpgePct}%` }} />
        </div>
      </div>

      {annualCost && (
        <div style={{ marginTop: 10, fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.5 }}>
          ~<strong style={{ color: 'var(--text)', fontWeight: 600 }}>
            ${annualCost.toLocaleString()}
          </strong>/yr electricity
          {savings > 0 && (
            <span style={{ color: 'var(--green)', marginLeft: 6, fontWeight: 500 }}>
              saves ${savings.toLocaleString()} vs gas
            </span>
          )}
        </div>
      )}

      {isSelected && (
        <div className="badge badge-amber" style={{ marginTop: 10 }}>
          ✓ Selected
        </div>
      )}
    </div>
  )
}
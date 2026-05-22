import { useStore } from '../store'

const MAX_RANGE = 530
const MAX_MPGE  = 150

export function VehicleCard({ vehicle }) {
  const { selected, toggleSelect } = useStore()
  const isSelected = selected.some(v => v.id === vehicle.id)

  const rangePct = Math.round((vehicle.range_miles  / MAX_RANGE) * 100)
  const mpgePct  = Math.round((vehicle.combined_mpge / MAX_MPGE)  * 100)

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
          <span className="amber">{vehicle.range_miles ? Math.round(vehicle.range_miles) : '—'} mi</span>
        </div>
        <div className="bar-track">
          <div className="bar-fill" style={{ width: `${rangePct}%` }} />
        </div>
      </div>

      <div className="bar-row">
        <div className="bar-label">
          <span>Efficiency</span>
          <span className="amber">{vehicle.combined_mpge ? Math.round(vehicle.combined_mpge) : '—'} MPGe</span>
        </div>
        <div className="bar-track">
          <div className="bar-fill" style={{ width: `${mpgePct}%` }} />
        </div>
      </div>

      {vehicle.msrp_base && (
        <div style={{ marginTop: 10, fontSize: 12, color: 'var(--text-dim)' }}>
          From <strong style={{ color: 'var(--text)' }}>
            ${vehicle.msrp_base.toLocaleString()}
          </strong>
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
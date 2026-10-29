import { useState } from 'react'
import { useStore } from '../store'
import { OperatingCost } from './OperatingCost'

// ─── VehicleRow ───────────────────────────────────────────────────────────────
function VehicleRow({ vehicle, isSelected, onToggle, compact = false }) {
  return (
    <div
      onClick={onToggle}
      role="button" tabIndex={0}
      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onToggle() } }}
      style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: compact ? '6px 10px' : '10px 12px',
        borderRadius: 6,
        background: isSelected ? 'var(--terra-glow)' : 'transparent',
        border: `1px solid ${isSelected ? 'var(--terra)' : 'var(--border)'}`,
        cursor: 'pointer', transition: 'all 0.12s',
      }}
      onMouseEnter={e => { if (!isSelected) e.currentTarget.style.borderColor = 'var(--border-hover)' }}
      onMouseLeave={e => { if (!isSelected) e.currentTarget.style.borderColor = 'var(--border)' }}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          fontFamily: 'var(--font-display)', fontSize: compact ? 13 : 15,
          fontWeight: 700, color: 'var(--text)', marginBottom: 4,
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
        }}>
          {vehicle.model}
        </div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--terra)' }}>
            {Math.round(vehicle.combined_mpge)} MPGe
          </span>
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--green)' }}>
            {Math.round(vehicle.range_miles)} mi
          </span>
          {vehicle.battery_capacity_kwh && (
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              {Math.round(vehicle.battery_capacity_kwh)} kWh
            </span>
          )}
          <OperatingCost mpge={vehicle.combined_mpge} />
          {vehicle.connector_type && (
            <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>{vehicle.connector_type}</span>
          )}
        </div>
      </div>
      {isSelected && (
        <span style={{ fontSize: 10, color: 'var(--terra)', whiteSpace: 'nowrap' }}>✓ selected</span>
      )}
    </div>
  )
}

// ─── ModelGroup ───────────────────────────────────────────────────────────────
function ModelGroup({ year, base, vehicles, sortBy = 'combined_mpge' }) {
  const { selected, toggleSelect } = useStore()
  const [expanded, setExpanded] = useState(false)

  function trimVal(v) {
    if (sortBy === 'year')        return v.year ?? 0
    if (sortBy === 'range_miles') return v.range_miles ?? 0
    return v.combined_mpge ?? 0
  }

  const best = [...vehicles].sort((a, b) => trimVal(b) - trimVal(a))[0]
  const hasMultiple = vehicles.length > 1
  const isAnySelected = vehicles.some(v => selected.find(s => s.id === v.id))
  const isBestSelected = !!selected.find(s => s.id === best.id)

  const displayName = `${year} ${base}`

  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', gap: 6, alignItems: 'stretch' }}>
        <div style={{ flex: 1 }}>
          <div
            onClick={() => toggleSelect(best)}
            role="button" tabIndex={0}
            onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleSelect(best) } }}
            style={{
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '14px 16px', borderRadius: 8,
              background: isBestSelected ? 'var(--terra-glow)' : 'transparent',
              border: `1px solid ${isBestSelected ? 'var(--terra)' : 'var(--border)'}`,
              cursor: 'pointer', transition: 'all 0.12s',
            }}
            onMouseEnter={e => { if (!isBestSelected) e.currentTarget.style.borderColor = 'var(--border-hover)' }}
            onMouseLeave={e => { if (!isBestSelected) e.currentTarget.style.borderColor = 'var(--border)' }}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{
                fontFamily: 'var(--font-display)', fontSize: 16,
                fontWeight: 700, color: 'var(--text)', marginBottom: 5,
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              }}>
                {displayName}
              </div>
              <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'center' }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--terra)' }}>
                  {Math.round(best.combined_mpge)} MPGe
                </span>
                <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--green)' }}>
                  {Math.round(best.range_miles)} mi
                </span>
                {best.battery_capacity_kwh && (
                  <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    {Math.round(best.battery_capacity_kwh)} kWh
                  </span>
                )}
                <OperatingCost mpge={best.combined_mpge} />
                {hasMultiple && (
                  <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                    {vehicles.length} trims available
                  </span>
                )}
              </div>
            </div>
            {isBestSelected && (
              <span style={{ fontSize: 10, color: 'var(--terra)', whiteSpace: 'nowrap' }}>✓ selected</span>
            )}
          </div>
        </div>

        {hasMultiple && (
          <button
            onClick={() => setExpanded(e => !e)}
            style={{
              padding: '0 10px', borderRadius: 6, cursor: 'pointer',
              border: `1px solid ${isAnySelected ? 'var(--terra)' : 'var(--border)'}`,
              background: expanded ? 'var(--surface-2)' : 'none',
              color: 'var(--text-muted)', fontSize: 10,
              fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap',
              transition: 'all 0.12s',
            }}
            title={`${vehicles.length} trims available`}
          >
            {expanded ? '▲' : '▼'} {vehicles.length} trims
          </button>
        )}
      </div>

      {expanded && hasMultiple && (
        <div style={{
          marginTop: 4, marginLeft: 16,
          display: 'flex', flexDirection: 'column', gap: 4,
          padding: '8px', background: 'var(--surface)',
          border: '1px solid var(--border)', borderRadius: 6,
        }}>
          {[...vehicles]
            .sort((a, b) => trimVal(b) - trimVal(a))
            .map(v => (
              <VehicleRow
                key={v.id}
                vehicle={v}
                isSelected={!!selected.find(s => s.id === v.id)}
                onToggle={() => toggleSelect(v)}
                compact
              />
            ))
          }
        </div>
      )}
    </div>
  )
}

// ─── MakeGrid ─────────────────────────────────────────────────────────────────
export function MakeGrid({ vehicles, defaultExpanded = 0, sortBy = 'combined_mpge' }) {
  // Group: byMake → { "year||base" → [vehicles] }
  const byMake = {}
  for (const v of vehicles) {
    const base = v.base_model || v.model
    const key  = `${v.year}||${base}`
    if (!byMake[v.make])       byMake[v.make] = {}
    if (!byMake[v.make][key])  byMake[v.make][key] = []
    byMake[v.make][key].push(v)
  }

  // Best value for a group of vehicles under the current sort field
  function bestVal(vlist) {
    if (sortBy === 'year') return Math.max(...vlist.map(v => v.year ?? 0))
    if (sortBy === 'range_miles') return Math.max(...vlist.map(v => v.range_miles ?? 0))
    return Math.max(...vlist.map(v => v.combined_mpge ?? 0))
  }

  const makes = Object.entries(byMake)
    .sort((a, b) => {
      const bestA = Math.max(...Object.values(a[1]).map(bestVal))
      const bestB = Math.max(...Object.values(b[1]).map(bestVal))
      return bestB - bestA
    })

  const [expanded, setExpanded] = useState(new Set(
    makes.slice(0, defaultExpanded).map(([make]) => make)
  ))

  function toggle(make) {
    setExpanded(prev => {
      const next = new Set(prev)
      next.has(make) ? next.delete(make) : next.add(make)
      return next
    })
  }

  const totalModelYears = makes.reduce((s, [, m]) => s + Object.keys(m).length, 0)

  return (
    <div>
      {/* Summary + controls */}
      <div style={{
        display: 'flex', gap: 8, marginBottom: 16,
        alignItems: 'center', flexWrap: 'wrap',
      }}>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          {`${totalModelYears} model-year groups · ${vehicles.length} records · ${makes.length} manufacturers`}
        </span>
        <button className="btn" style={{ padding: '4px 10px', fontSize: 10 }}
          onClick={() => setExpanded(new Set(makes.map(([m]) => m)))}>
          Expand all
        </button>
        <button className="btn" style={{ padding: '4px 10px', fontSize: 10 }}
          onClick={() => setExpanded(new Set())}>
          Collapse all
        </button>
      </div>

      {makes.map(([make, modelYearGroups]) => {
        const isOpen    = expanded.has(make)
        const numGroups = Object.keys(modelYearGroups).length

        return (
          <div key={make} style={{ marginBottom: 12 }}>
            <button
              onClick={() => toggle(make)}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                width: '100%', background: 'var(--surface)',
                border: '1px solid var(--border)', borderRadius: 8,
                padding: '10px 16px', cursor: 'pointer', textAlign: 'left',
                transition: 'border-color 0.15s', marginBottom: isOpen ? 8 : 0,
              }}
              onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--border-hover)'}
              onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
            >
              <span style={{
                fontFamily: 'var(--font-display)', fontSize: 16,
                fontWeight: 700, color: 'var(--text)', flex: 1,
              }}>
                {make}
              </span>
              <span className="badge badge-amber">
                {numGroups} model{numGroups !== 1 ? 's' : ''}
              </span>
              <span style={{ color: 'var(--text-muted)', fontSize: 14 }}>
                {isOpen ? '▲' : '▼'}
              </span>
            </button>

            {isOpen && (
              <div style={{ paddingLeft: 8 }}>
                {Object.entries(modelYearGroups)
                  .sort(([keyA, trimsA], [keyB, trimsB]) => {
                    if (sortBy === 'year') {
                      const [yearA] = keyA.split('||')
                      const [yearB] = keyB.split('||')
                      return Number(yearB) - Number(yearA)
                    }
                    const field = sortBy === 'range_miles' ? 'range_miles' : 'combined_mpge'
                    const bestA = Math.max(...trimsA.map(v => v[field] ?? 0))
                    const bestB = Math.max(...trimsB.map(v => v[field] ?? 0))
                    return bestB - bestA
                  })
                  .map(([key, trims]) => {
                    const [year, base] = key.split('||')
                    return (
                      <ModelGroup key={key} year={year} base={base} vehicles={trims} sortBy={sortBy} />
                    )
                  })
                }
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
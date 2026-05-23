import { Link } from 'react-router-dom'
import { api } from '../api'
import { useApi } from '../hooks/useApi'

export function HomePage() {
  const { data: summary } = useApi(() => api.summary(), [])
  const { data: segData }  = useApi(() => api.segments(), [])

  const kpis = [
    { label: 'Current EV models',  value: summary?.total_vehicles?.toLocaleString(), unit: '' },
    { label: 'Manufacturers',      value: summary?.total_manufacturers,               unit: '' },
    { label: 'Avg range',          value: summary?.avg_range_miles,                   unit: 'mi' },
    { label: 'Avg efficiency',     value: summary?.avg_mpge,                          unit: 'MPGe' },
    { label: 'Charging stations',  value: summary?.total_stations?.toLocaleString(),  unit: '' },
    { label: 'DC Fast chargers',   value: summary?.dc_fast_stations?.toLocaleString(), unit: '' },
  ]

  const cards = [
    {
      to: '/compare',
      title: 'Compare EVs',
      desc: 'Browse and filter every current EV by range, efficiency, and manufacturer. Select two to compare head-to-head.',
      tag: '1,200+ vehicles',
    },
    {
      to: '/chargers',
      title: 'Find Chargers',
      desc: 'Interactive map of 80,000+ US charging stations. Pan to explore — stations load as you move.',
      tag: '80k stations',
    },
    {
      to: '/segments',
      title: 'Market Segments',
      desc: 'Four buyer profiles — High Efficiency, Mainstream, Long Range, and Performance. Find your fit.',
      tag: '4 profiles',
    },
    {
      to: '/trends',
      title: 'Efficiency Trends',
      desc: 'See how each manufacturer\'s average efficiency has changed year-over-year since 2019.',
      tag: 'EPA data',
    },
  ]

  return (
    <div className="page">
      {/* Hero */}
      <div style={{ marginBottom: 40, paddingTop: 8 }}>
        <div style={{
          fontFamily: 'var(--font-display)',
          fontSize: 52, fontWeight: 900,
          lineHeight: 1, letterSpacing: '-0.01em',
          marginBottom: 16, color: 'var(--text)',
        }}>
          Find your<br />
          <span style={{ color: 'var(--terra)' }}>electric vehicle.</span>
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: 15, maxWidth: 480, lineHeight: 1.7 }}>
          Real EPA data. 80,000 charging stations. Built for people
          who want honest answers, not marketing copy.
        </p>
      </div>

      {/* KPIs */}
      <div className="grid-3" style={{ marginBottom: 40 }}>
        {kpis.map(({ label, value, unit }) => (
          <div key={label} className="stat-card fade-up">
            <div className="label">{label}</div>
            <div className="value">
              {value ?? '…'}<span className="unit">{unit}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Feature cards */}
      <div className="section-header"><h2>Explore</h2></div>
      <div className="grid-2">
        {cards.map(({ to, title, desc, tag }) => (
          <Link key={to} to={to} style={{ textDecoration: 'none' }}>
            <div className="card" style={{ height: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between',
                            alignItems: 'flex-start', marginBottom: 10 }}>
                <div style={{
                  fontFamily: 'var(--font-display)',
                  fontSize: 22, fontWeight: 700, color: 'var(--text)',
                }}>
                  {title}
                </div>
                <span className="badge badge-amber">{tag}</span>
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: 13, lineHeight: 1.7 }}>{desc}</p>
            </div>
          </Link>
        ))}
      </div>

      <p style={{ marginTop: 32, fontSize: 11, color: 'var(--text-dim)', textAlign: 'center' }}>
        Vehicle data: EPA fueleconomy.gov · Station data: NREL AFDC ·
        Segmentation: quantile cuts on deduplicated base models
      </p>
    </div>
  )
}
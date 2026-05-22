import { Link } from 'react-router-dom'
import { api } from '../api'
import { useApi } from '../hooks/useApi'

export function HomePage() {
  const { data: summary } = useApi(() => api.summary(), [])
  const { data: segData }  = useApi(() => api.segments(), [])
  // Total unique models = sum of segment counts (2022+ deduplicated)
  const uniqueModels = segData?.reduce((s, seg) => s + (seg.count ?? 0), 0)

  const kpis = [
    { label: 'Current EV models', value: summary?.total_vehicles?.toLocaleString(),     unit: '' },
    { label: 'Manufacturers',    value: summary?.total_manufacturers,                   unit: '' },
    { label: 'Avg range',        value: summary?.avg_range_miles,                       unit: 'mi' },
    { label: 'Avg efficiency',   value: summary?.avg_mpge,                              unit: 'MPGe' },
    { label: 'Charging stations',value: summary?.total_stations?.toLocaleString(),      unit: '' },
    { label: 'DC Fast stations', value: summary?.dc_fast_stations?.toLocaleString(),    unit: '' },
  ]

  const cards = [
    {
      to: '/compare',
      title: 'Compare EVs',
      desc: 'Filter, rank, and compare vehicles side-by-side. Scatter plot, bar charts, and head-to-head metrics.',
      tag: '1,600+ vehicles',
    },
    {
      to: '/chargers',
      title: 'Find Chargers',
      desc: 'Interactive map of 80,000+ US charging stations. Filter by network, state, and charger type.',
      tag: '80k stations',
    },
    {
      to: '/segments',
      title: 'Market Segments',
      desc: 'GMM-based clustering reveals distinct EV market segments. Toggle between 2D and 3D exploration.',
      tag: '2D + 3D',
    },
    {
      to: '/trends',
      title: 'Efficiency Trends',
      desc: 'Track how each manufacturer\'s average MPGe has evolved year-over-year since 2019.',
      tag: 'EPA data',
    },
  ]

  return (
    <div className="page">
      {/* Hero */}
      <div style={{ marginBottom: 40, paddingTop: 16 }}>
        <div className="display" style={{ fontSize: 48, fontWeight: 900, lineHeight: 1,
             letterSpacing: '-0.01em', marginBottom: 12 }}>
          <span style={{ color: 'var(--amber)' }}>ELECTRIC</span><br />
          VEHICLE EXPLORER
        </div>
        <p style={{ color: 'var(--text-muted)', fontSize: 14, maxWidth: 480 }}>
          Real EPA data. 80,000 charging stations. GMM segmentation.
          Built for consumers who want to understand the EV market — not just browse a spec sheet.
        </p>
      </div>

      {/* KPI strip */}
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
            <div className="vehicle-card" style={{ height: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between',
                            alignItems: 'flex-start', marginBottom: 8 }}>
                <div className="display" style={{ fontSize: 20, fontWeight: 700 }}>{title}</div>
                <span className="badge badge-amber">{tag}</span>
              </div>
              <p style={{ color: 'var(--text-muted)', fontSize: 12, lineHeight: 1.6 }}>{desc}</p>
            </div>
          </Link>
        ))}
      </div>

      <p style={{ marginTop: 32, fontSize: 10, color: 'var(--text-muted)', textAlign: 'center' }}>
        Vehicle data: EPA fueleconomy.gov · Station data: NREL AFDC ·
        Segmentation: Gaussian Mixture Model with BIC model selection
      </p>
    </div>
  )
}
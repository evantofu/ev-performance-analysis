import { NavLink } from 'react-router-dom'

export function Topbar() {
  return (
    <header className="topbar">
      <NavLink to="/" className="logo">
        EV <span>Explorer</span>
      </NavLink>
      <nav className="nav">
        {[
          ['/compare',  'Compare'],
          ['/chargers', 'Chargers'],
          ['/segments', 'Segments'],
          ['/trends',   'Trends'],
        ].map(([to, label]) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            {label}
          </NavLink>
        ))}
      </nav>
    </header>
  )
}
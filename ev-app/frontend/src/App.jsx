import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Topbar }       from './components/Topbar'
import { HomePage }     from './pages/HomePage'
import { ComparePage }  from './pages/ComparePage'
import { ChargersPage } from './pages/ChargersPage'
import { SegmentsPage } from './pages/SegmentsPage'
import { TrendsPage }   from './pages/TrendsPage'
import './index.css'

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Topbar />
        <main>
          <Routes>
            <Route path="/"         element={<HomePage />} />
            <Route path="/compare"  element={<ComparePage />} />
            <Route path="/chargers" element={<ChargersPage />} />
            <Route path="/segments" element={<SegmentsPage />} />
            <Route path="/trends"   element={<TrendsPage />} />
            <Route path="*"         element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

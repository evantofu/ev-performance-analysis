import { lazy, Suspense } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Topbar }       from './components/Topbar'
import { HomePage }     from './pages/HomePage'
const ComparePage = lazy(() => import('./pages/ComparePage').then(module => ({ default: module.ComparePage })))
const ChargersPage = lazy(() => import('./pages/ChargersPage').then(module => ({ default: module.ChargersPage })))
const SegmentsPage = lazy(() => import('./pages/SegmentsPage').then(module => ({ default: module.SegmentsPage })))
const TrendsPage = lazy(() => import('./pages/TrendsPage').then(module => ({ default: module.TrendsPage })))
import './index.css'

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Topbar />
        <main>
          <Suspense fallback={<div className="loading">Loading page…</div>}>
          <Routes>
            <Route path="/"         element={<HomePage />} />
            <Route path="/compare"  element={<ComparePage />} />
            <Route path="/chargers" element={<ChargersPage />} />
            <Route path="/segments" element={<SegmentsPage />} />
            <Route path="/trends"   element={<TrendsPage />} />
            <Route path="*"         element={<Navigate to="/" replace />} />
          </Routes>
          </Suspense>
        </main>
      </div>
    </BrowserRouter>
  )
}

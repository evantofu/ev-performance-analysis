import { useState, useEffect, useRef } from 'react'

/** fetchFn receives an AbortSignal; only the latest mounted request may update state. */
export function useApi(fetchFn, deps = []) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const fetchRef = useRef(fetchFn)
  fetchRef.current = fetchFn

  useEffect(() => {
    const controller = new AbortController()
    let active = true
    setLoading(true)
    setError(null)
    Promise.resolve().then(() => fetchRef.current(controller.signal))
      .then(result => { if (active) setData(result) })
      .catch(err => { if (active && err.name !== 'AbortError') setError(err.message) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false; controller.abort() }
    // Callers declare the values that should trigger a new request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { data, loading, error }
}

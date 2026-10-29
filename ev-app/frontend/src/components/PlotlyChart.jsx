import { useEffect, useRef } from 'react'
import Plotly from 'plotly.js-basic-dist-min'
const DEFAULT_CONFIG = {}

export function PlotlyChart({ data, layout, config = DEFAULT_CONFIG, style = {} }) {
  const ref = useRef(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return                          // unmounted before effect ran

    Plotly.react(el, data ?? [], layout ?? {}, {
      displayModeBar: false,
      responsive: true,
      ...config,
    })

  }, [data, layout, config])

  useEffect(() => {
    const element = ref.current
    return () => { if (element) Plotly.purge(element) }
  }, [])

  return <div ref={ref} style={{ width: '100%', ...style }} />
}
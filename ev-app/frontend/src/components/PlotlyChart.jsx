import { useEffect, useRef } from 'react'
import Plotly from 'plotly.js-dist-min'

export function PlotlyChart({ data, layout, config = {}, style = {} }) {
  const ref = useRef(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return                          // unmounted before effect ran

    Plotly.react(el, data ?? [], layout ?? {}, {
      displayModeBar: false,
      responsive: true,
      ...config,
    })

    return () => {
      if (ref.current) Plotly.purge(ref.current)   // only purge if still mounted
    }
  }, [data, layout])

  return <div ref={ref} style={{ width: '100%', ...style }} />
}
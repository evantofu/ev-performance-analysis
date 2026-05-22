import { useRef, useMemo, useEffect, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

const COLORS = ['#f5a623', '#22c55e', '#3b82f6', '#ef4444', '#a855f7']

function norm(arr) {
  const mn = Math.min(...arr), mx = Math.max(...arr)
  return arr.map(v => mx > mn ? (v - mn) / (mx - mn) : 0.5)
}

function jitter(i, scale) {
  return ((Math.sin(i * 127.1 + 311.7) * 43758.5453) % 1) * scale
}

// Orbit controller
function OrbitHandler() {
  const { camera, gl } = useThree()
  const drag = useRef(false)
  const last = useRef({ x: 0, y: 0 })
  const sph  = useRef({ theta: 0.5, phi: 1.2, radius: 6 })
  useEffect(() => {
    const c = gl.domElement
    const dn = e => { drag.current = true; last.current = { x: e.clientX, y: e.clientY } }
    const up = () => { drag.current = false }
    const mv = e => {
      if (!drag.current) return
      sph.current.theta -= (e.clientX - last.current.x) * 0.01
      sph.current.phi = Math.max(0.1, Math.min(Math.PI - 0.1, sph.current.phi + (e.clientY - last.current.y) * 0.01))
      last.current = { x: e.clientX, y: e.clientY }
    }
    const wh = e => { sph.current.radius = Math.max(3, Math.min(14, sph.current.radius + e.deltaY * 0.01)) }
    c.addEventListener('pointerdown', dn); c.addEventListener('pointerup', up)
    c.addEventListener('pointermove', mv); c.addEventListener('wheel', wh, { passive: true })
    return () => {
      c.removeEventListener('pointerdown', dn); c.removeEventListener('pointerup', up)
      c.removeEventListener('pointermove', mv); c.removeEventListener('wheel', wh)
    }
  }, [gl])
  useFrame(() => {
    const { theta, phi, radius } = sph.current
    camera.position.set(
      radius * Math.sin(phi) * Math.sin(theta),
      radius * Math.cos(phi),
      radius * Math.sin(phi) * Math.cos(theta)
    )
    camera.lookAt(0, 0, 0)
  })
  return null
}

// Points with raycasting hover
function Points({ vehicles, scaledPositions, onHover }) {
  const meshRef  = useRef()
  const { camera, gl, size } = useThree()
  const raycaster = useMemo(() => new THREE.Raycaster(), [])
  raycaster.params.Points = { threshold: 0.15 }

  const { positions, colors } = useMemo(() => {
    const positions = new Float32Array(scaledPositions.flat())
    const colors    = new Float32Array(vehicles.length * 3)
    const col       = new THREE.Color()
    vehicles.forEach((v, i) => {
      col.set(COLORS[(v.cluster ?? 0) % COLORS.length])
      colors[i * 3] = col.r; colors[i * 3 + 1] = col.g; colors[i * 3 + 2] = col.b
    })
    return { positions, colors }
  }, [vehicles, scaledPositions])

  useEffect(() => {
    const canvas = gl.domElement
    const onMove = e => {
      if (!meshRef.current) return
      const rect = canvas.getBoundingClientRect()
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width)  *  2 - 1,
        ((e.clientY - rect.top)  / rect.height) * -2 + 1
      )
      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObject(meshRef.current)
      if (hits.length > 0) {
        const idx = hits[0].index
        onHover({ vehicle: vehicles[idx], x: e.clientX, y: e.clientY })
      } else {
        onHover(null)
      }
    }
    canvas.addEventListener('pointermove', onMove)
    return () => canvas.removeEventListener('pointermove', onMove)
  }, [gl, camera, vehicles, raycaster, onHover])

  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color"    args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.08} vertexColors transparent opacity={0.9} sizeAttenuation />
    </points>
  )
}

function Axes() {
  const lines = [
    { from: [-2,-2,-2], to: [2,-2,-2],  color: '#f5a623' },
    { from: [-2,-2,-2], to: [-2,2,-2],  color: '#22c55e' },
    { from: [-2,-2,-2], to: [-2,-2, 2], color: '#3b82f6' },
  ]
  return (
    <>
      {lines.map(({ from, to, color }, i) => {
        const geo = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(...from), new THREE.Vector3(...to)
        ])
        return <lineSegments key={i} geometry={geo}><lineBasicMaterial color={color} /></lineSegments>
      })}
    </>
  )
}

function Grid() {
  const geo = useMemo(() => {
    const pts = []
    for (let i = -2; i <= 2; i++) {
      pts.push(new THREE.Vector3(-2,-2,i), new THREE.Vector3(2,-2,i))
      pts.push(new THREE.Vector3(i,-2,-2), new THREE.Vector3(i,-2,2))
    }
    return new THREE.BufferGeometry().setFromPoints(pts)
  }, [])
  return <lineSegments geometry={geo}><lineBasicMaterial color="#1e2435" /></lineSegments>
}

export function Segment3D({ vehicles }) {
  const [hovered, setHovered] = useState(null)   // { vehicle, x, y }

  const active = vehicles.filter(v => v.combined_mpge != null && v.range_miles != null)
  const clusters = [...new Set(active.map(v => v.cluster ?? 0))].sort()
  const yearMin = Math.min(...active.map(v => v.year ?? 2022))
  const yearMax = Math.max(...active.map(v => v.year ?? 2022))

  // Pre-compute scaled positions so Points and raycasting share the same coords
  const scaledPositions = useMemo(() => {
    const xs   = norm(active.map(v => v.combined_mpge))
    const ys   = norm(active.map(v => v.range_miles))
    const rawZ = norm(active.map(v => v.year ?? 2022))
    const zs   = rawZ.map((z, i) => z + jitter(i, 0.04))
    return active.map((_, i) => [
      (xs[i] - 0.5) * 4,
      (ys[i] - 0.5) * 4,
      (zs[i] - 0.5) * 4,
    ])
  }, [active])

  if (active.length < 10) {
    return <div className="empty">No data available for 3D view.</div>
  }

  return (
    <div>
      {/* Cluster legend */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 10, flexWrap: 'wrap' }}>
        {clusters.map(cid => (
          <div key={cid} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
            <div style={{ width: 9, height: 9, borderRadius: '50%',
                          background: COLORS[cid % COLORS.length] }} />
            <span style={{ color: 'var(--text-dim)' }}>Segment {cid}</span>
          </div>
        ))}
      </div>

      {/* Canvas */}
      <div style={{ position: 'relative', height: 460,
                    borderRadius: 'var(--radius-lg)', overflow: 'hidden',
                    border: '1px solid var(--border)' }}>
        <Canvas camera={{ position: [4, 3, 5], fov: 50 }}>
          <color attach="background" args={['#12151c']} />
          <ambientLight intensity={0.8} />
          <Grid />
          <Axes />
          <Points
            vehicles={active}
            scaledPositions={scaledPositions}
            onHover={setHovered}
          />
          <OrbitHandler />
        </Canvas>

        {/* Axis legend */}
        <div style={{
          position: 'absolute', bottom: 12, left: 12,
          background: 'rgba(10,12,16,0.92)', border: '1px solid var(--border)',
          borderRadius: 6, padding: '8px 12px', fontSize: 10, lineHeight: 1.9,
          fontFamily: 'var(--font-mono)', pointerEvents: 'none',
        }}>
          <div style={{ color: 'var(--text-muted)', fontSize: 9,
                        textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 2 }}>
            Axes
          </div>
          <div><span style={{ color: '#f5a623', fontWeight: 700 }}>——</span>
               <span style={{ color: 'var(--text-dim)', marginLeft: 6 }}>X · Efficiency (MPGe)</span></div>
          <div><span style={{ color: '#22c55e', fontWeight: 700 }}>——</span>
               <span style={{ color: 'var(--text-dim)', marginLeft: 6 }}>Y · Range (miles)</span></div>
          <div><span style={{ color: '#3b82f6', fontWeight: 700 }}>——</span>
               <span style={{ color: 'var(--text-dim)', marginLeft: 6 }}>Z · Model year ({yearMin}–{yearMax})</span></div>
        </div>

        {/* Interaction hint */}
        <div style={{
          position: 'absolute', top: 10, right: 12,
          fontSize: 10, color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)', pointerEvents: 'none',
        }}>
          Drag to rotate · Scroll to zoom · Hover dots for details
        </div>

        {/* Hover tooltip */}
        {hovered && (
          <div style={{
            position: 'fixed',
            left: hovered.x + 14,
            top:  hovered.y - 10,
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 6, padding: '8px 12px',
            fontSize: 11, lineHeight: 1.7,
            fontFamily: 'var(--font-mono)',
            pointerEvents: 'none', zIndex: 1000,
            whiteSpace: 'nowrap',
            boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
          }}>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: 13,
                          fontWeight: 700, marginBottom: 4, color: 'var(--text)' }}>
              {hovered.vehicle.year} {hovered.vehicle.make} {hovered.vehicle.model}
            </div>
            <div style={{ color: 'var(--text-muted)' }}>
              <span style={{ color: '#f5a623' }}>{hovered.vehicle.combined_mpge} MPGe</span>
              {' · '}
              <span style={{ color: '#22c55e' }}>{hovered.vehicle.range_miles} mi</span>
              {' · '}
              <span style={{ color: '#3b82f6' }}>{hovered.vehicle.year}</span>
            </div>
            {hovered.vehicle.battery_capacity_kwh && (
              <div style={{ color: 'var(--text-muted)' }}>
                Battery: {hovered.vehicle.battery_capacity_kwh} kWh
              </div>
            )}
            {hovered.vehicle.max_dc_kw && (
              <div style={{ color: 'var(--text-muted)' }}>
                Max DC: {hovered.vehicle.max_dc_kw} kW
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
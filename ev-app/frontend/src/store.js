import { create } from 'zustand'

export const useStore = create((set, get) => ({
  // ── Vehicle filters ─────────────────────────────────────────────────────
  filters: {
    make:         null,
    min_range:    null,
    max_price:    null,
    min_mpge:     null,
    year:         null,
    current_only: null,
    full_specs:   null,
    sort_by:      'combined_mpge',
    sort_desc:    true,
  },
  setFilter: (key, value) =>
    set(s => ({ filters: { ...s.filters, [key]: value || null } })),
  resetFilters: () =>
    set({ filters: { make: null, min_range: null, max_price: null,
                     min_mpge: null, year: null, current_only: null,
                     full_specs: null,
                     sort_by: 'combined_mpge', sort_desc: true } }),

  // ── Selected vehicles for comparison ────────────────────────────────────
  selected: [],   // array of vehicle objects, max 2
  toggleSelect: (vehicle) => {
    const { selected } = get()
    const exists = selected.find(v => v.id === vehicle.id)
    if (exists) {
      set({ selected: selected.filter(v => v.id !== vehicle.id) })
    } else {
      const next = selected.length >= 2
        ? [selected[1], vehicle]   // drop oldest
        : [...selected, vehicle]
      set({ selected: next })
    }
  },
  clearSelected: () => set({ selected: [] }),

  // ── Station filters ──────────────────────────────────────────────────────
  stationFilters: {
    network:      null,
    dc_fast_only: false,
    state:        null,
  },
  setStationFilter: (key, value) =>
    set(s => ({ stationFilters: { ...s.stationFilters, [key]: value } })),

  // ── UI state ─────────────────────────────────────────────────────────────
  segmentView: '2d',   // '2d' | '3d'
  setSegmentView: (v) => set({ segmentView: v }),
}))
// Explicit assumptions; this is a comparison estimate, not the EPA's own cost field.
export function annualElectricityCost(mpge, miles = 13500, rate = 0.16) {
  return Number.isFinite(mpge) && mpge > 0 ? Math.round(miles / mpge * 33.7 * rate) : null
}

export function OperatingCost({ mpge }) {
  const cost = annualElectricityCost(mpge)
  return cost == null ? null : (
    <span className="operating-cost" title="13,500 miles/year · $0.16/kWh · gas comparison: $3.50/gallon at 25 mpg">
      ~${cost.toLocaleString()}/yr electricity
      {1890 > cost && ` · saves ~$${(1890-cost).toLocaleString()} vs gas`}
    </span>
  )
}

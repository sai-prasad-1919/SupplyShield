import React from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchSupplier, fetchSupplierShipments } from '../../lib/api'
import { RiskBadge } from '../../components/RiskBadge'

export const Route = createFileRoute('/_authenticated/suppliers/$id')({
  component: SupplierDetail,
})

// ── Metric tile ──────────────────────────────────────────────────────────────
function MetricTile({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="glass-card p-4 flex flex-col gap-1">
      <div className="text-xs text-slate-400 uppercase tracking-wider">{label}</div>
      <div className="text-2xl font-bold text-white">{value}</div>
      {sub && <div className="text-xs text-slate-500">{sub}</div>}
    </div>
  )
}

// ── Reliability bar ──────────────────────────────────────────────────────────
function ReliabilityBar({ rate }: { rate: number }) {
  const pct = Math.round(rate * 100)
  const color = pct >= 70 ? '#22c55e' : pct >= 45 ? '#f59e0b' : '#ef4444'
  return (
    <div className="flex items-center gap-3 mt-1">
      <div className="flex-1 h-2 rounded-full bg-slate-700/60 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-sm font-semibold" style={{ color }}>{pct}%</span>
    </div>
  )
}

// ── Info row ─────────────────────────────────────────────────────────────────
function InfoRow({ label, value }: { label: string; value?: string | null }) {
  return (
    <div className="flex justify-between py-2.5 border-b border-border/50 last:border-0">
      <span className="text-slate-400 text-sm">{label}</span>
      <span className="text-white text-sm font-medium">{value ?? '—'}</span>
    </div>
  )
}

// ── Risk colours map (inline for the detail card) ────────────────────────────
const RISK_BG: Record<string, string> = {
  High:   'oklch(0.28 0.08 15)',
  Medium: 'oklch(0.28 0.08 60)',
  Low:    'oklch(0.25 0.05 145)',
}

// ── Main component ───────────────────────────────────────────────────────────
function SupplierDetail() {
  const { id } = Route.useParams()

  const { data: supplier, isLoading: supLoading } = useQuery({
    queryKey: ['supplier', id],
    queryFn:  () => fetchSupplier(id),
  })

  const { data: shipmentsData, isLoading: shipsLoading } = useQuery({
    queryKey: ['supplier-shipments', id],
    queryFn:  () => fetchSupplierShipments(id),
    enabled:  !!supplier,
  })

  if (supLoading) {
    return <div className="p-12 text-center text-slate-400 animate-pulse">Loading supplier profile…</div>
  }

  if (!supplier) {
    return (
      <div className="p-12 text-center text-slate-400">
        Supplier not found. <Link to="/suppliers" className="text-primary hover:underline">Go back</Link>
      </div>
    )
  }

  const m = supplier.metrics
  const shipments: any[] = shipmentsData?.shipments ?? []

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Back link */}
      <Link to="/suppliers" className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-white transition-colors w-fit">
        ← Back to Suppliers
      </Link>

      {/* Header card */}
      <div
        className="rounded-2xl p-6 border border-border"
        style={{ background: RISK_BG[supplier.risk_level] ?? 'oklch(0.18 0.02 250)' }}
      >
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <h1 className="text-2xl font-bold text-white">{supplier.name}</h1>
              {m.shipment_count > 0
                ? <RiskBadge level={supplier.risk_level} />
                : <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-700 text-slate-400">Unrated</span>
              }
            </div>
            <p className="text-slate-300 text-sm">
              {supplier.category} · <span className="capitalize">{supplier.region}</span> Region
            </p>
          </div>
          <div className="text-sm text-slate-400 space-y-0.5 md:text-right">
            <div>{supplier.contact_name}</div>
            <div>{supplier.contact_email}</div>
            <div>{supplier.contact_phone}</div>
          </div>
        </div>

        {/* On-time rate bar */}
        {m.shipment_count > 0 && (
          <div className="mt-5">
            <div className="text-xs text-slate-400 mb-1">Overall On-Time Delivery Rate</div>
            <ReliabilityBar rate={m.on_time_rate} />
          </div>
        )}
      </div>

      {/* Performance metrics grid */}
      <div>
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Performance Metrics</h2>
        {m.shipment_count === 0 ? (
          <div className="glass-card p-6 text-center text-slate-500 text-sm border-2 border-dashed border-border">
            No shipments linked to this supplier yet.
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricTile
              label="Total Shipments"
              value={m.shipment_count}
            />
            <MetricTile
              label="On-Time Deliveries"
              value={m.on_time_count}
              sub={`${Math.round(m.on_time_rate * 100)}% on-time rate`}
            />
            <MetricTile
              label="Delayed Shipments"
              value={m.delayed_count}
              sub={`${Math.round(m.delay_percentage * 100)}% delay rate`}
            />
            <MetricTile
              label="Avg Lead Time"
              value={`${supplier.lead_time_days}d`}
              sub="order to dispatch"
            />
            <MetricTile
              label="Avg Delivery Rating"
              value={m.avg_delivery_rating > 0 ? `★ ${m.avg_delivery_rating.toFixed(1)}` : '—'}
            />
            <MetricTile
              label="Avg Delivery Cost"
              value={m.avg_delivery_cost > 0 ? `₹${m.avg_delivery_cost.toFixed(0)}` : '—'}
            />
            <MetricTile
              label="Avg Delivery Time"
              value={m.avg_delivery_time_hrs > 0 ? `${m.avg_delivery_time_hrs.toFixed(1)}h` : '—'}
            />
            <MetricTile
              label="Avg Delay (when late)"
              value={m.avg_delay_hours > 0 ? `${m.avg_delay_hours.toFixed(1)}h` : '0h'}
              sub="time past expected"
            />
          </div>
        )}
      </div>

      {/* Supplier profile details */}
      <div className="glass-card p-6">
        <h2 className="text-base font-semibold text-white mb-3 border-b border-border pb-2">Supplier Profile</h2>
        <InfoRow label="Company Name"  value={supplier.name} />
        <InfoRow label="Category"      value={supplier.category} />
        <InfoRow label="Region"        value={supplier.region?.charAt(0).toUpperCase() + supplier.region?.slice(1)} />
        <InfoRow label="Lead Time"     value={`${supplier.lead_time_days} days`} />
        <InfoRow label="Contact"       value={supplier.contact_name} />
        <InfoRow label="Email"         value={supplier.contact_email} />
        <InfoRow label="Phone"         value={supplier.contact_phone} />
        <InfoRow label="Status"        value={supplier.active ? 'Active' : 'Inactive'} />
        <InfoRow label="Joined"        value={supplier.joined_at ? new Date(supplier.joined_at).toLocaleDateString('en-IN', { year: 'numeric', month: 'long', day: 'numeric' }) : null} />
      </div>

      {/* Recent shipments */}
      <div className="glass-card overflow-hidden">
        <div className="p-4 border-b border-border flex items-center justify-between">
          <h2 className="font-semibold text-white">Recent Shipments</h2>
          <span className="text-xs text-slate-400">{shipmentsData?.total ?? 0} linked · showing last 20</span>
        </div>

        {shipsLoading ? (
          <div className="p-8 text-center text-slate-400 animate-pulse">Loading shipments…</div>
        ) : shipments.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-sm">No shipments linked to this supplier.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-slate-400 text-xs uppercase tracking-wider"
                style={{ background: 'oklch(0.16 0.02 250)' }}>
                <tr>
                  <th className="px-4 py-3 font-medium">Shipment ID</th>
                  <th className="px-4 py-3 font-medium">Route</th>
                  <th className="px-4 py-3 font-medium">Package</th>
                  <th className="px-4 py-3 font-medium">Delay Prob.</th>
                  <th className="px-4 py-3 font-medium">Risk</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {shipments.map((s: any) => (
                  <tr key={s.id} className="hover:bg-white/[0.02] transition-colors">
                    <td className="px-4 py-3 font-mono text-primary text-xs">
                      <Link to={`/shipments/${s.id}`} className="hover:underline">{s.id}</Link>
                    </td>
                    <td className="px-4 py-3 text-slate-300">{s.route}</td>
                    <td className="px-4 py-3 text-slate-300">{s.package_type}</td>
                    <td className="px-4 py-3 text-slate-200 font-mono text-xs">
                      {(s.prediction.delay_probability * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3">
                      <RiskBadge level={s.prediction.risk_level} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

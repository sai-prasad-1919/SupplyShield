import React, { useState } from 'react'
import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchSuppliers, fetchSupplierKpis } from '../../lib/api'
import { RiskBadge } from '../../components/RiskBadge'
import { KpiCard } from '../../components/KpiCard'

export const Route = createFileRoute('/_authenticated/suppliers')({
  component: SuppliersPage,
})

// ── Reliability bar component ────────────────────────────────────────────────
function ReliabilityBar({ rate }: { rate: number }) {
  const pct = Math.round(rate * 100)
  const color =
    pct >= 70 ? '#22c55e'   // green
    : pct >= 45 ? '#f59e0b' // amber
    : '#ef4444'              // red

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-slate-700/60 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-xs font-mono text-slate-300 w-9 text-right">{pct}%</span>
    </div>
  )
}

// ── Region pill ──────────────────────────────────────────────────────────────
const REGION_COLORS: Record<string, string> = {
  north:   'bg-blue-500/15 text-blue-300',
  south:   'bg-purple-500/15 text-purple-300',
  east:    'bg-orange-500/15 text-orange-300',
  west:    'bg-teal-500/15 text-teal-300',
  central: 'bg-pink-500/15 text-pink-300',
}

function RegionPill({ region }: { region: string }) {
  const cls = REGION_COLORS[region.toLowerCase()] ?? 'bg-slate-700/40 text-slate-400'
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium capitalize ${cls}`}>
      {region}
    </span>
  )
}

// ── Main page ────────────────────────────────────────────────────────────────
function SuppliersPage() {
  const navigate = useNavigate()
  const [search,    setSearch]    = useState('')
  const [catFilter, setCatFilter] = useState('All')
  const [riskFilter,setRiskFilter]= useState('All')

  const { data: kpisData, isLoading: kpisLoading } = useQuery({
    queryKey: ['supplier-kpis'],
    queryFn:  fetchSupplierKpis,
  })

  const { data: suppliersData, isLoading: suppliersLoading } = useQuery({
    queryKey: ['suppliers'],
    queryFn:  fetchSuppliers,
  })

  const suppliers: any[] = suppliersData?.suppliers ?? []

  // derive unique categories for the filter dropdown
  const categories = ['All', ...Array.from(new Set(suppliers.map((s: any) => s.category))).sort()]

  const filtered = suppliers.filter((s: any) => {
    const matchSearch = !search ||
      s.name.toLowerCase().includes(search.toLowerCase()) ||
      s.category.toLowerCase().includes(search.toLowerCase())
    const matchCat  = catFilter  === 'All' || s.category  === catFilter
    const matchRisk = riskFilter === 'All' || s.risk_level === riskFilter
    return matchSearch && matchCat && matchRisk
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Supplier Intelligence</h1>
          <p className="text-sm text-slate-400 mt-1">
            Analytical performance view across all active suppliers
          </p>
        </div>
        <div className="text-xs text-slate-500 px-3 py-1.5 rounded-md"
          style={{ background: 'oklch(0.20 0.02 250)' }}>
          {suppliersData?.total ?? '—'} Active Suppliers
        </div>
      </div>

      {/* KPI Strip */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <KpiCard
          label="Total Suppliers"
          value={kpisLoading ? '…' : kpisData?.total_suppliers ?? '—'}
        />
        <KpiCard
          label="High-Risk Suppliers"
          value={kpisLoading ? '…' : kpisData?.high_risk_count ?? '—'}
        />
        <KpiCard
          label="Avg On-Time Rate"
          value={kpisLoading ? '…' : `${Math.round((kpisData?.avg_on_time_rate ?? 0) * 100)}%`}
        />
      </div>

      {/* Filters */}
      <div className="glass-card p-4 flex flex-wrap gap-3 items-center">
        <input
          type="text"
          placeholder="Search supplier or category…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="flex-1 min-w-[200px] bg-transparent border border-border rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-primary"
        />
        <select
          value={catFilter}
          onChange={e => setCatFilter(e.target.value)}
          className="bg-transparent border border-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary"
          style={{ background: 'oklch(0.15 0.02 250)' }}
        >
          {categories.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        <select
          value={riskFilter}
          onChange={e => setRiskFilter(e.target.value)}
          className="bg-transparent border border-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-primary"
          style={{ background: 'oklch(0.15 0.02 250)' }}
        >
          {['All', 'High', 'Medium', 'Low'].map(r => <option key={r} value={r}>{r} Risk</option>)}
        </select>
      </div>

      {/* Suppliers Table */}
      <div className="glass-card overflow-hidden">
        <div className="p-4 border-b border-border flex items-center justify-between">
          <h2 className="font-semibold text-white">Supplier Performance</h2>
          <span className="text-xs text-slate-400">{filtered.length} results</span>
        </div>

        {suppliersLoading ? (
          <div className="p-12 text-center text-slate-400 animate-pulse">
            Loading supplier intelligence…
          </div>
        ) : filtered.length === 0 ? (
          <div className="p-12 text-center text-slate-500 border-2 border-dashed border-border rounded-lg m-4">
            No suppliers match the current filters.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-slate-400 text-xs uppercase tracking-wider"
                style={{ background: 'oklch(0.16 0.02 250)' }}>
                <tr>
                  <th className="px-4 py-3 font-medium">Supplier</th>
                  <th className="px-4 py-3 font-medium">Category</th>
                  <th className="px-4 py-3 font-medium">Region</th>
                  <th className="px-4 py-3 font-medium">Shipments</th>
                  <th className="px-4 py-3 font-medium">On-Time Rate</th>
                  <th className="px-4 py-3 font-medium">Avg Rating</th>
                  <th className="px-4 py-3 font-medium">Lead Time</th>
                  <th className="px-4 py-3 font-medium">Risk</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {filtered.map((s: any) => (
                  <tr
                    key={s.id}
                    onClick={() => navigate({ to: '/suppliers/$id', params: { id: String(s.id) } })}
                    className="cursor-pointer transition-colors hover:bg-white/[0.03] group"
                  >
                    <td className="px-4 py-3">
                      <div className="font-medium text-white group-hover:text-primary transition-colors">
                        {s.name}
                      </div>
                      <div className="text-xs text-slate-500 mt-0.5">{s.contact_name}</div>
                    </td>
                    <td className="px-4 py-3 text-slate-300">{s.category}</td>
                    <td className="px-4 py-3">
                      <RegionPill region={s.region} />
                    </td>
                    <td className="px-4 py-3 text-slate-200 font-mono text-xs">
                      {s.metrics.shipment_count > 0 ? (
                        <span>{s.metrics.shipment_count}
                          <span className="text-slate-500 ml-1">
                            ({s.metrics.delayed_count} late)
                          </span>
                        </span>
                      ) : (
                        <span className="text-slate-500 italic">Unrated</span>
                      )}
                    </td>
                    <td className="px-4 py-3 w-36">
                      {s.metrics.shipment_count > 0
                        ? <ReliabilityBar rate={s.metrics.on_time_rate} />
                        : <span className="text-slate-500 italic text-xs">—</span>
                      }
                    </td>
                    <td className="px-4 py-3 text-slate-300">
                      {s.metrics.avg_delivery_rating > 0
                        ? `★ ${s.metrics.avg_delivery_rating.toFixed(1)}`
                        : '—'
                      }
                    </td>
                    <td className="px-4 py-3 text-slate-300">{s.lead_time_days}d</td>
                    <td className="px-4 py-3">
                      {s.metrics.shipment_count > 0
                        ? <RiskBadge level={s.risk_level as any} />
                        : <span className="text-xs text-slate-500 italic">Unrated</span>
                      }
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

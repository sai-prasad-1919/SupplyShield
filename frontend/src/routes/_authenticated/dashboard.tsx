import React from 'react'
import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchShipments, fetchKpis, fetchFederatedStatus, fetchFeedbackHistory } from '../../lib/api'
import { KpiCard } from '../../components/KpiCard'
import { RiskBadge } from '../../components/RiskBadge'
import { BrainIcon, ClipboardIcon } from '../../components/Icons'

export const Route = createFileRoute('/_authenticated/dashboard')({
  component: Dashboard,
})

function Dashboard() {
  const navigate = useNavigate()

  const { data: shipments, isLoading: shipmentsLoading } = useQuery({
    queryKey: ['shipments'],
    queryFn: () => fetchShipments(),
  })

  const { data: kpis, isLoading: kpisLoading } = useQuery({
    queryKey: ['kpis'],
    queryFn: () => fetchKpis(),
  })

  const { data: flStatus } = useQuery({
    queryKey: ['federated-status'],
    queryFn: fetchFederatedStatus,
    staleTime: 5 * 60 * 1000,
  })

  const { data: feedbackData } = useQuery({
    queryKey: ['feedback-history'],
    queryFn: fetchFeedbackHistory,
    staleTime: 60 * 1000,
  })

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight text-white">Dashboard Overview</h1>
        <div className="text-sm text-slate-400">Live predictions from active transit data</div>
      </div>
      
      {/* KPI Trio */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <KpiCard label="High Risk Shipments" value={kpisLoading ? '...' : kpis?.high_risk_count} />
        <KpiCard label="Average Delay Impact" value={kpisLoading ? '...' : kpis?.avg_delay} />
        <KpiCard label="Units in Transit" value={kpisLoading ? '...' : kpis?.units_in_transit} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <KpiCard label="Total Shipments" value={kpisLoading ? '...' : kpis?.total_shipments} />
        <KpiCard label="On-Time Count" value={kpisLoading ? '...' : kpis?.on_time_count} />
        <KpiCard label="Delayed Count" value={kpisLoading ? '...' : kpis?.delayed_count} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div 
          onClick={() => navigate({ to: '/federated' })}
          className="glass-card p-5 cursor-pointer hover:bg-[var(--sidebar-accent)] transition-colors flex items-center justify-between"
        >
          <div>
            <div className="flex items-center gap-2 mb-1">
              <BrainIcon className="w-5 h-5 text-primary" />
              <h3 className="font-semibold text-white">Federated Model</h3>
            </div>
            <p className="text-sm text-slate-400">
              {flStatus ? `Global AUC: ${(flStatus.final_metrics?.auc * 100).toFixed(2)}% | Rounds: ${flStatus.fl_rounds_completed}` : 'Loading FL Status...'}
            </p>
          </div>
          <div className="text-slate-500">→</div>
        </div>

        <div 
          onClick={() => navigate({ to: '/feedback' })}
          className="glass-card p-5 cursor-pointer hover:bg-[var(--sidebar-accent)] transition-colors flex items-center justify-between"
        >
          <div>
            <div className="flex items-center gap-2 mb-1">
              <ClipboardIcon className="w-5 h-5 text-emerald-400" />
              <h3 className="font-semibold text-white">Feedback History</h3>
            </div>
            <p className="text-sm text-slate-400">
              {feedbackData ? `${feedbackData.total} Decisions Logged | Confirm Rate: ${((feedbackData.summary?.confirm / feedbackData.total) * 100 || 0).toFixed(0)}%` : 'Loading Feedback Log...'}
            </p>
          </div>
          <div className="text-slate-500">→</div>
        </div>
      </div>

      {/* Shipments Table */}
      <div className="glass-card overflow-hidden">
        <div className="p-4 border-b border-border flex justify-between items-center">
          <h2 className="font-semibold text-white">Active Shipments</h2>
          <div className="text-xs text-slate-400 px-2 py-1 rounded bg-[var(--sidebar)]">
            Top 30 At-Risk · Click any row for full details
          </div>
        </div>
        
        {shipmentsLoading ? (
          <div className="p-8 text-center text-slate-400">Loading live predictions...</div>
        ) : shipments?.length === 0 ? (
          <div className="p-8 text-center border-2 border-dashed border-border rounded-lg m-4 text-slate-400">
            No shipments found for this organization.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-[var(--sidebar)] text-slate-400">
                <tr>
                  <th className="px-4 py-3 font-medium">ID</th>
                  <th className="px-4 py-3 font-medium">Route</th>
                  <th className="px-4 py-3 font-medium">Package Type</th>
                  <th className="px-4 py-3 font-medium">Vehicle</th>
                  <th className="px-4 py-3 font-medium">Weather</th>
                  <th className="px-4 py-3 font-medium">Risk Level</th>
                  <th className="px-4 py-3 font-medium">Delay Prob.</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {shipments?.map((shipment: any) => (
                  <tr
                    key={shipment.id}
                    onClick={() => navigate({ to: '/shipments/$id', params: { id: shipment.id } })}
                    className="cursor-pointer transition-colors hover:bg-[var(--sidebar-accent)]"
                  >
                    <td className="px-4 py-3 font-mono text-primary font-semibold">
                      {shipment.id}
                    </td>
                    <td className="px-4 py-3 text-slate-200">{shipment.route}</td>
                    <td className="px-4 py-3 text-slate-300">{shipment.package_type}</td>
                    <td className="px-4 py-3 text-slate-300 capitalize">
                      {shipment.features?.vehicle_type ?? '—'}
                    </td>
                    <td className="px-4 py-3 text-slate-300 capitalize">
                      {shipment.features?.weather_condition ?? '—'}
                    </td>
                    <td className="px-4 py-3">
                      <RiskBadge level={shipment.prediction.risk_level} />
                    </td>
                    <td className="px-4 py-3 text-slate-300">
                      {(shipment.prediction.delay_probability * 100).toFixed(1)}%
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


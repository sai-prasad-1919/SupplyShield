import React, { useState } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchShipments, fetchKpis } from '../../lib/api'
import { KpiCard } from '../../components/KpiCard'
import { RiskBadge } from '../../components/RiskBadge'
import { GuidancePanel } from '../../components/GuidancePanel'

export const Route = createFileRoute('/_authenticated/dashboard')({
  component: Dashboard,
})

function Dashboard() {
  const [selectedShipment, setSelectedShipment] = useState<any>(null)

  const { data: shipments, isLoading: shipmentsLoading } = useQuery({
    queryKey: ['shipments'],
    queryFn: () => fetchShipments(),
  })

  const { data: kpis, isLoading: kpisLoading } = useQuery({
    queryKey: ['kpis'],
    queryFn: () => fetchKpis(),
  })

  // Auto-select the highest risk shipment on load
  React.useEffect(() => {
    if (shipments && shipments.length > 0 && !selectedShipment) {
      setSelectedShipment(shipments[0])
    }
  }, [shipments])

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

      {/* Main Content: List + Guidance */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 glass-card overflow-hidden">
          <div className="p-4 border-b border-border flex justify-between items-center">
            <h2 className="font-semibold text-white">Active Shipments</h2>
            <div className="text-xs text-slate-400 px-2 py-1 rounded bg-[var(--sidebar)]">Top 30 At-Risk</div>
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
                    <th className="px-4 py-3 font-medium">Risk Level</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {shipments?.map((shipment: any) => (
                    <tr 
                      key={shipment.id} 
                      onClick={() => setSelectedShipment(shipment)}
                      className={`cursor-pointer transition-colors ${
                        selectedShipment?.id === shipment.id 
                          ? 'bg-[var(--sidebar-accent)]' 
                          : 'hover:bg-[var(--sidebar)]'
                      }`}
                    >
                      <td className="px-4 py-3 font-mono text-primary">
                        <Link to={`/shipments/${shipment.id}`} className="hover:underline">
                          {shipment.id}
                        </Link>
                      </td>
                      <td className="px-4 py-3 text-slate-200">{shipment.route}</td>
                      <td className="px-4 py-3 text-slate-300">{shipment.package_type}</td>
                      <td className="px-4 py-3">
                        <RiskBadge level={shipment.prediction.risk_level} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
        
        {/* Guidance Panel */}
        <div className="xl:col-span-1 hidden lg:block">
          {selectedShipment ? (
            <GuidancePanel shipment={selectedShipment} />
          ) : (
            <div className="glass-card p-6 h-full flex items-center justify-center text-slate-500 text-sm text-center">
              Select a shipment to view AI guidance
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

import React from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchShipment } from '../../lib/api'
import { RiskBadge } from '../../components/RiskBadge'
import { GuidancePanel } from '../../components/GuidancePanel'

export const Route = createFileRoute('/_authenticated/shipments/$id')({
  component: ShipmentDetail,
})

function ShipmentDetail() {
  const { id } = Route.useParams()

  const { data: shipment, isLoading } = useQuery({
    queryKey: ['shipment', id],
    queryFn: () => fetchShipment(id),
  })

  if (isLoading) {
    return <div className="p-8 text-center text-slate-400">Loading shipment details...</div>
  }

  if (!shipment) {
    return <div className="p-8 text-center text-slate-400">Shipment not found.</div>
  }

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <div className="flex items-center space-x-4 mb-2">
        <Link to="/dashboard" className="text-slate-400 hover:text-white transition-colors">
          &larr; Back to Dashboard
        </Link>
      </div>
      
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h1 className="text-3xl font-bold tracking-tight text-white font-mono">{shipment.id}</h1>
          <RiskBadge level={shipment.prediction.risk_level} />
        </div>
        <div className="text-sm text-slate-400">Route: {shipment.route}</div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="glass-card p-6">
            <h3 className="text-lg font-semibold text-white mb-4 border-b border-border pb-2">Shipment Details</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-slate-400 mb-1">Package Type</div>
                <div className="text-white">{shipment.package_type}</div>
              </div>
              <div>
                <div className="text-slate-400 mb-1">Weight</div>
                <div className="text-white">{shipment.features.package_weight_kg} kg</div>
              </div>
              <div>
                <div className="text-slate-400 mb-1">Vehicle Type</div>
                <div className="text-white capitalize">{shipment.features.vehicle_type}</div>
              </div>
              <div>
                <div className="text-slate-400 mb-1">Delivery Mode</div>
                <div className="text-white capitalize">{shipment.features.delivery_mode}</div>
              </div>
              <div>
                <div className="text-slate-400 mb-1">Region</div>
                <div className="text-white capitalize">{shipment.features.region}</div>
              </div>
              <div>
                <div className="text-slate-400 mb-1">Weather</div>
                <div className="text-white capitalize">{shipment.features.weather_condition}</div>
              </div>
              <div>
                <div className="text-slate-400 mb-1">Distance</div>
                <div className="text-white">{shipment.features.distance_km} km</div>
              </div>
              <div>
                <div className="text-slate-400 mb-1">Delay Probability</div>
                <div className="text-white">{(shipment.prediction.delay_probability * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="lg:col-span-1">
          <GuidancePanel shipment={shipment} />
        </div>
      </div>
    </div>
  )
}

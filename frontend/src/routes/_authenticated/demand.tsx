import React from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchDemandForecast } from '../../lib/api'
import { ForecastChart } from '../../components/ForecastChart'
import { KpiCard } from '../../components/KpiCard'

export const Route = createFileRoute('/_authenticated/demand')({
  component: DemandPage,
})

function trendLabel(history: { sales: number }[], forecast: { sales: number }[]): string {
  if (!history.length || !forecast.length) return '—'
  const histAvg = history.slice(-4).reduce((s, h) => s + h.sales, 0) / Math.min(4, history.length)
  const foreAvg = forecast.reduce((s, f) => s + f.sales, 0) / forecast.length
  if (histAvg === 0) return '—'
  const pct = ((foreAvg - histAvg) / histAvg) * 100
  return pct >= 0 ? `+${pct.toFixed(1)}% ↑` : `${pct.toFixed(1)}% ↓`
}

function formatSales(n: number): string {
  if (n >= 1_000_000) return `₹${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000) return `₹${(n / 1_000).toFixed(0)}K`
  return `₹${n}`
}

function DemandPage() {
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ['demand-forecast', 4],
    queryFn: () => fetchDemandForecast(4),
    staleTime: 5 * 60 * 1000,
  })

  const orgName = localStorage.getItem('supplyshield_org_name') ?? 'Your Organisation'

  // KPI calculations
  const nextWeekSales =
    data?.forecast?.length > 0 ? formatSales(data.forecast[0].sales) : '—'
  const trendStr =
    data ? trendLabel(data.history ?? [], data.forecast ?? []) : '—'
  const trendPositive = trendStr.startsWith('+')

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Demand Forecast</h1>
          <p className="text-sm text-slate-400 mt-0.5">
            LSTM-powered 4-week demand prediction · {orgName}
          </p>
        </div>
        <div className="text-xs text-slate-500 bg-[var(--sidebar)] border border-border px-3 py-1.5 rounded">
          Organisation locked to JWT session
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <KpiCard label="Forecast Horizon" value="4 Weeks" />
        <KpiCard label="Next Week Projected" value={isLoading ? '…' : nextWeekSales} />
        <div className="glass-card p-6 flex flex-col">
          <div className="text-sm font-medium text-slate-400 mb-2">Trend vs Prior 4 Weeks</div>
          <div
            className="text-3xl font-bold tracking-tight"
            style={{ color: trendPositive ? 'oklch(0.75 0.15 155)' : 'oklch(0.75 0.12 25)' }}
          >
            {isLoading ? '…' : trendStr}
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="glass-card p-6">
        <h2 className="font-semibold text-white mb-1">Weekly Sales Forecast</h2>
        <p className="text-xs text-slate-500 mb-4">
          Historical (actual) → Forecast (LSTM prediction)
        </p>

        {isLoading && (
          <div
            className="rounded animate-pulse"
            style={{ height: 300, background: 'var(--sidebar)' }}
          />
        )}

        {isError && (
          <div className="flex flex-col items-center justify-center gap-3" style={{ height: 300 }}>
            <p className="text-slate-400 text-sm">Unable to load forecast.</p>
            <button
              onClick={() => refetch()}
              className="px-4 py-1.5 rounded bg-primary text-white text-sm hover:bg-primary/90 transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {!isLoading && !isError && data && (
          <ForecastChart
            history={data.history ?? []}
            forecast={data.forecast ?? []}
            forecastStartDate={data.forecast_start_date ?? ''}
          />
        )}
      </div>

      {/* LSTM Model Info Card */}
      {data?.model_info && (
        <div className="glass-card p-5">
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">
            LSTM Model Information
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-4">
            <div>
              <p className="text-slate-500 text-xs mb-1">Model Type</p>
              <p className="text-slate-200">{data.model_info.type}</p>
            </div>
            <div>
              <p className="text-slate-500 text-xs mb-1">Architecture</p>
              <p className="text-slate-200">{data.model_info.layers} layers · {data.model_info.hidden_units} hidden units</p>
            </div>
            <div>
              <p className="text-slate-500 text-xs mb-1">Sequence Window</p>
              <p className="text-slate-200">{data.model_info.window_weeks} weeks input</p>
            </div>
            <div>
              <p className="text-slate-500 text-xs mb-1">Forecast Horizon</p>
              <p className="text-slate-200">4 weeks</p>
            </div>
          </div>
          <div
            className="rounded-lg p-3 text-xs text-slate-400 leading-relaxed"
            style={{ background: 'var(--sidebar)' }}
          >
            <strong className="text-slate-300">Training: </strong>
            {data.model_info.training}
            <br />
            <strong className="text-slate-300 mt-1 block">⚠ Note: </strong>
            Forecasts reflect learned seasonal patterns. Validate against live inventory data before making procurement decisions.
          </div>
        </div>
      )}
    </div>
  )
}

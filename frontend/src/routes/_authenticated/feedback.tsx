import React, { useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchFeedbackHistory } from '../../lib/api'
import { RiskBadge } from '../../components/RiskBadge'

export const Route = createFileRoute('/_authenticated/feedback')({
  component: FeedbackPage,
})

function FeedbackPage() {
  const [filter, setFilter] = useState<'all' | 'confirm' | 'override' | 'escalate'>('all')

  const { data, isLoading } = useQuery({
    queryKey: ['feedback-history'],
    queryFn: fetchFeedbackHistory,
    staleTime: 60 * 1000,
  })

  if (isLoading) {
    return <div className="text-slate-400 p-8">Loading feedback history...</div>
  }

  const entries = data?.entries ?? []
  const filteredEntries = filter === 'all' ? entries : entries.filter((e: any) => e.decision === filter)
  const summary = data?.summary ?? { confirm: 0, override: 0, escalate: 0, gemini_count: 0, rule_based_count: 0 }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Human-in-the-Loop Feedback Log</h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Audit trail of all manager decisions on AI shipment guidance
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="glass-card p-4 text-center">
          <div className="text-xs text-slate-400 mb-1">Total Decisions</div>
          <div className="text-2xl font-bold text-white">{data?.total ?? 0}</div>
        </div>
        <div className="glass-card p-4 text-center">
          <div className="text-xs text-slate-400 mb-1">Confirmed</div>
          <div className="text-2xl font-bold text-emerald-400">{summary.confirm}</div>
        </div>
        <div className="glass-card p-4 text-center">
          <div className="text-xs text-slate-400 mb-1">Overridden</div>
          <div className="text-2xl font-bold text-amber-400">{summary.override}</div>
        </div>
        <div className="glass-card p-4 text-center">
          <div className="text-xs text-slate-400 mb-1">Escalated</div>
          <div className="text-2xl font-bold text-rose-400">{summary.escalate}</div>
        </div>
        <div className="glass-card p-4 text-center">
          <div className="text-xs text-slate-400 mb-1">Gemini AI Used</div>
          <div className="text-2xl font-bold text-primary">{summary.gemini_count}</div>
        </div>
      </div>

      <div className="glass-card overflow-hidden">
        <div className="p-4 border-b border-border flex items-center justify-between">
          <h2 className="font-semibold text-white">Decision Timeline</h2>
          <div className="flex gap-2 bg-[var(--sidebar)] p-1 rounded-lg">
            {(['all', 'confirm', 'override', 'escalate'] as const).map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1 text-xs font-medium rounded-md capitalize transition-colors ${
                  filter === f ? 'bg-primary text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        {filteredEntries.length === 0 ? (
          <div className="p-8 text-center text-slate-400">
            No feedback entries found.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-[var(--sidebar)] text-slate-400">
                <tr>
                  <th className="px-4 py-3 font-medium">Timestamp</th>
                  <th className="px-4 py-3 font-medium">Shipment ID</th>
                  <th className="px-4 py-3 font-medium">Risk Level</th>
                  <th className="px-4 py-3 font-medium">AI Source</th>
                  <th className="px-4 py-3 font-medium">Decision</th>
                  <th className="px-4 py-3 font-medium">Override Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {filteredEntries.map((entry: any, i: number) => (
                  <tr key={i} className="hover:bg-[var(--sidebar-accent)] transition-colors">
                    <td className="px-4 py-3 text-slate-300 whitespace-nowrap">
                      {new Date(entry.timestamp).toLocaleString()}
                    </td>
                    <td className="px-4 py-3 font-mono text-primary font-medium">
                      {entry.shipment_id}
                    </td>
                    <td className="px-4 py-3">
                      <RiskBadge level={entry.risk_level} />
                    </td>
                    <td className="px-4 py-3 text-slate-300">
                      {entry.guidance_source === 'gemini' ? 'Gemini 2.5 Pro' : 'Rule-based'}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2.5 py-1 rounded-full text-xs font-medium ${
                        entry.decision === 'confirm' ? 'bg-emerald-500/10 text-emerald-400' :
                        entry.decision === 'override' ? 'bg-amber-500/10 text-amber-400' :
                        'bg-rose-500/10 text-rose-400'
                      }`}>
                        {entry.decision.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-400 max-w-xs truncate">
                      {entry.decision === 'override' ? (
                        <div title={entry.override_reason}>
                          <span className="font-medium text-slate-300">Reason:</span> {entry.override_reason || 'None provided'}
                        </div>
                      ) : (
                        '—'
                      )}
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

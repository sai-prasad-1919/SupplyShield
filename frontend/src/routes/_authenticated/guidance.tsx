import React, { useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useQuery, useMutation } from '@tanstack/react-query'
import { fetchShipments, explainShipment, submitFeedback, type FeedbackPayload } from '../../lib/api'
import { RiskBadge } from '../../components/RiskBadge'
import { ModelConsensus } from '../../components/ModelConsensus'
import { ShapChart } from '../../components/ShapChart'
import { RecommendationCard } from '../../components/RecommendationCard'

export const Route = createFileRoute('/_authenticated/guidance')({
  component: GuidancePage,
})

// ─── Human Decision Panel ────────────────────────────────────────────────────

interface DecisionPanelProps {
  analysisData: any
  shipmentId: string
  onSubmitted: () => void
}

function DecisionPanel({ analysisData, shipmentId, onSubmitted }: DecisionPanelProps) {
  const [decision, setDecision] = useState<'confirm' | 'override' | 'escalate' | null>(null)
  const [overrideReason, setOverrideReason] = useState('')
  const [altAction, setAltAction] = useState('')
  const [submitted, setSubmitted] = useState(false)

  const { mutate: submitDecision, isPending } = useMutation({
    mutationFn: (payload: FeedbackPayload) => submitFeedback(payload),
    onSuccess: () => {
      setSubmitted(true)
      onSubmitted()
    },
  })

  const handleSubmit = () => {
    if (!decision) return
    submitDecision({
      shipment_id: shipmentId,
      fl_probability: analysisData.consensus?.[0]?.probability ?? 0,
      xgb_probability: analysisData.consensus?.[1]?.probability ?? 0,
      risk_level: analysisData.prediction?.risk_level ?? '',
      recommendations: analysisData.recommendations ?? [],
      guidance_source: analysisData.guidance_source ?? 'rule_based',
      decision,
      override_reason: overrideReason,
      alternative_action: altAction,
    })
  }

  if (submitted) {
    return (
      <div className="glass-card p-4 text-center text-sm text-slate-400">
        ✅ Decision logged successfully. Your feedback helps improve the model.
      </div>
    )
  }

  return (
    <div className="glass-card p-5">
      <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4">
        What is your decision?
      </h3>
      <div className="flex gap-2 mb-4">
        {(['confirm', 'override', 'escalate'] as const).map(d => (
          <button
            key={d}
            onClick={() => setDecision(d)}
            className="flex-1 py-2 px-3 rounded-lg text-sm font-medium transition-all"
            style={{
              background: decision === d ? 'var(--primary)' : 'var(--sidebar)',
              color: decision === d ? 'white' : '#94a3b8',
              border: `1px solid ${decision === d ? 'var(--primary)' : 'var(--border)'}`,
            }}
          >
            {d === 'confirm' ? '✅ Confirm' : d === 'override' ? '✏️ Override' : '🔺 Escalate'}
          </button>
        ))}
      </div>

      {decision === 'override' && (
        <div className="space-y-3 mb-4">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Reason for Override</label>
            <textarea
              rows={2}
              value={overrideReason}
              onChange={e => setOverrideReason(e.target.value)}
              placeholder="Why are you overriding the AI recommendation?"
              className="w-full bg-[var(--sidebar)] border border-border rounded-lg p-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-primary resize-none"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Alternative Action</label>
            <textarea
              rows={2}
              value={altAction}
              onChange={e => setAltAction(e.target.value)}
              placeholder="What action will you take instead?"
              className="w-full bg-[var(--sidebar)] border border-border rounded-lg p-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-primary resize-none"
            />
          </div>
        </div>
      )}

      {decision && (
        <button
          onClick={handleSubmit}
          disabled={isPending}
          className="w-full py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {isPending ? 'Submitting…' : 'Submit Decision'}
        </button>
      )}
    </div>
  )
}

// ─── Main Guidance Page ──────────────────────────────────────────────────────

function GuidancePage() {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [decisionLogged, setDecisionLogged] = useState(false)

  const { data: shipments, isLoading: shipmentsLoading } = useQuery({
    queryKey: ['shipments'],
    queryFn: fetchShipments,
    staleTime: 2 * 60 * 1000,
  })

  const { mutate: analyse, isPending: analysing, isError: analyseError } = useMutation({
    mutationFn: (shipmentId: string) => explainShipment(shipmentId),
    onSuccess: data => {
      setAnalysisResult(data)
      setDecisionLogged(false)
    },
  })

  const handleAnalyse = () => {
    if (!selectedId) return
    setAnalysisResult(null)
    analyse(selectedId)
  }

  const dest_map: Record<string, string> = {
    north: 'Delhi', south: 'Bangalore', east: 'Kolkata', west: 'Mumbai', central: 'Nagpur',
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">AI Shipment Guidance</h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Select a shipment · Run full AI analysis · Make your decision
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        {/* ── Left: Shipment Selector ─────────────────────────────────────── */}
        <div className="lg:col-span-1 glass-card overflow-hidden">
          <div className="p-4 border-b border-border">
            <h2 className="font-semibold text-white text-sm">Select Shipment</h2>
          </div>

          {shipmentsLoading ? (
            <div className="p-4 text-sm text-slate-400">Loading shipments…</div>
          ) : (
            <div className="divide-y divide-border max-h-[60vh] overflow-y-auto">
              {(shipments ?? []).map((s: any) => {
                const region = s.features?.region ?? ''
                const dest = dest_map[region] ?? 'Hub'
                const isSelected = selectedId === s.id
                return (
                  <button
                    key={s.id}
                    onClick={() => setSelectedId(s.id)}
                    className="w-full text-left px-4 py-3 transition-colors"
                    style={{
                      background: isSelected ? 'var(--sidebar-accent)' : 'transparent',
                    }}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono text-xs font-semibold text-primary">{s.id}</span>
                      <RiskBadge level={s.prediction?.risk_level ?? 'Low'} />
                    </div>
                    <div className="text-xs text-slate-400">
                      Supplier → {dest}
                    </div>
                  </button>
                )
              })}
            </div>
          )}

          <div className="p-3 border-t border-border">
            <button
              onClick={handleAnalyse}
              disabled={!selectedId || analysing}
              className="w-full py-2 rounded-lg bg-primary text-white text-sm font-semibold transition-colors hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {analysing ? 'Analysing…' : 'Analyse Shipment'}
            </button>
          </div>
        </div>

        {/* ── Right: Analysis Output ──────────────────────────────────────── */}
        <div className="lg:col-span-2 space-y-4">
          {/* Idle */}
          {!selectedId && !analysing && !analysisResult && (
            <div className="glass-card p-12 text-center text-slate-400 text-sm">
              Select a shipment from the panel to begin AI analysis.
            </div>
          )}

          {/* Loading */}
          {analysing && (
            <div className="glass-card p-12 flex flex-col items-center gap-4 text-slate-400 text-sm">
              <div
                className="rounded-full border-2 border-primary border-t-transparent animate-spin"
                style={{ width: 40, height: 40 }}
              />
              <div className="text-center">
                <p className="text-white font-medium mb-1">Analysing Shipment {selectedId}…</p>
                <p className="text-xs text-slate-500">
                  Running prediction models · Generating SHAP explanation · Preparing guidance
                </p>
              </div>
            </div>
          )}

          {/* Error */}
          {analyseError && !analysing && !analysisResult && (
            <div className="glass-card p-8 flex flex-col items-center gap-3 text-slate-400 text-sm">
              <p>Unable to analyse shipment.</p>
              <button
                onClick={handleAnalyse}
                className="px-4 py-1.5 rounded bg-primary text-white text-sm hover:bg-primary/90 transition-colors"
              >
                Try Again
              </button>
            </div>
          )}

          {/* Results */}
          {analysisResult && !analysing && (
            <>
              {/* Shipment summary strip */}
              <div
                className="glass-card px-5 py-3 flex items-center justify-between"
                style={{ borderLeft: '3px solid var(--primary)' }}
              >
                <div>
                  <span className="font-mono font-bold text-primary text-sm">
                    {analysisResult.shipment_id}
                  </span>
                  <span className="ml-3 text-slate-400 text-sm">
                    {analysisResult.prediction?.verdict === 'Delayed' ? '🔴' : '🟢'}{' '}
                    {analysisResult.prediction?.verdict} ·{' '}
                    {(analysisResult.prediction?.risk_probability * 100).toFixed(1)}% risk
                  </span>
                </div>
                <RiskBadge level={analysisResult.prediction?.risk_level ?? 'Low'} />
              </div>

              {/* Model Consensus */}
              <ModelConsensus consensus={analysisResult.consensus ?? []} />

              {/* SHAP */}
              {analysisResult.shap?.length > 0 && (
                <div className="glass-card p-5">
                  <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-1">
                    Why the AI flagged this shipment
                  </h3>
                  <p className="text-xs text-slate-500 mb-3">
                    SHAP feature contributions (positive = increases delay risk)
                  </p>
                  <ShapChart drivers={analysisResult.shap} />
                </div>
              )}

              {/* Recommendations */}
              {analysisResult.recommendations?.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3">
                    AI Recommendations
                  </h3>
                  <div className="space-y-3">
                    {analysisResult.recommendations.map((rec: any, idx: number) => (
                      <RecommendationCard
                        key={idx}
                        recommendation={rec}
                        index={idx}
                        guidanceSource={analysisResult.guidance_source}
                        showSourceBadge={idx === analysisResult.recommendations.length - 1}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Human-in-the-loop */}
              <DecisionPanel
                analysisData={analysisResult}
                shipmentId={analysisResult.shipment_id}
                onSubmitted={() => setDecisionLogged(true)}
              />

              {decisionLogged && (
                <p className="text-xs text-slate-500 text-center">
                  Feedback stored in <code>checkpoints/feedback_log.jsonl</code> for Phase 8 model improvement.
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

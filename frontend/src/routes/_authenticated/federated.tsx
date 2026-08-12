import React from 'react'
import { createFileRoute } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { fetchFederatedStatus } from '../../lib/api'
import { ConvergenceChart } from '../../components/ConvergenceChart'

export const Route = createFileRoute('/_authenticated/federated')({
  component: FederatedPage,
})

function FederatedPage() {
  const { data: status, isLoading } = useQuery({
    queryKey: ['federated-status'],
    queryFn: fetchFederatedStatus,
    staleTime: 5 * 60 * 1000,
  })

  if (isLoading) {
    return <div className="text-slate-400 p-8">Loading FL model status...</div>
  }

  if (!status) {
    return <div className="text-slate-400 p-8">Unable to load FL status.</div>
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Federated Learning (FL) Status</h1>
          <p className="text-sm text-slate-400 mt-0.5">
            Global model convergence and participating organization metrics
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="glass-card p-5 text-center">
          <div className="text-sm text-slate-400 mb-1">Participating Orgs</div>
          <div className="text-3xl font-bold text-white">{status.organizations?.length ?? 3}</div>
        </div>
        <div className="glass-card p-5 text-center">
          <div className="text-sm text-slate-400 mb-1">FL Rounds Completed</div>
          <div className="text-3xl font-bold text-white">{status.fl_rounds_completed}</div>
        </div>
        <div className="glass-card p-5 text-center">
          <div className="text-sm text-slate-400 mb-1">Global AUC (Test)</div>
          <div className="text-3xl font-bold text-primary">
            {status.final_metrics?.auc ? (status.final_metrics.auc * 100).toFixed(2) + '%' : 'N/A'}
          </div>
        </div>
        <div className="glass-card p-5 text-center">
          <div className="text-sm text-slate-400 mb-1">Global F1 (Test)</div>
          <div className="text-3xl font-bold text-white">
            {status.final_metrics?.f1 ? status.final_metrics.f1.toFixed(4) : 'N/A'}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-card p-5">
          <h2 className="font-semibold text-white mb-2">Model Convergence (AUC vs Loss)</h2>
          <div className="text-sm text-slate-400 mb-4">
            Validation metrics recorded at the end of each federated communication round.
          </div>
          {status.convergence && status.convergence.length > 0 ? (
            <ConvergenceChart data={status.convergence} />
          ) : (
            <div className="text-center py-12 text-slate-500">No convergence data available</div>
          )}
        </div>

        <div className="lg:col-span-1 space-y-6">
          <div className="glass-card p-5">
            <h2 className="font-semibold text-white mb-4">Model Checkpoints</h2>
            <div className="space-y-4 text-sm">
              <div className="flex justify-between items-center border-b border-border pb-3">
                <span className="text-slate-400">FL Global Weights</span>
                <span className="text-white font-mono">{status.model_sizes?.fl_model_kb ?? '22.37'} KB</span>
              </div>
              <div className="flex justify-between items-center border-b border-border pb-3">
                <span className="text-slate-400">XGBoost Ensemble</span>
                <span className="text-white font-mono">{status.model_sizes?.xgboost_kb ?? '1292.76'} KB</span>
              </div>
              <div className="flex justify-between items-center pb-1">
                <span className="text-slate-400">Demand LSTM (per org)</span>
                <span className="text-white font-mono">{status.model_sizes?.lstm_kb_per_org ?? '201.29'} KB</span>
              </div>
            </div>
          </div>
          
          <div className="glass-card p-5">
            <h2 className="font-semibold text-white mb-4">Hyperparameters</h2>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">Local Epochs / Round</span>
                <span className="text-white">{status.local_epochs_per_round}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">FL Threshold</span>
                <span className="text-white">{status.fl_threshold?.toFixed(4)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">XGBoost Threshold</span>
                <span className="text-white">{status.xgb_threshold?.toFixed(4)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

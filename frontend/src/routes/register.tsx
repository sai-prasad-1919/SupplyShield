import React from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { ShieldCheckIcon, InfoIcon } from '../components/Icons'

export const Route = createFileRoute('/register')({
  component: Register,
})

function Register() {
  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-6">
      {/* Logo */}
      <div className="flex items-center space-x-3 text-white mb-10">
        <ShieldCheckIcon className="w-9 h-9 text-primary" />
        <span className="font-bold text-2xl tracking-tight">SupplyShield</span>
      </div>

      {/* Coming Soon Banner */}
      <div className="glass-card w-full max-w-md p-8">
        <div className="flex items-start space-x-3 bg-primary/10 border border-primary/30 rounded-lg p-4 mb-6">
          <InfoIcon className="w-5 h-5 text-primary mt-0.5 flex-shrink-0" />
          <div>
            <h3 className="text-sm font-semibold text-primary mb-1">Self-Registration Coming Soon</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              We're building a full self-service onboarding flow. For now, contact the SupplyShield team to register your organization and get your Org ID.
            </p>
          </div>
        </div>

        <h2 className="text-2xl font-bold text-white mb-1 tracking-tight">Register Organization</h2>
        <p className="text-sm text-slate-400 mb-8">Join the federated supply-chain network</p>

        {/* Disabled form preview */}
        <div className="space-y-5 opacity-50 pointer-events-none select-none">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1.5">Organization Name</label>
            <input
              type="text"
              placeholder="e.g. Acme Retail"
              disabled
              className="w-full bg-sidebar border border-border rounded-md px-4 py-2.5 text-slate-400 text-sm cursor-not-allowed"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1.5">Owner Name</label>
            <input
              type="text"
              placeholder="Full name"
              disabled
              className="w-full bg-sidebar border border-border rounded-md px-4 py-2.5 text-slate-400 text-sm cursor-not-allowed"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1.5">
              Password <span className="text-xs text-slate-500 font-normal">(min 6 characters)</span>
            </label>
            <input
              type="password"
              placeholder="••••••••••"
              disabled
              className="w-full bg-sidebar border border-border rounded-md px-4 py-2.5 text-slate-400 text-sm cursor-not-allowed"
            />
          </div>

          <button
            disabled
            className="w-full bg-primary/50 cursor-not-allowed text-white font-semibold py-2.5 rounded-md text-sm"
          >
            Create Organization
          </button>
        </div>

        <div className="mt-6 pt-6 border-t border-border text-center">
          <Link to="/login" className="text-sm text-slate-400 hover:text-white transition-colors">
            ← Back to login
          </Link>
        </div>
      </div>
    </div>
  )
}

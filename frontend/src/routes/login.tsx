import React, { useState } from 'react'
import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { ShieldCheckIcon } from '../components/Icons'
import { loginOrg, saveSession } from '../lib/api'

export const Route = createFileRoute('/login')({
  component: Login,
})

function Login() {
  const navigate = useNavigate()
  const [orgId, setOrgId] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!orgId.trim()) { setError('Please enter your Organization ID.'); return }
    if (!password) { setError('Please enter your password.'); return }

    setLoading(true)
    try {
      const data = await loginOrg(orgId, password)
      saveSession(data.access_token, data.org_id, data.company_name, data.owner_name)
      navigate({ to: '/dashboard' })
    } catch (err: any) {
      const detail = err?.response?.data?.detail
      setError(typeof detail === 'string' ? detail : 'Login failed. Check your credentials.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-6">
      {/* Logo */}
      <div className="flex items-center space-x-3 text-white mb-10">
        <ShieldCheckIcon className="w-9 h-9 text-primary" />
        <span className="font-bold text-2xl tracking-tight">SupplyShield</span>
      </div>

      {/* Card */}
      <div className="glass-card w-full max-w-md p-8">
        <h2 className="text-2xl font-bold text-white mb-1 text-center tracking-tight">Welcome back</h2>
        <p className="text-sm text-slate-400 text-center mb-8">Sign in to your organization dashboard</p>

        <form onSubmit={handleLogin} className="space-y-5">
          {/* Org ID */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1.5" htmlFor="org-id">
              Organization ID
            </label>
            <input
              id="org-id"
              type="text"
              placeholder="e.g. NVM-A1B2C3D4"
              value={orgId}
              onChange={e => setOrgId(e.target.value.toUpperCase())}
              className="w-full bg-sidebar border border-border rounded-md px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary text-sm font-mono tracking-widest"
              autoComplete="username"
              autoFocus
            />
            <p className="text-xs text-slate-500 mt-1.5">
              Your unique Org ID was shown when your organization was registered.
            </p>
          </div>

          {/* Password */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1.5" htmlFor="password">
              Password
            </label>
            <input
              id="password"
              type="password"
              placeholder="••••••••••"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="w-full bg-sidebar border border-border rounded-md px-4 py-2.5 text-white placeholder-slate-500 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary text-sm"
              autoComplete="current-password"
            />
          </div>

          {/* Error */}
          {error && (
            <div className="bg-risk-high/20 border border-risk-high/40 rounded-md px-4 py-3 text-sm text-risk-high-foreground">
              {error}
            </div>
          )}

          {/* Submit */}
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-primary hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-2.5 rounded-md transition-colors text-sm"
          >
            {loading ? 'Signing in…' : 'Sign In'}
          </button>
        </form>

        {/* Divider */}
        <div className="mt-6 pt-6 border-t border-border text-center">
          <p className="text-sm text-slate-500">
            Want to register your organization?{' '}
            <Link to="/register" className="text-primary hover:text-primary/80 transition-colors">
              Learn more
            </Link>
          </p>
        </div>
      </div>

      <p className="mt-6 text-xs text-slate-600 text-center max-w-sm">
        SupplyShield uses federated learning — your raw data stays on-premise. Only model insights are shared.
      </p>
    </div>
  )
}

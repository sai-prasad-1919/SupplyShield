import React from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { ShieldCheckIcon } from '../components/Icons'

export const Route = createFileRoute('/')({
  component: Landing,
})

function Landing() {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col items-center justify-center p-6">
      <div className="max-w-3xl text-center space-y-8">
        <div className="flex justify-center mb-8">
          <div className="flex items-center space-x-3 text-white">
            <ShieldCheckIcon className="w-10 h-10 text-primary" />
            <span className="font-bold text-3xl tracking-tight">SupplyShield</span>
          </div>
        </div>
        
        <div className="inline-flex items-center px-3 py-1 rounded-full border border-border bg-[var(--card)] text-sm text-slate-300">
          <span className="w-2 h-2 rounded-full bg-primary mr-2"></span>
          Federated Privacy Network
        </div>
        
        <h1 className="text-5xl font-extrabold tracking-tight text-white leading-tight">
          Predict supply chain disruptions <br className="hidden md:block"/>
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-blue-400">
            without sharing your data
          </span>
        </h1>
        
        <p className="text-xl text-slate-400 max-w-2xl mx-auto">
          SupplyShield uses federated learning to build world-class risk models across retailers, manufacturers, and logistics providers—while keeping your raw data safely on-premise.
        </p>
        
        <div className="flex items-center justify-center space-x-4 pt-4">
          <Link to="/login" className="px-8 py-3 rounded-md bg-primary text-white font-medium hover:bg-primary/90 transition-colors">
            Enter Dashboard
          </Link>
          <button className="px-8 py-3 rounded-md border border-border bg-[var(--card)] text-white font-medium hover:bg-border transition-colors">
            Read Docs
          </button>
        </div>
      </div>
    </div>
  )
}

import React from 'react';
import { AlertTriangleIcon } from './Icons';

interface GuidancePanelProps {
  shipment: any;
}

export const GuidancePanel = ({ shipment }: GuidancePanelProps) => {
  if (!shipment) return null;

  return (
    <div className="glass-card p-6 flex flex-col h-full">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
        <AlertTriangleIcon className="w-5 h-5 text-primary mr-2" />
        AI Guidance
      </h3>
      
      <div className="bg-[var(--sidebar)] border border-border rounded-lg p-4 mb-4">
        <p className="text-white text-sm leading-relaxed mb-3">
          {shipment.guidance?.text}
        </p>
        <div className="bg-background rounded px-3 py-2 text-xs font-mono text-slate-400">
          Why: {shipment.guidance?.rule_fired}
        </div>
      </div>
      
      <div className="mt-auto">
        <h4 className="text-sm font-medium text-slate-300 mb-2">Follow up</h4>
        <textarea 
          placeholder="Ask a question about this recommendation..." 
          className="w-full bg-[var(--sidebar)] border border-border rounded-lg p-3 text-sm focus:outline-none focus:border-primary text-white placeholder-slate-500 resize-none h-24"
        />
        <button className="mt-3 w-full bg-primary hover:bg-primary/90 text-white font-medium py-2 rounded-lg transition-colors text-sm">
          Send to AI
        </button>
      </div>
    </div>
  );
};

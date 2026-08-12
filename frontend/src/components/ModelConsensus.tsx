import React from 'react';

interface ConsensusEntry {
  model: string;
  probability: number;
  verdict: string;
}

interface ModelConsensusProps {
  consensus: ConsensusEntry[];
  threshold?: number;
}

export const ModelConsensus = ({ consensus, threshold }: ModelConsensusProps) => {
  return (
    <div className="glass-card p-5">
      <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4">
        Model Consensus
      </h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-slate-400 text-left border-b border-border">
            <th className="pb-2 font-medium">Model</th>
            <th className="pb-2 font-medium text-right">Probability</th>
            <th className="pb-2 font-medium text-right pr-1">Verdict</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {consensus.map((entry, idx) => (
            <tr key={idx} className="text-slate-200">
              <td className="py-3 text-slate-300">{entry.model}</td>
              <td className="py-3 text-right font-mono font-semibold text-white">
                {(entry.probability * 100).toFixed(1)}%
              </td>
              <td className="py-3 text-right pr-1">
                <span
                  className="inline-block px-2 py-0.5 rounded text-xs font-semibold"
                  style={{
                    background:
                      entry.verdict === 'Delayed'
                        ? 'var(--risk-high)'
                        : 'var(--risk-low)',
                    color:
                      entry.verdict === 'Delayed'
                        ? 'var(--risk-high-foreground)'
                        : 'var(--risk-low-foreground)',
                  }}
                >
                  {entry.verdict === 'Delayed' ? '🔴' : '🟢'} {entry.verdict}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {threshold !== undefined && (
        <p className="mt-3 text-xs text-slate-500">
          FL threshold: {(threshold * 100).toFixed(1)}% · XGBoost threshold: 19.7%
        </p>
      )}
    </div>
  );
};

import React from 'react';

interface Recommendation {
  title: string;
  description: string;
}

interface RecommendationCardProps {
  recommendation: Recommendation;
  index: number;
  guidanceSource?: string;
  showSourceBadge?: boolean;
}

const ICONS = ['🔧', '🚛', '📢'];

export const RecommendationCard = ({
  recommendation,
  index,
  guidanceSource,
  showSourceBadge = false,
}: RecommendationCardProps) => {
  const icon = ICONS[index % ICONS.length];

  return (
    <div
      className="glass-card p-4 flex gap-3"
      style={{
        borderLeft: '3px solid var(--primary)',
        animation: `fadeIn 0.3s ease ${index * 0.1}s both`,
      }}
    >
      <span className="text-xl flex-shrink-0 mt-0.5">{icon}</span>
      <div className="flex-1 min-w-0">
        <p className="text-white font-semibold text-sm mb-1">{recommendation.title}</p>
        <p className="text-slate-400 text-sm leading-relaxed">{recommendation.description}</p>
        {showSourceBadge && guidanceSource && (
          <span
            className="inline-block mt-2 text-xs px-2 py-0.5 rounded"
            style={{
              background: guidanceSource === 'gemini' ? 'oklch(0.30 0.05 270)' : 'oklch(0.28 0.03 260)',
              color: guidanceSource === 'gemini' ? 'oklch(0.75 0.15 270)' : '#64748b',
            }}
          >
            {guidanceSource === 'gemini' ? '✦ AI Generated' : '⚙ Rule-Based Fallback'}
          </span>
        )}
      </div>
    </div>
  );
};

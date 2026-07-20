import React from 'react';

type RiskLevel = 'High' | 'Medium' | 'Low';

export const RiskBadge = ({ level }: { level: RiskLevel }) => {
  const getColors = () => {
    switch (level) {
      case 'High':
        return 'bg-risk-high text-risk-high-foreground';
      case 'Medium':
        return 'bg-risk-medium text-risk-medium-foreground';
      case 'Low':
        return 'bg-risk-low text-risk-low-foreground';
      default:
        return 'bg-slate-800 text-slate-400';
    }
  };

  return (
    <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${getColors()}`}>
      {level}
    </span>
  );
};

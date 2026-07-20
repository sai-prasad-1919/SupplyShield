import React from 'react';

interface KpiCardProps {
  label: string;
  value: string | number;
}

export const KpiCard = ({ label, value }: KpiCardProps) => {
  return (
    <div className="glass-card p-6 flex flex-col">
      <div className="text-sm font-medium text-slate-400 mb-2">{label}</div>
      <div className="text-3xl font-bold tracking-tight text-white">{value}</div>
    </div>
  );
};

import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Cell, ReferenceLine, ResponsiveContainer,
} from 'recharts';

interface ShapDriver {
  feature: string;
  value: any;
  impact: number;
  direction: string;
}

interface ShapChartProps {
  drivers: ShapDriver[];
}

const formatFeatureName = (name: string) =>
  name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d: ShapDriver = payload[0]?.payload;
  if (!d) return null;
  return (
    <div
      style={{
        background: 'var(--card)',
        border: '1px solid var(--border)',
        borderRadius: '0.5rem',
        padding: '0.75rem 1rem',
        fontSize: '0.8rem',
      }}
    >
      <p style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: 4 }}>
        {formatFeatureName(d.feature)}
      </p>
      <p style={{ color: '#94a3b8', margin: '2px 0' }}>
        Value: <span style={{ color: '#e2e8f0' }}>{String(d.value)}</span>
      </p>
      <p style={{ color: d.impact >= 0 ? 'oklch(0.65 0.15 25)' : 'oklch(0.65 0.15 155)', margin: '2px 0' }}>
        SHAP Impact: {d.impact >= 0 ? '+' : ''}{d.impact.toFixed(3)}
      </p>
      <p style={{ color: '#64748b', fontSize: '0.72rem', marginTop: 4 }}>
        {d.impact >= 0 ? '↑ Increases delay probability' : '↓ Decreases delay probability'}
      </p>
    </div>
  );
};

export const ShapChart = ({ drivers }: ShapChartProps) => {
  // Take top 5 by absolute impact, already sorted by backend
  const topDrivers = drivers.slice(0, 5);

  return (
    <ResponsiveContainer width="100%" height={Math.max(180, topDrivers.length * 44)}>
      <BarChart
        layout="vertical"
        data={topDrivers}
        margin={{ top: 4, right: 40, left: 0, bottom: 4 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
        <XAxis
          type="number"
          tick={{ fill: '#64748b', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          tickFormatter={v => v.toFixed(2)}
        />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={130}
          tickFormatter={formatFeatureName}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'var(--border)', opacity: 0.3 }} />
        <ReferenceLine x={0} stroke="var(--border)" strokeWidth={1.5} />
        <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
          {topDrivers.map((d, idx) => (
            <Cell
              key={idx}
              fill={
                d.impact >= 0
                  ? 'oklch(0.55 0.15 25)'   // red/orange — increases delay
                  : 'oklch(0.50 0.14 155)'  // green — decreases delay
              }
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

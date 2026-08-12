import React from 'react';
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Legend,
} from 'recharts';

interface DataPoint {
  week: string;
  historical?: number;
  forecast?: number;
}

interface ForecastChartProps {
  history: { week: string; sales: number }[];
  forecast: { week: string; sales: number }[];
  forecastStartDate: string;
}

const formatSales = (value: number) => {
  if (value >= 1_000_000) return `₹${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000)     return `₹${(value / 1_000).toFixed(0)}K`;
  return `₹${value}`;
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
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
      <p style={{ color: '#94a3b8', marginBottom: 4 }}>{label}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color, margin: '2px 0' }}>
          {p.name === 'Historical' ? '● Historical: ' : '◆ Forecast: '}
          <strong>{formatSales(p.value)}</strong>
        </p>
      ))}
    </div>
  );
};

export const ForecastChart = ({ history, forecast, forecastStartDate }: ForecastChartProps) => {
  // Merge history + forecast into a single data array
  const data: DataPoint[] = [
    ...history.map(h => ({ week: h.week, historical: h.sales })),
    ...forecast.map(f => ({ week: f.week, forecast: f.sales })),
  ];

  // Connect the seam: last history point also appears as first forecast point for visual continuity
  if (history.length > 0 && forecast.length > 0) {
    const lastHistIdx = history.length - 1;
    data[lastHistIdx] = {
      ...data[lastHistIdx],
      forecast: history[lastHistIdx].sales,
    };
  }

  const primaryColor   = 'oklch(0.65 0.19 255)';
  const mutedColor     = 'oklch(0.55 0.05 260)';
  const forecastAreaFill = 'oklch(0.65 0.19 255 / 0.15)';

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
        <XAxis
          dataKey="week"
          tick={{ fill: '#64748b', fontSize: 11 }}
          tickFormatter={v => v.slice(5)}   // show MM-DD only
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tickFormatter={formatSales}
          tick={{ fill: '#64748b', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={60}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: '0.8rem', color: '#94a3b8', paddingTop: 8 }}
        />

        {/* Forecast area fill */}
        <Area
          dataKey="forecast"
          fill={forecastAreaFill}
          stroke="none"
          name="Forecast"
          legendType="none"
          activeDot={false}
          isAnimationActive
        />

        {/* Historical line */}
        <Line
          dataKey="historical"
          name="Historical"
          stroke={mutedColor}
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: mutedColor }}
          isAnimationActive
        />

        {/* Forecast line */}
        <Line
          dataKey="forecast"
          name="Forecast"
          stroke={primaryColor}
          strokeWidth={2.5}
          strokeDasharray="6 3"
          dot={{ r: 3, fill: primaryColor }}
          activeDot={{ r: 5, fill: primaryColor }}
          isAnimationActive
        />

        {/* Forecast Start separator */}
        <ReferenceLine
          x={forecastStartDate}
          stroke="var(--border)"
          strokeDasharray="4 2"
          label={{
            value: 'Forecast Start →',
            position: 'insideTopLeft',
            fill: '#64748b',
            fontSize: 11,
            dx: 6,
            dy: -4,
          }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

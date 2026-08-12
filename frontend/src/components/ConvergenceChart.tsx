import React from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts'

interface ConvergenceChartProps {
  data: any[]
}

export function ConvergenceChart({ data }: ConvergenceChartProps) {
  return (
    <div className="h-64 w-full mt-4">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
          <XAxis 
            dataKey="round" 
            stroke="#94a3b8" 
            fontSize={12} 
            tickLine={false} 
            axisLine={false}
          />
          <YAxis 
            yAxisId="left" 
            stroke="#94a3b8" 
            fontSize={12} 
            tickLine={false} 
            axisLine={false}
            domain={['auto', 'auto']}
          />
          <YAxis 
            yAxisId="right" 
            orientation="right" 
            stroke="#94a3b8" 
            fontSize={12} 
            tickLine={false} 
            axisLine={false}
            domain={['auto', 'auto']}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'var(--sidebar)', 
              borderColor: 'var(--border)', 
              borderRadius: '0.5rem',
              color: 'white'
            }}
            itemStyle={{ color: 'white' }}
          />
          <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="auc" 
            name="AUC-ROC"
            stroke="var(--primary)" 
            strokeWidth={2}
            dot={{ r: 3, fill: 'var(--primary)' }}
            activeDot={{ r: 5 }}
          />
          <Line 
            yAxisId="right"
            type="monotone" 
            dataKey="loss" 
            name="Validation Loss"
            stroke="#f59e0b" 
            strokeWidth={2}
            dot={{ r: 3, fill: '#f59e0b' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

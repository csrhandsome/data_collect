import { useMemo } from 'react'
import { nearestSample } from '../../lib/format'
import type { EEData } from '../../types/dataset'

const COLORS = ['#74d9b0', '#71aaf3', '#efbf77', '#dd8de4', '#ef8c8c', '#92c7cf', '#c2ce8a', '#bbb3ff']

export function SeriesPlot({ data, time, duration, onSeek }: {
  data: EEData; time: number; duration: number; onSeek: (value: number) => void
}) {
  const plot = useMemo(() => {
    const low = Math.min(...data.bounds.min)
    const high = Math.max(...data.bounds.max)
    const range = Math.max(high-low, 1e-6)
    const end = Math.max(duration, data.timestamps.at(-1) || 0, 0.001)
    return {
      end,
      paths: data.names.map((_, axis) => data.values.map((row, i) =>
        `${i ? 'L' : 'M'}${(40+data.timestamps[i]/end*460).toFixed(2)},${(180-(row[axis]-low)/range*145).toFixed(2)}`,
      ).join(' ')),
      low, high,
    }
  }, [data, duration])
  const current = data.values[nearestSample(data.timestamps, time)] || []
  return <div className="ee-trajectory">
    <svg viewBox="0 0 540 220" className="time-chart" role="img" aria-label={`${data.feature} 数值随时间变化`}
      onClick={event => onSeek(Math.max(0, Math.min(plot.end,
        ((event.clientX-event.currentTarget.getBoundingClientRect().left)/event.currentTarget.getBoundingClientRect().width*540-40)/460*plot.end)))}>
      <text x="5" y="30" className="chart-tick">{plot.high.toFixed(3)}</text>
      <text x="5" y="185" className="chart-tick">{plot.low.toFixed(3)}</text>
      {plot.paths.map((path, i) => <path key={data.names[i]} d={path} stroke={COLORS[i%COLORS.length]} className="chart-line" />)}
      <line x1={40+time/plot.end*460} x2={40+time/plot.end*460} y1="30" y2="185" className="playhead-line" />
      <text x="40" y="210" className="chart-tick">0 s</text>
      <text x="465" y="210" className="chart-tick">{plot.end.toFixed(2)} s</text>
    </svg>
    <div className="series-readout">{data.names.map((name, i) =>
      <span key={name} style={{color: COLORS[i%COLORS.length]}}>{name}: {current[i]?.toFixed(4)} {data.units[i]}</span>)}</div>
  </div>
}

import { useMemo, useState } from 'react'
import { nearestSample } from '../../lib/format'
import type { EEData } from '../../types/dataset'
import { Icon } from '../ui/Icon'

const COLORS = ['#5ce1bf', '#72a9ff', '#ecb36e']
const AXES = ['X', 'Y', 'Z']
const WIDTH = 540
const HEIGHT = 220
const PLOT = { left: 57, right: 18, top: 24, bottom: 34 }

interface EETrajectoryProps {
  data: EEData
  time: number
  duration: number
  onSeek: (time: number) => void
}

export function EETrajectory({ data, time, duration, onSeek }: EETrajectoryProps) {
  const [mode, setMode] = useState<'time' | 'space'>('time')
  const [hoverTime, setHoverTime] = useState<number | null>(null)
  const index = nearestSample(data.timestamps, time)
  const hoverIndex = hoverTime === null ? index : nearestSample(data.timestamps, hoverTime)
  const current = data.values[index] || []
  const inspected = data.values[hoverIndex] || []
  const timelineEnd = duration || data.timestamps.at(-1) || 1
  const chart = useMemo(() => {
    const plotWidth = WIDTH - PLOT.left - PLOT.right
    const plotHeight = HEIGHT - PLOT.top - PLOT.bottom
    let minimum = Math.min(...data.bounds.min.slice(0, 3))
    let maximum = Math.max(...data.bounds.max.slice(0, 3))
    const padding = Math.max((maximum - minimum) * 0.14, 0.015)
    minimum -= padding
    maximum += padding
    const x = (seconds: number) => PLOT.left + (seconds / timelineEnd) * plotWidth
    const y = (value: number) => PLOT.top + ((maximum - value) / (maximum - minimum)) * plotHeight
    const paths = AXES.map((_, axis) =>
      data.values
        .map(
          (row, sample) =>
            `${sample ? 'L' : 'M'}${x(data.timestamps[sample]).toFixed(2)},${y(row[axis]).toFixed(2)}`,
        )
        .join(' '),
    )
    const ticks = Array.from({ length: 5 }, (_, tick) => ({
      value: minimum + ((maximum - minimum) * tick) / 4,
      y: PLOT.top + plotHeight * (1 - tick / 4),
    }))
    return { paths, ticks, x, y, plotWidth, plotHeight }
  }, [data, timelineEnd])

  const spatial = useMemo(() => {
    const min = data.bounds.min
    const max = data.bounds.max
    const ranges = [0, 1, 2].map((axis) => Math.max(max[axis] - min[axis], 0.001))
    const largestRange = Math.max(...ranges)
    const normalize = (row: number[]) =>
      row.slice(0, 3).map((value, axis) => (value - (min[axis] + max[axis]) / 2) / largestRange)
    const project = (row: number[]) => ({
      x: WIDTH / 2 + (row[0] - row[1]) * 152,
      y: HEIGHT / 2 + (row[0] + row[1]) * 46 - row[2] * 138,
    })
    const points = data.values.map((row) => project(normalize(row)))
    return {
      points,
      path: points
        .map((point, sample) => `${sample ? 'L' : 'M'}${point.x.toFixed(2)},${point.y.toFixed(2)}`)
        .join(' '),
      project,
    }
  }, [data])

  function eventTime(event: { currentTarget: SVGSVGElement; clientX: number }) {
    const rect = event.currentTarget.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * WIDTH
    return Math.max(0, Math.min(timelineEnd, ((x - PLOT.left) / chart.plotWidth) * timelineEnd))
  }

  const spatialCurrent = spatial.points[index]
  return (
    <div className="ee-trajectory" data-testid="ee-trajectory">
      <div className="chart-toolbar">
        <div className="segmented-control" role="tablist" aria-label="末端视图">
          <button
            role="tab"
            aria-selected={mode === 'time'}
            className={mode === 'time' ? 'is-active' : ''}
            onClick={() => setMode('time')}
          >
            <Icon name="signal" size={13} />
            时间曲线
          </button>
          <button
            role="tab"
            aria-selected={mode === 'space'}
            className={mode === 'space' ? 'is-active' : ''}
            onClick={() => setMode('space')}
          >
            <Icon name="trajectory" size={13} />
            空间轨迹
          </button>
        </div>
        <div className="chart-legend">
          {AXES.map((axis, i) => (
            <span key={axis}>
              <i style={{ backgroundColor: COLORS[i] }} />
              {axis}
            </span>
          ))}
        </div>
      </div>
      {mode === 'time' ? (
        <div className="chart-stage">
          <svg
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            role="img"
            aria-label="末端 XYZ 位置随时间变化曲线，单位米"
            className="time-chart"
            onPointerMove={(event) => setHoverTime(eventTime(event))}
            onPointerLeave={() => setHoverTime(null)}
            onClick={(event) => onSeek(eventTime(event))}
          >
            <text x="13" y="15" className="axis-caption">
              位置 / m
            </text>
            {chart.ticks.map((tick, tickIndex) => (
              <g key={tickIndex}>
                <line
                  x1={PLOT.left}
                  x2={WIDTH - PLOT.right}
                  y1={tick.y}
                  y2={tick.y}
                  className="chart-grid-line"
                />
                <text x={PLOT.left - 10} y={tick.y + 4} textAnchor="end" className="chart-tick">
                  {tick.value.toFixed(2)}
                </text>
              </g>
            ))}
            {Array.from({ length: 7 }, (_, tick) => (
              <g key={tick}>
                <line
                  x1={chart.x((timelineEnd * tick) / 6)}
                  x2={chart.x((timelineEnd * tick) / 6)}
                  y1={PLOT.top}
                  y2={HEIGHT - PLOT.bottom}
                  className="chart-grid-line vertical"
                />
                <text
                  x={chart.x((timelineEnd * tick) / 6)}
                  y={HEIGHT - 13}
                  textAnchor="middle"
                  className="chart-tick"
                >
                  {((timelineEnd * tick) / 6).toFixed(1)}
                </text>
              </g>
            ))}
            {chart.paths.map((path, axis) => (
              <path key={axis} d={path} stroke={COLORS[axis]} className="chart-line" />
            ))}
            <line
              x1={chart.x(time)}
              x2={chart.x(time)}
              y1={PLOT.top}
              y2={HEIGHT - PLOT.bottom}
              className="playhead-line"
            />
            <path d={`M${chart.x(time) - 4} ${PLOT.top - 6}h8l-4 5Z`} fill="#e9f4ef" />
            {current.slice(0, 3).map((value, axis) => (
              <circle
                key={axis}
                cx={chart.x(data.timestamps[index])}
                cy={chart.y(value)}
                r="3.2"
                fill={COLORS[axis]}
                stroke="#151d22"
                strokeWidth="2"
              />
            ))}
            {hoverTime !== null ? (
              <line
                x1={chart.x(data.timestamps[hoverIndex])}
                x2={chart.x(data.timestamps[hoverIndex])}
                y1={PLOT.top}
                y2={HEIGHT - PLOT.bottom}
                className="hover-line"
              />
            ) : null}
            <text x={WIDTH - 18} y={HEIGHT - 2} textAnchor="end" className="axis-caption">
              时间 / s
            </text>
          </svg>
          <div className="chart-hint">
            {hoverTime === null
              ? '点击曲线定位时间'
              : `采样时间 ${data.timestamps[hoverIndex].toFixed(3)} s`}
          </div>
        </div>
      ) : (
        <div className="chart-stage spatial-stage">
          <svg
            viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
            role="img"
            aria-label="末端 XYZ 空间轨迹的等轴测投影，单位米"
          >
            <defs>
              <pattern
                id={`grid-${data.feature.replace(/[^a-zA-Z0-9_-]/g, '')}`}
                width="26"
                height="26"
                patternUnits="userSpaceOnUse"
                patternTransform="matrix(1 .32 -1 .32 270 135)"
              >
                <path d="M26 0H0V26" fill="none" stroke="#273339" strokeWidth=".7" />
              </pattern>
            </defs>
            <path
              d="M85 140 270 77 455 140 270 203Z"
              fill={`url(#grid-${data.feature.replace(/[^a-zA-Z0-9_-]/g, '')})`}
              opacity=".65"
            />
            <g className="spatial-axes">
              <line x1="72" y1="160" x2="111" y2="173" stroke={COLORS[0]} />
              <line x1="72" y1="160" x2="33" y2="173" stroke={COLORS[1]} />
              <line x1="72" y1="160" x2="72" y2="123" stroke={COLORS[2]} />
              <text x="116" y="178" fill={COLORS[0]}>
                X
              </text>
              <text x="22" y="178" fill={COLORS[1]}>
                Y
              </text>
              <text x="68" y="116" fill={COLORS[2]}>
                Z
              </text>
            </g>
            <path d={spatial.path} stroke="#385c56" className="spatial-path" />
            <path
              d={spatial.points
                .slice(0, index + 1)
                .map(
                  (point, sample) =>
                    `${sample ? 'L' : 'M'}${point.x.toFixed(2)},${point.y.toFixed(2)}`,
                )
                .join(' ')}
              stroke={COLORS[0]}
              className="spatial-path spatial-played"
            />
            {spatial.points[0] ? (
              <g>
                <circle cx={spatial.points[0].x} cy={spatial.points[0].y} r="3" fill="#829c94" />
                <text
                  x={spatial.points[0].x + 8}
                  y={spatial.points[0].y + 3}
                  className="chart-tick"
                >
                  起点
                </text>
              </g>
            ) : null}
            {spatialCurrent ? (
              <g>
                <circle
                  cx={spatialCurrent.x}
                  cy={spatialCurrent.y}
                  r="9"
                  fill={COLORS[0]}
                  opacity=".13"
                />
                <circle
                  cx={spatialCurrent.x}
                  cy={spatialCurrent.y}
                  r="4"
                  fill={COLORS[0]}
                  stroke="#e0fff5"
                  strokeWidth="1.2"
                />
              </g>
            ) : null}
            <text x={WIDTH - 18} y="20" textAnchor="end" className="axis-caption">
              XYZ · 等轴测投影
            </text>
            <text x={WIDTH - 18} y={HEIGHT - 12} textAnchor="end" className="axis-caption">
              坐标单位 / m
            </text>
          </svg>
          <div className="chart-hint">跟随时间轴显示当前位置</div>
        </div>
      )}
      <div className="position-readout" data-testid={`ee-values-${data.feature}`}>
        {AXES.map((axis, axisIndex) => (
          <div key={axis}>
            <span>
              <i style={{ background: COLORS[axisIndex] }} />
              {axis}
            </span>
            <strong>
              {(mode === 'time' ? inspected[axisIndex] : current[axisIndex])?.toFixed(4) || '—'}
              <small>m</small>
            </strong>
          </div>
        ))}
      </div>
      <div className="ee-details">
        <span>
          <span className="status-dot" />
          {data.point_count.toLocaleString()} / {data.total_points.toLocaleString()} 个采样点
        </span>
        {current.length > 3 ? (
          <code
            title={data.names
              .slice(3)
              .map((name, axis) => `${name} (${data.units[axis + 3] || '1'})`)
              .join(' / ')}
          >
            {current.length === 7 ? 'QUAT' : current.length === 6 ? 'RPY' : 'POSE'}{' '}
            {current
              .slice(3)
              .map((value) => value.toFixed(2))
              .join(' / ')}{' '}
            {data.units[3] === 'rad' ? 'rad' : ''}
          </code>
        ) : (
          <code>XYZ POSITION</code>
        )}
      </div>
    </div>
  )
}

import { SERIES_STYLES } from '../../lib/seriesStyles'
import { useId, useMemo, useState } from 'react'
import { nearestSample } from '../../lib/format'
import type { EEData } from '../../types/dataset'
import { Icon } from '../ui/Icon'

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
  const panelId = useId()
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
    <div className="bg-background" data-testid="ee-trajectory">
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 pt-4">
        <div
          className="inline-flex border border-foreground"
          role="tablist"
          aria-label="末端视图"
          onKeyDown={(event) => {
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
            event.preventDefault()
            const next =
              event.key === 'Home'
                ? 'time'
                : event.key === 'End'
                  ? 'space'
                  : mode === 'time'
                    ? 'space'
                    : 'time'
            setMode(next)
            event.currentTarget
              .querySelectorAll<HTMLButtonElement>('button')
              [next === 'time' ? 0 : 1]?.focus()
          }}
        >
          <button
            role="tab"
            aria-selected={mode === 'time'}
            id={`${panelId}-time`}
            aria-controls={panelId}
            tabIndex={mode === 'time' ? 0 : -1}
            className={`flex min-h-11 items-center gap-2 px-3 font-mono text-[11px] transition-none ${mode === 'time' ? 'bg-foreground text-background' : 'bg-background text-foreground hover:bg-muted'}`}
            onClick={() => setMode('time')}
          >
            <Icon name="signal" size={13} />
            时间曲线
          </button>
          <button
            role="tab"
            aria-selected={mode === 'space'}
            id={`${panelId}-space`}
            aria-controls={panelId}
            tabIndex={mode === 'space' ? 0 : -1}
            className={`flex min-h-11 items-center gap-2 px-3 font-mono text-[11px] transition-none ${mode === 'space' ? 'bg-foreground text-background' : 'bg-background text-foreground hover:bg-muted'}`}
            onClick={() => setMode('space')}
          >
            <Icon name="trajectory" size={13} />
            空间轨迹
          </button>
        </div>
        <div className="flex gap-3 font-mono text-[11px] [&>span]:flex [&>span]:items-center [&>span]:gap-1.5 [&_i]:inline-block [&_i]:w-4 [&_i]:border-t-2 [&_i]:border-foreground">
          {AXES.map((axis, i) => (
            <span key={axis}>
              <i style={{ borderTopStyle: SERIES_STYLES[i].border }} />
              {axis}
            </span>
          ))}
        </div>
      </div>
      <div id={panelId} role="tabpanel" aria-labelledby={`${panelId}-${mode}`}>
        {mode === 'time' ? (
          <div className="relative px-2 pt-4 pb-5">
            <svg
              viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
              className="block h-auto w-full"
              role="img"
              aria-label="末端 XYZ 位置随时间变化曲线，单位米"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
                  event.preventDefault()
                  onSeek(
                    Math.max(
                      0,
                      Math.min(timelineEnd, time + (event.key === 'ArrowRight' ? 0.1 : -0.1)),
                    ),
                  )
                }
              }}
              onPointerMove={(event) => setHoverTime(eventTime(event))}
              onPointerLeave={() => setHoverTime(null)}
              onClick={(event) => onSeek(eventTime(event))}
            >
              <text x="13" y="15" className="fill-muted-foreground font-mono text-[9px]">
                位置 / m
              </text>
              {chart.ticks.map((tick, tickIndex) => (
                <g key={tickIndex}>
                  <line
                    x1={PLOT.left}
                    x2={WIDTH - PLOT.right}
                    y1={tick.y}
                    y2={tick.y}
                    className="stroke-border-light [stroke-width:0.75]"
                  />
                  <text
                    x={PLOT.left - 10}
                    y={tick.y + 4}
                    textAnchor="end"
                    className="fill-muted-foreground font-mono text-[10px]"
                  >
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
                    className="stroke-border-light [stroke-width:0.75] [stroke-dasharray:2_4]"
                  />
                  <text
                    x={chart.x((timelineEnd * tick) / 6)}
                    y={HEIGHT - 13}
                    textAnchor="middle"
                    className="fill-muted-foreground font-mono text-[10px]"
                  >
                    {((timelineEnd * tick) / 6).toFixed(1)}
                  </text>
                </g>
              ))}
              {chart.paths.map((path, axis) => (
                <path
                  key={axis}
                  d={path}
                  stroke="var(--color-foreground)"
                  strokeDasharray={SERIES_STYLES[axis].dash}
                  className="fill-none [stroke-width:1.65] [stroke-linejoin:round]"
                />
              ))}
              <line
                x1={chart.x(time)}
                x2={chart.x(time)}
                y1={PLOT.top}
                y2={HEIGHT - PLOT.bottom}
                className="stroke-foreground [stroke-width:1]"
              />
              <path
                d={`M${chart.x(time) - 4} ${PLOT.top - 6}h8l-4 5Z`}
                fill="var(--color-foreground)"
              />
              {current.slice(0, 3).map((value, axis) => (
                <circle
                  key={axis}
                  cx={chart.x(data.timestamps[index])}
                  cy={chart.y(value)}
                  r="3.2"
                  fill="var(--color-foreground)"
                  stroke="var(--color-background)"
                  strokeWidth="2"
                />
              ))}
              {hoverTime !== null ? (
                <line
                  x1={chart.x(data.timestamps[hoverIndex])}
                  x2={chart.x(data.timestamps[hoverIndex])}
                  y1={PLOT.top}
                  y2={HEIGHT - PLOT.bottom}
                  className="stroke-muted-foreground [stroke-width:0.8] [stroke-dasharray:3_3]"
                />
              ) : null}
              <text
                x={WIDTH - 18}
                y={HEIGHT - 2}
                textAnchor="end"
                className="fill-muted-foreground font-mono text-[9px]"
              >
                时间 / s
              </text>
            </svg>
            <div className="pointer-events-none absolute bottom-1 left-4 font-mono text-[10px] text-muted-foreground">
              {hoverTime === null
                ? '点击曲线定位时间'
                : `采样时间 ${data.timestamps[hoverIndex].toFixed(3)} s`}
            </div>
          </div>
        ) : (
          <div className="relative px-2 pt-4 pb-5">
            <svg
              viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
              className="block h-auto w-full"
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
                  <path
                    d="M26 0H0V26"
                    fill="none"
                    stroke="var(--color-border-light)"
                    strokeWidth=".7"
                  />
                </pattern>
              </defs>
              <path
                d="M85 140 270 77 455 140 270 203Z"
                fill={`url(#grid-${data.feature.replace(/[^a-zA-Z0-9_-]/g, '')})`}
                opacity=".65"
              />
              <g className="[&_line]:[stroke-width:1.5] [&_text]:font-mono [&_text]:text-[10px]">
                <line x1="72" y1="160" x2="111" y2="173" stroke="var(--color-foreground)" />
                <line x1="72" y1="160" x2="33" y2="173" stroke="var(--color-foreground)" />
                <line x1="72" y1="160" x2="72" y2="123" stroke="var(--color-foreground)" />
                <text x="116" y="178" fill="var(--color-foreground)">
                  X
                </text>
                <text x="22" y="178" fill="var(--color-foreground)">
                  Y
                </text>
                <text x="68" y="116" fill="var(--color-foreground)">
                  Z
                </text>
              </g>
              <path
                d={spatial.path}
                stroke="var(--color-border-light)"
                className="fill-none [stroke-width:2]"
              />
              <path
                d={spatial.points
                  .slice(0, index + 1)
                  .map(
                    (point, sample) =>
                      `${sample ? 'L' : 'M'}${point.x.toFixed(2)},${point.y.toFixed(2)}`,
                  )
                  .join(' ')}
                stroke="var(--color-foreground)"
                className="fill-none [stroke-width:2] [stroke-width:2.3]"
              />
              {spatial.points[0] ? (
                <g>
                  <circle
                    cx={spatial.points[0].x}
                    cy={spatial.points[0].y}
                    r="3"
                    fill="var(--color-foreground)"
                  />
                  <text
                    x={spatial.points[0].x + 8}
                    y={spatial.points[0].y + 3}
                    className="fill-muted-foreground font-mono text-[10px]"
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
                    fill="var(--color-foreground)"
                    opacity=".13"
                  />
                  <circle
                    cx={spatialCurrent.x}
                    cy={spatialCurrent.y}
                    r="4"
                    fill="var(--color-foreground)"
                    stroke="var(--color-background)"
                    strokeWidth="1.2"
                  />
                </g>
              ) : null}
              <text
                x={WIDTH - 18}
                y="20"
                textAnchor="end"
                className="fill-muted-foreground font-mono text-[9px]"
              >
                XYZ · 等轴测投影
              </text>
              <text
                x={WIDTH - 18}
                y={HEIGHT - 12}
                textAnchor="end"
                className="fill-muted-foreground font-mono text-[9px]"
              >
                坐标单位 / m
              </text>
            </svg>
            <div className="pointer-events-none absolute bottom-1 left-4 font-mono text-[10px] text-muted-foreground">
              跟随时间轴显示当前位置
            </div>
          </div>
        )}
      </div>
      <div
        className="mx-4 mt-3 grid grid-cols-3 border-t border-foreground py-4 [&>div]:min-w-0 [&>div+div]:border-l [&>div+div]:border-border-light [&>div+div]:pl-3 [&_span]:mb-2 [&_span]:flex [&_span]:items-center [&_span]:gap-2 [&_span]:font-mono [&_span]:text-[11px] [&_i]:w-4 [&_i]:border-t-2 [&_i]:border-foreground [&_strong]:font-mono [&_strong]:text-xs [&_strong]:font-normal [&_strong]:sm:text-sm [&_small]:pl-1 [&_small]:text-[10px] [&_small]:text-muted-foreground"
        data-testid={`ee-values-${data.feature}`}
      >
        {AXES.map((axis, axisIndex) => (
          <div key={axis}>
            <span>
              <i style={{ borderTopStyle: SERIES_STYLES[axisIndex].border }} />
              {axis}
            </span>
            <strong>
              {(mode === 'time' ? inspected[axisIndex] : current[axisIndex])?.toFixed(4) || '—'}
              <small>m</small>
            </strong>
          </div>
        ))}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border-light px-4 py-3 font-mono text-[10px] text-muted-foreground [&>span]:flex [&>span]:items-center [&>span]:gap-2">
        <span>
          <span className="inline-block size-1.5 shrink-0 bg-current" />
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

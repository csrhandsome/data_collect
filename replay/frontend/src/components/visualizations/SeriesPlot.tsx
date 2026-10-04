import { SERIES_STYLES } from '../../lib/seriesStyles'
import { useMemo } from 'react'
import { nearestSample } from '../../lib/format'
import type { EEData } from '../../types/dataset'

export function SeriesPlot({
  data,
  time,
  duration,
  onSeek,
}: {
  data: EEData
  time: number
  duration: number
  onSeek: (value: number) => void
}) {
  const plot = useMemo(() => {
    const low = Math.min(...data.bounds.min)
    const high = Math.max(...data.bounds.max)
    const range = Math.max(high - low, 1e-6)
    const end = Math.max(duration, data.timestamps.at(-1) || 0, 0.001)
    return {
      end,
      paths: data.names.map((_, axis) =>
        data.values
          .map(
            (row, i) =>
              `${i ? 'L' : 'M'}${(40 + (data.timestamps[i] / end) * 460).toFixed(2)},${(180 - ((row[axis] - low) / range) * 145).toFixed(2)}`,
          )
          .join(' '),
      ),
      low,
      high,
    }
  }, [data, duration])
  const current = data.values[nearestSample(data.timestamps, time)] || []
  return (
    <div className="bg-background">
      <svg
        viewBox="0 0 540 220"
        className="block h-auto w-full cursor-crosshair touch-pan-y"
        role="img"
        aria-label={`${data.feature} 数值随时间变化`}
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
            event.preventDefault()
            onSeek(
              Math.max(0, Math.min(plot.end, time + (event.key === 'ArrowRight' ? 0.1 : -0.1))),
            )
          }
        }}
        onClick={(event) =>
          onSeek(
            Math.max(
              0,
              Math.min(
                plot.end,
                ((((event.clientX - event.currentTarget.getBoundingClientRect().left) /
                  event.currentTarget.getBoundingClientRect().width) *
                  540 -
                  40) /
                  460) *
                  plot.end,
              ),
            ),
          )
        }
      >
        <text x="5" y="30" className="fill-muted-foreground font-mono text-[10px]">
          {plot.high.toFixed(3)}
        </text>
        <text x="5" y="185" className="fill-muted-foreground font-mono text-[10px]">
          {plot.low.toFixed(3)}
        </text>
        {plot.paths.map((path, i) => (
          <path
            key={data.names[i]}
            d={path}
            stroke="var(--color-foreground)"
            strokeDasharray={SERIES_STYLES[i % SERIES_STYLES.length].dash}
            className="fill-none [stroke-width:1.65] [stroke-linejoin:round]"
          />
        ))}
        <line
          x1={40 + (time / plot.end) * 460}
          x2={40 + (time / plot.end) * 460}
          y1="30"
          y2="185"
          className="stroke-foreground [stroke-width:1]"
        />
        <text x="40" y="210" className="fill-muted-foreground font-mono text-[10px]">
          0 s
        </text>
        <text x="465" y="210" className="fill-muted-foreground font-mono text-[10px]">
          {plot.end.toFixed(2)} s
        </text>
      </svg>
      <div className="flex flex-wrap gap-x-4 gap-y-3 border-t border-border-light px-4 py-4 font-mono text-[11px] [&>span]:flex [&>span]:items-center [&>span]:gap-2">
        {data.names.map((name, i) => (
          <span key={name}>
            <svg aria-hidden="true" width="20" height="8">
              <line
                x1="0"
                x2="20"
                y1="4"
                y2="4"
                stroke="var(--color-foreground)"
                strokeWidth="2"
                strokeDasharray={SERIES_STYLES[i % SERIES_STYLES.length].dash}
              />
            </svg>
            {name}: {current[i]?.toFixed(4)} {data.units[i]}
          </span>
        ))}
      </div>
    </div>
  )
}

import { timeLabel } from '../../lib/format'
import { Icon } from '../ui/Icon'
import { Texture } from '../ui/Texture'

interface TimelineProps {
  time: number
  duration: number
  playing: boolean
  rate: number
  fps: number
  disabled: boolean
  onToggle: () => void
  onSeek: (time: number) => void
  onRateChange: (rate: number) => void
}

const frameButton =
  'inline-flex size-11 shrink-0 items-center justify-center border border-background/30 enabled:hover:bg-background enabled:hover:text-foreground'

export function Timeline({
  time,
  duration,
  playing,
  rate,
  fps,
  disabled,
  onToggle,
  onSeek,
  onRateChange,
}: TimelineProps) {
  const progress = duration > 0 ? (time / duration) * 100 : 0
  const frame = Math.min(Math.floor(time * fps), Math.max(0, Math.ceil(duration * fps) - 1))
  return (
    <section
      className="relative z-30 md:sticky md:bottom-0 isolate shrink-0 border-t-4 border-foreground bg-foreground px-4 py-4 text-background md:px-6 [&_:focus-visible]:outline-background"
      aria-label="工作区时间轴"
    >
      <Texture pattern="inverted" />
      <div className="flex items-center gap-3">
        <span className="font-mono text-[10px]">00:00</span>
        <div className="relative flex min-w-0 flex-1 items-center">
          <div
            className="pointer-events-none absolute inset-x-0 top-1 flex justify-between"
            aria-hidden="true"
          >
            {Array.from({ length: 13 }, (_, index) => (
              <i key={index} className="h-1.5 border-l border-background/30 nth-[3n+1]:h-2.5" />
            ))}
          </div>
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-x-0 h-px bg-background/30"
          >
            <span
              className="absolute inset-y-0 left-0 bg-background"
              style={{ width: `${progress}%` }}
            />
          </div>
          <input
            type="range"
            min={0}
            max={duration || 1}
            step={fps > 0 ? 1 / fps : 0.001}
            value={time}
            onChange={(event) => onSeek(Number(event.target.value))}
            disabled={disabled}
            className="relative z-10 m-0 h-11 w-full cursor-pointer appearance-none bg-transparent disabled:cursor-default [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-none [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-background [&::-webkit-slider-thumb]:bg-foreground [&::-moz-range-thumb]:h-5 [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:rounded-none [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-background [&::-moz-range-thumb]:bg-foreground"
            aria-label="工作区播放进度"
            aria-valuetext={`${time.toFixed(2)} 秒，共 ${duration.toFixed(2)} 秒`}
            data-testid="timeline-slider"
          />
        </div>
        <span className="font-mono text-[10px]">{timeLabel(duration)}</span>
      </div>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2 md:gap-3">
          <button
            type="button"
            className={frameButton}
            onClick={() => onSeek(time - 1 / fps)}
            disabled={disabled}
            title="上一帧"
            aria-label="上一帧"
          >
            <Icon name="previous" size={18} />
          </button>
          <button
            type="button"
            className="inline-flex size-11 shrink-0 items-center justify-center border-2 border-background bg-background text-foreground enabled:hover:bg-foreground enabled:hover:text-background focus-visible:outline-background!"
            onClick={onToggle}
            disabled={disabled}
            aria-label={playing ? '暂停工作区' : '播放工作区'}
            data-testid="play-toggle"
          >
            <Icon name={playing ? 'pause' : 'play'} size={20} />
          </button>
          <button
            type="button"
            className={frameButton}
            onClick={() => onSeek(time + 1 / fps)}
            disabled={disabled}
            title="下一帧"
            aria-label="下一帧"
          >
            <Icon name="next" size={18} />
          </button>
          <div className="ml-1 flex flex-col gap-1 font-mono sm:flex-row sm:items-baseline sm:gap-2">
            <strong className="text-sm font-normal" data-testid="timeline-time">
              {timeLabel(time, true)}
            </strong>
            <span className="text-[10px]">/ {timeLabel(duration, true)}</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <span className="hidden items-center gap-2 font-mono text-[10px] 2xl:flex">
            <Icon name={playing ? 'play' : 'pause'} size={12} />
            同步回放
          </span>
          <code className="hidden text-[10px] xl:block">
            F {frame.toString().padStart(4, '0')} · {fps} Hz
          </code>
          <select
            value={rate}
            onChange={(event) => onRateChange(Number(event.target.value))}
            className="min-h-11 min-w-14 border border-background bg-foreground px-2 font-mono text-xs text-background enabled:hover:bg-background enabled:hover:text-foreground focus-visible:outline-background!"
            aria-label="播放速度"
            disabled={disabled}
          >
            {[0.25, 0.5, 1, 1.5, 2].map((speed) => (
              <option key={speed} value={speed}>
                {speed}×
              </option>
            ))}
          </select>
        </div>
      </div>
    </section>
  )
}

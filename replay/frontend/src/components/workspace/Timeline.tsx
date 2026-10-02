import type { CSSProperties } from 'react'
import { timeLabel } from '../../lib/format'
import { Icon } from '../ui/Icon'

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
    <section className="timeline" aria-label="工作区时间轴">
      <div className="timeline-track-row">
        <span className="time-edge">00:00</span>
        <div className="timeline-track">
          <div className="timeline-ticks" aria-hidden="true">
            {Array.from({ length: 13 }, (_, index) => (
              <i key={index} />
            ))}
          </div>
          <input
            type="range"
            min={0}
            max={duration || 1}
            step={fps > 0 ? 1 / fps : 0.001}
            value={time}
            onChange={(event) => onSeek(Number(event.target.value))}
            disabled={disabled}
            className="timeline-range"
            style={{ '--progress': `${progress}%` } as CSSProperties}
            aria-label="工作区播放进度"
            data-testid="timeline-slider"
          />
        </div>
        <span className="time-edge">{timeLabel(duration)}</span>
      </div>
      <div className="timeline-controls">
        <div className="playback-buttons">
          <button
            className="icon-button frame-button"
            onClick={() => onSeek(time - 1 / fps)}
            disabled={disabled}
            title="上一帧"
            aria-label="上一帧"
          >
            <Icon name="previous" size={16} />
          </button>
          <button
            className="play-button"
            onClick={onToggle}
            disabled={disabled}
            aria-label={playing ? '暂停工作区' : '播放工作区'}
            data-testid="play-toggle"
          >
            <Icon name={playing ? 'pause' : 'play'} size={20} />
          </button>
          <button
            className="icon-button frame-button"
            onClick={() => onSeek(time + 1 / fps)}
            disabled={disabled}
            title="下一帧"
            aria-label="下一帧"
          >
            <Icon name="next" size={16} />
          </button>
          <div className="time-readout">
            <strong data-testid="timeline-time">{timeLabel(time, true)}</strong>
            <span>/ {timeLabel(duration, true)}</span>
          </div>
        </div>
        <div className="timeline-info">
          <span className={`sync-indicator ${playing ? 'is-playing' : ''}`}>
            <span className="status-dot" />
            同步回放
          </span>
          <code className="frame-readout">
            F {frame.toString().padStart(4, '0')} <span>· {fps} Hz</span>
          </code>
          <select
            value={rate}
            onChange={(event) => onRateChange(Number(event.target.value))}
            className="speed-select"
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

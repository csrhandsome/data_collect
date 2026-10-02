import { useState } from 'react'
import type { DragEvent, ReactNode } from 'react'
import { FEATURE_DRAG_TYPE } from '../../lib/api'
import type { DragFeature } from '../../types/dataset'
import { Icon } from '../ui/Icon'

interface ReferencePanelProps {
  children: ReactNode
  count: number
  disabled: boolean
  onDropFeature: (payload: DragFeature) => void
  onAddDefaults: () => void
}

function isDragFeature(value: unknown): value is DragFeature {
  if (!value || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  return (
    typeof record.datasetId === 'string' &&
    typeof record.feature === 'string' &&
    Number.isInteger(record.episodeIndex)
  )
}

export function ReferencePanel({
  children,
  count,
  disabled,
  onDropFeature,
  onAddDefaults,
}: ReferencePanelProps) {
  const [dragOver, setDragOver] = useState(false)
  function allowDrop(event: DragEvent<HTMLElement>) {
    if (disabled || !event.dataTransfer.types.includes(FEATURE_DRAG_TYPE)) return
    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
    setDragOver(true)
  }
  function drop(event: DragEvent<HTMLElement>) {
    event.preventDefault()
    setDragOver(false)
    if (disabled) return
    try {
      const payload: unknown = JSON.parse(event.dataTransfer.getData(FEATURE_DRAG_TYPE))
      if (isDragFeature(payload)) onDropFeature(payload)
    } catch {
      /* Ignore drags from outside this workspace. */
    }
  }

  return (
    <section
      className={`reference-panel ${dragOver ? 'is-drag-over' : ''} ${count ? 'has-cards' : ''}`}
      data-testid="reference-panel"
      aria-label="参考栏"
      onDragOver={allowDrop}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragOver(false)
      }}
      onDrop={drop}
    >
      {count ? (
        <>
          <div className="visualization-grid">{children}</div>
          <div className="drop-more">
            <Icon name="plus" size={17} />
            <span>继续拖入数据字段，添加一个视图</span>
            <code>DROP TO ADD</code>
          </div>
        </>
      ) : (
        <div className="empty-workspace">
          <div className="empty-illustration" aria-hidden="true">
            <div className="illustration-grid" />
            <div className="floating-card floating-card-video">
              <Icon name="video" size={22} />
              <span>CAMERA STREAM</span>
              <div className="mini-video">
                <span />
              </div>
            </div>
            <div className="floating-card floating-card-plot">
              <Icon name="trajectory" size={19} />
              <span>EE TRAJECTORY</span>
              <svg viewBox="0 0 140 55">
                <path d="M0 45C15 45 12 13 30 20S48 52 65 30 86 30 98 15s24-3 42-7" />
              </svg>
            </div>
            <span className="drop-cross">
              <Icon name="plus" size={20} />
            </span>
          </div>
          <p className="eyebrow">YOUR DATA, IN MOTION</p>
          <h2>把数据拖进来，开始回放。</h2>
          <p className="empty-description">
            从左侧选择相机、音频或状态数据，拖动到参考栏。
            <br />
            多个视图将跟随同一条时间轴同步播放。
          </p>
          <button className="button button-primary" disabled={disabled} onClick={onAddDefaults}>
            <Icon name="plus" size={16} />
            添加相机与 EE
            <Icon name="arrow" size={16} />
          </button>
          <div className="supported-views">
            <span>
              <Icon name="video" size={14} />
              相机图片与视频
            </span>
            <span>
              <Icon name="trajectory" size={14} />
              末端、关节曲线与音频
            </span>
          </div>
          <p className="empty-keyboard-note">也可以点击数据块上的「添加」按钮</p>
        </div>
      )}
      {dragOver ? (
        <div className="drop-overlay">
          <Icon name="plus" size={26} />
          <strong>松开，添加到参考栏</strong>
        </div>
      ) : null}
    </section>
  )
}

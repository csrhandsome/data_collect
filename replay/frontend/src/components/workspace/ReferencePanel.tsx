import { Texture } from '../ui/Texture'
import { Button } from '../ui/Button'
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
      className={`relative mb-6 min-h-80 min-w-0 flex-1 md:overflow-y-auto md:p-1 ${dragOver ? 'outline-4 outline-offset-4 outline-foreground' : ''}`}
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
          <div className="grid grid-cols-1 items-start gap-5 xl:grid-cols-2 2xl:grid-cols-3">
            {children}
          </div>
          <div className="mt-5 flex flex-wrap items-center justify-center gap-3 border-y border-border-light py-5 text-sm text-muted-foreground [&>code]:font-mono [&>code]:text-[10px] [&>code]:tracking-widest">
            <Icon name="plus" size={17} />
            <span>继续拖入数据字段，添加一个视图</span>
            <code>DROP TO ADD</code>
          </div>
        </>
      ) : (
        <div className="relative isolate flex min-h-[400px] md:min-h-full flex-col items-center justify-center overflow-hidden border border-foreground px-5 py-10 text-center ">
          <Texture pattern="grid" />
          <div aria-hidden="true" className="mb-7 grid grid-cols-2 border border-foreground">
            <div className="flex h-20 w-28 flex-col items-center justify-center gap-2 border-r border-foreground">
              <Icon name="video" size={24} />
              <span className="font-mono text-[9px] tracking-widest">01 / CAMERA</span>
            </div>
            <div className="flex h-20 w-28 flex-col items-center justify-center gap-2 bg-foreground text-background">
              <Icon name="trajectory" size={24} />
              <span className="font-mono text-[9px] tracking-widest">02 / MOTION</span>
            </div>
          </div>
          <p className="font-mono text-[10px] tracking-[0.2em] text-muted-foreground">
            YOUR DATA, IN MOTION
          </p>
          <h2 className="mt-3 max-w-lg font-display text-3xl leading-tight tracking-tight lg:text-[40px]">
            把数据拖进来，
            <br className="sm:hidden" />
            开始回放。
          </h2>
          <p className="my-5 max-w-lg text-base leading-relaxed text-muted-foreground">
            从左侧选择相机、音频或状态数据，拖动到参考栏。
            <br />
            多个视图将跟随同一条时间轴同步播放。
          </p>
          <Button disabled={disabled} onClick={onAddDefaults}>
            <Icon name="plus" size={16} />
            添加相机与 EE
            <Icon name="arrow" size={16} />
          </Button>
          <div className="mt-7 flex flex-wrap justify-center gap-5 [&>span]:flex [&>span]:items-center [&>span]:gap-2 [&>span]:font-mono [&>span]:text-[11px]">
            <span>
              <Icon name="video" size={14} />
              相机图片与视频
            </span>
            <span>
              <Icon name="trajectory" size={14} />
              末端、关节曲线与音频
            </span>
          </div>
          <p className="mt-4 font-mono text-[10px] text-muted-foreground">
            也可以点击数据块上的「添加」按钮
          </p>
        </div>
      )}
      {dragOver ? (
        <div className="absolute inset-0 z-20 flex min-h-60 flex-col items-center justify-center gap-4 border-4 border-foreground bg-background text-foreground pointer-events-none [&>strong]:font-display [&>strong]:text-3xl [&>strong]:font-normal">
          <Icon name="plus" size={26} />
          <strong>松开，添加到参考栏</strong>
        </div>
      ) : null}
    </section>
  )
}

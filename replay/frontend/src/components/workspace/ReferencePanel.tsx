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
      className={`relative mb-3 min-h-80 min-w-0 flex-1 md:min-h-40 md:overflow-y-auto ${dragOver ? 'outline-4 outline-offset-4 outline-foreground' : ''}`}
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
          <div className="grid grid-cols-1 items-start gap-3 xl:grid-cols-2 2xl:grid-cols-3">
            {children}
          </div>
          <div className="mt-5 flex flex-wrap items-center justify-center gap-3 border-y border-border-light py-5 text-sm text-muted-foreground [&>code]:font-mono [&>code]:text-[10px] [&>code]:tracking-widest">
            <Icon name="plus" size={17} />
            <span>继续拖入数据字段，添加一个视图</span>
            <code>DROP TO ADD</code>
          </div>
        </>
      ) : (
        <div className="relative isolate flex min-h-80 flex-col items-center justify-center overflow-hidden border border-foreground px-5 py-5 text-center md:min-h-full">
          <Texture pattern="grid" />
          <div aria-hidden="true" className="mb-4 grid grid-cols-2 border border-foreground">
            <div className="flex h-12 w-24 items-center justify-center border-r border-foreground">
              <Icon name="video" size={24} />
            </div>
            <div className="flex h-12 w-24 items-center justify-center bg-foreground text-background">
              <Icon name="trajectory" size={24} />
            </div>
          </div>
          <h2 className="max-w-lg font-display text-2xl leading-tight tracking-tight lg:text-3xl">
            把数据拖进来，
            <br className="sm:hidden" />
            开始回放。
          </h2>
          <p className="my-3 max-w-lg text-sm leading-relaxed text-muted-foreground">
            从左侧选择相机、音频或状态数据，拖动到参考栏。
            <br />
            多个视图将跟随同一条时间轴同步播放。
          </p>
          <Button disabled={disabled} onClick={onAddDefaults}>
            <Icon name="plus" size={16} />
            添加相机与 EE
            <Icon name="arrow" size={16} />
          </Button>
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

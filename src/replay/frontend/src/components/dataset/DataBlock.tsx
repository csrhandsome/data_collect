import type { DragEvent } from 'react'
import { FEATURE_DRAG_TYPE } from '../../lib/api'
import type { DataFeature, DragFeature } from '../../types/dataset'
import { Icon } from '../ui/Icon'

interface DataBlockProps {
  feature: DataFeature
  datasetId: string
  episodeIndex: number
  added: boolean
  onAdd: (feature: DataFeature) => void
}

export function DataBlock({ feature, datasetId, episodeIndex, added, onAdd }: DataBlockProps) {
  const supported = feature.kind !== 'unsupported'
  const icon = feature.kind === 'video' ? 'video' : feature.kind === 'ee' ? 'trajectory' : 'box'
  const kindLabel = {
    video: '视频',
    audio: '音频',
    ee: '末端 EE',
    series: '数值曲线',
    unsupported: '数据字段',
  }[feature.kind]
  function startDrag(event: DragEvent<HTMLElement>) {
    if (!supported) {
      event.preventDefault()
      return
    }
    const payload: DragFeature = { datasetId, episodeIndex, feature: feature.key }
    event.dataTransfer.setData(FEATURE_DRAG_TYPE, JSON.stringify(payload))
    event.dataTransfer.effectAllowed = 'copy'
  }

  return (
    <article
      className={`group min-w-0 shrink-0 border border-foreground p-2 transition-colors duration-100 ${supported ? 'cursor-grab bg-background hover:bg-foreground hover:text-background active:cursor-grabbing' : 'border-border-light bg-muted text-muted-foreground'} ${added ? 'border-l-4' : ''}`}
      draggable={supported}
      onDragStart={startDrag}
      data-testid={`data-block-${feature.key}`}
    >
      <div className="flex items-center gap-1.5">
        <span className="flex size-6 shrink-0 items-center justify-center border border-current">
          <Icon name={icon} size={14} />
        </span>
        <span className="min-w-0 flex-1 truncate text-[13px] font-semibold" title={feature.label}>
          {feature.label}
        </span>
        <span className="shrink-0 font-mono text-[9px]">{kindLabel}</span>
        {supported ? <Icon name="grip" size={12} className="shrink-0" /> : null}
        <button
          className="inline-flex size-7 shrink-0 items-center justify-center border border-current font-mono enabled:hover:bg-background enabled:hover:text-foreground disabled:cursor-default disabled:opacity-60 group-hover:focus-visible:outline-background"
          disabled={!supported || added}
          onClick={() => onAdd(feature)}
          aria-label={`${added ? '已添加' : '添加'} ${feature.label}`}
          title={added ? '已添加到参考栏' : supported ? '添加到参考栏' : '此字段暂不可视化'}
        >
          <Icon name={added ? 'check' : 'plus'} size={14} />
        </button>
      </div>
      <code
        className="mt-1 mb-1.5 block truncate text-[10px] leading-4 opacity-80"
        title={feature.key}
      >
        {feature.key}
      </code>
      <div className="flex flex-wrap gap-1 [&>code]:max-w-full [&>code]:border [&>code]:border-current [&>code]:px-1 [&>code]:text-[9px] [&>code]:[overflow-wrap:anywhere]">
        <code>{feature.dtype}</code>
        <code>[{feature.shape.join(' × ')}]</code>
      </div>
      {feature.names?.length ? (
        <details className="mt-1 text-[10px] [&>summary]:min-h-7 [&>summary]:cursor-pointer [&>summary]:content-center [&>div]:pt-1 [&>div]:font-mono [&>div]:text-[10px] [&>div]:leading-relaxed [&>div]:[overflow-wrap:anywhere]">
          <summary>维度名称 · {feature.names.length}</summary>
          <div>{feature.names.join(', ')}</div>
        </details>
      ) : null}
      {!supported ? <p className="mt-1 text-[10px]">暂未支持可视化</p> : null}
    </article>
  )
}

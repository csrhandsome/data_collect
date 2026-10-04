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
    image: '图像',
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
      className={`group min-w-0 shrink-0 border border-foreground p-4 transition-colors duration-100 ${supported ? 'cursor-grab bg-background hover:bg-foreground hover:text-background active:cursor-grabbing' : 'border-border-light bg-muted text-muted-foreground'} ${added ? 'border-l-4' : ''}`}
      draggable={supported}
      onDragStart={startDrag}
      data-testid={`data-block-${feature.key}`}
    >
      <div className="flex items-center gap-2">
        <span className="flex size-8 shrink-0 items-center justify-center border border-current">
          <Icon name={icon} size={17} />
        </span>
        <span className="min-w-0 flex-1 text-base font-semibold [overflow-wrap:anywhere]">
          {feature.label}
        </span>
        <span className="shrink-0 font-mono text-[10px]">{kindLabel}</span>
        {supported ? <Icon name="grip" size={14} className="shrink-0" /> : null}
      </div>
      <code
        className="mt-3 mb-2 block text-[11px] leading-relaxed opacity-80 [overflow-wrap:anywhere]"
        title={feature.key}
      >
        {feature.key}
      </code>
      <div className="flex flex-wrap gap-2 [&>code]:max-w-full [&>code]:border [&>code]:border-current [&>code]:px-1.5 [&>code]:py-0.5 [&>code]:text-[10px] [&>code]:[overflow-wrap:anywhere]">
        <code>{feature.dtype}</code>
        <code>[{feature.shape.join(' × ')}]</code>
      </div>
      {feature.names?.length ? (
        <details className="mt-3 text-xs [&>summary]:min-h-11 [&>summary]:cursor-pointer [&>summary]:content-center [&>div]:pt-1 [&>div]:font-mono [&>div]:text-[11px] [&>div]:leading-relaxed [&>div]:[overflow-wrap:anywhere]">
          <summary>维度名称 · {feature.names.length}</summary>
          <div>{feature.names.join(', ')}</div>
        </details>
      ) : null}
      <div className="mt-3 flex items-center justify-between gap-2 border-t border-current pt-2 [&>span]:font-mono [&>span]:text-[10px]">
        <span>{supported ? (added ? '已在参考栏中' : '拖动到右侧查看') : '暂未支持可视化'}</span>
        <button
          className="inline-flex min-h-11 shrink-0 items-center gap-1 border border-current px-2 font-mono text-[11px] enabled:hover:bg-background enabled:hover:text-foreground disabled:cursor-default disabled:opacity-60 group-hover:focus-visible:outline-background"
          disabled={!supported || added}
          onClick={() => onAdd(feature)}
          aria-label={`${added ? '已添加' : '添加'} ${feature.label}`}
          title={added ? '已添加到参考栏' : supported ? '添加到参考栏' : '此字段暂不可视化'}
        >
          <Icon name={added ? 'check' : 'plus'} size={14} />
          <span>{added ? '已添加' : '添加'}</span>
        </button>
      </div>
    </article>
  )
}

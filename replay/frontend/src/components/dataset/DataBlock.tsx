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
  const kindLabel =
    ({ video: '视频', image: '图像', audio: '音频', ee: '末端 EE', series: '数值曲线', unsupported: '数据字段' })[feature.kind]
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
      className={`data-block ${supported ? 'data-block-supported' : 'data-block-disabled'} ${added ? 'data-block-added' : ''}`}
      draggable={supported}
      onDragStart={startDrag}
      data-testid={`data-block-${feature.key}`}
    >
      <div className="data-block-heading">
        <span className={`field-icon field-icon-${feature.kind}`}>
          <Icon name={icon} size={17} />
        </span>
        <span className="data-block-title">{feature.label}</span>
        <span className="field-kind">{kindLabel}</span>
        {supported ? <Icon name="grip" size={14} className="drag-grip" /> : null}
      </div>
      <code className="field-key" title={feature.key}>
        {feature.key}
      </code>
      <div className="field-meta">
        <code>{feature.dtype}</code>
        <code>[{feature.shape.join(' × ')}]</code>
      </div>
      {feature.names?.length ? (
        <details className="field-names">
          <summary>维度名称 · {feature.names.length}</summary>
          <div>{feature.names.join(', ')}</div>
        </details>
      ) : null}
      <div className="data-block-footer">
        <span>{supported ? (added ? '已在参考栏中' : '拖动到右侧查看') : '暂未支持可视化'}</span>
        <button
          className={`add-button ${added ? 'is-added' : ''}`}
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

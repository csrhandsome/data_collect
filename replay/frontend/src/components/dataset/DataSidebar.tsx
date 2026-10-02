import { useState } from 'react'
import { episodeLabel } from '../../lib/format'
import type { DataFeature, DatasetDetail, DatasetSummary, EpisodeDetail } from '../../types/dataset'
import { Icon } from '../ui/Icon'
import { LoadingState } from '../ui/ResourceState'
import { DataBlock } from './DataBlock'

interface DataSidebarProps {
  datasets: DatasetSummary[]
  dataset: DatasetDetail | null
  selectedDataset: string
  selectedEpisode: number
  episode: EpisodeDetail | null
  loading: boolean
  error: string | null
  addedKeys: Set<string>
  onDatasetChange: (id: string) => void
  onEpisodeChange: (index: number) => void
  onAdd: (feature: DataFeature) => void
}

export function DataSidebar({
  datasets,
  dataset,
  selectedDataset,
  selectedEpisode,
  episode,
  loading,
  error,
  addedKeys,
  onDatasetChange,
  onEpisodeChange,
  onAdd,
}: DataSidebarProps) {
  const [search, setSearch] = useState('')
  const features = episode?.blocks || dataset?.features || []
  const query = search.toLowerCase().trim()
  const filtered = features
    .filter((field) =>
      `${field.key} ${field.label} ${field.dtype} ${field.names?.join(' ') || ''}`
        .toLowerCase()
        .includes(query),
    )
    .sort(
      (left, right) => Number(left.kind === 'unsupported') - Number(right.kind === 'unsupported'),
    )
  const supportedCount = features.filter((field) => field.kind !== 'unsupported').length
  return (
    <aside className="data-sidebar" aria-label="数据栏">
      <div className="sidebar-heading">
        <Icon name="database" />
        <h2>数据资源</h2>
        <span className="tiny-label">EXPLORER</span>
      </div>
      <div className="dataset-section">
        <label className="input-label" htmlFor="dataset-select">
          数据集
        </label>
        <div className="select-wrap">
          <select
            id="dataset-select"
            data-testid="dataset-select"
            value={selectedDataset}
            onChange={(event) => onDatasetChange(event.target.value)}
            disabled={!datasets.length}
          >
            <option value="" disabled>
              选择一个数据集
            </option>
            {selectedDataset && !datasets.some((item) => item.id === selectedDataset) ? (
              <option value={selectedDataset}>{selectedDataset}</option>
            ) : null}
            {datasets.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name}
              </option>
            ))}
          </select>
          <Icon name="chevron" size={14} />
        </div>
        {dataset ? (
          <>
            <div className="dataset-badges">
              <span className="format-badge">LeRobot {dataset.version}</span>
              <span className="robot-badge">{dataset.robot_type || '未标注机器人'}</span>
            </div>
            <div className="dataset-stats">
              <div>
                <strong>{dataset.total_episodes.toLocaleString()}</strong>
                <span>Episodes</span>
              </div>
              <div>
                <strong>{dataset.total_frames.toLocaleString()}</strong>
                <span>总帧数</span>
              </div>
              <div>
                <strong>
                  {dataset.fps}
                  <small> Hz</small>
                </strong>
                <span>数据采样率</span>
              </div>
            </div>
            <label className="input-label episode-input-label" htmlFor="episode-select">
              当前片段 <span>EPISODE</span>
            </label>
            <div className="select-wrap">
              <select
                id="episode-select"
                data-testid="episode-select"
                value={selectedEpisode}
                onChange={(event) => onEpisodeChange(Number(event.target.value))}
                disabled={!dataset.episodes.length}
              >
                {dataset.episodes.map((item) => (
                  <option key={item.episode_index} value={item.episode_index}>
                    {episodeLabel(item.episode_index)} · {item.duration_s.toFixed(2)} s
                  </option>
                ))}
              </select>
              <Icon name="chevron" size={14} />
            </div>
            {episode?.tasks.length ? (
              <p className="episode-task" title={episode.tasks.join(' / ')}>
                <span className="status-dot" />
                {episode.tasks.join(' / ')}
              </p>
            ) : null}
          </>
        ) : null}
      </div>
      <div className="fields-section">
        <div className="fields-heading">
          <h3>数据字段</h3>
          <span>{features.length.toString().padStart(2, '0')}</span>
        </div>
        <label className="field-search">
          <Icon name="search" size={15} />
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="搜索名称、类型或维度…"
            aria-label="搜索数据字段"
          />
        </label>
        {loading ? (
          <LoadingState label="正在读取数据字段…" />
        ) : error ? (
          <p className="sidebar-message">数据暂未就绪。请在工作区重试加载。</p>
        ) : features.length === 0 ? (
          <p className="sidebar-message">暂无数据字段。添加本地数据后重新加载。</p>
        ) : (
          <>
            <p className="field-description">
              <span>{supportedCount} 个可视化字段</span>
              <span>拖动或点击 +</span>
            </p>
            <div className="data-block-list">
              {filtered.map((feature) => (
                <DataBlock
                  key={feature.key}
                  feature={feature}
                  datasetId={selectedDataset}
                  episodeIndex={selectedEpisode}
                  added={addedKeys.has(feature.key)}
                  onAdd={onAdd}
                />
              ))}
            </div>
            {!filtered.length ? <p className="sidebar-message">未找到匹配的字段。</p> : null}
          </>
        )}
      </div>
      <div className="sidebar-footer">
        <Icon name="layers" size={14} />
        <span>完整 Schema · 图像 / 音频 / 状态可视化</span>
      </div>
    </aside>
  )
}

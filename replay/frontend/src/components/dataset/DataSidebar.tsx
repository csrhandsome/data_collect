import { Button } from '../ui/Button'
import { memo, useState } from 'react'
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
  scanning: boolean
  addedKeys: Set<string>
  onDatasetChange: (id: string) => void
  onScan: () => void
  onEpisodeChange: (index: number) => void
  onAdd: (feature: DataFeature) => void
}

export const DataSidebar = memo(function DataSidebar({
  datasets,
  dataset,
  selectedDataset,
  selectedEpisode,
  episode,
  loading,
  error,
  scanning,
  addedKeys,
  onDatasetChange,
  onScan,
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
    <aside
      className="relative flex min-h-0 min-w-0 flex-col border-b-4 border-foreground bg-background md:max-h-[max(960px,calc(100dvh-84px))] md:overflow-hidden md:border-r md:border-b-0"
      aria-label="数据栏"
    >
      <div className="flex shrink-0 items-center gap-3 border-b border-border-light px-5 py-4 md:px-6">
        <Icon name="database" />
        <h2 className="font-display text-2xl">数据资源</h2>
        <span className="ml-auto font-mono text-[10px] tracking-widest text-muted-foreground">
          EXPLORER
        </span>
      </div>
      <div className="shrink-0 border-b-4 border-foreground px-5 py-4 md:px-6">
        <div className="mb-2 flex items-center justify-between gap-2">
          <label className="block font-mono text-xs tracking-widest" htmlFor="dataset-select">
            数据集
          </label>
          <Button
            variant="secondary"
            className="px-3! text-[11px]!"
            onClick={onScan}
            disabled={scanning}
            data-testid="scan-datasets"
          >
            <Icon name="reset" size={12} />
            {scanning ? '扫描中…' : '重新扫描'}
          </Button>
        </div>
        <div className="relative [&>svg]:pointer-events-none [&>svg]:absolute [&>svg]:top-1/2 [&>svg]:right-3 [&>svg]:-translate-y-1/2">
          <select
            className="min-h-11 w-full appearance-none border-2 border-foreground bg-background py-3 pr-9 pl-3 font-mono text-xs enabled:hover:bg-muted"
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
        <p className="mt-3 font-mono text-[11px] text-muted-foreground" role="status">
          {scanning ? '正在扫描数据集目录…' : `发现 ${datasets.length} 个数据集`}
        </p>
        {dataset ? (
          <>
            <div className="mt-4 flex flex-wrap gap-2">
              <span className="border border-foreground bg-foreground px-2 py-1 font-mono text-[10px] text-background">
                LeRobot {dataset.version}
              </span>
              <span className="max-w-full border border-border-light px-2 py-1 font-mono text-[10px] text-muted-foreground [overflow-wrap:anywhere]">
                {dataset.robot_type || '未标注机器人'}
              </span>
            </div>
            <label
              className="mt-4 mb-2 flex justify-between font-mono text-xs tracking-widest [&>span]:text-muted-foreground"
              htmlFor="episode-select"
            >
              当前片段 <span>EPISODE</span>
            </label>
            <div className="relative [&>svg]:pointer-events-none [&>svg]:absolute [&>svg]:top-1/2 [&>svg]:right-3 [&>svg]:-translate-y-1/2">
              <select
                className="min-h-11 w-full appearance-none border-2 border-foreground bg-background py-3 pr-9 pl-3 font-mono text-xs enabled:hover:bg-muted"
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
              <p
                className="mt-3 flex items-start gap-2 text-sm leading-relaxed text-muted-foreground [&>.inline-block]:mt-2"
                title={episode.tasks.join(' / ')}
              >
                <span className="inline-block size-1.5 shrink-0 bg-current" />
                {episode.tasks.join(' / ')}
              </p>
            ) : null}
          </>
        ) : null}
      </div>
      <div className="flex min-h-0 flex-1 flex-col p-5 md:p-6">
        <div className="mb-4 flex shrink-0 items-center justify-between [&>h3]:font-display [&>h3]:text-xl [&>span]:font-mono [&>span]:text-xs">
          <h3>数据字段</h3>
          <span>{features.length.toString().padStart(2, '0')}</span>
        </div>
        <label className="flex min-h-11 items-center gap-2 shrink-0 border-b-2 border-foreground focus-within:border-b-4">
          <Icon name="search" size={15} />
          <input
            className="min-h-11 min-w-0 flex-1 bg-transparent py-2 text-sm placeholder:text-muted-foreground placeholder:italic focus:outline-none"
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
          <p className="py-5 text-base leading-relaxed text-muted-foreground">
            数据暂未就绪。请在工作区重试加载。
          </p>
        ) : features.length === 0 ? (
          <p className="py-5 text-base leading-relaxed text-muted-foreground">
            暂无数据字段。添加本地数据后重新加载。
          </p>
        ) : (
          <>
            <p className="my-4 flex shrink-0 flex-wrap justify-between gap-2 font-mono text-[10px] text-muted-foreground">
              <span>{supportedCount} 个可视化字段</span>
              <span>拖动或点击 +</span>
            </p>
            <div className="flex max-h-[360px] flex-col gap-3 overflow-y-auto px-1 py-1 md:max-h-none md:flex-1">
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
            {!filtered.length ? (
              <p className="py-5 text-base leading-relaxed text-muted-foreground">
                未找到匹配的字段。
              </p>
            ) : null}
          </>
        )}
      </div>
      <div className="mt-auto flex shrink-0 items-center gap-2 border-t border-border-light px-5 py-5 font-mono text-[10px] text-muted-foreground md:px-6">
        <Icon name="layers" size={14} />
        <span>完整 Schema · 图像 / 音频 / 状态可视化</span>
      </div>
    </aside>
  )
})

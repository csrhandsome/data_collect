import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { DataSidebar } from '../components/dataset/DataSidebar'
import { Icon } from '../components/ui/Icon'
import { ErrorState } from '../components/ui/ResourceState'
import { ReferencePanel } from '../components/workspace/ReferencePanel'
import { Timeline } from '../components/workspace/Timeline'
import { VisualizationCard } from '../components/workspace/VisualizationCard'
import { useApiResource } from '../hooks/useApiResource'
import { usePlayback } from '../hooks/usePlayback'
import { episodeUrl } from '../lib/api'
import { episodeLabel } from '../lib/format'
import type {
  DataFeature,
  DatasetDetail,
  DatasetSummary,
  DragFeature,
  EpisodeDetail,
} from '../types/dataset'

interface WorkspaceViews {
  selectionKey: string
  features: DataFeature[]
}

export function ReplayPage() {
  const [params, setParams] = useSearchParams()
  const list = useApiResource<{ datasets: DatasetSummary[] }>('/api/datasets')
  const health = useApiResource<{ status: string }>('/api/health')
  const datasets = list.data?.datasets || []
  const datasetId = params.get('dataset') || datasets[0]?.id || ''
  const episodeParam = Number(params.get('episode') || '0')
  const episodeIndex = Number.isSafeInteger(episodeParam) && episodeParam >= 0 ? episodeParam : 0
  const dataset = useApiResource<DatasetDetail>(
    datasetId ? `/api/datasets/${encodeURIComponent(datasetId)}` : null,
  )
  const episode = useApiResource<EpisodeDetail>(
    datasetId && dataset.data ? episodeUrl(datasetId, episodeIndex) : null,
  )
  const selectionKey = `${datasetId}:${episodeIndex}`
  const [workspace, setWorkspace] = useState<WorkspaceViews>({
    selectionKey: '',
    features: [],
  })
  const features = workspace.selectionKey === selectionKey ? workspace.features : []
  const addedKeys = useMemo(() => new Set(features.map((feature) => feature.key)), [features])
  const duration = episode.data?.duration_s || 0
  const playback = usePlayback(duration, selectionKey)
  const ready = Boolean(episode.data && !episode.error && !episode.loading)
  const error = list.error || dataset.error || episode.error
  const loading = list.loading || dataset.loading || episode.loading

  useEffect(() => {
    setWorkspace({ selectionKey, features: [] })
  }, [selectionKey])

  useEffect(() => {
    if (datasetId && !params.get('dataset')) {
      setParams({ dataset: datasetId, episode: episodeIndex.toString() }, { replace: true })
    }
  }, [datasetId, episodeIndex, params, setParams])

  useEffect(() => {
    if (
      dataset.data?.episodes.length &&
      !dataset.data.episodes.some((item) => item.episode_index === episodeIndex)
    ) {
      setParams(
        {
          dataset: datasetId,
          episode: dataset.data.episodes[0].episode_index.toString(),
        },
        { replace: true },
      )
    }
  }, [dataset.data, episodeIndex, datasetId, setParams])

  const addFeature = useCallback(
    (feature: DataFeature) => {
      if (!ready || feature.kind === 'unsupported') return
      setWorkspace((previous) => {
        const current = previous.selectionKey === selectionKey ? previous.features : []
        return current.some((item) => item.key === feature.key)
          ? previous
          : { selectionKey, features: [...current, feature] }
      })
    },
    [ready, selectionKey],
  )

  const dropFeature = useCallback(
    (payload: DragFeature) => {
      if (payload.datasetId !== datasetId || payload.episodeIndex !== episodeIndex || !episode.data)
        return
      const feature = episode.data.blocks.find((item) => item.key === payload.feature)
      if (feature) addFeature(feature)
    },
    [datasetId, episodeIndex, episode.data, addFeature],
  )

  function addDefaults() {
    const blocks = episode.data?.blocks || []
    const video = blocks.find((feature) => feature.kind === 'video' || feature.kind === 'image')
    const ee = blocks.find((feature) => feature.kind === 'ee')
    if (video) addFeature(video)
    if (ee) addFeature(ee)
  }

  function removeFeature(key: string) {
    if (features.length === 1) playback.pause()
    setWorkspace({
      selectionKey,
      features: features.filter((feature) => feature.key !== key),
    })
  }

  function clearWorkspace() {
    playback.pause()
    setWorkspace({ selectionKey, features: [] })
  }

  function retry() {
    list.retry()
    dataset.retry()
    episode.retry()
    health.retry()
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <a className="brand" href="/replay" aria-label="Replay 数据回放">
          <span className="brand-mark">
            <Icon name="play" size={19} />
            <i />
          </span>
          <strong>REPLAY</strong>
          <span className="brand-divider" />
          <span className="brand-caption">机器人数据工作区</span>
        </a>
        <div className="header-status">
          <span className="local-mode">LOCAL WORKSPACE</span>
          <span className={`api-status ${health.error ? 'api-offline' : ''}`}>
            <span className="status-dot" />
            {health.loading ? '连接中' : health.error ? 'API 未连接' : 'API 已连接'}
          </span>
        </div>
      </header>
      <div className="workspace-layout">
        <DataSidebar
          datasets={datasets}
          dataset={dataset.data}
          selectedDataset={datasetId}
          selectedEpisode={episodeIndex}
          episode={episode.data}
          loading={loading}
          error={error}
          addedKeys={addedKeys}
          onDatasetChange={(id) => setParams({ dataset: id, episode: '0' })}
          onEpisodeChange={(index) => setParams({ dataset: datasetId, episode: index.toString() })}
          onAdd={addFeature}
        />
        <main className="workspace-main">
          <div className="workspace-heading">
            <div>
              <div className="workspace-breadcrumb">
                <span>工作区</span>
                <span>/</span>
                <code>{datasetId ? episodeLabel(episodeIndex) : 'SELECT DATASET'}</code>
              </div>
              <h1>
                数据回放<span className="heading-dot">.</span>
              </h1>
              <p>在同一时刻，看见每一个视角。</p>
            </div>
            <div className="workspace-actions">
              <span className="readonly-badge">
                <Icon name="layers" size={13} />
                只读查看
              </span>
              <button
                className="button button-secondary clear-button"
                disabled={!features.length}
                onClick={clearWorkspace}
                aria-label="清空参考栏"
                data-testid="clear-workspace"
              >
                <Icon name="reset" size={14} />
                清空画布
              </button>
            </div>
          </div>
          <div className="canvas-heading">
            <div>
              <Icon name="layers" size={16} />
              <h2>参考栏</h2>
              <span className="canvas-count">{features.length} 个视图</span>
            </div>
            <span className="canvas-status">
              {loading
                ? '正在读取片段…'
                : episode.data
                  ? `${episode.data.length} 帧 · ${duration.toFixed(2)} 秒`
                  : '等待数据'}
            </span>
          </div>
          {error ? (
            <div className="workspace-error">
              <ErrorState message={error} onRetry={retry} />
            </div>
          ) : list.data && !datasets.length ? (
            <div className="workspace-error">
              <ErrorState
                message="尚未发现数据集。添加本地 LeRobot 数据后重新加载。"
                onRetry={retry}
              />
            </div>
          ) : null}
          <ReferencePanel
            count={features.length}
            disabled={!ready}
            onDropFeature={dropFeature}
            onAddDefaults={addDefaults}
          >
            {features.map((feature) => (
              <VisualizationCard
                key={`${selectionKey}:${feature.key}`}
                feature={feature}
                datasetId={datasetId}
                episodeIndex={episodeIndex}
                time={playback.time}
                duration={duration}
                playing={playback.playing}
                rate={playback.rate}
                onSeek={playback.seek}
                onToggle={playback.toggle}
                onRemove={() => removeFeature(feature.key)}
              />
            ))}
          </ReferencePanel>
          <Timeline
            time={playback.time}
            duration={duration}
            playing={playback.playing}
            rate={playback.rate}
            fps={dataset.data?.fps || 30}
            disabled={!ready || !features.length}
            onToggle={playback.toggle}
            onSeek={playback.seek}
            onRateChange={playback.setRate}
          />
          <footer className="workspace-footer">
            <span>
              <span className="status-dot" />
              LeRobot v2 / v3
            </span>
            <span>
              图像 · 音频 · 状态曲线<span className="footer-separator">/</span>
              所有视图共享时间轴
            </span>
            <code>REPLAY / 01</code>
          </footer>
        </main>
      </div>
    </div>
  )
}

import { Texture } from '../components/ui/Texture'
import { Button } from '../components/ui/Button'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { DataSidebar } from '../components/dataset/DataSidebar'
import { Icon } from '../components/ui/Icon'
import { ErrorState } from '../components/ui/ResourceState'
import { ReferencePanel } from '../components/workspace/ReferencePanel'
import { Timeline } from '../components/workspace/Timeline'
import { VisualizationCard } from '../components/workspace/VisualizationCard'
import { OperationPanel } from '../components/workspace/OperationPanel'
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
  const requestedDataset = params.get('dataset') || ''
  const datasetId = list.data
    ? datasets.find((item) => item.id === requestedDataset)?.id || datasets[0]?.id || ''
    : requestedDataset
  const episodeParam = Number(params.get('episode') || '0')
  const episodeIndex = Number.isSafeInteger(episodeParam) && episodeParam >= 0 ? episodeParam : 0
  const dataset = useApiResource<DatasetDetail>(
    datasetId ? `/api/datasets/${encodeURIComponent(datasetId)}` : null,
  )
  const episode = useApiResource<EpisodeDetail>(
    datasetId && dataset.data?.episodes.some((item) => item.episode_index === episodeIndex)
      ? episodeUrl(datasetId, episodeIndex)
      : null,
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
    if (datasetId && params.get('dataset') !== datasetId) {
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

  const retry = useCallback(() => {
    list.retry()
    dataset.retry()
    episode.retry()
    health.retry()
  }, [list.retry, dataset.retry, episode.retry, health.retry])

  const changeDataset = useCallback(
    (id: string) => {
      setParams({ dataset: id, episode: '0' })
    },
    [setParams],
  )
  const changeEpisode = useCallback(
    (index: number) => {
      setParams({ dataset: datasetId, episode: index.toString() })
    },
    [datasetId, setParams],
  )

  return (
    <div className="relative isolate min-h-dvh bg-background text-foreground">
      <Texture />
      <Texture pattern="lines" />
      <a
        href="#workspace"
        className="sr-only fixed top-3 left-3 z-50 bg-foreground px-5 py-3 font-mono text-sm text-background focus:not-sr-only"
      >
        跳转到回放工作区
      </a>
      <header className="relative z-10 flex min-h-20 items-center justify-between gap-4 border-b-4 border-foreground px-5 py-4 md:px-8">
        <a
          className="inline-flex min-h-11 items-center gap-3"
          href="/replay"
          aria-label="Replay 数据回放"
        >
          <span className="relative grid size-9 place-items-center border-2 border-foreground bg-foreground pl-1 text-background">
            <Icon name="play" size={19} />
            <i className="absolute top-2 left-1.5 h-4 border-l border-background" />
          </span>
          <strong className="font-display text-2xl font-normal tracking-tight">Replay.</strong>
          <span className="mx-2 hidden h-6 border-l border-foreground md:block" />
          <span className="hidden font-mono text-xs tracking-widest text-muted-foreground md:block">
            机器人数据工作区
          </span>
        </a>
        <div className="flex items-center gap-6">
          <span className="hidden font-mono text-xs tracking-widest text-muted-foreground xl:block">
            LOCAL WORKSPACE
          </span>
          <span
            className={`flex items-center gap-2 font-mono text-xs ${health.error ? 'underline decoration-2 underline-offset-4' : ''}`}
          >
            <span className="inline-block size-1.5 shrink-0 bg-current" />
            {health.loading ? '连接中' : health.error ? 'API 未连接' : 'API 已连接'}
          </span>
        </div>
      </header>
      <div className="relative grid min-h-[calc(100dvh-84px)] md:min-h-[960px] grid-cols-1 md:grid-cols-[300px_minmax(0,1fr)] xl:grid-cols-[320px_minmax(0,1fr)]">
        <DataSidebar
          datasets={datasets}
          dataset={dataset.data}
          selectedDataset={datasetId}
          selectedEpisode={episodeIndex}
          episode={episode.data}
          loading={loading}
          error={error}
          scanning={list.loading}
          addedKeys={addedKeys}
          onDatasetChange={changeDataset}
          onScan={retry}
          onEpisodeChange={changeEpisode}
          onAdd={addFeature}
        />
        <main
          id="workspace"
          tabIndex={-1}
          className="flex min-h-0 min-w-0 flex-col px-5 pt-7 md:px-7 xl:px-10"
        >
          <div className="mb-5 flex shrink-0 flex-wrap items-end justify-between gap-4 border-b-4 border-foreground pb-4">
            <div>
              <div className="mb-3 flex items-center gap-3 font-mono text-[11px] tracking-widest text-muted-foreground [&>code]:text-foreground">
                <span>01 / 数据回放</span>
                <span>/</span>
                <code>{datasetId ? episodeLabel(episodeIndex) : 'SELECT DATASET'}</code>
              </div>
              <h1 className="font-display text-2xl leading-tight font-normal md:text-3xl">
                遥操回放和控制平台
              </h1>
            </div>
            <div className="flex items-center gap-4">
              <span className="hidden items-center gap-2 font-mono text-[11px] text-muted-foreground xl:flex">
                <Icon name="layers" size={13} />
                本地工作台
              </span>
              <Button
                variant="secondary"
                disabled={!features.length}
                onClick={clearWorkspace}
                aria-label="清空参考栏"
                data-testid="clear-workspace"
              >
                <Icon name="reset" size={14} />
                清空画布
              </Button>
            </div>
          </div>
          <OperationPanel
            dataset={dataset.data}
            episodeIndex={episodeIndex}
            onFinished={() => {
              clearWorkspace()
              retry()
            }}
          />
          <div className="flex shrink-0 flex-wrap items-center justify-between gap-3 pb-4 [&>div]:flex [&>div]:items-center [&>div]:gap-2 [&_h2]:font-display [&_h2]:text-xl">
            <div>
              <span aria-hidden="true" className="size-3 border-2 border-foreground" />
              <h2>参考栏</h2>
              <span className="ml-2 border-l border-foreground pl-3 font-mono text-[11px]">
                {features.length} 个视图
              </span>
            </div>
            <span className="font-mono text-[11px] text-muted-foreground">
              {loading
                ? '正在读取片段…'
                : episode.data
                  ? `${episode.data.length} 帧 · ${duration.toFixed(2)} 秒`
                  : '等待数据'}
            </span>
          </div>
          {error ? (
            <div className="mb-5 shrink-0 border-l-4 border-foreground bg-muted">
              <ErrorState message={error} onRetry={retry} />
            </div>
          ) : list.data && !datasets.length ? (
            <div className="mb-5 shrink-0 border-l-4 border-foreground bg-muted">
              <ErrorState
                message="尚未发现数据集。将 LeRobot 数据集放入数据目录，再点击「重新扫描」。"
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
                seekVersion={playback.seekVersion}
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
          <footer className="mt-5 flex flex-wrap items-center gap-5 border-t-4 border-foreground py-5 font-mono text-[10px] text-muted-foreground [&>span:first-child]:flex [&>span:first-child]:items-center [&>span:first-child]:gap-2 [&>span:nth-child(2)]:hidden [&>span:nth-child(2)]:xl:block [&>code]:ml-auto [&>code]:tracking-widest">
            <span>
              <span className="inline-block size-1.5 shrink-0 bg-current" />
              LeRobot v2 / v3
            </span>
            <span>
              图像 · 音频 · 状态曲线<span className="px-3">/</span>
              所有视图共享时间轴
            </span>
            <code>REPLAY / 01</code>
          </footer>
        </main>
      </div>
    </div>
  )
}

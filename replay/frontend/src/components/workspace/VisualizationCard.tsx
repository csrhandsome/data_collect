import { useApiResource } from '../../hooks/useApiResource'
import { episodeUrl } from '../../lib/api'
import type { AudioData, DataFeature, EEData, VideoData } from '../../types/dataset'
import { AudioStream } from '../visualizations/AudioStream'
import { SeriesPlot } from '../visualizations/SeriesPlot'
import { Icon } from '../ui/Icon'
import { ErrorState, LoadingState } from '../ui/ResourceState'
import { EETrajectory } from '../visualizations/EETrajectory'
import { VideoPlayer } from '../visualizations/VideoPlayer'

interface VisualizationCardProps {
  feature: DataFeature
  datasetId: string
  episodeIndex: number
  time: number
  duration: number
  playing: boolean
  rate: number
  seekVersion: number
  onSeek: (time: number) => void
  onToggle: () => void
  onRemove: () => void
}

function VideoContent({
  feature,
  datasetId,
  episodeIndex,
  time,
  playing,
  rate,
  seekVersion,
  onToggle,
}: VisualizationCardProps) {
  const resource = useApiResource<VideoData>(
    `${episodeUrl(datasetId, episodeIndex)}/video?feature=${encodeURIComponent(feature.key)}`,
  )
  return resource.loading ? (
    <LoadingState label="正在准备相机视频，首次打开需稍等…" />
  ) : resource.error ? (
    <ErrorState message={resource.error} onRetry={resource.retry} />
  ) : resource.data ? (
    <VideoPlayer
      video={resource.data}
      time={time}
      playing={playing}
      rate={rate}
      seekVersion={seekVersion}
      onToggle={onToggle}
    />
  ) : null
}

function EEContent({
  feature,
  datasetId,
  episodeIndex,
  time,
  duration,
  onSeek,
}: VisualizationCardProps) {
  const resource = useApiResource<EEData>(
    `${episodeUrl(datasetId, episodeIndex)}/ee?feature=${encodeURIComponent(feature.key)}&max_points=2000`,
  )
  return resource.loading ? (
    <LoadingState label="正在载入末端轨迹…" />
  ) : resource.error ? (
    <ErrorState message={resource.error} onRetry={resource.retry} />
  ) : resource.data && resource.data.values.length ? (
    <EETrajectory data={resource.data} time={time} duration={duration} onSeek={onSeek} />
  ) : (
    <div className="flex min-h-44 flex-col items-center justify-center gap-4 p-5 text-center text-base leading-relaxed text-muted-foreground [&>p]:max-w-lg [&>p]:[overflow-wrap:anywhere]">
      此片段没有末端采样点。
    </div>
  )
}

function StreamContent(props: VisualizationCardProps) {
  const base = episodeUrl(props.datasetId, props.episodeIndex)
  const kind = props.feature.kind
  const endpoint = kind === 'audio' ? 'audio' : 'series'
  const resource = useApiResource<EEData | AudioData>(
    `${base}/${endpoint}?feature=${encodeURIComponent(props.feature.key)}`,
  )
  if (resource.loading) return <LoadingState label="正在读取数据…" />
  if (resource.error) return <ErrorState message={resource.error} onRetry={resource.retry} />
  if (!resource.data) return null
  if (kind === 'audio')
    return (
      <AudioStream
        data={resource.data as AudioData}
        url={`${base}/audio/file`}
        time={props.time}
        playing={props.playing}
        rate={props.rate}
      />
    )
  return (
    <SeriesPlot
      data={resource.data as EEData}
      time={props.time}
      duration={props.duration}
      onSeek={props.onSeek}
    />
  )
}

export function VisualizationCard(props: VisualizationCardProps) {
  const { feature, onRemove } = props
  return (
    <article
      className="min-w-0 overflow-hidden border border-foreground bg-background"
      data-testid={`visualization-${feature.key}`}
    >
      <header className="flex items-center gap-3 border-b border-foreground px-4 py-3 [&>div]:min-w-0 [&>div]:flex-1 [&_h3]:text-base [&_h3]:font-semibold [&_code]:block [&_code]:truncate [&_code]:text-[10px] [&_code]:text-muted-foreground">
        <span className="flex size-8 shrink-0 items-center justify-center border border-current">
          <Icon
            name={feature.kind === 'video' || feature.kind === 'image' ? 'video' : 'trajectory'}
            size={17}
          />
        </span>
        <div>
          <h3>{feature.label}</h3>
          <code title={feature.key}>{feature.key}</code>
        </div>
        <span className="hidden font-mono text-[10px] tracking-widest 2xl:block">
          {feature.kind.toUpperCase()}
        </span>
        <button
          className="inline-flex size-11 shrink-0 items-center justify-center border border-transparent bg-transparent enabled:hover:border-current enabled:hover:bg-foreground enabled:hover:text-background"
          onClick={onRemove}
          title="移除视图"
          aria-label={`移除 ${feature.label}`}
        >
          <Icon name="close" size={16} />
        </button>
      </header>
      {feature.kind === 'video' || feature.kind === 'image' ? (
        <VideoContent {...props} />
      ) : feature.kind === 'ee' ? (
        <EEContent {...props} />
      ) : (
        <StreamContent {...props} />
      )}
    </article>
  )
}

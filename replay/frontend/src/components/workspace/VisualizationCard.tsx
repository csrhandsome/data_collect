import { useApiResource } from '../../hooks/useApiResource'
import { episodeUrl } from '../../lib/api'
import type { AudioData, DataFeature, EEData, VideoData } from '../../types/dataset'
import { ImageStream } from '../visualizations/ImageStream'
import type { ImageMetadata } from '../visualizations/ImageStream'
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
  onToggle,
}: VisualizationCardProps) {
  const resource = useApiResource<VideoData>(
    `${episodeUrl(datasetId, episodeIndex)}/video?feature=${encodeURIComponent(feature.key)}`,
  )
  return resource.loading ? (
    <LoadingState label="正在载入相机视频…" />
  ) : resource.error ? (
    <ErrorState message={resource.error} onRetry={resource.retry} />
  ) : resource.data ? (
    <VideoPlayer
      video={resource.data}
      time={time}
      playing={playing}
      rate={rate}
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
    <div className="resource-state">此片段没有末端采样点。</div>
  )
}

function StreamContent(props: VisualizationCardProps) {
  const base = episodeUrl(props.datasetId, props.episodeIndex)
  const kind = props.feature.kind
  const endpoint = kind === 'image' ? 'image/metadata' : kind === 'audio' ? 'audio' : 'series'
  const resource = useApiResource<EEData | AudioData | ImageMetadata>(
    `${base}/${endpoint}?feature=${encodeURIComponent(props.feature.key)}`,
  )
  if (resource.loading) return <LoadingState label="正在读取数据…" />
  if (resource.error) return <ErrorState message={resource.error} onRetry={resource.retry} />
  if (!resource.data) return null
  if (kind === 'image') return <ImageStream baseUrl={base} feature={props.feature.key} time={props.time} metadata={resource.data as ImageMetadata} />
  if (kind === 'audio') return <AudioStream data={resource.data as AudioData} url={`${base}/audio/file`} time={props.time} playing={props.playing} rate={props.rate} />
  return <SeriesPlot data={resource.data as EEData} time={props.time} duration={props.duration} onSeek={props.onSeek} />
}

export function VisualizationCard(props: VisualizationCardProps) {
  const { feature, onRemove } = props
  return (
    <article
      className={`visualization-card visualization-card-${feature.kind}`}
      data-testid={`visualization-${feature.key}`}
    >
      <header className="visualization-heading">
        <span className={`field-icon field-icon-${feature.kind}`}>
          <Icon name={feature.kind === 'video' ? 'video' : 'trajectory'} size={17} />
        </span>
        <div>
          <h3>{feature.label}</h3>
          <code title={feature.key}>{feature.key}</code>
        </div>
        <span className="visualization-type">
          {feature.kind.toUpperCase()}
        </span>
        <button
          className="icon-button card-close"
          onClick={onRemove}
          title="移除视图"
          aria-label={`移除 ${feature.label}`}
        >
          <Icon name="close" size={16} />
        </button>
      </header>
      {feature.kind === 'video' ? <VideoContent {...props} /> : feature.kind === 'ee' ? <EEContent {...props} /> : <StreamContent {...props} />}
    </article>
  )
}

export interface DataFeature {
  key: string
  label: string
  kind: 'video' | 'ee' | 'image' | 'audio' | 'series' | 'unsupported'
  dtype: string
  shape: number[]
  names: string[] | null
}

export interface DatasetSummary {
  id: string
  name: string
  version: string
  robot_type: string | null
  fps: number
  total_episodes: number
  total_frames: number
}

export interface EpisodeSummary {
  episode_index: number
  length: number
  duration_s: number
  tasks: string[]
}

export interface DatasetDetail extends DatasetSummary {
  features: DataFeature[]
  episodes: EpisodeSummary[]
}

export interface EpisodeDetail extends EpisodeSummary {
  dataset_id: string
  blocks: DataFeature[]
  success: boolean | null
  saved_at_ns: string | null
}

export interface LatestEpisode {
  dataset_id: string
  episode_index: number
  saved_at_ns: string
}

export interface EEData {
  dataset_id: string
  episode_index: number
  feature: string
  names: string[]
  units: string[]
  timestamps: number[]
  values: number[][]
  point_count: number
  total_points: number
  bounds: { min: number[]; max: number[] }
}

export interface VideoData {
  dataset_id: string
  episode_index: number
  feature: string
  url: string
  start_time_s: number
  end_time_s: number
  duration_s: number
  fps: number
}

export interface DragFeature {
  datasetId: string
  episodeIndex: number
  feature: string
}

export interface AudioData {
  sample_rate: number
  channels: number
  num_samples: number
  duration_s: number
  offset_s: number
  waveform: number[]
  vad_segments: { start_sec?: number; end_sec?: number; start?: number; end?: number }[]
  instruction_audio_window: {
    start_sec: number | null
    end_sec: number | null
    audio_valid: boolean
  } | null
}

import { useEffect, useRef, useState } from 'react'
import { timeLabel } from '../../lib/format'
import type { VideoData } from '../../types/dataset'
import { Icon } from '../ui/Icon'

interface VideoPlayerProps {
  video: VideoData
  time: number
  playing: boolean
  rate: number
  onToggle: () => void
}

export function VideoPlayer({ video, time, playing, rate, onToggle }: VideoPlayerProps) {
  const element = useRef<HTMLVideoElement>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [buffering, setBuffering] = useState(false)
  const [playRejected, setPlayRejected] = useState(false)
  const frameDuration = 1 / video.fps
  // v3 can contain many episodes in one file; keep all seeks within this clip.
  const clipTime =
    video.start_time_s + Math.min(Math.max(0, time), Math.max(0, video.duration_s - frameDuration))

  useEffect(() => {
    const player = element.current
    if (!player || !ready) return
    if (Math.abs(player.currentTime - clipTime) > (playing ? 0.16 : 0.005))
      player.currentTime = clipTime
  }, [clipTime, playing, ready])

  useEffect(() => {
    const player = element.current
    if (!player || !ready) return
    player.playbackRate = rate
    if (playing) {
      void player
        .play()
        .then(() => setPlayRejected(false))
        .catch(() => setPlayRejected(true))
    } else {
      player.pause()
      setBuffering(false)
    }
  }, [playing, rate, ready])

  function loadedMetadata() {
    const player = element.current
    if (player) player.currentTime = clipTime
    setReady(true)
  }

  function enforceBounds() {
    const player = element.current
    if (!player) return
    if (player.currentTime < video.start_time_s) player.currentTime = video.start_time_s
    if (player.currentTime >= video.end_time_s - 0.002) {
      player.pause()
      player.currentTime = Math.max(video.start_time_s, video.end_time_s - frameDuration)
    }
  }

  return (
    <div className="video-player">
      <div className="video-stage">
        <video
          ref={element}
          src={video.url}
          muted
          playsInline
          preload="auto"
          onLoadedMetadata={loadedMetadata}
          onTimeUpdate={enforceBounds}
          onWaiting={() => setBuffering(true)}
          onPlaying={() => setBuffering(false)}
          onCanPlay={() => setBuffering(false)}
          onError={() => setError('视频无法解码或读取，请检查 MP4 文件。')}
          aria-label={`${video.feature} 相机视频`}
          data-testid={`video-${video.feature}`}
        />
        <div className="video-corner-mark">
          <span className="status-dot" />
          CAMERA FEED
        </div>
        <span className="video-timecode">{timeLabel(time, true)}</span>
        {!ready && !error ? (
          <div className="video-status">
            <span className="spinner" />
            <span>正在加载视频…</span>
          </div>
        ) : null}
        {error ? (
          <div className="video-status video-error" role="alert">
            <Icon name="warning" size={24} />
            <span>{error}</span>
          </div>
        ) : null}
        {ready && !error && !playing ? (
          <button className="video-play-overlay" onClick={onToggle} aria-label="播放视频与工作区">
            <Icon name="play" size={22} />
          </button>
        ) : null}
        {buffering && playing ? (
          <span className="buffering-badge" role="status">
            <span className="spinner" />
            视频缓冲中
          </span>
        ) : null}
      </div>
      <div className="video-details">
        <span>
          <Icon name="signal" size={13} />
          {video.fps} FPS <i />
          同步时间轴
        </span>
        <code title="当前 Episode 在源 MP4 中的时间范围">
          CLIP {video.start_time_s.toFixed(2)} — {video.end_time_s.toFixed(2)} s
        </code>
      </div>
      {playRejected ? (
        <p className="video-notice" role="alert">
          浏览器暂停了视频播放，请再次点击时间轴播放按钮。
        </p>
      ) : null}
    </div>
  )
}

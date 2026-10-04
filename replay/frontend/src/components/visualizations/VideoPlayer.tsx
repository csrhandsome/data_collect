import { useEffect, useRef, useState } from 'react'
import { timeLabel } from '../../lib/format'
import type { VideoData } from '../../types/dataset'
import { Icon } from '../ui/Icon'

interface VideoPlayerProps {
  video: VideoData
  time: number
  playing: boolean
  rate: number
  seekVersion: number
  onToggle: () => void
}

export function VideoPlayer({
  video,
  time,
  playing,
  rate,
  seekVersion,
  onToggle,
}: VideoPlayerProps) {
  const element = useRef<HTMLVideoElement>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [buffering, setBuffering] = useState(false)
  const [playRejected, setPlayRejected] = useState(false)
  const frameDuration = 1 / video.fps
  // v3 can contain many episodes in one file; keep all seeks within this clip.
  const clipTime =
    video.start_time_s + Math.min(Math.max(0, time), Math.max(0, video.duration_s - frameDuration))
  const latestClipTime = useRef(clipTime)
  latestClipTime.current = clipTime

  useEffect(() => {
    const player = element.current
    if (!player || !ready || playing) return
    if (Math.abs(player.currentTime - clipTime) > 0.005) player.currentTime = clipTime
  }, [clipTime, playing, ready])

  useEffect(() => {
    const player = element.current
    if (!player || !ready) return
    // Native playback owns decoding. Seek only on explicit user seeks or
    // playback transitions, never on the timeline's 30 Hz React updates.
    if (Math.abs(player.currentTime - latestClipTime.current) > 0.005)
      player.currentTime = latestClipTime.current
  }, [playing, ready, seekVersion])

  useEffect(() => {
    const player = element.current
    if (player) player.playbackRate = rate
  }, [rate, ready])

  useEffect(() => {
    const player = element.current
    if (!player || !ready) return
    let disposed = false
    if (playing) {
      void player
        .play()
        .then(() => {
          if (!disposed) setPlayRejected(false)
        })
        .catch(() => {
          if (!disposed) setPlayRejected(true)
        })
    } else {
      player.pause()
      setBuffering(false)
    }
    return () => {
      disposed = true
    }
  }, [playing, ready])

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

  useEffect(() => {
    const player = element.current
    if (!player || !playing || !ready) return
    let callback = 0
    const check = () => {
      enforceBounds()
      if (!player.paused) callback = player.requestVideoFrameCallback(check)
    }
    callback = player.requestVideoFrameCallback(check)
    return () => player.cancelVideoFrameCallback(callback)
  }, [playing, ready, video.start_time_s, video.end_time_s, frameDuration])

  return (
    <div className="bg-background">
      <div className="relative aspect-video overflow-hidden bg-foreground text-background">
        <video
          className="block h-full w-full object-contain"
          ref={element}
          src={video.url}
          muted
          playsInline
          preload="auto"
          onLoadedMetadata={loadedMetadata}
          onTimeUpdate={enforceBounds}
          onWaiting={() => setBuffering(true)}
          onPlaying={() => setBuffering(false)}
          onCanPlay={() => {
            setBuffering(false)
            // Recover once after a real stall, rather than repeatedly seeking
            // during normal playback or while a seek is still decoding.
            const player = element.current
            if (
              player &&
              playing &&
              !player.seeking &&
              Math.abs(player.currentTime - latestClipTime.current) > 0.5
            )
              player.currentTime = latestClipTime.current
          }}
          onError={() => setError('视频无法解码或读取，请检查 MP4 文件。')}
          aria-label={`${video.feature} 相机视频`}
          data-testid={`video-${video.feature}`}
        />
        <div className="absolute top-3 left-3 flex items-center gap-2 border border-background bg-foreground px-2 py-1 font-mono text-[10px] tracking-widest text-background">
          <span className="inline-block size-1.5 shrink-0 bg-current" />
          CAMERA FEED
        </div>
        <span className="absolute right-3 bottom-3 bg-foreground px-2 py-1 font-mono text-xs text-background">
          {timeLabel(time, true)}
        </span>
        {!ready && !error ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-foreground/90 p-5 text-center text-sm text-background">
            <span className="inline-block size-4 shrink-0 border-2 border-current border-r-transparent motion-safe:animate-pulse" />
            <span>正在加载视频…</span>
          </div>
        ) : null}
        {error ? (
          <div
            className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-foreground/90 p-5 text-center text-sm text-background"
            role="alert"
          >
            <Icon name="warning" size={24} />
            <span>{error}</span>
          </div>
        ) : null}
        {ready && !error && !playing ? (
          <button
            className="absolute top-1/2 left-1/2 grid size-14 -translate-x-1/2 -translate-y-1/2 place-items-center border-2 border-background bg-foreground text-background hover:bg-background hover:text-foreground focus-visible:outline-background!"
            onClick={onToggle}
            aria-label="播放视频与工作区"
          >
            <Icon name="play" size={22} />
          </button>
        ) : null}
        {buffering && playing ? (
          <span
            className="absolute bottom-3 left-3 flex items-center gap-2 bg-foreground px-2 py-1 font-mono text-[10px] text-background"
            role="status"
          >
            <span className="inline-block size-4 shrink-0 border-2 border-current border-r-transparent motion-safe:animate-pulse" />
            视频缓冲中
          </span>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border-light px-4 py-3 font-mono text-[10px] text-muted-foreground [&>span]:flex [&>span]:items-center [&>span]:gap-2 [&_i]:h-3 [&_i]:border-l [&_i]:border-border-light">
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
        <p className="px-4 pb-3 text-sm text-muted-foreground" role="alert">
          浏览器暂停了视频播放，请再次点击时间轴播放按钮。
        </p>
      ) : null}
    </div>
  )
}

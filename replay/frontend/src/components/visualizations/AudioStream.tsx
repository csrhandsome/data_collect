import { useEffect, useRef, useState } from 'react'
import type { AudioData } from '../../types/dataset'

export function AudioStream({ data, url, time, playing, rate }: {
  data: AudioData; url: string; time: number; playing: boolean; rate: number
}) {
  const element = useRef<HTMLAudioElement>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    const audio = element.current
    if (!audio || !ready) return
    const offset = time-data.offset_s
    if (offset < 0 || offset >= data.duration_s) { audio.pause(); return }
    if (Math.abs(audio.currentTime-offset) > (playing ? 0.15 : 0.01)) audio.currentTime = offset
    audio.playbackRate = rate
    if (playing) void audio.play().catch(() => setError('点击页面播放按钮以允许音频播放。'))
    else audio.pause()
  }, [time, playing, rate, ready, data])
  const samples = data.waveform.map((v, i) => `${i ? 'L' : 'M'}${i/Math.max(1, data.waveform.length-1)*500},${70-v*60}`).join(' ')
  const cursor = Math.max(0, Math.min(500, (time-data.offset_s)/data.duration_s*500))
  return <div className="ee-trajectory">
    <audio ref={element} src={url} preload="metadata" onLoadedMetadata={() => setReady(true)}
      onError={() => setError('音频读取或解码失败。')} data-testid="episode-audio" />
    <svg viewBox="0 0 500 100" role="img" aria-label="音频波形" className="time-chart">
      <path d={samples} className="chart-line" stroke="#74d9b0" />
      <line x1={cursor} x2={cursor} y1="5" y2="90" className="playhead-line" />
    </svg>
    {error ? <p role="alert">{error}</p> : null}
    <div className="ee-details"><span>{data.sample_rate} Hz · {data.channels} 声道 · {data.duration_s.toFixed(2)} s</span>
      <span>VAD {data.vad_segments.length} 段</span></div>
    {data.instruction_audio_window?.start_sec != null ? <p className="audio-window">
      指令窗口 {data.instruction_audio_window.start_sec.toFixed(2)}–{data.instruction_audio_window.end_sec?.toFixed(2)} s
    </p> : null}
  </div>
}

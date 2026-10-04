import { useCallback, useEffect, useRef, useState } from 'react'

export function usePlayback(duration: number, selectionKey: string) {
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [rate, setRate] = useState(1)
  const [seekVersion, setSeekVersion] = useState(0)
  const clock = useRef({ time: 0, lastTick: 0, lastPaint: 0 })

  useEffect(() => {
    clock.current.time = 0
    setTime(0)
    setPlaying(false)
  }, [selectionKey])

  useEffect(() => {
    if (!playing || duration <= 0) return
    let frame = 0
    clock.current.lastTick = performance.now()
    clock.current.lastPaint = 0
    const tick = (now: number) => {
      const next = Math.min(
        duration,
        clock.current.time + ((now - clock.current.lastTick) / 1000) * rate,
      )
      clock.current.lastTick = now
      clock.current.time = next
      if (now - clock.current.lastPaint >= 1000 / 30 || next >= duration) {
        setTime(next)
        clock.current.lastPaint = now
      }
      if (next >= duration) setPlaying(false)
      else frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [playing, duration, rate])

  const seek = useCallback(
    (next: number) => {
      const clamped = Math.max(0, Math.min(duration, next))
      clock.current.time = clamped
      setTime(clamped)
      setSeekVersion((value) => value + 1)
    },
    [duration],
  )

  const toggle = useCallback(() => {
    if (duration <= 0) return
    if (clock.current.time >= duration) {
      clock.current.time = 0
      setTime(0)
      setSeekVersion((value) => value + 1)
    }
    setPlaying((value) => !value)
  }, [duration])

  const pause = useCallback(() => setPlaying(false), [])
  return { time, playing, rate, seekVersion, setRate, seek, toggle, pause }
}

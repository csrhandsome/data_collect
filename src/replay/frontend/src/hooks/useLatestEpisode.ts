import { useEffect, useRef, useState } from 'react'
import { getJson } from '../lib/api'
import type { LatestEpisode } from '../types/dataset'

export function useLatestEpisode(enabled: boolean, onSaved: (episode: LatestEpisode) => void) {
  const [error, setError] = useState<string | null>(null)
  const callback = useRef(onSaved)
  const follow = useRef(enabled)
  const previous = useRef<string | null | undefined>(undefined)
  callback.current = onSaved
  follow.current = enabled

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      try {
        const response = await getJson<{ episode: LatestEpisode | null }>(
          '/api/datasets/latest-episode',
          controller.signal,
        )
        if (controller.signal.aborted) return
        const episode = response.episode
        const key = episode
          ? `${episode.dataset_id}:${episode.episode_index}:${episode.saved_at_ns}`
          : null
        // The initial result is a baseline, so reloading an older episode
        // preserves the user's selection. Later saves open automatically.
        const changed = previous.current !== undefined && previous.current !== key
        previous.current = key
        setError(null)
        if (changed && episode && follow.current) callback.current(episode)
      } catch (error) {
        if (!controller.signal.aborted)
          setError(error instanceof Error ? error.message : '无法检测新采集片段')
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 1500)
      }
    }
    void poll()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [])

  return error
}

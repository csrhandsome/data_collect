import { useEffect, useState } from 'react'
import { getJson } from '../lib/api'
import type { Ros2Health } from '../types/ros2Health'

export function useRos2Health() {
  const [data, setData] = useState<Ros2Health | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    async function poll() {
      try {
        const result = await getJson<Ros2Health>('/api/ros2/health', controller.signal)
        if (!controller.signal.aborted) {
          setData(result)
          setError(null)
        }
      } catch {
        if (!controller.signal.aborted) setError('监控接口暂不可用，正在重连')
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 1000)
      }
    }
    void poll()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [])
  return { data, error }
}

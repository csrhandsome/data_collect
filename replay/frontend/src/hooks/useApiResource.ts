import { useCallback, useEffect, useState } from 'react'
import { getJson } from '../lib/api'

interface ResourceState<T> {
  url: string | null
  data: T | null
  error: string | null
  loading: boolean
}

export function useApiResource<T>(url: string | null) {
  const [attempt, setAttempt] = useState(0)
  const [state, setState] = useState<ResourceState<T>>({
    url: null,
    data: null,
    error: null,
    loading: false,
  })
  useEffect(() => {
    if (!url) return
    const controller = new AbortController()
    setState({ url, data: null, error: null, loading: true })
    getJson<T>(url, controller.signal).then(
      (data) => {
        if (!controller.signal.aborted) setState({ url, data, error: null, loading: false })
      },
      (error: unknown) => {
        if (!controller.signal.aborted) {
          setState({
            url,
            data: null,
            error: error instanceof Error ? error.message : '读取数据失败',
            loading: false,
          })
        }
      },
    )
    return () => controller.abort()
  }, [url, attempt])

  const retry = useCallback(() => setAttempt((value) => value + 1), [])
  const current =
    state.url === url ? state : { url, data: null, error: null, loading: Boolean(url) }
  return { ...current, retry }
}

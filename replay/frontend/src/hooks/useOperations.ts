import { useCallback, useEffect, useRef, useState } from 'react'
import { getJson, postJson } from '../lib/api'
import { isOperationActive, type Operation, type OperationKind } from '../types/operation'

export function useOperations(onFinished: (operation: Operation) => void) {
  const [operation, setOperation] = useState<Operation | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)
  const [pending, setPending] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const previous = useRef<Operation | null>(null)
  const finishedCallback = useRef(onFinished)
  const submitting = useRef(false)
  const mutationVersion = useRef(0)
  finishedCallback.current = onFinished

  const accept = useCallback((next: Operation | null) => {
    const before = previous.current
    previous.current = next
    setOperation(next)
    if (
      next &&
      !isOperationActive(next) &&
      (isOperationActive(before) || (before && before.id !== next.id))
    ) {
      finishedCallback.current(next)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      const version = mutationVersion.current
      try {
        const response = await getJson<{ operation: Operation | null }>(
          '/api/operations',
          controller.signal,
        )
        if (!controller.signal.aborted) {
          if (!submitting.current && version === mutationVersion.current) accept(response.operation)
          setReady(true)
          setError(null)
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          setReady(false)
          setError(error instanceof Error ? error.message : '无法读取任务状态')
        }
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 1000)
      }
    }
    void poll()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [attempt, accept])

  async function mutate(url: string, body?: unknown) {
    if (submitting.current) return false
    submitting.current = true
    mutationVersion.current += 1
    setPending(true)
    setActionError(null)
    try {
      accept(await postJson<Operation>(url, body))
      return true
    } catch (error) {
      setActionError(error instanceof Error ? error.message : '操作失败')
      setReady(false)
      return false
    } finally {
      submitting.current = false
      mutationVersion.current += 1
      setPending(false)
      setAttempt((value) => value + 1)
    }
  }

  return {
    operation,
    error: actionError || error,
    ready,
    pending,
    busy: pending || isOperationActive(operation),
    start: (kind: OperationKind, datasetId?: string, episodeIndex?: number) =>
      mutate('/api/operations', { kind, dataset_id: datasetId, episode_index: episodeIndex }),
    stop: () =>
      operation ? mutate(`/api/operations/${operation.id}/stop`) : Promise.resolve(false),
    retry: () => {
      setActionError(null)
      setAttempt((value) => value + 1)
    },
  }
}

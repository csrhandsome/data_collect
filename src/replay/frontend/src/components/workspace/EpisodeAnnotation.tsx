import { useEffect, useRef, useState } from 'react'
import { episodeUrl, postJson } from '../../lib/api'
import { episodeLabel } from '../../lib/format'
import type { EpisodeDetail } from '../../types/dataset'
import { Button } from '../ui/Button'

export function EpisodeAnnotation({
  episode,
  onSaved,
  disabled = false,
  inferenceResult = false,
}: {
  episode: EpisodeDetail
  onSaved: (episode: EpisodeDetail) => void
  disabled?: boolean
  inferenceResult?: boolean
}) {
  const [success, setSuccess] = useState(episode.success ?? null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  const submitting = useRef(false)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  useEffect(() => {
    setSuccess(episode.success ?? null)
  }, [episode.success])

  async function annotate(value: boolean | null) {
    if (submitting.current || disabled) return
    submitting.current = true
    setPending(true)
    setError(null)
    setSaved(false)
    try {
      const result = await postJson<EpisodeDetail>(
        `${episodeUrl(episode.dataset_id, episode.episode_index)}/annotation`,
        { success: value, expected_saved_at_ns: episode.saved_at_ns },
      )
      if (mounted.current) {
        setSuccess(result.success)
        setSaved(true)
        onSaved(result)
      }
    } catch (error) {
      if (mounted.current) setError(error instanceof Error ? error.message : '标注保存失败，请重试')
    } finally {
      submitting.current = false
      if (mounted.current) setPending(false)
    }
  }

  const status = success === null ? '未标注' : success ? '成功' : '失败'
  return (
    <section
      aria-label="片段结果标注"
      className="mb-3 shrink-0 border border-foreground bg-background"
      data-testid="episode-annotation"
      aria-busy={pending}
    >
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2">
        <div>
          <h2 className="font-display text-lg">
            {inferenceResult ? '本次推理结果' : '片段结果'} · {episodeLabel(episode.episode_index)}
          </h2>
          <p role="status" className="mt-1 font-mono text-[11px] text-muted-foreground">
            {pending ? '正在保存标注…' : `当前结果：${status}${saved ? ' · 已保存' : ''}`}
          </p>
          {disabled ? (
            <p className="mt-1 text-xs text-muted-foreground">
              推理正在运行或返回起始姿态，结束后再标注本次结果。
            </p>
          ) : null}
        </div>
        <div className="flex flex-wrap gap-2" role="group" aria-label="选择片段结果">
          {(
            [
              [true, '成功'],
              [false, '失败'],
              [null, '未标注'],
            ] as const
          ).map(([value, label]) => (
            <Button
              key={label}
              variant={success === value ? 'primary' : 'secondary'}
              aria-pressed={success === value}
              disabled={pending || disabled}
              onClick={() => void annotate(value)}
            >
              {label}
            </Button>
          ))}
        </div>
      </div>
      {error ? (
        <p role="alert" className="px-4 pb-3 text-sm">
          {error}
        </p>
      ) : null}
    </section>
  )
}

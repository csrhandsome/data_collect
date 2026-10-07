import { useEffect, useRef, useState } from 'react'
import { episodeUrl, postJson } from '../../lib/api'
import { episodeLabel } from '../../lib/format'
import type { EpisodeDetail } from '../../types/dataset'
import { Button } from '../ui/Button'

export function EpisodeAnnotation({
  episode,
  onSaved,
}: {
  episode: EpisodeDetail
  onSaved: (episode: EpisodeDetail) => void
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
    if (submitting.current) return
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
      className="mb-5 shrink-0 border-2 border-foreground bg-background"
      data-testid="episode-annotation"
      aria-busy={pending}
    >
      <div className="flex flex-wrap items-center justify-between gap-4 px-4 py-4">
        <div>
          <h2 className="font-display text-xl">片段结果 · {episodeLabel(episode.episode_index)}</h2>
          <p className="mt-1 text-xs text-muted-foreground">查看回放后标注结果，选择后自动保存。</p>
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
              disabled={pending}
              onClick={() => void annotate(value)}
            >
              {label}
            </Button>
          ))}
        </div>
      </div>
      <p role="status" className="border-t border-foreground px-4 py-2 font-mono text-xs">
        {pending ? '正在保存标注…' : `当前结果：${status}${saved ? ' · 已保存' : ''}`}
      </p>
      {error ? (
        <p role="alert" className="px-4 pb-3 text-sm">
          {error}
        </p>
      ) : null}
    </section>
  )
}

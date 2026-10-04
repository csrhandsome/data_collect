import { useEffect, useState, type ReactNode } from 'react'

export function WaitingState({
  label = '正在读取数据…',
  description,
  startedAt,
  compact = false,
  children,
}: {
  label?: string
  description?: string
  startedAt?: string
  compact?: boolean
  children?: ReactNode
}) {
  const [now, setNow] = useState(Date.now)
  useEffect(() => {
    if (!startedAt) return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [startedAt])
  const elapsed = startedAt ? Math.max(0, Math.floor((now - Date.parse(startedAt)) / 1000)) : null
  return (
    <div
      className={`flex gap-4 ${compact ? 'flex-wrap items-center' : 'min-h-44 flex-col items-center justify-center p-5 text-center'}`}
      aria-busy="true"
    >
      <span
        aria-hidden="true"
        className="relative grid size-9 shrink-0 place-items-center border-2 border-foreground"
      >
        <span className="size-4 border-2 border-foreground border-t-transparent motion-safe:animate-spin" />
        <span className="absolute -right-1 -bottom-1 size-2 bg-foreground" />
      </span>
      <div className="min-w-0 flex-1" role="status" aria-live="polite">
        <p className="text-sm leading-relaxed">{label}</p>
        {description ? (
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{description}</p>
        ) : null}
      </div>
      {elapsed !== null ? (
        <span
          className="shrink-0 font-mono text-xs tabular-nums"
          aria-label={`已等待 ${elapsed} 秒`}
        >
          {Math.floor(elapsed / 60)
            .toString()
            .padStart(2, '0')}
          :{(elapsed % 60).toString().padStart(2, '0')}
        </span>
      ) : null}
      {children}
    </div>
  )
}

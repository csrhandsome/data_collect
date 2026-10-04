import { useEffect, useId, useRef, type ReactNode } from 'react'
import { Button } from './Button'

export function ConfirmDialog({
  title,
  children,
  confirmLabel,
  pending,
  onConfirm,
  onClose,
}: {
  title: string
  children: ReactNode
  confirmLabel: string
  pending: boolean
  onConfirm: () => void
  onClose: () => void
}) {
  const ref = useRef<HTMLDialogElement>(null)
  const titleId = useId()
  const descriptionId = useId()
  useEffect(() => {
    const dialog = ref.current
    dialog?.showModal()
    return () => dialog?.close()
  }, [])
  return (
    <dialog
      ref={ref}
      aria-labelledby={titleId}
      aria-describedby={descriptionId}
      onCancel={(event) => {
        event.preventDefault()
        if (!pending) onClose()
      }}
      className="fixed inset-0 m-auto w-[calc(100%-2rem)] max-w-lg border-4 border-foreground bg-background p-6 text-foreground backdrop:bg-foreground/40 md:p-8"
    >
      <p className="mb-4 font-mono text-[10px] tracking-widest text-muted-foreground">
        LOCAL WORKSPACE / CONFIRM
      </p>
      <h2 id={titleId} className="font-display text-3xl">
        {title}
      </h2>
      <div id={descriptionId} className="mt-4 text-sm leading-7 [overflow-wrap:anywhere]">
        {children}
      </div>
      <div className="mt-6 flex flex-wrap justify-end gap-3">
        <Button variant="secondary" autoFocus disabled={pending} onClick={onClose}>
          取消
        </Button>
        <Button disabled={pending} onClick={onConfirm}>
          {pending ? '正在提交…' : confirmLabel}
        </Button>
      </div>
    </dialog>
  )
}

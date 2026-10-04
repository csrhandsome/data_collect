import { Button } from './Button'
import { Icon } from './Icon'
import { WaitingState } from './WaitingState'

export function LoadingState({ label = '正在读取数据…' }: { label?: string }) {
  return <WaitingState label={label} />
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      className="flex min-h-44 flex-col items-center justify-center gap-4 p-5 text-center text-base leading-relaxed text-foreground [&>p]:max-w-lg [&>p]:[overflow-wrap:anywhere]"
      role="alert"
    >
      <Icon name="warning" size={24} />
      <p>{message}</p>
      {onRetry ? (
        <Button variant="secondary" onClick={onRetry}>
          <Icon name="reset" size={14} />
          重试加载
        </Button>
      ) : null}
    </div>
  )
}

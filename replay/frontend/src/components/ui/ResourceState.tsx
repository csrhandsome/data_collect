import { Icon } from './Icon'

export function LoadingState({ label = '正在读取数据…' }: { label?: string }) {
  return (
    <div className="resource-state" role="status">
      <span className="spinner" />
      {label}
    </div>
  )
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="resource-state error-state" role="alert">
      <Icon name="warning" size={24} />
      <p>{message}</p>
      {onRetry ? (
        <button className="button button-secondary" onClick={onRetry}>
          <Icon name="reset" size={14} />
          重试加载
        </button>
      ) : null}
    </div>
  )
}

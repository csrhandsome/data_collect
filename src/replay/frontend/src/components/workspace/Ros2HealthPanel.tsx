import { useState } from 'react'
import { useRos2Health } from '../../hooks/useRos2Health'
import { Icon } from '../ui/Icon'

const stateLabels = { ok: '接收正常', warning: 'WARNING', waiting: '等待启动', stopped: '已停止' }

export function Ros2HealthPanel() {
  const { data, error } = useRos2Health()
  const [expanded, setExpanded] = useState(false)
  const warnings = data?.topics.filter((topic) => topic.state === 'warning') || []
  const offline = Boolean(error) || data?.state === 'offline'
  const waiting = !data || data.state === 'not_started' || data.state === 'disabled'
  const hasWarning = offline || warnings.length > 0
  const hasActiveSubscriber = data?.topics.some((topic) => topic.state === 'ok')
  const summary = error || data?.message || '正在读取监控状态'
  const label = offline
    ? '监控未连接'
    : warnings.length
      ? `${warnings.length} 项 WARNING`
      : waiting
        ? data?.state === 'disabled'
          ? '监控已关闭'
          : '等待启动'
        : hasActiveSubscriber
          ? '订阅监控在线'
          : '等待采集订阅'

  return (
    <section
      aria-label="ROS2 订阅监控"
      data-testid="ros2-health"
      className={`relative z-10 shrink-0 border-b border-foreground ${hasWarning ? 'bg-[#fff8e6] text-[#754500]' : 'bg-muted'}`}
    >
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-5 py-2 md:px-6">
        <span className="inline-flex items-center gap-2 font-mono text-xs font-bold" role="status">
          <Icon name={hasWarning ? 'warning' : hasActiveSubscriber ? 'check' : 'pause'} size={15} />
          ROS2 · {label}
        </span>
        <span className="min-w-0 flex-1 truncate text-xs" title={summary}>
          {warnings[0] && !offline ? `${warnings[0].topic} · ${warnings[0].message}` : summary}
        </span>
        <button
          type="button"
          aria-expanded={expanded}
          aria-controls="ros2-health-details"
          className="min-h-9 shrink-0 border-b border-current font-mono text-xs"
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? '收起监控' : '查看监控'}
        </button>
      </div>
      {expanded && (
        <div
          id="ros2-health-details"
          className="max-h-80 overflow-auto border-t border-current/20 px-5 py-4 md:px-6"
        >
          <p className="mb-3 text-xs">
            监控仅提示异常，原有控制保护继续生效。频率下降表示可能跳帧或处理滞后，不代表精确丢帧数。
          </p>
          {waiting ? (
            <p className="font-mono text-xs">
              在 Docker 内运行 bash scripts/deploy_ros2.sh 启动监控。
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[620px] text-left text-xs">
                <caption className="sr-only">实际订阅接收状态与驱动发现结果</caption>
                <thead className="border-b border-current/30 font-mono">
                  <tr>
                    <th className="py-2 pr-4">话题 / 订阅节点</th>
                    <th className="pr-4">状态</th>
                    <th className="pr-4">接收 Hz</th>
                    <th className="pr-4">最近接收</th>
                    <th>说明</th>
                  </tr>
                </thead>
                <tbody>
                  {data?.topics.map((topic) => (
                    <tr key={topic.id} className="border-b border-current/10 align-top">
                      <td className="py-2 pr-4">
                        <span className="block break-all font-mono">{topic.topic}</span>
                        <span className="opacity-65">
                          {topic.component} ·{' '}
                          {topic.source === 'subscriber' ? '实际订阅' : '驱动发现'}
                        </span>
                      </td>
                      <td className="py-2 pr-4 font-mono font-bold">
                        {offline ? '状态未知' : stateLabels[topic.state]}
                      </td>
                      <td className="py-2 pr-4 font-mono">
                        {topic.received_hz === null ? '—' : topic.received_hz.toFixed(1)}
                        {!!topic.min_hz && (
                          <span className="block opacity-65">最低 {topic.min_hz.toFixed(1)}</span>
                        )}
                      </td>
                      <td className="py-2 pr-4 font-mono">
                        {topic.age_s === null ? '—' : `${topic.age_s.toFixed(2)} s`}
                      </td>
                      <td className="py-2">{topic.message}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {!!data?.events.length && (
            <div className="mt-4">
              <h2 className="mb-2 font-mono text-xs font-bold">最近的 WARNING 与恢复</h2>
              <ol className="space-y-2 text-xs">
                {data.events.slice(0, 10).map((event, index) => (
                  <li
                    key={`${event.time}-${event.topic}-${index}`}
                    className="flex flex-wrap gap-x-3"
                  >
                    <time
                      className="font-mono opacity-65"
                      dateTime={new Date(event.time * 1000).toISOString()}
                    >
                      {new Date(event.time * 1000).toLocaleTimeString('zh-CN', { hour12: false })}
                    </time>
                    <strong>{event.state === 'warning' ? 'WARNING' : '恢复 / 已停止'}</strong>
                    <span className="break-all font-mono">{event.topic}</span>
                    <span>{event.message}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </section>
  )
}

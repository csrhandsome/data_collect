import { useRef, useState } from 'react'
import { useOperations } from '../../hooks/useOperations'
import { isOperationActive, type Operation, type OperationKind } from '../../types/operation'
import type { DatasetDetail } from '../../types/dataset'
import { episodeLabel } from '../../lib/format'
import { Button } from '../ui/Button'
import { Icon } from '../ui/Icon'
import { WaitingState } from '../ui/WaitingState'
import { ConfirmDialog } from '../ui/ConfirmDialog'
import { buttonClasses } from '../ui/buttonStyles'

const labels = { collect: 'VR 采集', inference: '策略推理', replay: '真机回放', delete: '删除片段' }
const states = {
  running: '运行中',
  stopping: '正在停止',
  succeeded: '已完成',
  failed: '失败',
  cancelled: '已停止',
}

export function OperationPanel({
  dataset,
  episodeIndex,
  onFinished,
  onChanged,
}: {
  dataset: DatasetDetail | null
  episodeIndex: number
  onFinished: (operation: Operation) => void
  onChanged?: (operation: Operation | null) => void
}) {
  const tasks = useOperations(onFinished, onChanged)
  const [selectedMode, setSelectedMode] = useState<'collect' | 'inference' | null>(null)
  const more = useRef<HTMLDetailsElement>(null)
  const [confirmation, setConfirmation] = useState<{
    kind: OperationKind
    datasetId?: string
    episodeIndex?: number
  } | null>(null)
  const latest =
    dataset?.episodes.reduce((max, item) => Math.max(max, item.episode_index), -1) ?? -1
  const hasEpisode = Boolean(dataset?.episodes.some((item) => item.episode_index === episodeIndex))
  const canDelete = ['v2.0', 'v2.1', 'v3.0'].includes(dataset?.version ?? '')
  const operation = tasks.operation
  const active = isOperationActive(operation)
  const disabled = !tasks.ready || tasks.busy
  const mode =
    active && (operation?.kind === 'collect' || operation?.kind === 'inference')
      ? operation.kind
      : (selectedMode ?? (operation?.kind === 'inference' ? 'inference' : 'collect'))
  const canStop = active && operation?.kind !== 'delete'
  const stopLabel =
    operation?.kind === 'inference'
      ? '结束推理'
      : operation?.kind === 'collect'
        ? '结束采集'
        : '停止任务'

  function requestConfirmation(value: NonNullable<typeof confirmation>) {
    if (more.current) more.current.open = false
    setConfirmation(value)
  }

  async function confirm() {
    if (!confirmation) return
    if (await tasks.start(confirmation.kind, confirmation.datasetId, confirmation.episodeIndex))
      setConfirmation(null)
  }

  return (
    <section
      aria-label="采集推理与数据操作"
      className="mb-5 shrink-0 border-2 border-foreground bg-background"
    >
      <div className="flex flex-wrap items-center justify-between gap-4 px-4 py-4">
        <div>
          <h2 className="font-display text-xl">采集与推理</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            {mode === 'inference'
              ? '结束推理后保存本次片段并回到起始姿态，再查看结果并标注。'
              : '手柄保存片段后，在下方查看回放并标注成功或失败。'}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex" role="group" aria-label="选择操作模式">
            {(
              [
                ['collect', '采集'],
                ['inference', '推理'],
              ] as const
            ).map(([value, label]) => (
              <Button
                key={value}
                variant={mode === value ? 'primary' : 'secondary'}
                aria-pressed={mode === value}
                disabled={tasks.busy}
                onClick={() => setSelectedMode(value)}
              >
                {label}
              </Button>
            ))}
          </div>
          {canStop ? (
            <Button
              disabled={tasks.pending || operation?.state === 'stopping'}
              onClick={() => void tasks.stop()}
            >
              <Icon name="close" size={14} />
              {operation?.state === 'stopping' ? '正在结束…' : stopLabel}
            </Button>
          ) : (
            <Button disabled={disabled} onClick={() => requestConfirmation({ kind: mode })}>
              <Icon name={mode === 'inference' ? 'play' : 'plus'} size={14} />
              {mode === 'inference' ? '开始推理' : '开启采集'}
            </Button>
          )}
          <details ref={more} className="relative">
            <summary
              className={`${buttonClasses('secondary')} cursor-pointer list-none [&::-webkit-details-marker]:hidden`}
            >
              更多操作
            </summary>
            <div className="absolute right-0 z-40 mt-2 flex min-w-48 flex-col gap-2 border-2 border-foreground bg-background p-2 shadow-lg">
              <Button
                variant="secondary"
                disabled={disabled || latest < 0}
                onClick={() =>
                  requestConfirmation({
                    kind: 'replay',
                    datasetId: dataset!.id,
                    episodeIndex: latest,
                  })
                }
              >
                <Icon name="reset" size={14} />
                真机回放上一次
              </Button>
              <Button
                variant="secondary"
                disabled={disabled || !hasEpisode || !canDelete}
                onClick={() =>
                  requestConfirmation({ kind: 'delete', datasetId: dataset!.id, episodeIndex })
                }
              >
                <Icon name="close" size={14} />
                删除当前片段
              </Button>
            </div>
          </details>
        </div>
      </div>
      {dataset && !canDelete ? (
        <p className="px-4 pb-3 text-xs text-muted-foreground">此数据格式暂不支持删除。</p>
      ) : null}
      {tasks.error ? (
        <div
          role="alert"
          className="flex flex-wrap items-center gap-3 border-t border-foreground px-4 py-3 text-sm"
        >
          <Icon name="warning" size={16} />
          <span className="flex-1">{tasks.error}</span>
          <Button variant="secondary" onClick={tasks.retry}>
            刷新任务状态
          </Button>
        </div>
      ) : null}
      {tasks.pending && !active ? (
        <div className="border-t border-foreground p-4">
          <WaitingState label="正在提交操作…" compact />
        </div>
      ) : null}
      {operation ? (
        <div
          className="border-t border-foreground bg-muted px-4 py-3"
          data-testid="operation-status"
        >
          {active ? (
            <WaitingState
              compact
              startedAt={operation.started_at}
              label={`${labels[operation.kind]} · ${states[operation.state]}`}
              description={
                operation.state === 'stopping'
                  ? operation.kind === 'inference'
                    ? '正在停止推理、保存片段并返回起始姿态，请等待完成。'
                    : '等待设备停止及文件保存，请保持服务运行。'
                  : operation.kind === 'delete'
                    ? '正在删除文件并重排编号，请等待完成。'
                    : '任务运行中，可继续查看数据。'
              }
            />
          ) : (
            <p className="flex items-center gap-2 text-sm" role="status">
              <Icon name={operation.state === 'failed' ? 'warning' : 'check'} size={16} />
              {labels[operation.kind]} · {states[operation.state]}
              {operation.state === 'failed' ? `（退出码 ${operation.return_code}）` : ''}
            </p>
          )}
          {operation.dataset_id ? (
            <p className="mt-2 font-mono text-[10px] text-muted-foreground [overflow-wrap:anywhere]">
              {operation.dataset_id} / {episodeLabel(operation.episode_index ?? 0)}
            </p>
          ) : null}
          <details className="mt-2" open={operation.state === 'failed' ? true : undefined}>
            <summary className="font-mono text-[11px] text-muted-foreground">
              运行日志 · 最近 {operation.logs.length} 行
            </summary>
            <pre className="mt-2 max-h-40 overflow-auto border-l-2 border-foreground bg-background p-3 font-mono text-[11px] leading-5 whitespace-pre-wrap [overflow-wrap:anywhere]">
              {operation.logs.join('\n') || '等待进程输出…'}
            </pre>
          </details>
        </div>
      ) : null}
      {confirmation ? (
        <ConfirmDialog
          title={
            confirmation.kind === 'collect'
              ? '开启 VR 采集'
              : confirmation.kind === 'inference'
                ? '开始策略推理'
                : confirmation.kind === 'replay'
                  ? '在真机上回放'
                  : '删除当前片段'
          }
          confirmLabel={
            confirmation.kind === 'collect'
              ? '确认开启采集'
              : confirmation.kind === 'inference'
                ? '确认开始推理'
                : confirmation.kind === 'replay'
                  ? '确认真机回放'
                  : '确认删除'
          }
          pending={tasks.pending}
          onConfirm={() => void confirm()}
          onClose={() => setConfirmation(null)}
        >
          {confirmation.kind === 'collect' ? (
            <p>
              采集将按 panda.yaml
              的设备和保存目录配置启动。机械臂可能移动至起始姿态，请确认工作区域已准备好。
            </p>
          ) : confirmation.kind === 'inference' ? (
            <p>
              推理将使用 panda.yaml 的策略服务和任务配置，控制机械臂与夹爪。
              结束后保存本次推理片段，并将机械臂返回配置的起始姿态。请确认工作区域已准备好。
            </p>
          ) : confirmation.kind === 'replay' ? (
            <p>
              将回放 <strong>{confirmation.datasetId}</strong> 最后保存的{' '}
              <strong>{episodeLabel(confirmation.episodeIndex!)}</strong>
              ，包含机械臂移动与夹爪操作。请确认工作区域已准备好。
            </p>
          ) : (
            <p>
              将永久删除 <strong>{confirmation.datasetId}</strong> 的{' '}
              <strong>{episodeLabel(confirmation.episodeIndex!)}</strong>
              ，同时删除图像、视频、音频与采样记录，后续 episode 会重新编号。此操作无法撤销。
            </p>
          )}
        </ConfirmDialog>
      ) : null}
    </section>
  )
}

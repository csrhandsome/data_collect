import type { LatestEpisode } from './dataset'

export type OperationKind = 'collect' | 'inference' | 'replay' | 'delete'

export interface Operation {
  id: string
  kind: OperationKind
  state: 'running' | 'stopping' | 'succeeded' | 'failed' | 'cancelled'
  dataset_id: string | null
  episode_index: number | null
  started_at: string
  finished_at: string | null
  return_code: number | null
  result_episode?: LatestEpisode | null
  logs: string[]
}

export function isOperationActive(operation: Operation | null) {
  return operation?.state === 'running' || operation?.state === 'stopping'
}

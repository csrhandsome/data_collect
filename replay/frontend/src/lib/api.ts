export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export async function getJson<T>(url: string, signal: AbortSignal): Promise<T> {
  let response: Response
  try {
    response = await fetch(url, {
      signal,
      headers: { Accept: 'application/json' },
    })
  } catch (error) {
    if (signal.aborted) throw error
    throw new ApiError('无法连接后端，请确认 FastAPI 已启动。', 0)
  }
  if (!response.ok) {
    const body: { detail?: unknown } = await response.json().catch(() => ({}))
    const detail = typeof body.detail === 'string' ? body.detail : null
    throw new ApiError(detail || `数据请求失败（HTTP ${response.status}）`, response.status)
  }
  return response.json() as Promise<T>
}

export function episodeUrl(datasetId: string, episodeIndex: number) {
  return `/api/datasets/${encodeURIComponent(datasetId)}/episodes/${episodeIndex}`
}

export const FEATURE_DRAG_TYPE = 'application/x-replay-feature'

export async function postJson<T>(url: string, body?: unknown): Promise<T> {
  let response: Response
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch {
    throw new ApiError('无法连接后端。请先刷新任务状态，确认操作是否已启动。', 0)
  }
  if (!response.ok) {
    const body: { detail?: unknown } = await response.json().catch(() => ({}))
    throw new ApiError(
      typeof body.detail === 'string' ? body.detail : `操作失败（HTTP ${response.status}）`,
      response.status,
    )
  }
  return response.json() as Promise<T>
}

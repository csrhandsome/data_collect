import { expect, test, type Page } from '@playwright/test'
import { cp, readFile, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const demoRoot = fileURLToPath(new URL('../../demo_data/', import.meta.url))

interface FakeOperation {
  id: string
  kind: string
  state: string
  dataset_id: string | null
  episode_index: number | null
  started_at: string
  finished_at: string | null
  return_code: number | null
  logs: string[]
  result_episode?: { dataset_id: string; episode_index: number; saved_at_ns: string } | null
}

async function mockOperations(page: Page, deferStop = false) {
  let operation: FakeOperation | null = null
  const requests: Record<string, unknown>[] = []
  await page.route('**/api/operations{,/**}', async (route) => {
    const request = route.request()
    if (request.method() === 'GET') {
      await route.fulfill({ json: { operation } })
    } else if (request.url().endsWith('/stop')) {
      operation = {
        ...operation!,
        state: deferStop ? 'stopping' : 'cancelled',
        finished_at: deferStop ? null : new Date().toISOString(),
        return_code: deferStop ? null : 0,
      }
      await route.fulfill({ json: operation })
    } else {
      const body = request.postDataJSON() as Record<string, unknown>
      requests.push(body)
      operation = {
        id: `test-${requests.length}`,
        kind: String(body.kind),
        state: 'running',
        dataset_id: typeof body.dataset_id === 'string' ? body.dataset_id : null,
        episode_index: typeof body.episode_index === 'number' ? body.episode_index : null,
        started_at: new Date().toISOString(),
        finished_at: null,
        return_code: null,
        logs: ['临时测试任务，未连接机器人'],
      }
      await route.fulfill({ status: 202, json: operation })
    }
  })
  return {
    requests,
    finish: (state = 'succeeded', result_episode?: FakeOperation['result_episode']) => {
      operation = {
        ...operation!,
        state,
        result_episode,
        finished_at: new Date().toISOString(),
        return_code: 0,
      }
    },
  }
}

async function openDataset(page: Page, dataset = 'demo_v21', episode = 0) {
  await page.goto(`/replay?dataset=${dataset}&episode=${episode}`)
  await expect(page.getByRole('button', { name: '开启采集', exact: true })).toBeEnabled()
  await expect(page.getByTestId('episode-select')).toBeEnabled()
}

test('collection requires confirmation, keeps status after reload, and stops', async ({ page }) => {
  const fake = await mockOperations(page)
  await openDataset(page)
  await page.getByRole('button', { name: '开启采集', exact: true }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
  expect(fake.requests).toHaveLength(0)
  await expect(page.getByRole('button', { name: '取消', exact: true })).toBeFocused()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).not.toBeVisible()
  await page.getByRole('button', { name: '开启采集', exact: true }).click()
  await page.getByRole('button', { name: '确认开启采集' }).click()
  await expect(page.getByTestId('operation-status')).toContainText('VR 采集 · 运行中')
  expect(fake.requests).toEqual([{ kind: 'collect' }])
  await page.getByText('更多操作', { exact: true }).click()
  await expect(page.getByRole('button', { name: '真机回放上一次' })).toBeDisabled()
  await page.reload()
  await expect(page.getByTestId('operation-status')).toContainText('VR 采集 · 运行中')
  await expect(page.getByLabel(/已等待 \d+ 秒/)).toBeVisible()
  await page.getByRole('button', { name: '结束采集' }).click()
  await expect(page.getByTestId('operation-status')).toContainText('VR 采集 · 已停止')
  await expect(page.getByRole('button', { name: '开启采集', exact: true })).toBeEnabled()
})

test('real replay selects last saved episode even when viewing the first', async ({ page }) => {
  const fake = await mockOperations(page)
  await openDataset(page)
  await page.getByText('更多操作', { exact: true }).click()
  await page.getByRole('button', { name: '真机回放上一次' }).click()
  await expect(page.getByRole('dialog')).toContainText('Episode 002')
  await page.getByRole('button', { name: '确认真机回放' }).click()
  expect(fake.requests).toEqual([{ kind: 'replay', dataset_id: 'demo_v21', episode_index: 2 }])
  await expect(page.getByTestId('operation-status')).toContainText('真机回放 · 运行中')
})

test('delete confirms selected episode, waits and refreshes the renumbered list', async ({
  page,
}) => {
  const fake = await mockOperations(page)
  let deleted = false
  let refreshed = false
  await page.route('**/api/datasets/demo_v21', async (route) => {
    const response = await route.fetch()
    const data = await response.json()
    if (deleted) {
      refreshed = true
      data.episodes = data.episodes.slice(0, 2)
      data.total_episodes = 2
      data.total_frames = 360
    }
    await route.fulfill({ response, json: data })
  })
  await openDataset(page, 'demo_v21', 1)
  await page.getByText('更多操作', { exact: true }).click()
  await page.getByRole('button', { name: '删除当前片段' }).click()
  await expect(page.getByRole('dialog')).toContainText('Episode 001')
  await expect(page.getByRole('dialog')).toContainText('无法撤销')
  await page.getByRole('button', { name: '确认删除', exact: true }).click()
  expect(fake.requests).toEqual([{ kind: 'delete', dataset_id: 'demo_v21', episode_index: 1 }])
  await expect(page.getByTestId('operation-status')).toContainText('删除片段 · 运行中')
  await expect(page.getByRole('button', { name: '停止任务' })).toHaveCount(0)
  deleted = true
  fake.finish()
  await expect(page.getByTestId('operation-status')).toContainText('删除片段 · 已完成')
  await expect.poll(() => refreshed).toBe(true)
  await expect(page.getByTestId('episode-select').locator('option')).toHaveCount(2)
})

test('v3 delete is available and narrow layouts remain usable', async ({ page }) => {
  await mockOperations(page)
  await page.setViewportSize({ width: 390, height: 844 })
  await openDataset(page, 'demo_v30')
  await page.getByText('更多操作', { exact: true }).click()
  await expect(page.getByRole('button', { name: '删除当前片段' })).toBeEnabled()
  await expect(page.getByRole('button', { name: '真机回放上一次' })).toBeEnabled()
  await page.getByRole('button', { name: '开启采集', exact: true }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )
})

test('inference stops, waits for reset, then opens and labels its own saved episode', async ({
  page,
}) => {
  const fake = await mockOperations(page, true)
  const name = 'test_inference_result'
  const root = join(demoRoot, name)
  await rm(root, { recursive: true, force: true })
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  try {
    await openDataset(page)
    await page.getByRole('checkbox', { name: '自动打开新保存片段' }).uncheck()
    await page.getByRole('button', { name: '推理', exact: true }).click()
    await page.getByRole('button', { name: '开始推理', exact: true }).click()
    await expect(page.getByRole('dialog')).toContainText('起始姿态')
    expect(fake.requests).toHaveLength(0)
    await page.getByRole('button', { name: '确认开始推理' }).click()
    expect(fake.requests).toEqual([{ kind: 'inference' }])
    await expect(page.getByTestId('operation-status')).toContainText('策略推理 · 运行中')
    await expect(page.getByRole('button', { name: '采集', exact: true })).toBeDisabled()
    await expect(
      page.getByTestId('episode-annotation').getByRole('button', { name: '成功', exact: true }),
    ).toBeDisabled()
    await page.reload()
    await expect(page.getByRole('button', { name: '推理', exact: true })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    await expect(page.getByRole('button', { name: '结束推理', exact: true })).toBeEnabled()
    await page.getByRole('checkbox', { name: '自动打开新保存片段' }).uncheck()
    await page.getByRole('button', { name: '结束推理', exact: true }).click()
    await expect(page.getByTestId('operation-status')).toContainText('策略推理 · 正在停止')
    await expect(page.getByTestId('operation-status')).toContainText('返回起始姿态')
    await expect(page.getByRole('button', { name: '正在结束…', exact: true })).toBeDisabled()
    await expect(
      page.getByTestId('episode-annotation').getByRole('button', { name: '失败', exact: true }),
    ).toBeDisabled()

    await cp(join(demoRoot, 'demo_v21'), root, { recursive: true })
    const sync = join(root, 'episode_000002.sync.json')
    const savedAt = Date.now() * 1_000_000
    await writeFile(
      sync,
      JSON.stringify({
        episode_index: 2,
        success: null,
        source: 'inference',
        saved_at_ns: savedAt,
      }),
    )
    fake.finish('cancelled', { dataset_id: name, episode_index: 2, saved_at_ns: String(savedAt) })
    await expect(page.getByTestId('operation-status')).toContainText('策略推理 · 已停止')
    await expect(page.getByTestId('dataset-select')).toHaveValue(name)
    await expect(page.getByTestId('episode-select')).toHaveValue('2')
    const annotation = page.getByTestId('episode-annotation')
    await expect(annotation).toContainText('本次推理结果 · Episode 002')
    await expect(page.getByTestId('visualization-wrist_image_left')).toBeVisible()
    await annotation.getByRole('button', { name: '成功', exact: true }).click()
    await expect(annotation).toContainText('当前结果：成功 · 已保存')
    expect(JSON.parse(await readFile(sync, 'utf8')).success).toBe(true)
    await annotation.getByRole('button', { name: '失败', exact: true }).click()
    await expect(annotation).toContainText('当前结果：失败 · 已保存')
    expect(JSON.parse(await readFile(sync, 'utf8')).success).toBe(false)
    await annotation.getByRole('button', { name: '未标注', exact: true }).click()
    await expect(annotation).toContainText('当前结果：未标注 · 已保存')
    expect(JSON.parse(await readFile(sync, 'utf8')).success).toBe(null)
    await expect(page.getByRole('button', { name: '开始推理', exact: true })).toBeEnabled()
    await page.setViewportSize({ width: 390, height: 844 })
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
    expect(errors).toEqual([])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

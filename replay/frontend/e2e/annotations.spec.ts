import { expect, test, type Page } from '@playwright/test'
import { cp, mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const demoRoot = fileURLToPath(new URL('../../demo_data/', import.meta.url))

async function mockCollection(page: Page) {
  await page.route('**/api/operations{,/**}', async (route) => {
    expect(route.request().method()).toBe('GET')
    await route.fulfill({
      json: {
        operation: {
          id: 'annotation-collection',
          kind: 'collect',
          state: 'running',
          dataset_id: null,
          episode_index: null,
          started_at: new Date().toISOString(),
          finished_at: null,
          return_code: null,
          logs: ['模拟采集；未连接机器人'],
        },
      },
    })
  })
}

test('a save opens the new dataset and both cameras, then labels persist after reload', async ({
  page,
}) => {
  const name = 'test_annotation_flow'
  const root = join(demoRoot, name)
  await rm(root, { recursive: true, force: true })
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await mockCollection(page)
  try {
    const baseline = page.waitForResponse('**/api/datasets/latest-episode')
    await page.goto('/replay?dataset=demo_v21&episode=0')
    await baseline
    await expect(page.getByTestId('episode-annotation')).toContainText('当前结果：未标注')
    await cp(join(demoRoot, 'demo_v21'), root, { recursive: true })
    const sync = join(root, 'episode_000002.sync.json')
    const payload = { episode_index: 2, success: null, saved_at_ns: Date.now() * 1_000_000 }
    await writeFile(`${sync}.tmp`, JSON.stringify(payload))
    await rename(`${sync}.tmp`, sync)

    await expect(page.getByTestId('dataset-select')).toHaveValue(name)
    await expect(page.getByTestId('episode-select')).toHaveValue('2')
    await expect(page.getByTestId('visualization-exterior_image_1_left')).toBeVisible()
    await expect(page.getByTestId('visualization-wrist_image_left')).toBeVisible()
    await expect(page.getByTestId('visualization-ee_pose')).toBeVisible()
    await expect(page.getByTestId('operation-status')).toContainText('VR 采集 · 运行中')
    const video = page.getByTestId('visualization-exterior_image_1_left').locator('video')
    await expect
      .poll(() => video.evaluate((element: HTMLVideoElement) => element.readyState))
      .toBeGreaterThanOrEqual(2)
    await page.getByTestId('play-toggle').click()
    await expect
      .poll(async () => Number(await page.getByTestId('timeline-slider').inputValue()))
      .toBeGreaterThan(0.2)
    await page.getByTestId('play-toggle').click()

    const annotation = page.getByTestId('episode-annotation')
    await annotation.getByRole('button', { name: '失败', exact: true }).click()
    await expect(annotation).toContainText('当前结果：失败 · 已保存')
    expect(JSON.parse(await readFile(sync, 'utf8')).success).toBe(false)
    await page.reload()
    await expect(annotation.getByRole('button', { name: '失败', exact: true })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    await annotation.getByRole('button', { name: '成功', exact: true }).click()
    await expect(annotation).toContainText('当前结果：成功 · 已保存')
    expect(JSON.parse(await readFile(sync, 'utf8')).success).toBe(true)
    await annotation.getByRole('button', { name: '未标注', exact: true }).click()
    await expect(annotation).toContainText('当前结果：未标注 · 已保存')
    expect(JSON.parse(await readFile(sync, 'utf8')).success).toBe(null)
    // Updating annotations must not look like a new collection save.
    await page.getByTestId('episode-select').selectOption('0')
    await page.waitForTimeout(1800)
    await expect(page.getByTestId('episode-select')).toHaveValue('0')
    await page.setViewportSize({ width: 390, height: 844 })
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
    expect(errors).toEqual([])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('automatic opening can be disabled and failed annotation requests can be retried', async ({
  page,
}) => {
  const name = 'test_annotation_manual'
  const root = join(demoRoot, name)
  await rm(root, { recursive: true, force: true })
  await cp(join(demoRoot, 'demo_v21'), root, { recursive: true })
  await mockCollection(page)
  try {
    const baseline = page.waitForResponse('**/api/datasets/latest-episode')
    await page.goto(`/replay?dataset=${name}&episode=0`)
    await baseline
    await expect(page.getByTestId('episode-annotation')).toBeVisible()
    await page.getByRole('checkbox', { name: '自动打开新保存片段' }).uncheck()
    await mkdir(join(root, 'audio'), { recursive: true })
    const sync = join(root, 'audio', 'episode_000002.sync.json')
    await writeFile(
      sync,
      JSON.stringify({ episode_index: 2, success: null, saved_at_ns: Date.now() * 1_000_000 }),
    )
    await page.waitForTimeout(1800)
    await expect(page.getByTestId('episode-select')).toHaveValue('0')
    await page.getByTestId('episode-select').selectOption('2')
    const annotation = page.getByTestId('episode-annotation')
    await expect(annotation).toContainText('Episode 002')
    let reject = true
    await page.route(`**/api/datasets/${name}/episodes/2/annotation`, async (route) => {
      if (reject) {
        reject = false
        await route.fulfill({ status: 503, json: { detail: '测试保存失败，请重试' } })
      } else await route.continue()
    })
    await annotation.getByRole('button', { name: '成功', exact: true }).click()
    await expect(annotation.getByRole('alert')).toContainText('测试保存失败')
    await expect(annotation).toContainText('当前结果：未标注')
    expect(JSON.parse(await readFile(sync, 'utf8')).success).toBe(null)
    await annotation.getByRole('button', { name: '成功', exact: true }).click()
    await expect(annotation).toContainText('当前结果：成功 · 已保存')
    expect(JSON.parse(await readFile(sync, 'utf8')).success).toBe(true)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

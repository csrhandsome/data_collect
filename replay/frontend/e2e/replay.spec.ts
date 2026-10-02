import { expect, test, type Page } from '@playwright/test'

const runtimeErrors = new WeakMap<Page, string[]>()

test.beforeEach(async ({ page }) => {
  const errors: string[] = []
  runtimeErrors.set(page, errors)
  page.on('pageerror', (error) => errors.push(error.message))
})

test.afterEach(async ({ page }) => {
  expect(runtimeErrors.get(page)).toEqual([])
})

async function selectEpisode(page: Page, dataset: string, episode = 0) {
  await page.goto('/')
  await expect(page.getByTestId('dataset-select')).toBeEnabled()
  await page.getByTestId('dataset-select').selectOption(dataset)
  await expect(page.getByTestId('episode-select')).toBeEnabled()
  await page.getByTestId('episode-select').selectOption(String(episode))
  await expect(page.getByTestId('data-block-ee_pose')).toBeVisible()
}

test('drag video and EE blocks into the workspace and synchronize playback', async ({ page }) => {
  const consoleErrors: string[] = []
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text())
  })
  await selectEpisode(page, 'demo_v21')
  const panel = page.getByTestId('reference-panel')
  await page.getByTestId('data-block-exterior_image_1_left').dragTo(panel)
  const videoCard = page.getByTestId('visualization-exterior_image_1_left')
  await expect(videoCard).toBeVisible()
  await page.getByTestId('data-block-ee_pose').dragTo(panel)
  const eeCard = page.getByTestId('visualization-ee_pose')
  await expect(eeCard).toBeVisible()
  await expect(eeCard.getByRole('img', { name: /随时间/ })).toBeVisible()
  const initialEE = await page.getByTestId('ee-values-ee_pose').innerText()
  const video = videoCard.locator('video')
  await expect
    .poll(() => video.evaluate((element: HTMLVideoElement) => element.readyState))
    .toBeGreaterThanOrEqual(2)
  expect(await video.evaluate((element: HTMLVideoElement) => element.videoWidth)).toBeGreaterThan(0)
  await page.getByTestId('timeline-slider').fill('2.5')
  await expect
    .poll(() => video.evaluate((element: HTMLVideoElement) => Math.abs(element.currentTime - 2.5)))
    .toBeLessThan(0.15)
  await expect(page.getByTestId('ee-values-ee_pose')).not.toHaveText(initialEE)
  await eeCard.getByRole('tab', { name: /空间轨迹/ }).click()
  await expect(eeCard.getByRole('img', { name: /等轴测/ })).toBeVisible()
  await page.getByTestId('play-toggle').click()
  await expect
    .poll(async () => Number(await page.getByTestId('timeline-slider').inputValue()))
    .toBeGreaterThan(2.7)
  await page.getByTestId('play-toggle').click()
  await expect(page.getByTestId('play-toggle')).toHaveAttribute('aria-label', '播放工作区')
  await expect(video).toHaveJSProperty('paused', true)
  expect(consoleErrors).toEqual([])
})

test('v3 episode 1 video uses a nonzero clip offset and stops at its boundary', async ({
  page,
  request,
}) => {
  await selectEpisode(page, 'demo_v30', 1)
  const response = await request.get(
    '/api/datasets/demo_v30/episodes/1/video?feature=exterior_image_1_left',
  )
  expect(response.ok()).toBeTruthy()
  const metadata: { start_time_s: number; end_time_s: number; duration_s: number } =
    await response.json()
  expect(metadata.start_time_s).toBeGreaterThan(0)
  await page.getByTestId('data-block-exterior_image_1_left').getByRole('button').click()
  const video = page.getByTestId('visualization-exterior_image_1_left').locator('video')
  await expect
    .poll(() => video.evaluate((element: HTMLVideoElement) => element.readyState))
    .toBeGreaterThanOrEqual(2)
  await expect
    .poll(() =>
      video.evaluate(
        (element: HTMLVideoElement, offset: number) => Math.abs(element.currentTime - offset),
        metadata.start_time_s,
      ),
    )
    .toBeLessThan(0.15)
  await expect(video).toHaveJSProperty('controls', false)
  await page.getByTestId('timeline-slider').fill('2.5')
  await expect
    .poll(() =>
      video.evaluate(
        (element: HTMLVideoElement, target: number) => Math.abs(element.currentTime - target),
        metadata.start_time_s + 2.5,
      ),
    )
    .toBeLessThan(0.15)
  await page.getByTestId('timeline-slider').fill(String(metadata.duration_s - 0.1))
  await page.getByTestId('play-toggle').click()
  await expect(page.getByTestId('play-toggle')).toHaveAttribute('aria-label', '播放工作区')
  expect(
    await video.evaluate((element: HTMLVideoElement) => element.currentTime),
  ).toBeLessThanOrEqual(metadata.end_time_s + 0.04)
  await expect(video).toHaveJSProperty('paused', true)
})

test('numeric and unsupported fields remain visible and selection changes clear stale cards', async ({
  page,
}) => {
  await selectEpisode(page, 'demo_v21')
  const joints = page.getByTestId('data-block-joint_position')
  await expect(joints).toBeVisible()
  await expect(joints.getByRole('button')).toBeEnabled()
  await joints.getByRole('button').click()
  await expect(page.getByTestId('visualization-joint_position')).toBeVisible()
  await expect(joints).toHaveAttribute('draggable', 'true')
  await page.getByTestId('data-block-ee_pose').getByRole('button').click()
  await expect(page.getByTestId('visualization-ee_pose')).toBeVisible()
  await page.getByTestId('episode-select').selectOption('1')
  await expect(page.getByTestId('visualization-ee_pose')).toHaveCount(0)
  await page.getByTestId('data-block-ee_pose').getByRole('button').click()
  await expect(page.getByTestId('visualization-ee_pose')).toBeVisible()
  await page.getByTestId('episode-select').selectOption('2')
  await expect(page.getByTestId('visualization-ee_pose')).toHaveCount(0)
  await page.getByTestId('episode-select').selectOption('1')
  await expect(page.getByTestId('visualization-ee_pose')).toHaveCount(0)
  await page.getByTestId('dataset-select').selectOption('demo_v30')
  await expect(page.getByTestId('visualization-ee_pose')).toHaveCount(0)
})

test('adding and removing views works on a narrow screen without horizontal overflow', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await selectEpisode(page, 'demo_v30')
  await page.getByTestId('data-block-ee_pose').getByRole('button').click()
  const card = page.getByTestId('visualization-ee_pose')
  await expect(card).toBeVisible()
  await expect(card.getByRole('img', { name: /随时间/ })).toBeVisible()
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1),
  ).toBeTruthy()
  await card.getByRole('button', { name: /移除/ }).click()
  await expect(card).toHaveCount(0)
})

test('API failure is shown and can recover through retry', async ({ page }) => {
  let unavailable = true
  await page.route('**/api/datasets', (route) =>
    unavailable
      ? route.fulfill({
          status: 503,
          contentType: 'application/json',
          body: JSON.stringify({ detail: '演示接口暂时不可用' }),
        })
      : route.continue(),
  )
  await page.goto('/')
  await expect(page.getByText('演示接口暂时不可用').first()).toBeVisible()
  unavailable = false
  await page.getByRole('button', { name: /重试/ }).first().click()
  await expect(page.getByTestId('dataset-select')).toBeEnabled()
  await expect(page.getByTestId('data-block-ee_pose')).toBeVisible()
})

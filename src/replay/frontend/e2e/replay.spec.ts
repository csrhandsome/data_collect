import { expect, test, type Page } from '@playwright/test'
import { createServer } from 'node:http'

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

async function expectTimelineFrozen(page: Page) {
  const times = await page.getByTestId('timeline-slider').evaluate(async (element) => {
    const samples: number[] = []
    for (let frame = 0; frame < 25; frame++) {
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()))
      samples.push(Number((element as HTMLInputElement).value))
    }
    return samples
  })
  expect(Math.max(...times) - Math.min(...times)).toBeLessThan(0.01)
}

test('adding a video during playback keeps existing media running and joins at the current time', async ({
  page,
  request,
}) => {
  await selectEpisode(page, 'demo_v30', 1)
  await page.getByTestId('data-block-exterior_image_1_left').getByRole('button').click()
  const first = page.getByTestId('video-exterior_image_1_left')
  await expect
    .poll(() => first.evaluate((v: HTMLVideoElement) => v.readyState))
    .toBeGreaterThanOrEqual(2)
  await page.getByTestId('timeline-slider').fill('1')
  await page.getByTestId('play-toggle').click()
  await expect
    .poll(async () => Number(await page.getByTestId('timeline-slider').inputValue()))
    .toBeGreaterThan(1.2)

  let releaseMetadata!: () => void
  let releaseMedia!: () => void
  const metadataGate = new Promise<void>((resolve) => {
    releaseMetadata = resolve
  })
  const mediaGate = new Promise<void>((resolve) => {
    releaseMedia = resolve
  })
  await page.route('**/video?feature=wrist_image_left', async (route) => {
    await metadataGate
    await route.continue()
  })
  await page.route('**/video/content?feature=wrist_image_left*', async (route) => {
    await mediaGate
    await route.continue()
  })
  try {
    await page
      .getByTestId('data-block-wrist_image_left')
      .dragTo(page.getByTestId('reference-panel'))
    await expect(page.getByText('正在准备相机视频，首次打开需稍等…')).toBeVisible()
    await expect(first).toHaveJSProperty('paused', false)
    const metadataTime = Number(await page.getByTestId('timeline-slider').inputValue())
    const firstBeforeMetadata = await first.evaluate((v: HTMLVideoElement) => v.currentTime)
    await expect
      .poll(async () => Number(await page.getByTestId('timeline-slider').inputValue()))
      .toBeGreaterThan(metadataTime + 0.4)
    expect(await first.evaluate((v: HTMLVideoElement) => v.currentTime)).toBeGreaterThan(
      firstBeforeMetadata + 0.3,
    )
    releaseMetadata()
    const second = page.getByTestId('video-wrist_image_left')
    await expect(second).toBeAttached()
    await expect(page.getByText('正在加载视频…')).toBeVisible()
    await expect(second).toHaveJSProperty('paused', true)
    const mediaTime = Number(await page.getByTestId('timeline-slider').inputValue())
    const firstBeforeMedia = await first.evaluate((v: HTMLVideoElement) => v.currentTime)
    await expect
      .poll(async () => Number(await page.getByTestId('timeline-slider').inputValue()))
      .toBeGreaterThan(mediaTime + 0.4)
    expect(await first.evaluate((v: HTMLVideoElement) => v.currentTime)).toBeGreaterThan(
      firstBeforeMedia + 0.3,
    )
    await expect(first).toHaveJSProperty('paused', false)
    const resumeTime = Number(await page.getByTestId('timeline-slider').inputValue())
    releaseMedia()
    await expect(first).toHaveJSProperty('paused', false)
    await expect(second).toHaveJSProperty('paused', false)
    await expect
      .poll(async () => Number(await page.getByTestId('timeline-slider').inputValue()))
      .toBeGreaterThan(resumeTime + 0.3)
    const response = await request.get(
      '/api/datasets/demo_v30/episodes/1/video?feature=wrist_image_left',
    )
    const metadata: { start_time_s: number } = await response.json()
    expect(metadata.start_time_s).toBeGreaterThan(0)
    const drift = await page.evaluate((offset) => {
      const videos = Array.from(document.querySelectorAll('video'))
      const timeline = Number(
        (document.querySelector('[data-testid="timeline-slider"]') as HTMLInputElement).value,
      )
      return Math.max(...videos.map((video) => Math.abs(video.currentTime - offset - timeline)))
    }, metadata.start_time_s)
    expect(drift).toBeLessThan(0.2)
    await page.getByTestId('play-toggle').click()
    await expect(second).toHaveJSProperty('paused', true)
  } finally {
    releaseMetadata()
    releaseMedia()
  }
})

test('a new video keeps its loading state until media beyond the MP4 metadata arrives', async ({
  page,
  request,
}) => {
  const response = await request.get(
    '/api/datasets/demo_v30/episodes/0/video?feature=wrist_image_left',
  )
  const metadata: { url: string } = await response.json()
  const media = await request.get(metadata.url)
  const bytes = await media.body()
  // Send the headers and initial frames, withholding the later playback position.
  let mediaStart = 0
  while (
    mediaStart + 8 <= bytes.length &&
    bytes.toString('ascii', mediaStart + 4, mediaStart + 8) !== 'mdat'
  ) {
    const size = bytes.readUInt32BE(mediaStart)
    expect(size).toBeGreaterThanOrEqual(8)
    mediaStart += size
  }
  expect(mediaStart).toBeLessThan(bytes.length - 8)
  const initialBytes = Math.max(mediaStart + 8, 64 * 1024)
  let release!: () => void
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  const server = createServer(async (incoming, outgoing) => {
    const range = /^bytes=(\d+)-(\d*)$/.exec(incoming.headers.range || '')
    const start = range ? Number(range[1]) : 0
    const end = range?.[2] ? Math.min(Number(range[2]), bytes.length - 1) : bytes.length - 1
    outgoing.writeHead(range ? 206 : 200, {
      'Content-Type': 'video/mp4',
      'Content-Length': end - start + 1,
      'Accept-Ranges': 'bytes',
      ...(range ? { 'Content-Range': `bytes ${start}-${end}/${bytes.length}` } : {}),
    })
    const split = Math.min(end + 1, Math.max(start, initialBytes))
    outgoing.write(bytes.subarray(start, split))
    await gate
    outgoing.end(bytes.subarray(split, end + 1))
  })
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  try {
    const address = server.address()
    if (!address || typeof address === 'string') throw new Error('Missing media server address')
    await page.route('**/video?feature=wrist_image_left', (route) =>
      route.fulfill({
        json: {
          ...metadata,
          url: `http://127.0.0.1:${address.port}/video.mp4`,
        },
      }),
    )
    await selectEpisode(page, 'demo_v30')
    await page.getByTestId('data-block-ee_pose').getByRole('button').click()
    await page.getByRole('combobox', { name: '播放速度' }).selectOption('0.25')
    await page.getByTestId('timeline-slider').fill('4')
    await page.getByTestId('play-toggle').click()
    await page.getByTestId('data-block-wrist_image_left').getByRole('button').click()
    const video = page.getByTestId('video-wrist_image_left')
    await expect.poll(() => video.evaluate((v: HTMLVideoElement) => v.readyState)).toBe(1)
    await expect(page.getByText('正在加载视频…')).toBeVisible()
    await expect(video).toHaveJSProperty('paused', true)
    const time = Number(await page.getByTestId('timeline-slider').inputValue())
    await expect
      .poll(async () => Number(await page.getByTestId('timeline-slider').inputValue()))
      .toBeGreaterThan(time + 0.3)
    release()
    await expect(video).toHaveJSProperty('paused', false)
    await expect(page.getByText('正在加载视频…')).toHaveCount(0)
    await expect
      .poll(async () => video.evaluate((v: HTMLVideoElement) => v.currentTime))
      .toBeGreaterThan(time + 0.5)
  } finally {
    release()
    server.closeAllConnections()
    await new Promise<void>((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    )
  }
})

for (const action of ['pause', 'remove'] as const) {
  test(`${action} while an added video loads preserves the playback intent`, async ({ page }) => {
    await selectEpisode(page, 'demo_v30')
    await page.getByTestId('data-block-ee_pose').getByRole('button').click()
    await page.getByTestId('play-toggle').click()
    await expect
      .poll(async () => Number(await page.getByTestId('timeline-slider').inputValue()))
      .toBeGreaterThan(0.2)
    let release!: () => void
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    await page.route('**/video?feature=wrist_image_left', async (route) => {
      await gate
      await route.continue()
    })
    try {
      await page.getByTestId('data-block-wrist_image_left').getByRole('button').click()
      await expect(page.getByText('正在准备相机视频，首次打开需稍等…')).toBeVisible()
      const loadingTime = Number(await page.getByTestId('timeline-slider').inputValue())
      await expect
        .poll(async () => Number(await page.getByTestId('timeline-slider').inputValue()))
        .toBeGreaterThan(loadingTime + 0.2)
      if (action === 'pause') {
        await page.getByTestId('play-toggle').click()
        release()
        const video = page.getByTestId('video-wrist_image_left')
        await expect
          .poll(() => video.evaluate((v: HTMLVideoElement) => v.readyState))
          .toBeGreaterThanOrEqual(2)
        await expect(page.getByTestId('play-toggle')).toHaveAttribute('aria-label', '播放工作区')
        await expect(video).toHaveJSProperty('paused', true)
        await expectTimelineFrozen(page)
      } else {
        const previousTime = Number(await page.getByTestId('timeline-slider').inputValue())
        await page
          .getByTestId('visualization-wrist_image_left')
          .getByRole('button', { name: /移除/ })
          .click()
        await expect
          .poll(async () => Number(await page.getByTestId('timeline-slider').inputValue()))
          .toBeGreaterThan(previousTime + 0.2)
        release()
        // Re-adding a removed loading video should join ongoing playback.
        await page.getByTestId('data-block-wrist_image_left').getByRole('button').click()
        await expect(page.getByTestId('video-wrist_image_left')).toHaveJSProperty('paused', false)
      }
    } finally {
      release()
    }
  })
}

for (const failure of ['metadata', 'media'] as const) {
  test(`video ${failure} failure leaves playback running and recovery joins at the current time`, async ({
    page,
  }) => {
    await selectEpisode(page, 'demo_v30')
    await page.getByTestId('data-block-ee_pose').getByRole('button').click()
    await page.getByTestId('play-toggle').click()
    let failed = true
    await page.route(
      failure === 'metadata'
        ? '**/video?feature=wrist_image_left'
        : '**/video/content?feature=wrist_image_left*',
      (route) =>
        failed
          ? route.fulfill(
              failure === 'metadata'
                ? {
                    status: 503,
                    contentType: 'application/json',
                    body: JSON.stringify({ detail: '相机暂时不可用' }),
                  }
                : { status: 200, contentType: 'video/mp4', body: 'invalid MP4' },
            )
          : route.continue(),
    )
    await page.getByTestId('data-block-wrist_image_left').getByRole('button').click()
    const card = page.getByTestId('visualization-wrist_image_left')
    await expect(card.getByRole('alert')).toBeVisible()
    const time = Number(await page.getByTestId('timeline-slider').inputValue())
    await expect
      .poll(async () => Number(await page.getByTestId('timeline-slider').inputValue()))
      .toBeGreaterThan(time + 0.2)
    failed = false
    if (failure === 'metadata') {
      await card.getByRole('button', { name: '重试加载' }).click()
    } else {
      await card.getByRole('button', { name: /移除/ }).click()
      await page.getByTestId('data-block-wrist_image_left').getByRole('button').click()
    }
    await expect(page.getByTestId('video-wrist_image_left')).toHaveJSProperty('paused', false)
    await expect(card.getByRole('alert')).toHaveCount(0)
  })
}

test('drag video and EE blocks into the workspace and synchronize playback', async ({ page }) => {
  const consoleErrors: string[] = []
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text())
  })
  await selectEpisode(page, 'demo_v30')
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
  await selectEpisode(page, 'demo_v30')
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

test('rescanning recovers when the selected v3 dataset reappears', async ({ page }) => {
  let available = true
  await page.route('**/api/datasets', async (route) => {
    const response = await route.fetch()
    const body = await response.json()
    await route.fulfill({ response, json: { datasets: available ? body.datasets : [] } })
  })
  await page.goto('/')
  const select = page.getByTestId('dataset-select')
  await expect(select).toHaveValue('demo_v30')
  await page.getByTestId('data-block-ee_pose').getByRole('button').click()
  await expect(page.getByTestId('visualization-ee_pose')).toBeVisible()
  available = false
  await page.getByTestId('scan-datasets').click()
  await expect(select.locator('option[value="demo_v30"]')).toHaveCount(0)
  await expect(page.getByTestId('visualization-ee_pose')).toHaveCount(0)
  available = true
  await page.getByTestId('scan-datasets').click()
  await expect(select).toHaveValue('demo_v30')
  await expect(page.getByTestId('data-block-ee_pose')).toBeVisible()
})

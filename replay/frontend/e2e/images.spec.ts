import { expect, test } from '@playwright/test'

test('embedded PNG/JPEG cameras decode as MP4 and play without frame requests or repeated seeks', async ({
  page,
}) => {
  const errors: string[] = []
  const images: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  page.on('request', (request) => {
    if (/\/image[/?]/.test(request.url())) images.push(request.url())
  })
  await page.goto('/?dataset=demo_z_images&episode=0')
  for (const camera of ['camera_png', 'camera_jpeg']) {
    await page.getByTestId(`data-block-${camera}`).getByRole('button').click()
    await expect
      .poll(() =>
        page.getByTestId(`video-${camera}`).evaluate((v: HTMLVideoElement) => v.readyState),
      )
      .toBeGreaterThanOrEqual(3)
  }
  const video = page.getByTestId('video-camera_png')
  await video.evaluate((v: HTMLVideoElement) => {
    const measured = v as HTMLVideoElement & { seeks: number; presented: number }
    measured.seeks = 0
    measured.presented = 0
    v.addEventListener('seeking', () => measured.seeks++)
    const frame = () => {
      measured.presented++
      v.requestVideoFrameCallback(frame)
    }
    v.requestVideoFrameCallback(frame)
  })
  await page.getByTestId('play-toggle').click()
  await page.waitForTimeout(3000)
  const metrics = await video.evaluate((v: HTMLVideoElement) => {
    const m = v as HTMLVideoElement & { seeks: number; presented: number }
    return {
      seeks: m.seeks,
      presented: m.presented,
      quality: {
        totalVideoFrames: v.getVideoPlaybackQuality().totalVideoFrames,
        droppedVideoFrames: v.getVideoPlaybackQuality().droppedVideoFrames,
      },
      time: v.currentTime,
    }
  })
  expect(metrics.seeks).toBeLessThanOrEqual(1)
  expect(metrics.presented).toBeGreaterThan(65)
  expect(metrics.time).toBeGreaterThan(2.5)
  expect(metrics.quality.droppedVideoFrames / metrics.quality.totalVideoFrames).toBeLessThan(0.1)
  await page.getByTestId('timeline-slider').fill('5')
  await expect
    .poll(() => video.evaluate((v: HTMLVideoElement) => v.currentTime))
    .toBeGreaterThan(4.8)
  await page.getByTestId('play-toggle').click()
  await expect(video).toHaveJSProperty('paused', true)
  await page.getByTestId('timeline-slider').fill('1')
  await expect
    .poll(() => video.evaluate((v: HTMLVideoElement) => Math.abs(v.currentTime - 1)))
    .toBeLessThan(0.05)
  await expect(video).toHaveJSProperty('videoWidth', 320)
  await page.getByRole('combobox', { name: '播放速度' }).selectOption('2')
  await page.getByTestId('play-toggle').click()
  await expect(video).toHaveJSProperty('playbackRate', 2)
  await page.getByTestId('timeline-slider').fill('7.8')
  await expect(page.getByTestId('play-toggle')).toHaveAttribute('aria-label', '播放工作区')
  await expect(video).toHaveJSProperty('paused', true)
  expect(images).toEqual([])
  expect(errors).toEqual([])
})

test('MP4 preparation failure can be retried', async ({ page }) => {
  let unavailable = true
  await page.route('**/episodes/0/video?feature=camera_png', (route) =>
    unavailable
      ? route.fulfill({ status: 503, json: { detail: '视频转换暂时失败' } })
      : route.continue(),
  )
  await page.goto('/?dataset=demo_z_images&episode=0')
  await page.getByTestId('data-block-camera_png').getByRole('button').click()
  await expect(page.getByRole('alert')).toContainText('视频转换暂时失败')
  unavailable = false
  await page.getByRole('button', { name: '重试加载' }).click()
  await expect
    .poll(() =>
      page.getByTestId('video-camera_png').evaluate((v: HTMLVideoElement) => v.readyState),
    )
    .toBeGreaterThanOrEqual(2)
  await expect(page.getByRole('alert')).toHaveCount(0)
})

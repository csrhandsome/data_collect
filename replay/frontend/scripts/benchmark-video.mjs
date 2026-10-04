/** Measure real composited frames, decode drops, stalls, seeks and HTTP requests. */
import { chromium } from '@playwright/test'
import { parseArgs } from 'node:util'
import { mkdir, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

const { values } = parseArgs({
  options: {
    url: { type: 'string', default: 'http://127.0.0.1:5173' },
    dataset: { type: 'string' },
    episode: { type: 'string', default: '0' },
    seconds: { type: 'string', default: '8' },
    cameras: {
      type: 'string',
      default: 'exterior_image_1_left,exterior_image_2_left,wrist_image_left',
    },
    'latency-ms': { type: 'string', default: '0' },
    output: { type: 'string' },
  },
})
if (!values.dataset) throw new Error('Required: --dataset DATASET_ID')
const seconds = Number(values.seconds)
const latency = Number(values['latency-ms'])
if (!(seconds > 1 && seconds <= 60) || !(latency >= 0))
  throw new Error('Invalid duration or latency')
const browser = await chromium.launch({
  channel: process.env.REPLAY_BROWSER_CHANNEL || 'chrome',
  args: [
    '--no-sandbox',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
  ],
})
try {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1600 } })
  const errors = []
  const requests = []
  page.on('pageerror', (error) => errors.push(error.message))
  page.on('request', (request) => {
    if (/\/video\/content|\/image[/?]/.test(request.url())) requests.push(request.url())
  })
  if (latency) {
    const session = await page.context().newCDPSession(page)
    await session.send('Network.enable')
    await session.send('Network.emulateNetworkConditions', {
      offline: false,
      latency,
      downloadThroughput: 50 * 1024 * 1024,
      uploadThroughput: 50 * 1024 * 1024,
    })
  }
  const url = new URL(values.url)
  url.searchParams.set('dataset', values.dataset)
  url.searchParams.set('episode', values.episode)
  await page.goto(url.href)
  const cameras = values.cameras.split(',')
  const preparations = []
  for (const camera of cameras) {
    const start = performance.now()
    await page.getByTestId(`data-block-${camera}`).getByRole('button').click()
    await page.waitForFunction(
      (key) => {
        const video = document.querySelector(`[data-testid="video-${key}"]`)
        return video?.readyState >= 3
      },
      camera,
      { timeout: 120_000 },
    )
    const readyMs = performance.now() - start
    const metadataUrl = new URL(
      `/api/datasets/${encodeURIComponent(values.dataset)}/episodes/${values.episode}/video`,
      values.url,
    )
    metadataUrl.searchParams.set('feature', camera)
    const metadata = await (await page.request.get(metadataUrl.href)).json()
    preparations.push({ camera, readyMs, expectedFps: metadata.fps })
  }
  const max = Number(await page.getByTestId('timeline-slider').getAttribute('max'))
  if (seconds + 1 > max)
    throw new Error(`Choose a clip longer than ${seconds + 1}s (clip: ${max}s)`)
  await page.evaluate(() => {
    window.__videoBenchmark = Array.from(document.querySelectorAll('article video')).map(
      (video) => {
        const measurement = {
          key: video.dataset.testid,
          times: [],
          firstPresented: null,
          lastPresented: 0,
          seeks: 0,
          waits: 0,
          quality: video.getVideoPlaybackQuality(),
          startTime: video.currentTime,
          video,
        }
        video.addEventListener('seeking', () => measurement.seeks++)
        video.addEventListener('waiting', () => measurement.waits++)
        const frame = (now, metadata) => {
          measurement.firstPresented ??= metadata.presentedFrames
          measurement.lastPresented = metadata.presentedFrames
          measurement.times.push(now)
          video.requestVideoFrameCallback(frame)
        }
        video.requestVideoFrameCallback(frame)
        return measurement
      },
    )
  })
  const start = performance.now()
  await page.getByTestId('play-toggle').click()
  await page.waitForTimeout(seconds * 1000)
  const metrics = await page.evaluate(() => {
    const time = Number(document.querySelector('[data-testid="timeline-slider"]').value)
    return window.__videoBenchmark.map((m) => {
      const quality = m.video.getVideoPlaybackQuality()
      const times = m.times
      const gaps = times
        .slice(1)
        .map((t, i) => t - times[i])
        .sort((a, b) => a - b)
      return {
        camera: m.key,
        presentedFrames: m.firstPresented === null ? 0 : m.lastPresented - m.firstPresented + 1,
        frameCallbacks: times.length,
        decodedFrames: quality.totalVideoFrames - m.quality.totalVideoFrames,
        droppedFrames: quality.droppedVideoFrames - m.quality.droppedVideoFrames,
        seeks: m.seeks,
        waits: m.waits,
        maxCallbackGapMs: gaps.at(-1) || 0,
        p95CallbackGapMs: gaps[Math.floor(gaps.length * 0.95)] || 0,
        mediaElapsedS: m.video.currentTime - m.startTime,
        timelineS: time,
        width: m.video.videoWidth,
        height: m.video.videoHeight,
      }
    })
  })
  const elapsedS = (performance.now() - start) / 1000
  await page.getByTestId('play-toggle').click()
  const result = {
    dataset: values.dataset,
    episode: Number(values.episode),
    latencyMs: latency,
    elapsedS,
    preparations,
    cameras: metrics.map((m) => ({
      ...m,
      presentedFps: m.presentedFrames / elapsedS,
      dropRatio: m.decodedFrames ? m.droppedFrames / m.decodedFrames : 0,
    })),
    videoRequests: requests.filter((url) => url.includes('/video/content')).length,
    imageRequests: requests.filter((url) => /\/image[/?]/.test(url)).length,
    errors,
  }
  const json = JSON.stringify(result, null, 2)
  console.log(json)
  if (values.output) {
    await mkdir(dirname(values.output), { recursive: true })
    await writeFile(values.output, json + '\n')
  }
  if (
    errors.length ||
    result.imageRequests ||
    metrics.some(
      (m, i) =>
        m.presentedFrames / elapsedS < preparations[i].expectedFps * 0.85 ||
        m.droppedFrames / Math.max(1, m.decodedFrames) > 0.05 ||
        m.seeks > 1 ||
        m.waits > 0,
    )
  )
    process.exitCode = 1
} finally {
  await browser.close()
}

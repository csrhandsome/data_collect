import { expect, test } from '@playwright/test'
import { execFile } from 'node:child_process'
import { rm } from 'node:fs/promises'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { fileURLToPath } from 'node:url'

const execute = promisify(execFile)
const repoRoot = fileURLToPath(new URL('../../../../', import.meta.url))
const dataRoot = join(repoRoot, 'src/replay/demo_data')
const datasetId = 'test_compaction_live'
const root = join(dataRoot, datasetId)

async function python(code: string) {
  return execute(join(repoRoot, '.venv/bin/python'), ['-c', code], {
    cwd: repoRoot,
    env: {
      ...process.env,
      HF_DATASETS_CACHE: '/tmp/data-collect-e2e-hf-cache',
      HF_HUB_OFFLINE: '1',
    },
  })
}

async function appendEpisodes(count: number) {
  await python(`
from dataclasses import replace
import numpy as np
from control.collection.dataset import open_dataset
from control.collection.recording import EpisodeRecorder
from control.collection.devices import CameraPair
from control.direct.fake import FakeBackend
config = {"dataset": {"repo_id": "local/test_compaction", "date": "live", "root": ${JSON.stringify(dataRoot)}, "instruction": "Move", "action_space": "ee"}, "camera": {"image_hw": 32, "fps": 30}, "control": {"frequency_hz": 100}}
dataset, root = open_dataset(config)
recorder = EpisodeRecorder(dataset, root, config)
base = FakeBackend().snapshot()
first = dataset.meta.total_episodes
try:
    for episode in range(first, first + ${count}):
        recorder.start()
        for frame in range(30):
            ns = base.sampled_monotonic_ns + (episode * 30 + frame) * 40_000_000
            state = replace(base, sampled_monotonic_ns=ns)
            image = np.full((32, 32, 3), 50 + 40 * episode, dtype=np.uint8)
            recorder.frame(state, CameraPair(image, image, ns, ns, ns / 1e6, ns / 1e6))
        recorder.finish(state, publish=True)
finally:
    recorder.close()
`)
}

async function compact() {
  await python(
    `from pathlib import Path\nfrom data_analysis.compact_v3 import compact_dataset\nassert compact_dataset(Path(${JSON.stringify(root)}))`,
  )
}

test('repeated appends to shared MP4 remain seekable, isolated and annotated in the browser', async ({
  page,
  request,
}) => {
  test.setTimeout(120_000)
  await rm(root, { recursive: true, force: true })
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  try {
    await appendEpisodes(2)
    await page.goto(`/replay?dataset=${datasetId}&episode=1`)
    await expect(page.getByTestId('episode-select')).toHaveValue('1')
    await page.getByTestId('data-block-observation.exterior_image').getByRole('button').click()
    const video = page.getByTestId('video-observation.exterior_image')
    await expect
      .poll(() => video.evaluate((el: HTMLVideoElement) => el.readyState))
      .toBeGreaterThanOrEqual(2)
    await page.getByTestId('timeline-slider').fill('0.3')
    const annotation = page.getByTestId('episode-annotation')
    await annotation.getByRole('button', { name: '失败', exact: true }).click()
    await expect(annotation).toContainText('当前结果：失败 · 已保存')
    const endpoint = `/api/datasets/${datasetId}/episodes/1/video?feature=observation.exterior_image`
    const old = await (await request.get(endpoint)).json()
    expect(old.start_time_s).toBe(0)
    await compact()
    expect((await request.get(old.url)).status()).toBe(409)

    // Simulate a metadata response crossing the directory exchange. The player
    // must recover its new URL/offset automatically when the old range fails.
    await page.route(`**${endpoint}`, (route) => route.fulfill({ json: old }), { times: 1 })
    await page.getByRole('button', { name: /重新扫描/ }).click()
    const metadata = await (await request.get(endpoint)).json()
    expect(metadata.start_time_s).toBeGreaterThan(0)
    await expect(video).toHaveAttribute('src', metadata.url)
    await expect
      .poll(() => video.evaluate((el: HTMLVideoElement) => el.readyState))
      .toBeGreaterThanOrEqual(2)
    await expect
      .poll(() => video.evaluate((el: HTMLVideoElement) => el.currentTime))
      .toBeCloseTo(metadata.start_time_s + 0.3, 1)
    await expect(annotation).toContainText('当前结果：失败')
    await page.getByTestId('play-toggle').click()
    await expect
      .poll(async () => Number(await page.getByTestId('timeline-slider').inputValue()))
      .toBeGreaterThan(0.5)
    await page.getByTestId('play-toggle').click()
    // Three fresh writers append to the same previously compacted dataset.
    for (const index of [2, 3, 4]) {
      await appendEpisodes(1)
      await compact()
      await page.getByRole('button', { name: /重新扫描/ }).click()
      await expect(
        page.getByTestId('episode-select').locator(`option[value="${index}"]`),
      ).toHaveCount(1)
    }
    // Check every old and new clip after all append/compact rounds, through
    // the real browser media pipeline, with seeks relative to that episode.
    for (const index of [0, 1, 2, 3, 4]) {
      await page.getByTestId('episode-select').selectOption(String(index))
      await page.getByTestId('data-block-observation.exterior_image').getByRole('button').click()
      const clip = await (
        await request.get(endpoint.replace('/episodes/1/', `/episodes/${index}/`))
      ).json()
      expect(clip.start_time_s).toBeCloseTo(index, 3)
      expect(clip.duration_s).toBeCloseTo(1, 3)
      await expect(video).toHaveAttribute('src', clip.url)
      await expect
        .poll(() => video.evaluate((el: HTMLVideoElement) => el.readyState))
        .toBeGreaterThanOrEqual(2)
      await page.getByTestId('timeline-slider').fill('0.3')
      await expect
        .poll(() => video.evaluate((el: HTMLVideoElement) => el.currentTime))
        .toBeCloseTo(index + 0.3, 1)
      // Read rendered pixels so a correct timestamp cannot hide a wrong clip.
      const brightness = await video.evaluate((el: HTMLVideoElement) => {
        const canvas = document.createElement('canvas')
        canvas.width = canvas.height = 32
        const context = canvas.getContext('2d')!
        context.drawImage(el, 0, 0, 32, 32)
        return context.getImageData(16, 16, 1, 1).data[0]
      })
      expect(Math.abs(brightness - (50 + 40 * index))).toBeLessThanOrEqual(3)
      if (index === 1) await expect(annotation).toContainText('当前结果：失败')
    }
    expect(errors).toEqual([])
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

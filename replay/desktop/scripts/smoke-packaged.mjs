import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { access, mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { _electron as electron, expect } from '@playwright/test'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../../..')
let unpacked = process.env.DATA_COLLECT_PACKAGE_DIR || join(repo, 'dist/electron-cpu/linux-unpacked')
let executable = process.env.DATA_COLLECT_ELECTRON_EXECUTABLE || join(unpacked, 'panda-data-workbench')
const results = process.env.DATA_COLLECT_TEST_RESULTS || join(repo, 'replay/desktop/test-results/cpu')
const temporary = await mkdtemp(join(tmpdir(), 'panda-desktop-smoke-'))
const appImage = executable.endsWith('.AppImage') ? executable : null
if (appImage) {
  const extractDir = join(temporary, 'extracted')
  await mkdir(extractDir)
  const extracted = spawnSync(appImage, ['--appimage-extract'], {
    cwd: extractDir, env: process.env, stdio: ['ignore', 'ignore', 'pipe'],
    encoding: 'utf8', timeout: 180_000,
  })
  assert.equal(extracted.status, 0, `${extracted.stderr}\n${extracted.error || ''}`)
  unpacked = join(extractDir, 'squashfs-root')
  executable = join(unpacked, 'panda-data-workbench')
}
const launcher = join(unpacked, 'resources/runtime/bin/data-collect-runtime')
const datasets = join(temporary, 'datasets')
const config = join(temporary, 'panda.yaml')
const userData = join(temporary, 'user-data')
const readyFile = join(temporary, 'ready.json')
await mkdir(results, { recursive: true })
await access(executable)
const env = {
  ...process.env,
  DATA_COLLECT_USER_DATA: userData,
  DATA_COLLECT_CONFIG: config,
  DATA_COLLECT_SIMULATE: '1',
  REPLAY_DATA_ROOT: datasets,
  DATA_COLLECT_READY_FILE: readyFile,
  HF_HOME: join(temporary, 'huggingface'),
  XDG_DATA_HOME: join(temporary, 'xdg'),
  XDG_DATA_DIRS: '/usr/share',
  PYTHONPATH: '/does-not-exist/poisoned-source-path',
  PYTHONHOME: '/does-not-exist/poisoned-python-home',
}
function runtime(args) {
  const result = spawnSync(launcher, args, { cwd: temporary, env, encoding: 'utf8', timeout: 120_000, maxBuffer: 8 * 1024 * 1024 })
  if (result.status !== 0) throw new Error(`Runtime failed: ${args[0]}\n${result.stdout}\n${result.stderr}\n${result.error || ''}`)
  return result.stdout
}
const checks = []
function passed(name) { checks.push(name); console.log(`PASS ${name}`) }
if (appImage) passed('final AppImage extracts successfully; all following tests use its contents')
runtime(['doctor'])
passed('native imports, compiled modules, bundled assets and spawned workers')
if (process.env.DATA_COLLECT_EXPECT_CPU !== '0') {
  runtime(['-c', `
import importlib.metadata
import numpy as np
import torch, torchvision, torchaudio
assert torch.__version__.endswith('+cpu'), torch.__version__
assert torch.version.cuda is None
packages = {d.metadata['Name'].lower().replace('_', '-') for d in importlib.metadata.distributions()}
assert not any(name.startswith('nvidia-') or name in {'triton', 'cuda-bindings', 'nuitka', 'ruff'} for name in packages), packages
from data_analysis.preprocess_vad import _load_silero, _detect_segments
model, timestamp_fn, info = _load_silero()
segments, sample_rate = _detect_segments(
    audio=np.zeros(16000, dtype=np.float32), sample_rate=16000,
    model=model, timestamp_fn=timestamp_fn, threshold=0.5,
    min_speech_duration_ms=250, max_speech_duration_s=30,
    min_silence_duration_ms=100, speech_pad_ms=30, neg_threshold=None,
)
assert segments == [], segments
assert sample_rate == 16000
print('CPU VAD OK', torch.__version__, info)
`])
  passed('CPU PyTorch/audio/vision, no GPU/Nuitka/Ruff dependencies, compiled Silero inference')
}
for (const source of ['control/config.py', 'replay/backend/main.py', 'vr_collect.py', 'data_analysis/preprocess_vad.py', 'data_analysis/instruction_audio_window.py', 'data_analysis/dataset_io.py']) {
  await assert.rejects(access(join(unpacked, 'resources/runtime', source)))
  await assert.rejects(access(join(unpacked, 'resources/runtime/lib/python3.12/site-packages', source)))
}
passed('distribution contains no original project business source')
runtime(['generate-demo', datasets])
runtime(['-c', `
import yaml
from pathlib import Path
c = yaml.safe_load(Path(${JSON.stringify(join(unpacked, 'resources/config/train/panda.yaml'))}).read_text())
assert 'password' not in c['robot'] and 'username' not in c['robot']
c['dataset'].update(root=${JSON.stringify(datasets)}, repo_id='local/desktop', date='smoke')
c['audio'].update(enabled=False, vad_enabled=False)
c['robot']['move_to_start'] = False
c['reactive_desk']['reactive_desk_enabled'] = False
c['camera']['image_hw'] = 64
Path(${JSON.stringify(config)}).write_text(yaml.safe_dump(c))
`])
passed('distributed template has no robot credentials')

let application
let page
let backendPid
const errors = []
try {
  application = await electron.launch({
    executablePath: executable, args: ['--no-sandbox', '--disable-gpu'],
    cwd: temporary, env, timeout: 120_000,
  })
  page = await application.firstWindow({ timeout: 120_000 })
  page.on('pageerror', (error) => errors.push(error.message))
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()) })
  await expect(page.getByTestId('dataset-select')).toBeEnabled({ timeout: 30_000 })
  await expect.poll(async () => { try { return JSON.parse(await readFile(readyFile, 'utf8')) } catch { return null } }).not.toBeNull()
  const { origin, backendPid: pid } = JSON.parse(await readFile(readyFile, 'utf8'))
  backendPid = pid
  passed('packaged Electron starts from a directory outside the repository')
  if (process.env.DATA_COLLECT_BROWSER_CHECK === '1') {
    const browser = join(repo, 'replay/desktop/node_modules/.bin/agent-browser')
    const session = `panda-package-${Date.now()}`
    function browserCommand(args) {
      const result = spawnSync(browser, ['--session', session, ...args], { env, encoding: 'utf8', timeout: 30_000 })
      assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`)
      return result.stdout
    }
    try {
      browserCommand(['open', origin])
      browserCommand(['wait', '--load', 'networkidle'])
      const snapshot = browserCommand(['snapshot', '-i'])
      assert.ok(snapshot.includes('开启采集') && snapshot.includes('重新扫描'))
      assert.ok(browserCommand(['eval', 'document.body.innerText.trim().length > 0']).includes('true'))
      assert.ok(browserCommand(['eval', 'document.querySelector(".vite-error-overlay") === null']).includes('true'))
      browserCommand(['screenshot', join(results, 'browser-workbench.png')])
      passed('agent-browser verifies page content, controls and absence of an error overlay')
    } finally { browserCommand(['close']) }
  }
  assert.equal(await page.evaluate(() => typeof window.require), 'undefined')
  assert.equal(await page.evaluate(() => typeof window.process), 'undefined')
  passed('renderer has no Node.js access')

  async function request(path, method = 'GET', body) {
    const response = await fetch(`${origin}${path}`, { method, headers: { 'Content-Type': 'application/json' }, ...(body ? { body: JSON.stringify(body) } : {}) })
    assert.ok(response.ok, `${method} ${path}: ${response.status} ${await response.clone().text()}`)
    return response.json()
  }
  async function finished(id) {
    let last
    await expect.poll(async () => {
      last = (await request('/api/operations')).operation
      return last?.id === id && !['running', 'stopping'].includes(last.state)
    }, { timeout: 60_000 }).toBe(true)
    assert.equal(last.state, 'succeeded', JSON.stringify(last))
    return last
  }
  for (const dataset of ['demo_v21', 'demo_v30']) {
    await page.getByTestId('dataset-select').selectOption(dataset)
    await expect(page.getByTestId('episode-select')).toBeEnabled()
    await page.getByTestId('episode-select').selectOption(dataset === 'demo_v30' ? '1' : '0')
    await expect(page.getByTestId('data-block-ee_pose')).toBeVisible()
    await page.getByTestId('data-block-exterior_image_1_left').getByRole('button').click()
    await page.getByTestId('data-block-ee_pose').getByRole('button').click()
    const video = page.getByTestId('visualization-exterior_image_1_left').locator('video')
    await expect.poll(() => video.evaluate((v) => v.readyState), { timeout: 30_000 }).toBeGreaterThanOrEqual(2)
    assert.ok(await video.evaluate((v) => v.videoWidth > 0))
    const meta = await request(`/api/datasets/${dataset}/episodes/${dataset === 'demo_v30' ? 1 : 0}/video?feature=exterior_image_1_left`)
    await page.getByTestId('timeline-slider').fill('2.5')
    await expect.poll(() => video.evaluate((v, target) => Math.abs(v.currentTime - target), meta.start_time_s + 2.5)).toBeLessThan(0.15)
    await page.getByTestId('play-toggle').click()
    await expect.poll(async () => Number(await page.getByTestId('timeline-slider').inputValue())).toBeGreaterThan(2.7)
    await page.getByTestId('play-toggle').click()
    passed(`${dataset}: video decoding, seek, playback and EE view`)
  }
  await page.screenshot({ path: join(results, 'electron-workbench.png'), fullPage: true })
  const renderedImage = await request('/api/datasets/demo_z_images/episodes/0/video?feature=camera_png')
  assert.ok(renderedImage.duration_s > 0)
  passed('image dataset converts to a playable cached video')

  await page.getByRole('button', { name: '开启采集', exact: true }).click()
  await page.getByRole('button', { name: '确认开启采集', exact: true }).click()
  await expect.poll(async () => (await request('/api/operations')).operation?.kind).toBe('collect')
  const collected = (await request('/api/operations')).operation
  await finished(collected.id)
  assert.ok((await request('/api/datasets')).datasets.some((d) => d.id === 'desktop_smoke'))
  passed('UI starts compiled synthetic collection and saves a v3 dataset')

  const replay = await request('/api/operations', 'POST', { kind: 'replay', dataset_id: 'demo_v21', episode_index: 2 })
  await finished(replay.id)
  passed('compiled robot replay completes with the simulated arm')
  for (const dataset of ['demo_v21', 'demo_v30']) {
    const deleted = await request('/api/operations', 'POST', { kind: 'delete', dataset_id: dataset, episode_index: 1 })
    await finished(deleted.id)
    const detail = await request(`/api/datasets/${dataset}`)
    assert.equal(detail.episodes.length, 2)
    assert.deepEqual(detail.episodes.map((e) => e.episode_index), [0, 1])
    passed(`${dataset}: compiled deletion commits and renumbers temporary data`)
  }
  await page.goto(`${origin}/replay`)
  await expect(page.getByTestId('dataset-select')).toBeEnabled()
  passed('production route /replay loads without Vite')
  assert.deepEqual(errors, [])
  passed('renderer has no console or JavaScript errors')

  // Close while a real compiled task is active; the desktop must wait for it.
  const active = await request('/api/operations', 'POST', { kind: 'replay', dataset_id: 'demo_v21', episode_index: 1 })
  assert.equal(active.state, 'running')
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].close())
  await application.waitForEvent('close', { timeout: 60_000 })
  await expect.poll(() => {
    try { process.kill(backendPid, 0); return false } catch (error) { return error.code === 'ESRCH' }
  }).toBe(true)
  await assert.rejects(fetch(`${origin}/api/health`))
  passed('closing a window during a task stops the task and releases the backend port')
  application = null
  await writeFile(join(results, 'packaged-smoke.json'), JSON.stringify({ passed: true, checks, temporary, executable, appImage, backendPid }, null, 2))
  console.log(`Verified ${checks.length} checks. Report: ${join(results, 'packaged-smoke.json')}`)
} catch (error) {
  if (page && !page.isClosed()) await page.screenshot({ path: join(results, 'electron-failure.png'), fullPage: true }).catch(() => {})
  await writeFile(join(results, 'packaged-smoke.json'), JSON.stringify({ passed: false, checks, temporary, errors, error: String(error) }, null, 2))
  throw error
} finally {
  if (application) {
    await application.evaluate(({ app }) => app.quit()).catch(() => {})
    await application.waitForEvent('close', { timeout: 60_000 }).catch(() => {})
  }
}

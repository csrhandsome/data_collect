import { app, BrowserWindow, dialog, shell } from 'electron'
import { spawn } from 'node:child_process'
import { createWriteStream } from 'node:fs'
import { copyFile, mkdir, writeFile } from 'node:fs/promises'
import { constants } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { stopBackend, waitForBackend } from './lifecycle.mjs'

if (process.env.DATA_COLLECT_USER_DATA) app.setPath('userData', resolve(process.env.DATA_COLLECT_USER_DATA))
const hasLock = app.requestSingleInstanceLock()
if (!hasLock) app.quit()

let backend
let mainWindow
let shuttingDown = false
let canQuit = false
let shutdownPromise
let logStream

app.on('second-instance', () => {
  if (mainWindow?.isMinimized()) mainWindow.restore()
  mainWindow?.focus()
})

function shutdown() {
  if (shutdownPromise) return shutdownPromise
  shuttingDown = true
  mainWindow?.setTitle('正在停止任务并保存数据…')
  shutdownPromise = stopBackend(backend).finally(() => {
    logStream?.end()
    canQuit = true
    app.quit()
  })
  return shutdownPromise
}

app.on('before-quit', (event) => {
  if (canQuit) return
  event.preventDefault()
  void shutdown()
})
app.on('window-all-closed', () => app.quit())

async function start() {
  const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
  const resources = app.isPackaged ? process.resourcesPath : join(repoRoot, 'build/desktop')
  const userData = app.getPath('userData')
  const workDir = join(userData, 'workspace')
  await mkdir(workDir, { recursive: true })
  await mkdir(join(userData, 'logs'), { recursive: true })
  const configPath = process.env.DATA_COLLECT_CONFIG ? resolve(process.env.DATA_COLLECT_CONFIG) : join(userData, 'panda.yaml')
  if (!process.env.DATA_COLLECT_CONFIG) {
    await copyFile(join(resources, 'config/panda.yaml'), configPath, constants.COPYFILE_EXCL).catch((error) => {
      if (error.code !== 'EEXIST') throw error
    })
  }
  const frontend = app.isPackaged ? join(resources, 'frontend') : join(repoRoot, 'replay/frontend/dist')
  const command = join(resources, 'runtime/bin/data-collect-runtime')
  const args = ['server', '--config', configPath, '--frontend', frontend, '--work-dir', workDir]
  if (process.env.REPLAY_DATA_ROOT) args.push('--data-root', resolve(process.env.REPLAY_DATA_ROOT))
  if (process.env.DATA_COLLECT_SIMULATE === '1') args.push('--simulate')
  logStream = createWriteStream(join(userData, 'logs/backend.log'), { flags: 'a' })
  const childEnv = {
    ...process.env, PYTHONUNBUFFERED: '1',
    HF_HOME: process.env.HF_HOME || join(userData, 'cache/huggingface'),
    MPLCONFIGDIR: process.env.MPLCONFIGDIR || join(userData, 'cache/matplotlib'),
  }
  delete childEnv.PYTHONPATH
  delete childEnv.PYTHONHOME
  backend = spawn(command, args, { cwd: workDir, env: childEnv, stdio: ['ignore', 'pipe', 'pipe'] })
  backend.stdout.pipe(logStream, { end: false })
  backend.stderr.pipe(logStream, { end: false })
  const origin = await waitForBackend(backend)
  if (shuttingDown) return
  backend.on('exit', (code, signal) => {
    if (!shuttingDown) {
      void dialog.showMessageBox({ type: 'error', title: '后端已退出', message: `后端意外退出 (${code ?? signal})。`, detail: `日志：${join(userData, 'logs/backend.log')}` }).then(() => app.quit())
    }
  })
  mainWindow = new BrowserWindow({
    width: 1440, height: 960, minWidth: 800, minHeight: 600,
    title: 'Panda Data Workbench', show: false,
    webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true },
  })
  mainWindow.removeMenu()
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) void shell.openExternal(url)
    return { action: 'deny' }
  })
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (new URL(url).origin !== origin) event.preventDefault()
  })
  mainWindow.on('close', (event) => {
    if (!canQuit) { event.preventDefault(); app.quit() }
  })
  await mainWindow.loadURL(origin)
  if (!shuttingDown) mainWindow.show()
  // An opt-in diagnostic file lets tests inspect the real packaged process.
  if (process.env.DATA_COLLECT_READY_FILE) {
    await writeFile(process.env.DATA_COLLECT_READY_FILE, JSON.stringify({ origin, backendPid: backend.pid }))
  }
}

if (hasLock) {
  app.whenReady().then(start).catch(async (error) => {
    if (!shuttingDown) await dialog.showMessageBox({ type: 'error', title: '无法启动工作台', message: error.message, detail: `日志目录：${join(app.getPath('userData'), 'logs')}` })
    app.quit()
  })
}

import test from 'node:test'
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { waitForBackend, stopBackend } from '../lifecycle.mjs'

test('startup uses reported random port and shutdown waits for cleanup', async () => {
  const child = spawn(process.execPath, ['-e', `
    console.log('ordinary startup log');
    console.log(JSON.stringify({event:'ready',port:12345}));
    process.on('SIGTERM', () => setTimeout(() => process.exit(0), 250));
    setInterval(() => {}, 1000);
  `])
  try {
    assert.equal(await waitForBackend(child, 5000), 'http://127.0.0.1:12345')
    const start = Date.now()
    await stopBackend(child)
    assert.ok(Date.now() - start >= 200)
    assert.equal(child.exitCode, 0)
  } finally { child.kill('SIGKILL') }
})

test('startup failure rejects and shutdown accepts an exited child', async () => {
  const child = spawn(process.execPath, ['-e', 'process.exit(7)'])
  await assert.rejects(waitForBackend(child, 5000), /后端启动失败/)
  await stopBackend(child)
})

test('startup timeout is reported and the caller can stop the child', async () => {
  const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'])
  try {
    await assert.rejects(waitForBackend(child, 100), /启动超时/)
    await stopBackend(child)
  } finally { child.kill('SIGKILL') }
})

test('a failed spawn can be shut down without hanging', async () => {
  const child = spawn('/does-not-exist/data-collect-runtime')
  await assert.rejects(waitForBackend(child, 5000), /ENOENT/)
  await stopBackend(child)
})

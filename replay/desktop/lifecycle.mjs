import { createInterface } from 'node:readline'

export function waitForBackend(child, timeoutMs = 120_000) {
  return new Promise((resolve, reject) => {
    const lines = createInterface({ input: child.stdout })
    const timer = setTimeout(() => finish(new Error('后端启动超时，请查看运行日志。')), timeoutMs)
    function finish(error, port) {
      clearTimeout(timer)
      lines.close()
      child.off('error', onError)
      child.off('exit', onExit)
      error ? reject(error) : resolve(`http://127.0.0.1:${port}`)
    }
    const onError = (error) => finish(error)
    const onExit = (code, signal) => finish(new Error(`后端启动失败 (${code ?? signal})`))
    child.once('error', onError)
    child.once('exit', onExit)
    lines.on('line', (line) => {
      try {
        const message = JSON.parse(line)
        if (message.event === 'ready' && Number.isInteger(message.port) && message.port > 0 && message.port < 65536) {
          finish(null, message.port)
        }
      } catch { /* Ordinary backend logs are not readiness messages. */ }
    })
  })
}

export function stopBackend(child) {
  if (!child?.pid || child.exitCode !== null || child.signalCode !== null) return Promise.resolve()
  return new Promise((resolve) => {
    child.once('exit', resolve)
    child.kill('SIGTERM')
    // Never force-kill: collection must save, and deletion must finish its transaction.
  })
}

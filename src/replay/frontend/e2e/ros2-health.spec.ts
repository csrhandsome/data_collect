import { expect, test } from '@playwright/test'
import { mkdir, rename, rm, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

test('shared monitor snapshot reaches the browser through the real API', async ({
  page,
  request,
}) => {
  const path = process.env.ROS2_HEALTH_PATH
  test.skip(
    !path,
    'Set ROS2_HEALTH_PATH to an isolated test file for the backend and browser tests',
  )
  if (!path) return
  await mkdir(dirname(path), { recursive: true })
  const snapshot = {
    updated_at: Date.now() / 1000,
    state: 'warning',
    message: '订阅存在 warning',
    topics: [
      {
        id: 'real-api-camera',
        topic: '/camera/external/color/image_raw',
        component: 'camera',
        source: 'subscriber',
        state: 'warning',
        message: '消息中断，已 1.20 秒未收到数据',
        received_hz: 0,
        min_hz: 24,
        age_s: 1.2,
      },
    ],
    events: [],
  }
  try {
    await writeFile(`${path}.tmp`, JSON.stringify(snapshot))
    await rename(`${path}.tmp`, path)
    const response = await request.get('/api/ros2/health')
    expect(response.ok()).toBeTruthy()
    expect((await response.json()).topics[0].id).toBe('real-api-camera')
    await page.goto('/')
    const panel = page.getByTestId('ros2-health')
    await expect(panel).toContainText('1 项 WARNING')
    await expect(panel).toContainText('消息中断')
    await panel.getByRole('button', { name: '查看监控' }).click()
    await expect(panel.getByRole('table')).toContainText('/camera/external/color/image_raw')
    await page.screenshot({ path: '/tmp/data-collect-ros2-health-desktop.png', fullPage: true })
    await page.setViewportSize({ width: 390, height: 844 })
    await expect(panel.getByRole('button', { name: '收起监控' })).toBeVisible()
    await page.screenshot({ path: '/tmp/data-collect-ros2-health-mobile.png', fullPage: true })
  } finally {
    await rm(path, { force: true })
  }
})

test('ROS2 warnings appear, recover, and become unknown when monitoring disconnects', async ({
  page,
}) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  let state = 'warning'
  await page.route('**/api/ros2/health', (route) =>
    route.fulfill({
      json: {
        state,
        message: state === 'offline' ? '监控心跳中断' : 'ROS2 监控运行中',
        topics: [
          {
            id: 'camera',
            topic: '/camera/wrist/color/image_raw',
            component: 'camera',
            source: 'subscriber',
            state: state === 'warning' ? 'warning' : 'ok',
            message: state === 'warning' ? '接收频率偏低，可能跳帧' : '接收正常',
            received_hz: state === 'warning' ? 12 : 30,
            min_hz: 24,
            age_s: 0.02,
          },
        ],
        events: [
          {
            time: 1700000000,
            topic: '/camera/wrist/color/image_raw',
            state: 'warning',
            message: '接收频率偏低',
          },
        ],
      },
    }),
  )
  await page.goto('/')
  const panel = page.getByTestId('ros2-health')
  await expect(panel).toContainText('1 项 WARNING')
  await panel.getByRole('button', { name: '查看监控' }).click()
  await expect(panel.getByRole('table')).toContainText('12.0')
  await expect(panel.getByRole('table')).toContainText('实际订阅')
  state = 'ok'
  await expect(panel).toContainText('订阅监控在线')
  await expect(panel.getByRole('table')).toContainText('30.0')
  state = 'offline'
  await expect(panel).toContainText('监控未连接')
  await expect(panel.getByRole('table')).toContainText('状态未知')
  expect(errors).toEqual([])
})

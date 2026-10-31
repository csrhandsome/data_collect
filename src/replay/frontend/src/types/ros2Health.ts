export interface Ros2TopicHealth {
  id: string
  topic: string
  component: string
  source: 'subscriber' | 'publisher'
  state: 'ok' | 'warning' | 'waiting' | 'stopped'
  message: string
  received_hz: number | null
  min_hz: number | null
  age_s: number | null
  received?: number
  invalid?: number
}

export interface Ros2Health {
  state: 'ok' | 'warning' | 'offline' | 'not_started' | 'disabled'
  message: string
  age_s?: number
  topics: Ros2TopicHealth[]
  events: { time: number; topic: string; state: string; message: string }[]
}

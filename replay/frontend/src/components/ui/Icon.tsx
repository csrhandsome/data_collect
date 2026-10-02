import type { CSSProperties, ReactNode } from 'react'

export type IconName =
  | 'play'
  | 'pause'
  | 'video'
  | 'trajectory'
  | 'database'
  | 'plus'
  | 'close'
  | 'search'
  | 'grip'
  | 'arrow'
  | 'chevron'
  | 'previous'
  | 'next'
  | 'reset'
  | 'layers'
  | 'warning'
  | 'external'
  | 'check'
  | 'signal'
  | 'box'

const paths: Record<IconName, ReactNode> = {
  play: <path d="m9 5 11 7-11 7V5Z" />,
  pause: (
    <>
      <path d="M8 5v14M16 5v14" />
    </>
  ),
  video: (
    <>
      <rect x="3" y="5" width="13" height="14" rx="2" />
      <path d="m16 9 5-3v12l-5-3" />
    </>
  ),
  trajectory: (
    <>
      <path d="M3 18c2-8 5-1 8-7s7-8 10-5" />
      <circle cx="3" cy="18" r="1.5" />
      <circle cx="21" cy="6" r="1.5" />
      <path d="M3 3v18h18" />
    </>
  ),
  database: (
    <>
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0" />
    </>
  ),
  plus: <path d="M12 5v14M5 12h14" />,
  close: <path d="m6 6 12 12M18 6 6 18" />,
  search: (
    <>
      <circle cx="10.5" cy="10.5" r="6.5" />
      <path d="m16 16 5 5" />
    </>
  ),
  grip: (
    <>
      <circle cx="8" cy="6" r=".6" />
      <circle cx="16" cy="6" r=".6" />
      <circle cx="8" cy="12" r=".6" />
      <circle cx="16" cy="12" r=".6" />
      <circle cx="8" cy="18" r=".6" />
      <circle cx="16" cy="18" r=".6" />
    </>
  ),
  arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,
  chevron: <path d="m6 9 6 6 6-6" />,
  previous: (
    <>
      <path d="M6 5v14m12-14-9 7 9 7V5Z" />
    </>
  ),
  next: (
    <>
      <path d="M18 5v14M6 5l9 7-9 7V5Z" />
    </>
  ),
  reset: (
    <>
      <path d="M4 10a8 8 0 1 1 1 8M4 4v6h6" />
    </>
  ),
  layers: (
    <>
      <path d="m3 7 9-5 9 5-9 5-9-5Zm0 5 9 5 9-5M3 17l9 5 9-5" />
    </>
  ),
  warning: (
    <>
      <path d="M12 3 2 21h20L12 3ZM12 9v5m0 3v.1" />
    </>
  ),
  external: (
    <>
      <path d="M14 3h7v7M21 3 10 14M10 3H4v17h17v-6" />
    </>
  ),
  check: <path d="m5 12 4 4L19 6" />,
  signal: (
    <>
      <path d="M4 19v-3m5 3v-7m5 7V8m5 11V4" />
    </>
  ),
  box: (
    <>
      <path d="m12 2 9 5v10l-9 5-9-5V7l9-5Zm0 10v10M3 7l9 5 9-5M7 5l10 5" />
    </>
  ),
}

export function Icon({
  name,
  size = 18,
  className = '',
  style,
}: {
  name: IconName
  size?: number
  className?: string
  style?: CSSProperties
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      style={style}
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  )
}

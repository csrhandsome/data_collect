const patterns = {
  paper:
    "bg-[url('data:image/svg+xml,%3Csvg%20viewBox=%220%200%20256%20256%22%20xmlns=%22http://www.w3.org/2000/svg%22%3E%3Cfilter%20id=%22noise%22%3E%3CfeTurbulence%20type=%22fractalNoise%22%20baseFrequency=%220.8%22%20numOctaves=%224%22%20stitchTiles=%22stitch%22/%3E%3C/filter%3E%3Crect%20width=%22100%25%22%20height=%22100%25%22%20filter=%22url(%23noise)%22/%3E%3C/svg%3E')] opacity-[0.02]",
  lines:
    'bg-[repeating-linear-gradient(0deg,transparent,transparent_1px,#000_1px,#000_2px)] bg-size-[100%_4px] opacity-[0.015]',
  grid: 'bg-[linear-gradient(#000_1px,transparent_1px),linear-gradient(90deg,#000_1px,transparent_1px)] bg-size-[40px_40px] opacity-[0.025]',
  inverted:
    'bg-[repeating-linear-gradient(90deg,transparent,transparent_1px,#fff_1px,#fff_2px)] bg-size-[4px_100%] opacity-[0.03]',
}

export function Texture({ pattern = 'paper' }: { pattern?: keyof typeof patterns }) {
  return (
    <span
      aria-hidden="true"
      className={`pointer-events-none absolute inset-0 -z-10 ${patterns[pattern]}`}
    />
  )
}

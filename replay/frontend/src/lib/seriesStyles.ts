// Line patterns carry meaning without relying on color. SVG lines and legends share these styles.
export const SERIES_STYLES = [
  { dash: undefined, border: 'solid' },
  { dash: '8 4', border: 'dashed' },
  { dash: '2 3', border: 'dotted' },
  { dash: '10 3 2 3', border: 'dashed' },
  { dash: '12 4', border: 'dashed' },
  { dash: '2 2 2 6', border: 'dotted' },
  { dash: '6 3 2 3 2 3', border: 'dashed' },
  { dash: '16 3 4 3', border: 'dashed' },
] as const

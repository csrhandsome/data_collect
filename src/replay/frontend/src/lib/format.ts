export function timeLabel(seconds: number, precise = false): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0)
  const minutes = Math.floor(safe / 60)
    .toString()
    .padStart(2, '0')
  const secs = Math.floor(safe % 60)
    .toString()
    .padStart(2, '0')
  return `${minutes}:${secs}${
    precise
      ? `.${Math.floor((safe % 1) * 1000)
          .toString()
          .padStart(3, '0')}`
      : ''
  }`
}

export function episodeLabel(index: number) {
  return `Episode ${index.toString().padStart(3, '0')}`
}

export function nearestSample(timestamps: number[], time: number): number {
  let low = 0
  let high = timestamps.length - 1
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (timestamps[middle] < time) low = middle + 1
    else high = middle
  }
  if (low > 0 && time - timestamps[low - 1] < timestamps[low] - time) return low - 1
  return low
}

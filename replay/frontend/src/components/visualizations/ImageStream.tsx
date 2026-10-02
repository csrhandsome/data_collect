import { useState } from 'react'
import { nearestSample } from '../../lib/format'

export interface ImageMetadata { timestamps: number[]; length: number }

export function ImageStream({ baseUrl, feature, time, metadata }: {
  baseUrl: string; feature: string; time: number; metadata: ImageMetadata
}) {
  const frame = nearestSample(metadata.timestamps, time)
  const [error, setError] = useState(false)
  const src = `${baseUrl}/image?feature=${encodeURIComponent(feature)}&frame_index=${frame}`
  return <div className="video-player">
    <div className="video-stage">
      <img className="embedded-frame" src={src} alt={`${feature} 帧 ${frame}`} data-testid={`image-${feature}`}
        onLoad={() => setError(false)} onError={() => setError(true)} />
      {error ? <div role="alert" className="resource-state">图像读取失败，重新定位时间可重试。</div> : null}
    </div>
    <div className="ee-details"><span>帧 {frame+1} / {metadata.length}</span><code>{time.toFixed(3)} s</code></div>
  </div>
}

import { buttonClasses } from '../components/ui/buttonStyles'
import { Texture } from '../components/ui/Texture'
import { Link } from 'react-router-dom'
import { Icon } from '../components/ui/Icon'

export function NotFoundPage() {
  return (
    <main className="relative isolate flex min-h-dvh flex-col items-center justify-center gap-8 px-5 text-center">
      <Texture />
      <span className="font-mono text-[10px] tracking-[0.2em] text-muted-foreground">
        REPLAY / 404
      </span>
      <span
        aria-hidden="true"
        className="font-display text-[128px] leading-none tracking-tight md:text-[160px]"
      >
        404.
      </span>
      <h1 className="font-display text-3xl md:text-4xl">没有找到这个页面。</h1>
      <Link className={buttonClasses()} to="/replay">
        打开回放工作区
        <Icon name="arrow" size={16} />
      </Link>
    </main>
  )
}

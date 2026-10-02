import { Link } from 'react-router-dom'
import { Icon } from '../components/ui/Icon'

export function NotFoundPage() {
  return (
    <main className="not-found">
      <span className="eyebrow">REPLAY / 404</span>
      <h1>没有找到这个页面。</h1>
      <Link className="button button-primary" to="/replay">
        打开回放工作区
        <Icon name="arrow" size={16} />
      </Link>
    </main>
  )
}

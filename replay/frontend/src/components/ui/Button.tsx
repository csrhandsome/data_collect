import type { ButtonHTMLAttributes } from 'react'

import { buttonClasses } from './buttonStyles'
import type { ButtonVariant } from './buttonStyles'

export function Button({
  variant = 'primary',
  className = '',
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant }) {
  return <button type="button" className={`${buttonClasses(variant)} ${className}`} {...props} />
}

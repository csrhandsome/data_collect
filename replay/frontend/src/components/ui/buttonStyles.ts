const variants = {
  primary:
    'border-foreground bg-foreground text-background hover:bg-background hover:text-foreground',
  secondary:
    'border-foreground bg-background text-foreground hover:bg-foreground hover:text-background',
}

export type ButtonVariant = keyof typeof variants

export function buttonClasses(variant: ButtonVariant = 'primary') {
  return `inline-flex min-h-11 items-center justify-center gap-2 border-2 px-5 py-3 font-mono text-xs tracking-widest uppercase transition-none disabled:pointer-events-none ${variants[variant]}`
}

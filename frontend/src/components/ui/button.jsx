/* eslint-disable react-refresh/only-export-components -- shadcn-style: export variants with Button */
import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva } from 'class-variance-authority'

import { cn } from '@/lib/utils'

const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap text-sm font-medium ring-offset-background transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0',
  {
    variants: {
      variant: {
        default:
          'bg-primary text-primary-foreground hover:bg-primary/90 rounded-xl shadow-sm hover:shadow-md active:scale-[0.98]',
        destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90 rounded-xl',
        outline: 'border border-input bg-background hover:bg-accent hover:text-accent-foreground rounded-xl',
        secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80 rounded-xl',
        ghost: 'hover:bg-accent hover:text-accent-foreground rounded-xl',
        link: 'text-primary underline-offset-4 hover:underline',
        warm:
          'bg-gradient-to-r from-primary to-warm-coral text-primary-foreground rounded-xl shadow-md hover:shadow-lg hover:brightness-110 active:scale-[0.98]',
        'warm-outline': 'border-2 border-primary/30 text-primary hover:bg-primary/5 rounded-xl',
        glass: 'bg-card/60 backdrop-blur-md border border-border/50 text-foreground hover:bg-card/80 rounded-xl shadow-sm',
        chip: 'bg-secondary text-secondary-foreground rounded-full text-xs font-medium hover:bg-primary/10 hover:text-primary',
        'chip-active': 'bg-primary text-primary-foreground rounded-full text-xs font-semibold shadow-sm',
      },
      size: {
        default: 'h-10 px-5 py-2',
        sm: 'h-9 rounded-xl px-3.5 text-xs',
        lg: 'h-12 rounded-xl px-8 text-base',
        icon: 'h-10 w-10 rounded-xl',
        chip: 'h-7 px-3 py-1',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  },
)

const Button = React.forwardRef(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button'
    return <Comp className={cn(buttonVariants({ variant, size }), className)} ref={ref} {...props} />
  },
)
Button.displayName = 'Button'

export { Button, buttonVariants }

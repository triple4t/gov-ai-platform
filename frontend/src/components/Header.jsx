import React from 'react'
import { Sparkles } from 'lucide-react'

export default function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-card/70 backdrop-blur-xl">
      <div className="flex h-16 w-full items-center justify-between px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <img
            src="/docu-atlas-logo.png"
            alt=""
            width={40}
            height={40}
            className="h-10 w-10 shrink-0 object-contain"
            decoding="async"
          />
          <div>
            <h1 className="font-display text-lg font-bold tracking-tight text-foreground">DocuAtlas</h1>
            <p className="text-xs text-muted-foreground">Chat · RAG · Document generation</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden items-center gap-2 rounded-full bg-secondary/60 px-4 py-1.5 text-xs text-muted-foreground sm:flex">
            <Sparkles className="h-3 w-3 text-primary" fill="currentColor" />
            <span>AI-Powered</span>
          </div>
          <div className="text-right">
            <p className="text-xs font-medium text-foreground">Navigate code &amp; documents</p>
          </div>
        </div>
      </div>
    </header>
  )
}

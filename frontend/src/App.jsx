import React from 'react'
import Header from './components/Header'
import PRDPlatformPanel from './components/PRDPlatformPanel'

export default function App() {
  return (
    <div className="flex min-h-screen flex-col bg-background">
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 h-80 w-80 rounded-full bg-primary/5 blur-3xl animate-float" />
        <div
          className="absolute top-1/2 -left-40 h-96 w-96 rounded-full bg-warm-coral/5 blur-3xl animate-float"
          style={{ animationDelay: '3s' }}
        />
        <div
          className="absolute -bottom-20 right-1/4 h-64 w-64 rounded-full bg-warm-gold/5 blur-3xl animate-float"
          style={{ animationDelay: '1.5s' }}
        />
      </div>

      <Header />

      <main className="relative z-10 flex min-h-0 w-full flex-1 flex-col px-3 pb-3 pt-2 sm:px-5 sm:pb-4 lg:px-8">
        <PRDPlatformPanel />
      </main>
    </div>
  )
}

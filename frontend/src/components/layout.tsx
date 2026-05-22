import type { ReactNode } from 'react'

export type SidebarNavItem = {
  key: string
  label: string
  short: string
  description: string
  badge?: number
  tourId?: string
}

export function AppRoutes({ children }: { children: ReactNode }) {
  return <>{children}</>
}

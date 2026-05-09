import type { ReactNode } from 'react'

export type SidebarNavItem = {
  key: string
  label: string
  short: string
  description: string
  badge?: number
  tourId?: string
}

type SidebarNavProps = {
  items: SidebarNavItem[]
  activeKey: string
  onNavigate: (key: string) => void
}

export function AppRoutes({ children }: { children: ReactNode }) {
  return <>{children}</>
}

export function SidebarNav({ items, activeKey, onNavigate }: SidebarNavProps) {
  return (
    <nav className="nav-list" aria-label="Dashboard views">
      {items.map((item) => (
        <button
          key={item.key}
          className={activeKey === item.key ? 'nav-item nav-item--active' : 'nav-item'}
          type="button"
          onClick={() => onNavigate(item.key)}
          data-tour={item.tourId}
        >
          <span className="nav-code">{item.short}</span>
          <span className="nav-copy">
            <strong>{item.label}</strong>
            <small>{item.description}</small>
          </span>
          {item.badge ? <b>{item.badge}</b> : null}
        </button>
      ))}
    </nav>
  )
}

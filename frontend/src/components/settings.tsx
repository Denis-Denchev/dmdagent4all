import type { ReactNode } from 'react'
import { StatusBadge } from './cockpit'

export type ConfigHubEntry = {
  key: string
  title: string
  description: string
  status?: string
  tone?: 'normal' | 'warning' | 'danger'
  group?: string
}

type ConfigHubProps = {
  entries: ConfigHubEntry[]
  onOpen: (key: string) => void
}

type ConfigSectionPageProps = {
  title: string
  description: string
  onBack: () => void
  children: ReactNode
}

function configIcon(key: string) {
  const icons: Record<string, string> = {
    general: 'ID',
    models: 'MD',
    tools: 'TL',
    workspace: 'WS',
    security: 'SC',
    memory: 'MY',
    telegram: 'TG',
    emergency: 'EM',
    advanced: 'SY',
  }
  return icons[key] ?? 'CF'
}

export function ConfigHub({ entries, onOpen }: ConfigHubProps) {
  const groups = entries.reduce<Array<{ label: string; entries: ConfigHubEntry[] }>>((currentGroups, entry) => {
    const label = entry.group ?? 'Settings'
    const existing = currentGroups.find((group) => group.label === label)
    if (existing) {
      existing.entries.push(entry)
    } else {
      currentGroups.push({ label, entries: [entry] })
    }
    return currentGroups
  }, [])

  return (
    <section className="config-hub" data-tour="tour-config">
      <div className="settings-page-header">
        <div>
          <p className="eyebrow">Settings index</p>
          <h2>Configuration</h2>
          <p>Core local-agent settings grouped by the areas you actually configure.</p>
        </div>
      </div>
      <div className="settings-index">
        {groups.map((group) => (
          <div className="settings-index-group" key={group.label}>
            {groups.length > 1 ? <div className="settings-index-group-label">{group.label}</div> : null}
            {group.entries.map((entry) => (
              <button
                key={entry.key}
                className={`settings-index-row settings-index-row--${entry.tone ?? 'normal'}`}
                type="button"
                onClick={() => onOpen(entry.key)}
              >
                <span className="settings-index-icon">{configIcon(entry.key)}</span>
                <span className="settings-index-main">
                  <strong>{entry.title}</strong>
                  <small>{entry.description}</small>
                </span>
                {entry.status ? <StatusBadge tone={entry.tone}>{entry.status}</StatusBadge> : null}
                <span className="settings-index-arrow">Open</span>
              </button>
            ))}
          </div>
        ))}
      </div>
    </section>
  )
}

export function ConfigSectionPage({ title, description, onBack, children }: ConfigSectionPageProps) {
  return (
    <section className="config-section-page">
      <div className="settings-page-header">
        <div>
          <button className="button button-ghost settings-back" type="button" onClick={onBack}>
            Back to Config
          </button>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
      </div>
      {children}
    </section>
  )
}

export function SettingGroup({
  title,
  description,
  children,
}: {
  title: string
  description?: string
  children: ReactNode
}) {
  return (
    <section className="setting-group">
      <div className="setting-group-header">
        <strong>{title}</strong>
        {description ? <span>{description}</span> : null}
      </div>
      <div className="setting-group-body">{children}</div>
    </section>
  )
}

export function SettingRow({
  label,
  help,
  children,
}: {
  label: string
  help?: string
  children: ReactNode
}) {
  return (
    <label className="setting-row">
      <span>
        <strong>{label}</strong>
        {help ? <small>{help}</small> : null}
      </span>
      <span className="setting-row-control">{children}</span>
    </label>
  )
}

export function DangerZone({ children }: { children: ReactNode }) {
  return <section className="danger-zone">{children}</section>
}

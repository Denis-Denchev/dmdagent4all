import type { ReactNode } from 'react'
import type { AgentResponse, AgentTraceEvent, Approval, AuditEvent, Status } from '../api'
import { approvalIdFromResponse, dataObject, formatDate, traceFromResponse, visibleTrace } from '../utils'

type Tone = 'normal' | 'warning' | 'danger' | 'success' | 'muted'

export type CockpitNavItem = {
  key: string
  label: string
  short: string
  description: string
  badge?: number
  tourId?: string
}

export type ChatMessageModel = {
  id: string
  role: 'user' | 'agent' | 'system'
  text: string
  data?: unknown
  response?: AgentResponse
}

function shortJson(value: unknown) {
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function riskTone(risk: number | null | undefined): Tone {
  if ((risk ?? 0) >= 5) return 'danger'
  if ((risk ?? 0) >= 3) return 'warning'
  if ((risk ?? 0) >= 1) return 'normal'
  return 'success'
}

export function StatusBadge({
  children,
  tone = 'normal',
}: {
  children: ReactNode
  tone?: Tone
}) {
  return <span className={`status-badge status-badge--${tone}`}>{children}</span>
}

export function CommandCard({
  eyebrow,
  title,
  meta,
  actions,
  children,
  className = '',
}: {
  eyebrow?: string
  title?: string
  meta?: ReactNode
  actions?: ReactNode
  children?: ReactNode
  className?: string
}) {
  return (
    <section className={['panel command-card', className].filter(Boolean).join(' ')}>
      {title || eyebrow || meta || actions ? (
        <div className="command-card-header">
          <div>
            {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
            {title ? <strong>{title}</strong> : null}
          </div>
          {meta || actions ? <div className="command-card-actions">{meta}{actions}</div> : null}
        </div>
      ) : null}
      {children}
    </section>
  )
}

export function AppShell({
  collapsed,
  autonomy,
  children,
}: {
  collapsed: boolean
  autonomy: boolean
  children: ReactNode
}) {
  return (
    <div
      className={[
        'app-shell',
        collapsed ? 'app-shell--rail-collapsed' : '',
        autonomy ? 'app-shell--autonomy' : '',
      ].filter(Boolean).join(' ')}
    >
      {children}
    </div>
  )
}

export function Sidebar({
  collapsed,
  items,
  activeKey,
  provider,
  model,
  version,
  enabledToolCount,
  pendingCount,
  autonomyEnabled,
  emergencyActive,
  onToggle,
  onNavigate,
}: {
  collapsed: boolean
  items: CockpitNavItem[]
  activeKey: string
  provider?: string
  model?: string
  version?: string
  enabledToolCount: number
  pendingCount: number
  autonomyEnabled: boolean
  emergencyActive: boolean
  onToggle: () => void
  onNavigate: (key: string) => void
}) {
  return (
    <aside className="command-rail">
      <div className="rail-header">
        <div className="brand" data-tour="tour-help">
          <div className="brand-mark">D</div>
          <div className="brand-copy">
            <strong>DMD Agent 4 All</strong>
            <em>v2.0</em>
          </div>
        </div>
        <button
          className="rail-toggle"
          type="button"
          onClick={onToggle}
          aria-label={collapsed ? 'Expand navigation' : 'Collapse navigation'}
          aria-expanded={!collapsed}
        >
          <span />
          <span />
          <span />
        </button>
      </div>

      <div className="rail-identity">
        <span className="rail-identity-role">DMD CORE</span>
        <span className="rail-identity-sub">LOCAL-FIRST AGENT</span>
      </div>

      <nav className="nav-list" aria-label="Dashboard views">
        {items.map((item) => (
          <button
            key={item.key}
            className={activeKey === item.key ? 'nav-item nav-item--active' : 'nav-item'}
            type="button"
            onClick={() => onNavigate(item.key)}
            data-tour={item.tourId}
          >
            <span className="nav-icon" aria-hidden="true">{item.short}</span>
            <span className="nav-copy">
              <strong>{item.label}</strong>
              <small>{item.description}</small>
            </span>
            {item.badge ? <b className="nav-badge">{item.badge}</b> : null}
          </button>
        ))}
      </nav>

      <div className="rail-status">
        <div className="rail-status-head">
          <span className={emergencyActive ? 'status-led status-led--danger' : 'status-led'} />
          <strong>{emergencyActive ? 'Agent locked' : 'Agent online'}</strong>
        </div>
        <div className="rail-status-rows">
          <div className="rail-stat">
            <span>Runtime</span>
            <strong>{provider ?? '—'} / {model || '—'}</strong>
          </div>
          <div className="rail-stat">
            <span>Tools enabled</span>
            <strong>{enabledToolCount}</strong>
          </div>
          <div className="rail-stat">
            <span>Pending</span>
            <strong>{pendingCount}</strong>
          </div>
          <div className="rail-stat">
            <span>Safety</span>
            <strong>{autonomyEnabled ? 'Danger mode' : 'Safe mode'}</strong>
          </div>
          {version ? (
            <div className="rail-stat">
              <span>Version</span>
              <strong>{version}</strong>
            </div>
          ) : null}
        </div>
      </div>
    </aside>
  )
}

export function TopStatusBar({
  activeLabel,
  status,
  autonomyEnabled,
  toggleLocked,
  pendingCount,
  enabledToolCount,
  emergencyActive,
  busy,
  onToggleAutonomy,
  onEmergencyStop,
  onEmergencyReset,
  onStartTour,
  onRefresh,
}: {
  activeLabel: string
  status: Status | null
  autonomyEnabled: boolean
  toggleLocked: boolean
  pendingCount: number
  enabledToolCount: number
  emergencyActive: boolean
  busy: boolean
  onToggleAutonomy: () => void
  onEmergencyStop: () => void
  onEmergencyReset: () => void
  onStartTour: () => void
  onRefresh: () => void
}) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <p className="topbar-eyebrow">LOCAL-FIRST COCKPIT</p>
        <h1 className="topbar-title">{activeLabel}</h1>
      </div>

      <div className="status-strip" aria-label="Runtime status">
        <div className="status-cell">
          <span className="status-cell-label">Mode</span>
          <strong className={autonomyEnabled ? 'status-cell-value status-cell-value--danger' : 'status-cell-value'}>
            {autonomyEnabled ? 'Danger' : 'Safe'}
          </strong>
        </div>
        <div className="status-cell">
          <span className="status-cell-label">Runtime model</span>
          <strong className="status-cell-value">{status?.llm.model ?? '—'}</strong>
        </div>
        <div className="status-cell">
          <span className="status-cell-label">Safety</span>
          <strong className={emergencyActive ? 'status-cell-value status-cell-value--danger' : 'status-cell-value status-cell-value--ok'}>
            {emergencyActive ? 'Locked' : 'Engaged'}
          </strong>
        </div>
        <div className="status-cell">
          <span className="status-cell-label">Approvals</span>
          <strong className={pendingCount > 0 ? 'status-cell-value status-cell-value--warn' : 'status-cell-value'}>
            {pendingCount}
          </strong>
        </div>
        <div className="status-cell">
          <span className="status-cell-label">Tools</span>
          <strong className="status-cell-value">{enabledToolCount}</strong>
        </div>
      </div>

      <div className="topbar-actions">
        <button
          className={autonomyEnabled ? 'cmd-btn cmd-btn--danger-mode cmd-btn--active' : 'cmd-btn cmd-btn--danger-mode'}
          type="button"
          onClick={onToggleAutonomy}
          disabled={busy || toggleLocked}
          title={toggleLocked ? 'LOCAL_DEV_AUTONOMY env var is forcing autonomy mode.' : 'Switch runtime orchestration mode'}
        >
          <span className="autonomy-dot" />
          <span>Danger Mode</span>
        </button>
        {emergencyActive ? (
          <button className="cmd-btn cmd-btn--secondary" type="button" onClick={onEmergencyReset}>
            Reset Emergency
          </button>
        ) : null}
        <button
          className="cmd-btn cmd-btn--emergency"
          type="button"
          onClick={onEmergencyStop}
          data-tour="tour-emergency"
        >
          Emergency Stop
        </button>
        <button className="cmd-btn cmd-btn--secondary" type="button" onClick={onStartTour}>
          Tour
        </button>
        <button className="cmd-btn cmd-btn--secondary" type="button" onClick={onRefresh}>
          Refresh
        </button>
      </div>
    </header>
  )
}

export function ToolTrace({ events, live = false }: { events: AgentTraceEvent[]; live?: boolean }) {
  const visible = visibleTrace(events)
  if (!visible.length) return null
  const latest = visible[visible.length - 1]
  return (
    <details className={live ? 'reasoning-panel reasoning-panel--live' : 'reasoning-panel'} open={live}>
      <summary>
        <span className="reasoning-summary-main">
          <span className={live ? 'reasoning-dot reasoning-dot--live' : 'reasoning-dot'} />
          <span>{live ? latest.title : 'Tool trace'}</span>
        </span>
        <span>{visible.length} steps</span>
      </summary>
      <div className="reasoning-events">
        {visible.slice(live ? -8 : 0).map((event, index) => (
          <div className="reasoning-event" key={`${event.at ?? index}-${event.kind}-${event.title}`}>
            <span className={`reasoning-status reasoning-status--${event.status}`}>{event.status}</span>
            <span className="reasoning-event-copy">
              <strong>{event.title}</strong>
              {event.detail ? <small>{event.detail}</small> : null}
            </span>
            {event.tool ? <code>{event.tool}</code> : null}
          </div>
        ))}
      </div>
    </details>
  )
}

export function ChatMessage({
  message,
  onApprove,
  onDeny,
}: {
  message: ChatMessageModel
  onApprove: (id: number) => void
  onDeny: (id: number) => void
}) {
  const trace = visibleTrace(traceFromResponse(message.response))
  const responseData = dataObject(message.response?.data)
  const approvalId = approvalIdFromResponse(message.response)
  const tool = typeof responseData.tool === 'string'
    ? responseData.tool
    : [...trace].reverse().find((event) => event.tool)?.tool
  const timestamp = trace[0]?.at
  const to = typeof responseData.to === 'string' ? responseData.to : ''
  const subject = typeof responseData.subject === 'string' ? responseData.subject : ''
  const bodyPreview = typeof responseData.body_preview === 'string' ? responseData.body_preview : ''
  const roleLabel = message.role === 'user' ? 'YOU' : message.role === 'system' ? 'SYS' : 'AI'

  return (
    <article className={`message message--${message.role}`}>
      <div className="message-avatar">{roleLabel}</div>
      <div className="message-body">
        <div className="message-meta">
          <span className="message-role-label">
            {message.role === 'user' ? 'User command' : message.role === 'system' ? 'System' : 'Agent response'}
          </span>
          {message.response?.status ? (
            <StatusBadge tone={message.response.status === 'approval_required' ? 'warning' : 'success'}>
              {message.response.status}
            </StatusBadge>
          ) : null}
          {tool ? <code>{tool}</code> : null}
          {trace.length ? <span className="message-steps">{trace.length} steps</span> : null}
          {timestamp ? <span className="message-time">{formatDate(timestamp)}</span> : null}
        </div>
        <p>{message.text}</p>
        <ToolTrace events={trace} />
        {approvalId !== null ? (
          <div className="approval-inline approval-inline--danger">
            <div className="approval-inline-head">
              <StatusBadge tone="warning">Approval required</StatusBadge>
              <strong>{String(responseData.tool ?? 'tool action')}</strong>
            </div>
            {to || subject || bodyPreview ? (
              <div className="email-preview">
                {to ? <div><span>To</span><strong>{to}</strong></div> : null}
                {subject ? <div><span>Subject</span><strong>{subject}</strong></div> : null}
                {bodyPreview ? <pre>{bodyPreview}</pre> : null}
              </div>
            ) : null}
            <div className="row-actions">
              <button className="button" type="button" onClick={() => onApprove(approvalId)}>
                Approve
              </button>
              <button className="button button-danger" type="button" onClick={() => onDeny(approvalId)}>
                Deny
              </button>
            </div>
          </div>
        ) : null}
      </div>
    </article>
  )
}

export function ApprovalCard({
  approval,
  onApprove,
  onDeny,
  wide = false,
}: {
  approval: Approval
  onApprove: (id: number) => void
  onDeny: (id: number) => void
  wide?: boolean
}) {
  const tone = riskTone(approval.risk)
  return (
    <article className={wide ? 'approval-card approval-card--wide' : 'approval-card'}>
      <div className="approval-card-top">
        <div>
          <span className="eyebrow">Risk-control request</span>
          <strong>#{approval.id} {approval.tool}</strong>
        </div>
        <StatusBadge tone={tone}>Risk {approval.risk ?? '—'}</StatusBadge>
      </div>
      <p>{approval.reason ?? approval.request_reason ?? approval.decision_reason ?? 'Needs approval before execution.'}</p>
      <div className="approval-card-meta">
        <span>Requested</span>
        <strong>{formatDate(approval.created_at)}</strong>
        <span>Status</span>
        <strong>{approval.status}</strong>
      </div>
      <details className="approval-details">
        <summary>Action details</summary>
        <code>{shortJson(approval.args)}</code>
      </details>
      <div className="row-actions">
        <button className="button" type="button" onClick={() => onApprove(approval.id)}>Approve</button>
        <button className="button button-danger" type="button" onClick={() => onDeny(approval.id)}>Deny</button>
      </div>
    </article>
  )
}

export function ToolCard({
  title,
  description,
  status,
  risk,
  enabled,
  approvalRequired,
  onToggle,
}: {
  title: string
  description: string
  status?: string
  risk: number | null
  enabled: boolean
  approvalRequired?: boolean
  onToggle?: () => void
}) {
  return (
    <article className={enabled ? 'tool-card tool-card--enabled' : 'tool-card'}>
      <div className="tool-card-head">
        <div>
          <strong>{title}</strong>
          <span>{description}</span>
        </div>
        <span className={enabled ? 'status-led' : 'status-led status-led--muted'} />
      </div>
      <div className="tool-meta">
        <StatusBadge tone={enabled ? 'success' : 'muted'}>{status ?? (enabled ? 'Enabled' : 'Disabled')}</StatusBadge>
        <StatusBadge tone={riskTone(risk)}>Risk {risk ?? '—'}</StatusBadge>
        <StatusBadge tone={approvalRequired ? 'warning' : 'success'}>{approvalRequired ? 'Approval' : 'Direct'}</StatusBadge>
      </div>
      {onToggle ? (
        <button className={enabled ? 'toggle toggle-on' : 'toggle'} type="button" onClick={onToggle} aria-pressed={enabled}>
          {enabled ? 'Enabled' : 'Disabled'}
        </button>
      ) : null}
    </article>
  )
}

export function AgentCorePanel({
  status,
  autonomyEnabled,
  emergencyActive,
  enabledToolCount,
  pendingCount,
  memoryCount,
}: {
  status: Status | null
  autonomyEnabled: boolean
  emergencyActive: boolean
  enabledToolCount: number
  pendingCount: number
  memoryCount: number
}) {
  const stateLabel = emergencyActive ? 'LOCKED' : autonomyEnabled ? 'THINKING' : 'IDLE'
  const stateClass = emergencyActive ? 'danger' : autonomyEnabled ? 'active' : 'idle'
  return (
    <section className="agent-core-panel panel">
      <p className="panel-eyebrow">Agent Overview</p>
      <div className="ai-core-wrap">
        <div className={`ai-core ai-core--${stateClass}`}>
          <div className="ai-pulse ai-pulse--1" />
          <div className="ai-pulse ai-pulse--2" />
          <svg className="ai-globe" viewBox="0 0 240 240" aria-hidden="true">
            <defs>
              <radialGradient id="ai-globe-core" cx="38%" cy="28%" r="72%">
                <stop offset="0%" stopColor="rgba(238, 247, 255, 0.95)" />
                <stop offset="22%" stopColor="rgba(0, 245, 255, 0.55)" />
                <stop offset="58%" stopColor="rgba(0, 255, 198, 0.18)" />
                <stop offset="100%" stopColor="rgba(0, 18, 24, 0.1)" />
              </radialGradient>
              <clipPath id="ai-globe-clip">
                <circle cx="120" cy="120" r="82" />
              </clipPath>
            </defs>
            <circle className="ai-globe-shell" cx="120" cy="120" r="82" />
            <g className="ai-globe-grid" clipPath="url(#ai-globe-clip)">
              <ellipse cx="120" cy="120" rx="82" ry="18" />
              <ellipse cx="120" cy="120" rx="82" ry="34" />
              <ellipse cx="120" cy="120" rx="82" ry="58" />
              <ellipse cx="120" cy="120" rx="82" ry="78" />
              <ellipse cx="120" cy="120" rx="28" ry="82" />
              <ellipse cx="120" cy="120" rx="52" ry="82" />
              <ellipse cx="120" cy="120" rx="78" ry="82" />
              <ellipse cx="120" cy="120" rx="82" ry="30" transform="rotate(34 120 120)" />
              <ellipse cx="120" cy="120" rx="82" ry="30" transform="rotate(-34 120 120)" />
            </g>
            <g className="ai-neural-links" clipPath="url(#ai-globe-clip)">
              <path d="M64 92 C88 56, 130 50, 158 78" />
              <path d="M54 134 C82 108, 126 104, 180 122" />
              <path d="M74 166 C92 132, 130 124, 168 152" />
              <path d="M82 70 C116 96, 136 130, 150 174" />
              <path d="M51 112 C90 128, 122 96, 187 95" />
              <path d="M94 184 C116 146, 150 114, 174 64" />
              <path d="M59 151 C97 159, 126 80, 179 89" />
            </g>
            <g className="ai-data-streams" clipPath="url(#ai-globe-clip)">
              <path d="M43 120 C75 88, 119 78, 198 111" />
              <path d="M62 160 C103 135, 138 135, 184 158" />
              <path d="M76 70 C108 116, 132 146, 158 190" />
            </g>
            <g className="ai-neural-nodes">
              <circle cx="64" cy="92" r="3.2" />
              <circle cx="158" cy="78" r="3.6" />
              <circle cx="54" cy="134" r="2.8" />
              <circle cx="180" cy="122" r="3.1" />
              <circle cx="74" cy="166" r="3" />
              <circle cx="168" cy="152" r="3.4" />
              <circle cx="82" cy="70" r="2.7" />
              <circle cx="150" cy="174" r="3" />
              <circle cx="187" cy="95" r="2.8" />
              <circle cx="94" cy="184" r="2.6" />
              <circle cx="120" cy="120" r="4.4" />
              <circle cx="126" cy="80" r="2.4" />
              <circle cx="116" cy="149" r="2.4" />
            </g>
            <circle className="ai-globe-vignette" cx="120" cy="120" r="82" />
          </svg>
          <span className="ai-orbit-node ai-orbit-node--1" />
          <span className="ai-orbit-node ai-orbit-node--2" />
          <span className="ai-orbit-node ai-orbit-node--3" />
          <div className="ai-ring ai-ring--equator" />
          <div className="ai-ring ai-ring--3" />
          <div className="ai-ring ai-ring--2" />
          <div className="ai-ring ai-ring--1" />
          <div className="ai-scan" />
          <div className="ai-core-inner">
            <div className="ai-core-glow" />
            <span className="ai-core-label">AI</span>
          </div>
        </div>
      </div>
      <div className="agent-state-label">
        <span className={`agent-state agent-state--${stateClass}`}>{stateLabel}</span>
        <strong>{status?.llm.provider ?? 'runtime'} / {status?.llm.model ?? 'core'}</strong>
      </div>
      <div className="agent-core-grid">
        <div className="agent-stat"><span>Tools</span><strong>{enabledToolCount}</strong></div>
        <div className="agent-stat"><span>Memory</span><strong>{memoryCount}</strong></div>
        <div className="agent-stat"><span>Safety</span><strong>{emergencyActive ? 'Stop' : 'On'}</strong></div>
        <div className="agent-stat"><span>Approvals</span><strong>{pendingCount}</strong></div>
      </div>
    </section>
  )
}

export function ActivityFeed({ events, limit = 6 }: { events: AuditEvent[]; limit?: number }) {
  const recent = events.slice(0, limit)
  return (
    <section className="activity-feed panel">
      <div className="section-heading">
        <strong>Recent Activity</strong>
        <span>{recent.length} events</span>
      </div>
      {recent.length === 0 ? <p className="empty">No recorded activity.</p> : null}
      <div className="activity-list">
        {recent.map((event) => (
          <article className="activity-row" key={`${event.created_at}-${event.event_type}-${event.tool ?? ''}`}>
            <span className={event.result_status === 'error' ? 'status-led status-led--danger' : 'status-led'} />
            <div className="activity-row-copy">
              <strong>{event.event_type}</strong>
              <small>{event.tool ?? 'system'} / {event.result_status ?? 'recorded'}</small>
            </div>
            <time>{formatDate(event.created_at)}</time>
          </article>
        ))}
      </div>
    </section>
  )
}

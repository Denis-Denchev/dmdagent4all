import { useEffect, useMemo, useState } from 'react'
import {
  type AgentResponse,
  type Approval,
  type AuditEvent,
  api,
  type MemoryFile,
  type ModelMode,
  type PermissionItem,
  type Status,
  type Tool,
} from './api'

type View = 'chat' | 'approvals' | 'tools' | 'permissions' | 'memory' | 'audit' | 'models'

type ChatMessage = {
  role: 'user' | 'agent'
  text: string
  data?: unknown
}

const views: Array<{ key: View; label: string }> = [
  { key: 'chat', label: 'Chat' },
  { key: 'approvals', label: 'Approvals' },
  { key: 'tools', label: 'Tools' },
  { key: 'permissions', label: 'Permissions' },
  { key: 'memory', label: 'Memory' },
  { key: 'audit', label: 'Audit' },
  { key: 'models', label: 'Models' },
]

export function App() {
  const [activeView, setActiveView] = useState<View>('chat')
  const [status, setStatus] = useState<Status | null>(null)
  const [tools, setTools] = useState<Tool[]>([])
  const [permissions, setPermissions] = useState<PermissionItem[]>([])
  const [approvals, setApprovals] = useState<Approval[]>([])
  const [audit, setAudit] = useState<AuditEvent[]>([])
  const [memoryFiles, setMemoryFiles] = useState<string[]>([])
  const [selectedMemory, setSelectedMemory] = useState<MemoryFile | null>(null)
  const [memoryDraft, setMemoryDraft] = useState('')
  const [models, setModels] = useState<ModelMode[]>([])
  const [customModel, setCustomModel] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'agent',
      text: 'Local control center is ready. Tools, memory, approvals, and model settings are available from the sidebar.',
    },
  ])
  const [chatInput, setChatInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)

  const pendingCount = approvals.length
  const enabledToolCount = useMemo(
    () => tools.filter((tool) => tool.enabled).length,
    [tools],
  )

  useEffect(() => {
    void refreshAll()
  }, [])

  async function refreshAll() {
    const [
      statusResult,
      toolsResult,
      permissionsResult,
      approvalsResult,
      auditResult,
      memoryResult,
      modelsResult,
    ] =
      await Promise.all([
        api.status(),
        api.tools(),
        api.permissions(),
        api.approvals(),
        api.audit(),
        api.memory(),
        api.models(),
      ])
    setStatus(statusResult)
    setTools(toolsResult)
    setPermissions(permissionsResult.available)
    setApprovals(approvalsResult)
    setAudit(auditResult)
    setMemoryFiles(memoryResult.files)
    setModels(modelsResult.modes)
    setCustomModel(modelsResult.current.model)
  }

  async function runAction(action: () => Promise<unknown>, message?: string) {
    setBusy(true)
    setNotice(null)
    try {
      await action()
      if (message) setNotice(message)
      await refreshAll()
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Request failed')
    } finally {
      setBusy(false)
    }
  }

  async function sendChat() {
    const message = chatInput.trim()
    if (!message) return
    setChatInput('')
    setMessages((current) => [...current, { role: 'user', text: message }])
    setBusy(true)
    try {
      const response = await api.chat(message)
      setMessages((current) => [
        ...current,
        { role: 'agent', text: `[${response.status}] ${response.message}`, data: response.data },
      ])
      await refreshAll()
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: 'agent',
          text: error instanceof Error ? error.message : 'Request failed',
        },
      ])
    } finally {
      setBusy(false)
    }
  }

  async function loadMemoryFile(path: string) {
    const file = await api.memoryFile(path)
    setSelectedMemory(file)
    setMemoryDraft(file.content)
  }

  async function requestMemorySave() {
    if (!selectedMemory) return
    await runAction(
      async () => {
        const response = await api.writeMemory(selectedMemory.path, memoryDraft)
        setNotice(`${response.status}: ${response.message}`)
      },
      undefined,
    )
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">D</div>
          <div>
            <strong>DMD Agent</strong>
            <span>Local control center</span>
          </div>
        </div>

        <nav className="nav-list" aria-label="Dashboard views">
          {views.map((view) => (
            <button
              key={view.key}
              className={activeView === view.key ? 'nav-item nav-item--active' : 'nav-item'}
              type="button"
              onClick={() => setActiveView(view.key)}
            >
              {view.label}
              {view.key === 'approvals' && pendingCount > 0 ? (
                <span className="pill">{pendingCount}</span>
              ) : null}
            </button>
          ))}
        </nav>

        <div className="sidebar-status">
          <span>Model</span>
          <strong>{status?.llm.model ?? 'loading'}</strong>
          <span>Tools enabled</span>
          <strong>{enabledToolCount}</strong>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Local-first AI</p>
            <h1>{views.find((view) => view.key === activeView)?.label}</h1>
          </div>
          <button className="button button-secondary" type="button" onClick={() => void refreshAll()}>
            Refresh
          </button>
        </header>

        {notice ? <div className="notice">{notice}</div> : null}

        {activeView === 'chat' ? (
          <section className="panel chat-panel">
            <div className="chat-log">
              {messages.map((message, index) => (
                <article key={`${message.role}-${index}`} className={`message message--${message.role}`}>
                  <p>{message.text}</p>
                  {message.data ? <pre>{JSON.stringify(message.data, null, 2)}</pre> : null}
                </article>
              ))}
            </div>
            <form
              className="chat-form"
              onSubmit={(event) => {
                event.preventDefault()
                void sendChat()
              }}
            >
              <input
                value={chatInput}
                onChange={(event) => setChatInput(event.target.value)}
                placeholder="Ask the local agent"
              />
              <button className="button" type="submit" disabled={busy}>
                Send
              </button>
            </form>
          </section>
        ) : null}

        {activeView === 'approvals' ? (
          <section className="panel">
            <div className="table-list">
              {approvals.length === 0 ? <p className="empty">No pending approvals.</p> : null}
              {approvals.map((approval) => (
                <article className="approval-row" key={approval.id}>
                  <div>
                    <strong>#{approval.id} {approval.tool}</strong>
                    <span>Risk {approval.risk ?? '-'} - {approval.reason ?? 'Pending approval'}</span>
                    <code>{JSON.stringify(approval.args)}</code>
                  </div>
                  <div className="row-actions">
                    <button
                      className="button"
                      type="button"
                      onClick={() => void runAction(() => api.approve(approval.id), 'Approval executed.')}
                    >
                      Approve
                    </button>
                    <button
                      className="button button-danger"
                      type="button"
                      onClick={() => void runAction(() => api.deny(approval.id), 'Approval denied.')}
                    >
                      Deny
                    </button>
                  </div>
                </article>
              ))}
            </div>
          </section>
        ) : null}

        {activeView === 'tools' ? (
          <section className="panel">
            <div className="tools-grid">
              {tools.map((tool) => (
                <article className="tool-card" key={tool.name}>
                  <div>
                    <strong>{tool.name}</strong>
                    <span>{tool.description}</span>
                  </div>
                  <div className="tool-meta">
                    <span>Risk {tool.risk}</span>
                    <span>{tool.approval_required ? 'Approval' : 'No approval'}</span>
                  </div>
                  <button
                    className={tool.enabled ? 'toggle toggle-on' : 'toggle'}
                    type="button"
                    onClick={() =>
                      void runAction(
                        () => api.setTool(tool.name, !tool.enabled),
                        `${tool.name} ${tool.enabled ? 'disabled' : 'enabled'}.`,
                      )
                    }
                    aria-pressed={tool.enabled}
                  >
                    {tool.enabled ? 'Enabled' : 'Disabled'}
                  </button>
                </article>
              ))}
            </div>
          </section>
        ) : null}

        {activeView === 'permissions' ? (
          <section className="panel">
            <div className="table-list">
              {permissions.map((permission) => (
                <article className="permission-row" key={permission.name}>
                  <div>
                    <strong>{permission.name}</strong>
                    <span>{permission.tools.join(', ')}</span>
                  </div>
                  <button
                    className={permission.granted ? 'toggle toggle-on' : 'toggle'}
                    type="button"
                    onClick={() =>
                      void runAction(
                        () => api.setPermission(permission.name, !permission.granted),
                        `${permission.name} ${permission.granted ? 'revoked' : 'granted'}.`,
                      )
                    }
                    aria-pressed={permission.granted}
                  >
                    {permission.granted ? 'Granted' : 'Not Granted'}
                  </button>
                </article>
              ))}
            </div>
          </section>
        ) : null}

        {activeView === 'memory' ? (
          <section className="split-view">
            <div className="panel file-list">
              {memoryFiles.map((file) => (
                <button key={file} type="button" onClick={() => void loadMemoryFile(file)}>
                  {file}
                </button>
              ))}
            </div>
            <div className="panel editor-panel">
              {selectedMemory ? (
                <>
                  <div className="editor-header">
                    <strong>{selectedMemory.path}</strong>
                    <button className="button" type="button" onClick={() => void requestMemorySave()}>
                      Request Save
                    </button>
                  </div>
                  <textarea
                    value={memoryDraft}
                    onChange={(event) => setMemoryDraft(event.target.value)}
                    spellCheck={false}
                  />
                </>
              ) : (
                <p className="empty">Select a memory file.</p>
              )}
            </div>
          </section>
        ) : null}

        {activeView === 'audit' ? (
          <section className="panel">
            <div className="table-list">
              {audit.map((event, index) => (
                <article className="audit-row" key={`${event.created_at}-${index}`}>
                  <strong>{event.event_type}</strong>
                  <span>{event.created_at}</span>
                  <span>{event.tool ?? 'system'} - {event.result_status ?? 'recorded'}</span>
                </article>
              ))}
            </div>
          </section>
        ) : null}

        {activeView === 'models' ? (
          <section className="panel model-panel">
            <div className="model-current">
              <span>Current mode</span>
              <strong>{status?.llm.mode ?? '-'}</strong>
              <span>Current model</span>
              <strong>{status?.llm.model ?? '-'}</strong>
            </div>
            <div className="models-grid">
              {models.map((mode) => (
                <article className="model-card" key={mode.key}>
                  <strong>{mode.label}</strong>
                  <code>{mode.default_model}</code>
                  <span>{mode.description}</span>
                  <button
                    className="button button-secondary"
                    type="button"
                    onClick={() => void runAction(() => api.setModelMode(mode.key), `${mode.label} selected.`)}
                  >
                    Use Mode
                  </button>
                </article>
              ))}
            </div>
            <form
              className="model-form"
              onSubmit={(event) => {
                event.preventDefault()
                void runAction(() => api.setModel(customModel), `Model set to ${customModel}.`)
              }}
            >
              <input value={customModel} onChange={(event) => setCustomModel(event.target.value)} />
              <button className="button" type="submit">Set Custom Model</button>
            </form>
          </section>
        ) : null}
      </main>
    </div>
  )
}

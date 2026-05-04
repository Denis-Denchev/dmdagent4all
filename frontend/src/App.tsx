import { useEffect, useMemo, useState } from 'react'
import {
  type AgentResponse,
  type Approval,
  type AuditEvent,
  api,
  type ConnectorStatus,
  type DoctorResponse,
  type MemoryFile,
  type ModelMode,
  type PermissionItem,
  type Status,
  type TelegramStatus,
  type TerminalStatus,
  type Tool,
} from './api'

type View =
  | 'chat'
  | 'doctor'
  | 'connectors'
  | 'approvals'
  | 'tools'
  | 'permissions'
  | 'terminal'
  | 'telegram'
  | 'memory'
  | 'audit'
  | 'models'

type ChatMessage = {
  role: 'user' | 'agent'
  text: string
  data?: unknown
}

const views: Array<{ key: View; label: string }> = [
  { key: 'chat', label: 'Chat' },
  { key: 'doctor', label: 'Doctor' },
  { key: 'connectors', label: 'Connectors' },
  { key: 'approvals', label: 'Approvals' },
  { key: 'tools', label: 'Tools' },
  { key: 'permissions', label: 'Permissions' },
  { key: 'terminal', label: 'Terminal' },
  { key: 'telegram', label: 'Telegram' },
  { key: 'memory', label: 'Memory' },
  { key: 'audit', label: 'Audit' },
  { key: 'models', label: 'Models' },
]

export function App() {
  const [activeView, setActiveView] = useState<View>('chat')
  const [status, setStatus] = useState<Status | null>(null)
  const [tools, setTools] = useState<Tool[]>([])
  const [permissions, setPermissions] = useState<PermissionItem[]>([])
  const [connectors, setConnectors] = useState<ConnectorStatus[]>([])
  const [terminal, setTerminal] = useState<TerminalStatus | null>(null)
  const [telegram, setTelegram] = useState<TelegramStatus | null>(null)
  const [approvals, setApprovals] = useState<Approval[]>([])
  const [audit, setAudit] = useState<AuditEvent[]>([])
  const [doctor, setDoctor] = useState<DoctorResponse | null>(null)
  const [memoryFiles, setMemoryFiles] = useState<string[]>([])
  const [selectedMemory, setSelectedMemory] = useState<MemoryFile | null>(null)
  const [memoryDraft, setMemoryDraft] = useState('')
  const [models, setModels] = useState<ModelMode[]>([])
  const [customModel, setCustomModel] = useState('')
  const [terminalCommand, setTerminalCommand] = useState('')
  const [terminalRunCommand, setTerminalRunCommand] = useState('')
  const [terminalCwd, setTerminalCwd] = useState('')
  const [terminalTimeout, setTerminalTimeout] = useState('')
  const [terminalMaxOutput, setTerminalMaxOutput] = useState('')
  const [terminalAutoApprove, setTerminalAutoApprove] = useState(false)
  const [telegramUserId, setTelegramUserId] = useState('')
  const [telegramTokenEnv, setTelegramTokenEnv] = useState('')
  const [telegramToken, setTelegramToken] = useState('')
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
      connectorsResult,
      terminalResult,
      telegramResult,
      approvalsResult,
      auditResult,
      doctorResult,
      memoryResult,
      modelsResult,
    ] =
      await Promise.all([
        api.status(),
        api.tools(),
        api.permissions(),
        api.connectors(),
        api.terminal(),
        api.telegram(),
        api.approvals(),
        api.audit(),
        api.doctor(),
        api.memory(),
        api.models(),
      ])
    setStatus(statusResult)
    setTools(toolsResult)
    setPermissions(permissionsResult.available)
    setConnectors(connectorsResult)
    setTerminal(terminalResult)
    setTelegram(telegramResult)
    setTerminalTimeout(String(terminalResult.timeout_seconds))
    setTerminalMaxOutput(String(terminalResult.max_output_chars))
    setTerminalAutoApprove(terminalResult.auto_approve_allowlisted)
    setTelegramTokenEnv(telegramResult.bot_token_env)
    setApprovals(approvalsResult)
    setAudit(auditResult)
    setDoctor(doctorResult)
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

  function appendAgentResponse(response: AgentResponse, focusChat = false) {
    setMessages((current) => [
      ...current,
      { role: 'agent', text: `[${response.status}] ${response.message}`, data: response.data },
    ])
    if (focusChat) setActiveView('chat')
  }

  async function runAgentAction(
    action: () => Promise<AgentResponse>,
    options: { focusChat?: boolean } = {},
  ) {
    setBusy(true)
    setNotice(null)
    try {
      const response = await action()
      appendAgentResponse(response, options.focusChat ?? false)
      setNotice(`${response.status}: ${response.message}`)
      await refreshAll()
      return response
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Request failed'
      setNotice(message)
      setMessages((current) => [...current, { role: 'agent', text: message }])
      return null
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
      appendAgentResponse(response)
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
    await runAgentAction(() => api.writeMemory(selectedMemory.path, memoryDraft))
  }

  async function saveTerminalSettings() {
    await runAction(
      () =>
        api.updateTerminalSettings({
          workspace_only: terminal?.workspace_only ?? true,
          timeout_seconds: Number(terminalTimeout),
          max_output_chars: Number(terminalMaxOutput),
          auto_approve_allowlisted: terminalAutoApprove,
        }),
      'Terminal settings updated.',
    )
  }

  async function approveRequest(approvalId: number) {
    await runAgentAction(() => api.approve(approvalId), { focusChat: true })
  }

  async function denyRequest(approvalId: number) {
    await runAgentAction(() => api.deny(approvalId))
  }

  async function allowTerminalCommand() {
    const command = terminalCommand.trim()
    if (!command) return
    await runAction(() => api.allowTerminalCommand(command), `Allowlisted: ${command}`)
    setTerminalCommand('')
  }

  async function requestTerminalRun() {
    const command = terminalRunCommand.trim()
    if (!command) return
    const response = await runAgentAction(
      () => api.runTerminalCommand(command, terminalCwd.trim()),
      { focusChat: true },
    )
    if (response?.status === 'ok') setTerminalRunCommand('')
  }

  async function allowTelegramUser() {
    const userId = Number(telegramUserId)
    if (!Number.isInteger(userId)) return
    await runAction(() => api.allowTelegramUser(userId), `Telegram user allowed: ${userId}`)
    setTelegramUserId('')
  }

  async function saveTelegramTokenEnv() {
    const value = telegramTokenEnv.trim()
    if (!value) return
    await runAction(() => api.setTelegramTokenEnv(value), `Telegram token env set to ${value}.`)
  }

  async function loadTelegramToken() {
    const token = telegramToken.trim()
    if (!token) return
    await runAction(() => api.loadTelegramToken(token), 'Telegram token loaded into this API process.')
    setTelegramToken('')
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
          <span>Version</span>
          <strong>{status?.version ?? 'loading'}</strong>
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

        {activeView === 'doctor' ? (
          <section className="panel doctor-panel">
            <div className="doctor-summary">
              <div>
                <span>OK</span>
                <strong>{doctor?.summary.ok ?? 0}</strong>
              </div>
              <div>
                <span>Warnings</span>
                <strong>{doctor?.summary.warn ?? 0}</strong>
              </div>
              <div>
                <span>Failures</span>
                <strong>{doctor?.summary.fail ?? 0}</strong>
              </div>
            </div>
            <div className="table-list">
              {doctor?.checks.map((check, index) => (
                <article className={`doctor-row doctor-row--${check.status}`} key={`${check.area}-${index}`}>
                  <span>{check.status.toUpperCase()}</span>
                  <div>
                    <strong>{check.area}</strong>
                    <p>{check.message}</p>
                    {check.hint ? <code>{check.hint}</code> : null}
                  </div>
                </article>
              ))}
            </div>
          </section>
        ) : null}

        {activeView === 'connectors' ? (
          <section className="panel">
            <div className="connector-grid">
              {connectors.map((connector) => (
                <article className="connector-card" key={connector.name}>
                  <div className="connector-card__header">
                    <strong>{connector.name}</strong>
                    <span className={`status-chip status-chip--${connector.status}`}>
                      {connector.status}
                    </span>
                  </div>
                  {connector.detail ? <p>{connector.detail}</p> : null}
                  <div className="connector-meter">
                    <span>Tools</span>
                    <strong>
                      {connector.enabled_tools.length}/{connector.tools_total}
                    </strong>
                    <span>Permissions</span>
                    <strong>
                      {connector.permissions_granted.length}/{connector.permissions_required.length}
                    </strong>
                  </div>
                  {connector.enabled_tools.length ? (
                    <code>{connector.enabled_tools.join(', ')}</code>
                  ) : null}
                </article>
              ))}
            </div>
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
                      onClick={() => void approveRequest(approval.id)}
                    >
                      Approve
                    </button>
                    <button
                      className="button button-danger"
                      type="button"
                      onClick={() => void denyRequest(approval.id)}
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

        {activeView === 'terminal' ? (
          <section className="terminal-layout">
            <div className="panel settings-panel">
              <div className="settings-header">
                <div>
                  <strong>Terminal Policy</strong>
                  <span>{terminal?.ready ? 'Ready for approval-gated runs' : 'Not ready'}</span>
                </div>
                <button
                  className={terminal?.enabled ? 'button button-danger' : 'button'}
                  type="button"
                  onClick={() =>
                    void runAction(
                      () => (terminal?.enabled ? api.disableTerminal() : api.enableTerminal()),
                      terminal?.enabled ? 'Terminal disabled.' : 'Terminal enabled.',
                    )
                  }
                >
                  {terminal?.enabled ? 'Disable' : 'Enable'}
                </button>
              </div>
              <div className="status-grid">
                <span>Policy</span>
                <strong>{terminal?.enabled ? 'enabled' : 'disabled'}</strong>
                <span>Tool</span>
                <strong>{terminal?.tool_enabled ? 'enabled' : 'disabled'}</strong>
                <span>Permission</span>
                <strong>{terminal?.permission_granted ? 'granted' : 'missing'}</strong>
              </div>
              <div className="settings-form">
                <label>
                  Timeout seconds
                  <input
                    value={terminalTimeout}
                    onChange={(event) => setTerminalTimeout(event.target.value)}
                    inputMode="numeric"
                  />
                </label>
                <label>
                  Max output chars
                  <input
                    value={terminalMaxOutput}
                    onChange={(event) => setTerminalMaxOutput(event.target.value)}
                    inputMode="numeric"
                  />
                </label>
                <label className="setting-check">
                  <input
                    type="checkbox"
                    checked={terminalAutoApprove}
                    onChange={(event) => setTerminalAutoApprove(event.target.checked)}
                  />
                  <span>
                    <strong>Auto-approve exact allowlist</strong>
                    <small>Runs matching allowlisted commands without creating a new approval.</small>
                  </span>
                </label>
                <button className="button button-secondary" type="button" onClick={() => void saveTerminalSettings()}>
                  Save Settings
                </button>
              </div>
            </div>

            <div className="panel command-panel">
              <div className="settings-header">
                <div>
                  <strong>Exact Allowlist</strong>
                  <span>Command arrays are parsed from shell-like text.</span>
                </div>
              </div>
              <form
                className="inline-form"
                onSubmit={(event) => {
                  event.preventDefault()
                  void allowTerminalCommand()
                }}
              >
                <input
                  value={terminalCommand}
                  onChange={(event) => setTerminalCommand(event.target.value)}
                  placeholder="git status"
                />
                <button className="button" type="submit">Allow</button>
              </form>
              <div className="table-list table-list--compact">
                {terminal?.allowed_commands.map((command) => (
                  <article className="command-row" key={command.join('\u0000')}>
                    <code>{command.join(' ')}</code>
                    <button
                      className="button button-danger"
                      type="button"
                      onClick={() =>
                        void runAction(
                          () => api.removeTerminalCommand(command),
                          `Removed: ${command.join(' ')}`,
                        )
                      }
                    >
                      Remove
                    </button>
                  </article>
                ))}
              </div>
            </div>

            <div className="panel command-panel">
              <div className="settings-header">
                <div>
                  <strong>Request Run</strong>
                  <span>Runs create an approval item unless policy blocks them.</span>
                </div>
              </div>
              <form
                className="run-form"
                onSubmit={(event) => {
                  event.preventDefault()
                  void requestTerminalRun()
                }}
              >
                <input
                  value={terminalRunCommand}
                  onChange={(event) => setTerminalRunCommand(event.target.value)}
                  placeholder="pwd"
                />
                <input
                  value={terminalCwd}
                  onChange={(event) => setTerminalCwd(event.target.value)}
                  placeholder="workspace-relative cwd"
                />
                <button className="button" type="submit" disabled={busy}>Request</button>
              </form>
            </div>
          </section>
        ) : null}

        {activeView === 'telegram' ? (
          <section className="telegram-layout">
            <div className="panel settings-panel">
              <div className="settings-header">
                <div>
                  <strong>Telegram Remote Access</strong>
                  <span>{telegram?.ready ? 'Ready for allowlisted polling' : 'Needs token, enablement, and an allowlisted user'}</span>
                </div>
                <button
                  className={telegram?.enabled ? 'button button-danger' : 'button'}
                  type="button"
                  onClick={() =>
                    void runAction(
                      () => (telegram?.enabled ? api.disableTelegram() : api.enableTelegram()),
                      telegram?.enabled ? 'Telegram disabled.' : 'Telegram enabled.',
                    )
                  }
                >
                  {telegram?.enabled ? 'Disable' : 'Enable'}
                </button>
              </div>
              <div className="status-grid">
                <span>Token env</span>
                <strong>{telegram?.bot_token_env ?? '-'}</strong>
                <span>Token loaded</span>
                <strong>{telegram?.bot_token_available ? 'yes' : 'no'}</strong>
                <span>Allowed users</span>
                <strong>{telegram?.allowed_user_ids.length ?? 0}</strong>
              </div>
            </div>

            <div className="panel command-panel">
              <div className="settings-header">
                <div>
                  <strong>Token Environment</strong>
                  <span>Token values are loaded only into the running API process.</span>
                </div>
              </div>
              <form
                className="inline-form"
                onSubmit={(event) => {
                  event.preventDefault()
                  void saveTelegramTokenEnv()
                }}
              >
                <input
                  value={telegramTokenEnv}
                  onChange={(event) => setTelegramTokenEnv(event.target.value)}
                  placeholder="DMDAGENT_TELEGRAM_BOT_TOKEN"
                />
                <button className="button button-secondary" type="submit">Save Env</button>
              </form>
              <form
                className="inline-form"
                onSubmit={(event) => {
                  event.preventDefault()
                  void loadTelegramToken()
                }}
              >
                <input
                  type="password"
                  value={telegramToken}
                  onChange={(event) => setTelegramToken(event.target.value)}
                  placeholder="BotFather token"
                />
                <button className="button" type="submit">Load Token</button>
              </form>
            </div>

            <div className="panel command-panel">
              <div className="settings-header">
                <div>
                  <strong>Allowed Users</strong>
                  <span>Send /id to the bot to discover the numeric user ID.</span>
                </div>
              </div>
              <form
                className="inline-form"
                onSubmit={(event) => {
                  event.preventDefault()
                  void allowTelegramUser()
                }}
              >
                <input
                  value={telegramUserId}
                  onChange={(event) => setTelegramUserId(event.target.value)}
                  placeholder="123456789"
                  inputMode="numeric"
                />
                <button className="button" type="submit">Allow User</button>
              </form>
              <div className="table-list table-list--compact">
                {telegram?.allowed_user_ids.length ? null : <p className="empty">No allowed Telegram users.</p>}
                {telegram?.allowed_user_ids.map((userId) => (
                  <article className="command-row" key={userId}>
                    <code>{userId}</code>
                    <button
                      className="button button-danger"
                      type="button"
                      onClick={() =>
                        void runAction(
                          () => api.removeTelegramUser(userId),
                          `Removed Telegram user: ${userId}`,
                        )
                      }
                    >
                      Remove
                    </button>
                  </article>
                ))}
              </div>
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

import { type KeyboardEvent, useEffect, useMemo, useRef, useState } from 'react'
import {
  type AgentConfiguration,
  type AgentResponse,
  type Approval,
  type AuditEvent,
  api,
  type ConnectorStatus,
  type DeepSeekStatus,
  type DoctorResponse,
  type MemoryFile,
  type ModelMode,
  type OpenAIStatus,
  type PermissionItem,
  type Status,
  type TelegramStatus,
  type TerminalStatus,
  type Tool,
  type WorkspaceFile,
  type WorkspaceFileContent,
  type WorkspaceFilesResponse,
} from './api'

type View = 'chat' | 'downloads' | 'models' | 'config' | 'telegram' | 'memory' | 'tools' | 'logs' | 'help'

type ChatMessage = {
  id: string
  role: 'user' | 'agent' | 'system'
  text: string
  data?: unknown
  response?: AgentResponse
}

type TourStep = {
  id: string
  view: View
  title: string
  body: string
}

type TourRect = {
  top: number
  left: number
  width: number
  height: number
}

type ConfigDraft = {
  downloadsRoot: string
  agentName: string
  userName: string
  preferredLanguage: string
  responseLanguage: string
  plannerMaxTokens: string
  plannerTemperature: string
  plannerThink: boolean
  plannerSystemPrompt: string
  answerSystemPrompt: string
  sendChatHistoryToCloud: boolean
  browserTimeout: string
  browserMaxResponseBytes: string
  browserMaxTextChars: string
  approvalRisk: string
}

const views: Array<{ key: View; label: string; short: string; description: string }> = [
  { key: 'chat', label: 'Chat', short: 'CH', description: 'Work with the agent' },
  { key: 'downloads', label: 'Downloads', short: 'DL', description: 'Files from web work' },
  { key: 'models', label: 'Models', short: 'MD', description: 'Runtime and token controls' },
  { key: 'config', label: 'Config', short: 'CF', description: 'Agent settings and paths' },
  { key: 'telegram', label: 'Telegram', short: 'TG', description: 'Remote access' },
  { key: 'memory', label: 'Memory', short: 'MY', description: 'Local long-term context' },
  { key: 'tools', label: 'Tools', short: 'TL', description: 'Permissions and connectors' },
  { key: 'logs', label: 'Logs', short: 'LG', description: 'Audit trail' },
  { key: 'help', label: 'Help', short: 'HP', description: 'Manual and guide' },
]

const tourSteps: TourStep[] = [
  {
    id: 'tour-chat',
    view: 'chat',
    title: 'Chat workspace',
    body: 'This is the main control surface. Ask the agent to plan, scrape, use tools, remember facts, or run approved local actions.',
  },
  {
    id: 'tour-approvals',
    view: 'chat',
    title: 'Approvals in context',
    body: 'Risky or cloud-context actions appear beside the chat and inside the chat thread. Approve or deny without leaving the conversation.',
  },
  {
    id: 'tour-downloads',
    view: 'downloads',
    title: 'Internet files',
    body: 'Scraped Markdown and browser downloads are collected here. The file reader is limited to approved download folders.',
  },
  {
    id: 'tour-models',
    view: 'models',
    title: 'Model cockpit',
    body: 'Local modes, OpenAI, DeepSeek, API keys, model lists, and token usage live in one place.',
  },
  {
    id: 'tour-config',
    view: 'config',
    title: 'Configuration deck',
    body: 'Change agent identity, planner limits, browser scrape limits, terminal policy, approval threshold, and download storage paths.',
  },
  {
    id: 'tour-logs',
    view: 'logs',
    title: 'Action log',
    body: 'The log shows what the bot did: tool calls, approvals, notifications, and recorded system events.',
  },
  {
    id: 'tour-help',
    view: 'help',
    title: 'Manual on demand',
    body: 'The Help page keeps the full guide after the first-run tour is completed or skipped.',
  },
]

function formatUsd(value: number | null | undefined, digits = 2) {
  return value === null || value === undefined ? '-' : `$${value.toFixed(digits)}`
}

function formatBytes(value: number) {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / 1024 / 1024).toFixed(1)} MB`
}

function formatDate(value: string) {
  if (!value) return '-'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

function dataObject(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : {}
}

function approvalIdFromResponse(response?: AgentResponse) {
  if (response?.status !== 'approval_required') return null
  const approvalId = dataObject(response?.data).approval_id
  return typeof approvalId === 'number' ? approvalId : null
}

function initialConfigDraft(): ConfigDraft {
  return {
    downloadsRoot: '',
    agentName: '',
    userName: '',
    preferredLanguage: 'auto',
    responseLanguage: 'auto',
    plannerMaxTokens: '192',
    plannerTemperature: '0',
    plannerThink: false,
    plannerSystemPrompt: '',
    answerSystemPrompt: '',
    sendChatHistoryToCloud: false,
    browserTimeout: '15',
    browserMaxResponseBytes: '1000000',
    browserMaxTextChars: '12000',
    approvalRisk: '3',
  }
}

export function App() {
  const [activeView, setActiveView] = useState<View>('chat')
  const [railCollapsed, setRailCollapsed] = useState(false)
  const chatLogRef = useRef<HTMLDivElement | null>(null)
  const chatEndRef = useRef<HTMLDivElement | null>(null)
  const [status, setStatus] = useState<Status | null>(null)
  const [configuration, setConfiguration] = useState<AgentConfiguration | null>(null)
  const [tools, setTools] = useState<Tool[]>([])
  const [permissions, setPermissions] = useState<PermissionItem[]>([])
  const [connectors, setConnectors] = useState<ConnectorStatus[]>([])
  const [terminal, setTerminal] = useState<TerminalStatus | null>(null)
  const [telegram, setTelegram] = useState<TelegramStatus | null>(null)
  const [openai, setOpenAI] = useState<OpenAIStatus | null>(null)
  const [deepseek, setDeepSeek] = useState<DeepSeekStatus | null>(null)
  const [approvals, setApprovals] = useState<Approval[]>([])
  const [audit, setAudit] = useState<AuditEvent[]>([])
  const [doctor, setDoctor] = useState<DoctorResponse | null>(null)
  const [memoryFiles, setMemoryFiles] = useState<string[]>([])
  const [selectedMemory, setSelectedMemory] = useState<MemoryFile | null>(null)
  const [memoryDraft, setMemoryDraft] = useState('')
  const [workspaceFiles, setWorkspaceFiles] = useState<WorkspaceFilesResponse | null>(null)
  const [selectedWorkspaceFile, setSelectedWorkspaceFile] = useState<WorkspaceFileContent | null>(null)
  const [models, setModels] = useState<ModelMode[]>([])
  const [customModel, setCustomModel] = useState('')
  const [configDraft, setConfigDraft] = useState<ConfigDraft>(initialConfigDraft)
  const [terminalCommand, setTerminalCommand] = useState('')
  const [terminalRunCommand, setTerminalRunCommand] = useState('')
  const [terminalCwd, setTerminalCwd] = useState('')
  const [terminalWorkspaceRoot, setTerminalWorkspaceRoot] = useState('')
  const [terminalTimeout, setTerminalTimeout] = useState('')
  const [terminalMaxOutput, setTerminalMaxOutput] = useState('')
  const [terminalAutoApprove, setTerminalAutoApprove] = useState(false)
  const [telegramUserId, setTelegramUserId] = useState('')
  const [telegramTokenEnv, setTelegramTokenEnv] = useState('')
  const [telegramToken, setTelegramToken] = useState('')
  const [openaiKey, setOpenAIKey] = useState('')
  const [openaiModels, setOpenAIModels] = useState<string[]>([])
  const [openaiSelectedModel, setOpenAISelectedModel] = useState('')
  const [openaiLimitDraft, setOpenAILimitDraft] = useState('')
  const [deepseekKey, setDeepSeekKey] = useState('')
  const [deepseekModels, setDeepSeekModels] = useState<string[]>([])
  const [deepseekSelectedModel, setDeepSeekSelectedModel] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'system',
      text: 'Control center ready. Chat, tools, downloads, approvals, models, memory, Telegram, logs, and configuration are available from the command rail.',
    },
  ])
  const [chatInput, setChatInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [tourOpen, setTourOpen] = useState(false)
  const [tourStep, setTourStep] = useState(0)
  const [tourRect, setTourRect] = useState<TourRect | null>(null)

  const pendingCount = approvals.length
  const enabledToolCount = useMemo(() => tools.filter((tool) => tool.enabled).length, [tools])
  const activeLabel = views.find((view) => view.key === activeView)?.label ?? 'Dashboard'

  const openaiModelOptions = useMemo(() => {
    const options = new Set(openaiModels)
    if (openai?.provider === 'openai' && openai.model) options.add(openai.model)
    if (openaiSelectedModel) options.add(openaiSelectedModel)
    return Array.from(options).sort((left, right) => left.localeCompare(right))
  }, [openai?.model, openai?.provider, openaiModels, openaiSelectedModel])

  const deepseekModelOptions = useMemo(() => {
    const options = new Set(deepseek?.default_models ?? ['deepseek-v4-flash', 'deepseek-v4-pro'])
    deepseekModels.forEach((model) => options.add(model))
    if (deepseek?.provider === 'deepseek' && deepseek.model) options.add(deepseek.model)
    if (deepseekSelectedModel) options.add(deepseekSelectedModel)
    return Array.from(options).sort((left, right) => left.localeCompare(right))
  }, [deepseek?.default_models, deepseek?.model, deepseek?.provider, deepseekModels, deepseekSelectedModel])

  useEffect(() => {
    void refreshAll()
    if (localStorage.getItem('dmdagent4all:tutorial-complete') !== '1') {
      setTourOpen(true)
      setTourStep(0)
      setActiveView(tourSteps[0].view)
    }
  }, [])

  useEffect(() => {
    if (!tourOpen) return
    const step = tourSteps[tourStep]
    if (step.view !== activeView) {
      setActiveView(step.view)
      return
    }
    const update = () => {
      const target = document.querySelector(`[data-tour="${step.id}"]`)
      if (!target) {
        setTourRect(null)
        return
      }
      const rect = target.getBoundingClientRect()
      setTourRect({
        top: rect.top,
        left: rect.left,
        width: rect.width,
        height: rect.height,
      })
    }
    const frame = window.requestAnimationFrame(update)
    window.addEventListener('resize', update)
    return () => {
      window.cancelAnimationFrame(frame)
      window.removeEventListener('resize', update)
    }
  }, [activeView, tourOpen, tourStep])

  useEffect(() => {
    if (activeView !== 'chat') return
    const frame = window.requestAnimationFrame(() => {
      const chatLog = chatLogRef.current
      if (chatLog) {
        chatLog.scrollTop = chatLog.scrollHeight
      }
      chatEndRef.current?.scrollIntoView({ block: 'end' })
    })
    return () => window.cancelAnimationFrame(frame)
  }, [activeView, busy, messages.length])

  async function refreshAll() {
    const [
      statusResult,
      toolsResult,
      permissionsResult,
      connectorsResult,
      terminalResult,
      telegramResult,
      openaiResult,
      deepseekResult,
      approvalsResult,
      auditResult,
      doctorResult,
      configurationResult,
      memoryResult,
      workspaceFilesResult,
      modelsResult,
    ] = await Promise.all([
      api.status(),
      api.tools(),
      api.permissions(),
      api.connectors(),
      api.terminal(),
      api.telegram(),
      api.openai(),
      api.deepseek(),
      api.approvals(),
      api.audit(),
      api.doctor(),
      api.configuration(),
      api.memory(),
      api.workspaceFiles(),
      api.models(),
    ])
    setStatus(statusResult)
    setTools(toolsResult)
    setPermissions(permissionsResult.available)
    setConnectors(connectorsResult)
    setTerminal(terminalResult)
    setTelegram(telegramResult)
    setOpenAI(openaiResult)
    setDeepSeek(deepseekResult)
    setApprovals(approvalsResult)
    setAudit(auditResult)
    setDoctor(doctorResult)
    setConfiguration(configurationResult)
    setMemoryFiles(memoryResult.files)
    setWorkspaceFiles(workspaceFilesResult)
    setModels(modelsResult.modes)
    setCustomModel(modelsResult.current.model)
    setTerminalWorkspaceRoot(terminalResult.workspace_root)
    setTerminalTimeout(String(terminalResult.timeout_seconds))
    setTerminalMaxOutput(String(terminalResult.max_output_chars))
    setTerminalAutoApprove(terminalResult.auto_approve_allowlisted)
    setTelegramTokenEnv(telegramResult.bot_token_env)
    setOpenAISelectedModel(openaiResult.provider === 'openai' ? openaiResult.model : openaiSelectedModel)
    setOpenAILimitDraft(openaiResult.usage.limit_usd === null ? '' : String(openaiResult.usage.limit_usd))
    setDeepSeekSelectedModel(
      deepseekResult.provider === 'deepseek'
        ? deepseekResult.model
        : deepseekResult.default_models[0] ?? 'deepseek-v4-flash',
    )
    setConfigDraft({
      downloadsRoot: String(configurationResult.storage.downloads_root ?? ''),
      agentName: String(configurationResult.setup.agent_name ?? 'DMD Agent'),
      userName: String(configurationResult.setup.user_name ?? ''),
      preferredLanguage: String(configurationResult.setup.preferred_language ?? 'auto'),
      responseLanguage: String(configurationResult.llm.response_language ?? 'auto'),
      plannerMaxTokens: String(configurationResult.llm.planner_max_tokens ?? 192),
      plannerTemperature: String(configurationResult.llm.planner_temperature ?? 0),
      plannerThink: Boolean(configurationResult.llm.planner_think),
      plannerSystemPrompt: configurationResult.system_prompts.planner.effective,
      answerSystemPrompt: configurationResult.system_prompts.answer.effective,
      sendChatHistoryToCloud: Boolean(configurationResult.privacy.send_chat_history_to_cloud),
      browserTimeout: String(configurationResult.browser.timeout_seconds ?? 15),
      browserMaxResponseBytes: String(configurationResult.browser.max_response_bytes ?? 1000000),
      browserMaxTextChars: String(configurationResult.browser.max_text_chars ?? 12000),
      approvalRisk: String(configurationResult.permissions.approval_required_at_risk ?? 3),
    })
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
      {
        id: `${Date.now()}-${current.length}`,
        role: 'agent',
        text: response.message,
        data: response.data,
        response,
      },
    ])
    if (focusChat) setActiveView('chat')
  }

  async function runAgentAction(action: () => Promise<AgentResponse>, options: { focusChat?: boolean } = {}) {
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
      setMessages((current) => [...current, { id: `${Date.now()}-error`, role: 'agent', text: message }])
      return null
    } finally {
      setBusy(false)
    }
  }

  async function sendChat() {
    const message = chatInput.trim()
    if (!message) return
    setChatInput('')
    setMessages((current) => [...current, { id: `${Date.now()}-user`, role: 'user', text: message }])
    setBusy(true)
    try {
      const response = await api.chat(message, 'dashboard')
      appendAgentResponse(response)
      await refreshAll()
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: `${Date.now()}-error`,
          role: 'agent',
          text: error instanceof Error ? error.message : 'Request failed',
        },
      ])
    } finally {
      setBusy(false)
    }
  }

  function handleChatKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== 'Enter' || event.shiftKey) return
    event.preventDefault()
    if (!busy && chatInput.trim()) {
      void sendChat()
    }
  }

  async function approveRequest(approvalId: number) {
    await runAgentAction(() => api.approve(approvalId), { focusChat: true })
  }

  async function denyRequest(approvalId: number) {
    await runAgentAction(() => api.deny(approvalId), { focusChat: true })
  }

  async function loadWorkspaceFile(file: WorkspaceFile) {
    if (!file.previewable) {
      setSelectedWorkspaceFile({ ...file, content: '', truncated: false })
      return
    }
    setSelectedWorkspaceFile(await api.workspaceFile(file.path))
  }

  async function loadMemoryFile(path: string) {
    const file = await api.memoryFile(path)
    setSelectedMemory(file)
    setMemoryDraft(file.content)
  }

  async function requestMemorySave() {
    if (!selectedMemory) return
    await runAgentAction(() => api.writeMemory(selectedMemory.path, memoryDraft), { focusChat: true })
  }

  async function saveTerminalSettings() {
    await runAction(
      () =>
        api.updateTerminalSettings({
          workspace_only: terminal?.workspace_only ?? true,
          workspace_root: terminalWorkspaceRoot,
          timeout_seconds: Number(terminalTimeout),
          max_output_chars: Number(terminalMaxOutput),
          auto_approve_allowlisted: terminalAutoApprove,
        }),
      'Terminal settings updated.',
    )
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
    const response = await runAgentAction(() => api.runTerminalCommand(command, terminalCwd.trim()), { focusChat: true })
    if (response?.status === 'ok') setTerminalRunCommand('')
  }

  async function saveConfiguration() {
    const toNumber = (value: string) => Number(value.trim())
    await runAction(
      () =>
        api.updateConfiguration({
          downloads_root: configDraft.downloadsRoot,
          agent_name: configDraft.agentName,
          user_name: configDraft.userName,
          preferred_language: configDraft.preferredLanguage,
          response_language: configDraft.responseLanguage,
          planner_max_tokens: toNumber(configDraft.plannerMaxTokens),
          planner_temperature: toNumber(configDraft.plannerTemperature),
          planner_think: configDraft.plannerThink,
          planner_system_prompt: configDraft.plannerSystemPrompt,
          answer_system_prompt: configDraft.answerSystemPrompt,
          send_chat_history_to_cloud: configDraft.sendChatHistoryToCloud,
          browser_timeout_seconds: toNumber(configDraft.browserTimeout),
          browser_max_response_bytes: toNumber(configDraft.browserMaxResponseBytes),
          browser_max_text_chars: toNumber(configDraft.browserMaxTextChars),
          approval_required_at_risk: toNumber(configDraft.approvalRisk),
        }),
      'Configuration saved.',
    )
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
    await runAction(() => api.loadTelegramToken(token), 'Telegram token loaded.')
    setTelegramToken('')
  }

  async function toggleTelegramPolling() {
    await runAction(
      () => (telegram?.polling ? api.stopTelegram() : api.startTelegram()),
      telegram?.polling ? 'Telegram polling stopped.' : 'Telegram polling started.',
    )
  }

  async function loadOpenAIKey() {
    const key = openaiKey.trim()
    if (!key) return
    await runAction(() => api.loadOpenAIKey(key), 'OpenAI API key loaded for this process.')
    setOpenAIKey('')
  }

  async function refreshOpenAIModels() {
    setBusy(true)
    setNotice(null)
    try {
      const response = await api.openAIModels()
      setOpenAIModels(response.models)
      setOpenAI(response.data)
      if (!openaiSelectedModel && response.models.length > 0) setOpenAISelectedModel(response.models[0])
      setNotice(`Loaded ${response.models.length} OpenAI models.`)
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Request failed')
    } finally {
      setBusy(false)
    }
  }

  async function saveOpenAIModel() {
    const model = openaiSelectedModel.trim()
    if (!model) return
    await runAction(() => api.setOpenAIModel(model), `OpenAI model set to ${model}.`)
  }

  async function saveOpenAILimit() {
    const trimmed = openaiLimitDraft.trim()
    const limit = trimmed ? Number(trimmed) : null
    if (limit !== null && (!Number.isFinite(limit) || limit < 0)) {
      setNotice('OpenAI limit must be empty or a positive number.')
      return
    }
    await runAction(() => api.setOpenAILimit(limit), 'OpenAI local spending limit updated.')
  }

  async function resetOpenAIUsage() {
    await runAction(() => api.resetOpenAIUsage(), 'OpenAI local usage counters reset.')
  }

  async function loadDeepSeekKey() {
    const key = deepseekKey.trim()
    if (!key) return
    await runAction(() => api.loadDeepSeekKey(key), 'DeepSeek API key loaded for this process.')
    setDeepSeekKey('')
  }

  async function refreshDeepSeekModels() {
    setBusy(true)
    setNotice(null)
    try {
      const response = await api.deepSeekModels()
      setDeepSeekModels(response.models)
      setDeepSeek(response.data)
      if (!deepseekSelectedModel && response.models.length > 0) setDeepSeekSelectedModel(response.models[0])
      setNotice(`Loaded ${response.models.length} DeepSeek models.`)
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Request failed')
    } finally {
      setBusy(false)
    }
  }

  async function saveDeepSeekModel() {
    const model = deepseekSelectedModel.trim()
    if (!model) return
    await runAction(() => api.setDeepSeekModel(model), `DeepSeek model set to ${model}.`)
  }

  function finishTour() {
    localStorage.setItem('dmdagent4all:tutorial-complete', '1')
    setTourOpen(false)
  }

  function goTourStep(nextStep: number) {
    if (nextStep < 0) return
    if (nextStep >= tourSteps.length) {
      finishTour()
      return
    }
    setTourStep(nextStep)
    setActiveView(tourSteps[nextStep].view)
  }

  const currentTour = tourSteps[tourStep]
  const tourBubbleStyle = tourRect
    ? {
        left: Math.min(Math.max(tourRect.left, 18), Math.max(window.innerWidth - 390, 18)),
        top:
          tourRect.top + tourRect.height + 18 > window.innerHeight - 260
            ? Math.max(18, tourRect.top - 250)
            : tourRect.top + tourRect.height + 18,
      }
    : { left: 24, top: 96 }

  return (
    <div className={railCollapsed ? 'app-shell app-shell--rail-collapsed' : 'app-shell'}>
      <aside className="command-rail">
        <button
          className="rail-toggle"
          type="button"
          onClick={() => setRailCollapsed(!railCollapsed)}
          aria-label={railCollapsed ? 'Expand navigation' : 'Collapse navigation'}
          aria-expanded={!railCollapsed}
        >
          <span />
          <span />
          <span />
        </button>
        <div className="brand" data-tour="tour-help">
          <div className="brand-mark">DMD</div>
          <div className="brand-copy">
            <strong>DMD Agent</strong>
            <span>local command deck</span>
          </div>
        </div>

        <nav className="nav-list" aria-label="Dashboard views">
          {views.map((view) => (
            <button
              key={view.key}
              className={activeView === view.key ? 'nav-item nav-item--active' : 'nav-item'}
              type="button"
              onClick={() => setActiveView(view.key)}
              data-tour={view.key === 'downloads' ? 'tour-downloads' : view.key === 'models' ? 'tour-models' : view.key === 'config' ? 'tour-config' : view.key === 'logs' ? 'tour-logs' : undefined}
            >
              <span className="nav-code">{view.short}</span>
              <span className="nav-copy">
                <strong>{view.label}</strong>
                <small>{view.description}</small>
              </span>
              {view.key === 'chat' && pendingCount > 0 ? <b>{pendingCount}</b> : null}
            </button>
          ))}
        </nav>

        <div className="rail-status">
          <span>Runtime</span>
          <strong>{status?.llm.provider ?? 'loading'} / {status?.llm.model ?? '-'}</strong>
          <span>Enabled tools</span>
          <strong>{enabledToolCount}</strong>
          <span>Pending approvals</span>
          <strong>{pendingCount}</strong>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Local-first agent cockpit</p>
            <h1>{activeLabel}</h1>
          </div>
          <div className="topbar-actions">
            <button className="button button-secondary" type="button" onClick={() => setTourOpen(true)}>
              Start Tour
            </button>
            <button className="button button-secondary" type="button" onClick={() => void refreshAll()}>
              Refresh
            </button>
          </div>
        </header>

        {notice ? <div className="notice">{notice}</div> : null}

        {activeView === 'chat' ? (
          <section className="chat-workspace" data-tour="tour-chat">
            <div className="chat-main panel">
              <div className="chat-log" ref={chatLogRef}>
                {messages.map((message) => (
                  <article key={message.id} className={`message message--${message.role}`}>
                    <div className="message-avatar">{message.role === 'user' ? 'YOU' : message.role === 'system' ? 'SYS' : 'AI'}</div>
                    <div className="message-body">
                      <p>{message.text}</p>
                      {approvalIdFromResponse(message.response) !== null ? (
                        <div className="approval-inline">
                          <span>Approval required for {String(dataObject(message.response?.data).tool ?? 'tool')}</span>
                          <div className="row-actions">
                            <button className="button" type="button" onClick={() => void approveRequest(approvalIdFromResponse(message.response) ?? 0)}>
                              Approve
                            </button>
                            <button className="button button-danger" type="button" onClick={() => void denyRequest(approvalIdFromResponse(message.response) ?? 0)}>
                              Deny
                            </button>
                          </div>
                        </div>
                      ) : null}
                      {message.data ? <pre>{JSON.stringify(message.data, null, 2)}</pre> : null}
                    </div>
                  </article>
                ))}
                <div ref={chatEndRef} aria-hidden="true" />
              </div>
              <form
                className="chat-composer"
                onSubmit={(event) => {
                  event.preventDefault()
                  void sendChat()
                }}
              >
                <textarea
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  onKeyDown={handleChatKeyDown}
                  placeholder="Message the agent. Example: scrape fibank.bg and return Markdown"
                  rows={3}
                />
                <button className="button" type="submit" disabled={busy || !chatInput.trim()}>
                  Send
                </button>
              </form>
            </div>
            <aside className="panel chat-side" data-tour="tour-approvals">
              <div className="section-heading">
                <strong>Pending Approvals</strong>
                <span>{pendingCount} waiting</span>
              </div>
              {approvals.length === 0 ? <p className="empty">No pending approvals.</p> : null}
              {approvals.map((approval) => (
                <article className="approval-card" key={approval.id}>
                  <strong>#{approval.id} {approval.tool}</strong>
                  <span>Risk {approval.risk ?? '-'} - {approval.reason ?? approval.decision_reason ?? 'Needs approval'}</span>
                  <code>{JSON.stringify(approval.args)}</code>
                  <div className="row-actions">
                    <button className="button" type="button" onClick={() => void approveRequest(approval.id)}>Approve</button>
                    <button className="button button-danger" type="button" onClick={() => void denyRequest(approval.id)}>Deny</button>
                  </div>
                </article>
              ))}
            </aside>
          </section>
        ) : null}

        {activeView === 'downloads' ? (
          <section className="split-view">
            <div className="panel file-list">
              <div className="file-list-header">
                <strong>Internet Files</strong>
                <span>{workspaceFiles?.files.length ?? 0} files</span>
              </div>
              {workspaceFiles?.roots.map((root) => (
                <div className="folder-row" key={root.name}>
                  <span>{root.label}</span>
                  <strong>{root.count}</strong>
                </div>
              ))}
              {workspaceFiles?.files.length ? null : <p className="empty">No files yet.</p>}
              {workspaceFiles?.files.map((file) => (
                <button
                  key={file.path}
                  className={selectedWorkspaceFile?.path === file.path ? 'file-list-item file-list-item--active' : 'file-list-item'}
                  type="button"
                  onClick={() => void loadWorkspaceFile(file)}
                >
                  <strong>{file.name}</strong>
                  <span>{file.label} - {formatBytes(file.size)}</span>
                </button>
              ))}
            </div>
            <div className="panel editor-panel">
              {selectedWorkspaceFile ? (
                <>
                  <div className="editor-header">
                    <div>
                      <strong>{selectedWorkspaceFile.path}</strong>
                      <span>{selectedWorkspaceFile.content_type} - {formatBytes(selectedWorkspaceFile.size)}</span>
                    </div>
                    <a className="button button-secondary" href={api.workspaceFileDownloadUrl(selectedWorkspaceFile.path)}>
                      Download
                    </a>
                  </div>
                  {selectedWorkspaceFile.previewable ? (
                    <textarea
                      value={selectedWorkspaceFile.truncated ? `${selectedWorkspaceFile.content}\n\n[preview truncated]` : selectedWorkspaceFile.content}
                      readOnly
                      spellCheck={false}
                    />
                  ) : (
                    <p className="empty">Preview unavailable for this file type.</p>
                  )}
                </>
              ) : (
                <p className="empty">Select a file.</p>
              )}
            </div>
          </section>
        ) : null}

        {activeView === 'models' ? (
          <section className="model-console" data-tour="tour-models">
            <div className="panel command-panel">
              <div className="section-heading">
                <strong>Current Runtime</strong>
                <span>{status?.llm.provider ?? '-'} provider</span>
              </div>
              <div className="metric-grid">
                <div><span>Mode</span><strong>{status?.llm.mode ?? '-'}</strong></div>
                <div><span>Model</span><strong>{status?.llm.model ?? '-'}</strong></div>
                <div><span>Planner</span><strong>{status?.llm.planner_model ?? status?.llm.model ?? '-'}</strong></div>
                <div><span>Language</span><strong>{status?.llm.response_language ?? 'auto'}</strong></div>
              </div>
            </div>

            <div className="panel command-panel">
              <div className="section-heading">
                <strong>Local Modes</strong>
                <span>Ollama/local profiles</span>
              </div>
              <div className="models-grid">
                {models.map((mode) => (
                  <article className="model-card" key={mode.key}>
                    <strong>{mode.label}</strong>
                    <code>{mode.default_model}</code>
                    <span>{mode.description}</span>
                    <button className="button button-secondary" type="button" onClick={() => void runAction(() => api.setModelMode(mode.key), `${mode.label} selected.`)}>
                      Use Mode
                    </button>
                  </article>
                ))}
              </div>
              <form
                className="inline-form"
                onSubmit={(event) => {
                  event.preventDefault()
                  void runAction(() => api.setModel(customModel), `Model set to ${customModel}.`)
                }}
              >
                <input value={customModel} onChange={(event) => setCustomModel(event.target.value)} />
                <button className="button" type="submit">Set Custom Local Model</button>
              </form>
            </div>

            <div className="two-column">
              <div className="panel command-panel">
                <div className="section-heading">
                  <strong>OpenAI</strong>
                  <span>{openai?.api_key_available ? 'key loaded' : 'key not loaded'}</span>
                </div>
                <div className="status-grid">
                  <span>Model</span><strong>{openai?.model || '-'}</strong>
                  <span>Base URL</span><strong>{openai?.base_url || '-'}</strong>
                  <span>Key env</span><strong>{openai?.api_key_env || '-'}</strong>
                  <span>Estimated spend</span><strong>{formatUsd(openai?.usage.estimated_cost_usd, 4)}</strong>
                  <span>Total tokens</span><strong>{openai?.usage.total_tokens ?? 0}</strong>
                  <span>Remaining</span><strong>{formatUsd(openai?.usage.remaining_usd)}</strong>
                </div>
                <form className="inline-form" onSubmit={(event) => { event.preventDefault(); void loadOpenAIKey() }}>
                  <input type="password" value={openaiKey} onChange={(event) => setOpenAIKey(event.target.value)} placeholder="OpenAI API key" />
                  <button className="button" type="submit" disabled={busy}>Load Key</button>
                </form>
                <div className="inline-form">
                  <select value={openaiSelectedModel} onChange={(event) => setOpenAISelectedModel(event.target.value)}>
                    {openaiModelOptions.map((model) => <option value={model} key={model}>{model}</option>)}
                  </select>
                  <button className="button button-secondary" type="button" onClick={() => void refreshOpenAIModels()} disabled={busy || !openai?.api_key_available}>Load Models</button>
                  <button className="button" type="button" onClick={() => void saveOpenAIModel()} disabled={busy || !openaiSelectedModel}>Use</button>
                </div>
                <form className="inline-form" onSubmit={(event) => { event.preventDefault(); void saveOpenAILimit() }}>
                  <input value={openaiLimitDraft} onChange={(event) => setOpenAILimitDraft(event.target.value)} placeholder="USD limit, empty for none" inputMode="decimal" />
                  <button className="button button-secondary" type="submit">Save Limit</button>
                  <button className="button button-danger" type="button" onClick={() => void resetOpenAIUsage()}>Reset Usage</button>
                </form>
              </div>

              <div className="panel command-panel">
                <div className="section-heading">
                  <strong>DeepSeek</strong>
                  <span>{deepseek?.api_key_available ? 'key loaded' : 'key not loaded'}</span>
                </div>
                <div className="status-grid">
                  <span>Model</span><strong>{deepseek?.model || '-'}</strong>
                  <span>Base URL</span><strong>{deepseek?.base_url || '-'}</strong>
                  <span>Key env</span><strong>{deepseek?.api_key_env || '-'}</strong>
                  <span>Planner</span><strong>{deepseek?.planner_model || '-'}</strong>
                </div>
                <form className="inline-form" onSubmit={(event) => { event.preventDefault(); void loadDeepSeekKey() }}>
                  <input type="password" value={deepseekKey} onChange={(event) => setDeepSeekKey(event.target.value)} placeholder="DeepSeek API key" />
                  <button className="button" type="submit" disabled={busy}>Load Key</button>
                </form>
                <div className="inline-form">
                  <select value={deepseekSelectedModel} onChange={(event) => setDeepSeekSelectedModel(event.target.value)}>
                    {deepseekModelOptions.map((model) => <option value={model} key={model}>{model}</option>)}
                  </select>
                  <button className="button button-secondary" type="button" onClick={() => void refreshDeepSeekModels()} disabled={busy || !deepseek?.api_key_available}>Load Models</button>
                  <button className="button" type="button" onClick={() => void saveDeepSeekModel()} disabled={busy || !deepseekSelectedModel}>Use</button>
                </div>
              </div>
            </div>
          </section>
        ) : null}

        {activeView === 'config' ? (
          <section className="config-console" data-tour="tour-config">
            <div className="panel command-panel">
              <div className="section-heading">
                <strong>Agent Configuration</strong>
                <span>Storage, planner, browser, identity, and approvals</span>
              </div>
              <div className="settings-form">
                <label className="setting-wide">Downloads root
                  <input value={configDraft.downloadsRoot} onChange={(event) => setConfigDraft({ ...configDraft, downloadsRoot: event.target.value })} placeholder={configuration?.paths.workspace ?? '/absolute/path'} />
                </label>
                <label>Agent name
                  <input value={configDraft.agentName} onChange={(event) => setConfigDraft({ ...configDraft, agentName: event.target.value })} />
                </label>
                <label>User name
                  <input value={configDraft.userName} onChange={(event) => setConfigDraft({ ...configDraft, userName: event.target.value })} />
                </label>
                <label>Preferred language
                  <input value={configDraft.preferredLanguage} onChange={(event) => setConfigDraft({ ...configDraft, preferredLanguage: event.target.value })} />
                </label>
                <label>Response language
                  <input value={configDraft.responseLanguage} onChange={(event) => setConfigDraft({ ...configDraft, responseLanguage: event.target.value })} />
                </label>
                <label>Planner max tokens
                  <input value={configDraft.plannerMaxTokens} onChange={(event) => setConfigDraft({ ...configDraft, plannerMaxTokens: event.target.value })} inputMode="numeric" />
                </label>
                <label>Planner temperature
                  <input value={configDraft.plannerTemperature} onChange={(event) => setConfigDraft({ ...configDraft, plannerTemperature: event.target.value })} inputMode="decimal" />
                </label>
                <label>Browser timeout seconds
                  <input value={configDraft.browserTimeout} onChange={(event) => setConfigDraft({ ...configDraft, browserTimeout: event.target.value })} inputMode="numeric" />
                </label>
                <label>Max response bytes
                  <input value={configDraft.browserMaxResponseBytes} onChange={(event) => setConfigDraft({ ...configDraft, browserMaxResponseBytes: event.target.value })} inputMode="numeric" />
                </label>
                <label>Max text chars
                  <input value={configDraft.browserMaxTextChars} onChange={(event) => setConfigDraft({ ...configDraft, browserMaxTextChars: event.target.value })} inputMode="numeric" />
                </label>
                <label>Approval risk threshold
                  <input value={configDraft.approvalRisk} onChange={(event) => setConfigDraft({ ...configDraft, approvalRisk: event.target.value })} inputMode="numeric" />
                </label>
                <label className="setting-check">
                  <input type="checkbox" checked={configDraft.plannerThink} onChange={(event) => setConfigDraft({ ...configDraft, plannerThink: event.target.checked })} />
                  <span><strong>Planner think mode</strong><small>Passes the think flag to compatible local/cloud providers.</small></span>
                </label>
                <label className="setting-check">
                  <input type="checkbox" checked={configDraft.sendChatHistoryToCloud} onChange={(event) => setConfigDraft({ ...configDraft, sendChatHistoryToCloud: event.target.checked })} />
                  <span><strong>Send recent chat context to cloud models</strong><small>Allows OpenAI, DeepSeek, and compatible cloud providers to receive the recent chat window for follow-up questions.</small></span>
                </label>
                <button className="button" type="button" onClick={() => void saveConfiguration()}>Save Configuration</button>
              </div>
            </div>

            <div className="panel command-panel prompt-panel">
              <div className="section-heading">
                <strong>System Prompts</strong>
                <span>Runtime instructions sent to the selected model</span>
              </div>
              <div className="prompt-grid">
                <label>Planner system prompt
                  <textarea
                    value={configDraft.plannerSystemPrompt}
                    onChange={(event) => setConfigDraft({ ...configDraft, plannerSystemPrompt: event.target.value })}
                    spellCheck={false}
                  />
                  <span className="field-note">
                    {configuration?.system_prompts.planner.customized ? 'Custom prompt is active.' : 'Currently using the code default.'}
                  </span>
                </label>
                <label>Answer system prompt
                  <textarea
                    value={configDraft.answerSystemPrompt}
                    onChange={(event) => setConfigDraft({ ...configDraft, answerSystemPrompt: event.target.value })}
                    spellCheck={false}
                  />
                  <span className="field-note">
                    {configuration?.system_prompts.answer.customized ? 'Custom prompt is active.' : 'Currently using the code default.'}
                  </span>
                </label>
              </div>
              <div className="row-actions">
                <button className="button" type="button" onClick={() => void saveConfiguration()}>
                  Save Prompts
                </button>
                <button
                  className="button button-secondary"
                  type="button"
                  onClick={() =>
                    setConfigDraft({
                      ...configDraft,
                      plannerSystemPrompt: configuration?.system_prompts.planner.default ?? '',
                      answerSystemPrompt: configuration?.system_prompts.answer.default ?? '',
                    })
                  }
                >
                  Load Code Defaults
                </button>
              </div>
            </div>

            <div className="two-column">
              <div className="panel command-panel">
                <div className="section-heading"><strong>System Paths</strong><span>Current local storage map</span></div>
                <div className="status-grid">
                  <span>Data</span><strong>{configuration?.paths.data_dir ?? '-'}</strong>
                  <span>Config</span><strong>{configuration?.paths.config ?? '-'}</strong>
                  <span>Memory</span><strong>{configuration?.paths.memory ?? '-'}</strong>
                  <span>Workspace</span><strong>{configuration?.paths.workspace ?? '-'}</strong>
                  <span>Downloads</span><strong>{configuration?.paths.downloads_root ?? '-'}</strong>
                  <span>Audit DB</span><strong>{configuration?.paths.audit_db ?? '-'}</strong>
                </div>
              </div>

              <div className="panel command-panel">
                <div className="section-heading"><strong>Readiness</strong><span>Doctor and connectors</span></div>
                <div className="metric-grid">
                  <div><span>OK</span><strong>{doctor?.summary.ok ?? 0}</strong></div>
                  <div><span>Warnings</span><strong>{doctor?.summary.warn ?? 0}</strong></div>
                  <div><span>Failures</span><strong>{doctor?.summary.fail ?? 0}</strong></div>
                  <div><span>Connectors</span><strong>{connectors.length}</strong></div>
                </div>
                <div className="mini-list">
                  {connectors.map((connector) => (
                    <div key={connector.name}><strong>{connector.name}</strong><span>{connector.status}</span></div>
                  ))}
                </div>
              </div>
            </div>

            <div className="panel command-panel">
              <div className="section-heading">
                <strong>Terminal Policy</strong>
                <span>{terminal?.ready ? 'ready' : 'not ready'}</span>
              </div>
              <div className="settings-form">
                <label className="setting-wide">Workspace root
                  <input value={terminalWorkspaceRoot} onChange={(event) => setTerminalWorkspaceRoot(event.target.value)} placeholder="/path/to/project" />
                </label>
                <label>Timeout seconds
                  <input value={terminalTimeout} onChange={(event) => setTerminalTimeout(event.target.value)} inputMode="numeric" />
                </label>
                <label>Max output chars
                  <input value={terminalMaxOutput} onChange={(event) => setTerminalMaxOutput(event.target.value)} inputMode="numeric" />
                </label>
                <label className="setting-check">
                  <input type="checkbox" checked={terminalAutoApprove} onChange={(event) => setTerminalAutoApprove(event.target.checked)} />
                  <span><strong>Auto-approve exact allowlist</strong><small>Only commands matching the exact allowlist can bypass approval.</small></span>
                </label>
                <div className="row-actions">
                  <button className={terminal?.enabled ? 'button button-danger' : 'button'} type="button" onClick={() => void runAction(() => (terminal?.enabled ? api.disableTerminal() : api.enableTerminal()), terminal?.enabled ? 'Terminal disabled.' : 'Terminal enabled.')}>
                    {terminal?.enabled ? 'Disable Terminal' : 'Enable Terminal'}
                  </button>
                  <button className="button button-secondary" type="button" onClick={() => void saveTerminalSettings()}>Save Terminal</button>
                </div>
              </div>
              <form className="inline-form" onSubmit={(event) => { event.preventDefault(); void allowTerminalCommand() }}>
                <input value={terminalCommand} onChange={(event) => setTerminalCommand(event.target.value)} placeholder="git status" />
                <button className="button" type="submit">Allow Command</button>
              </form>
              <form className="run-form" onSubmit={(event) => { event.preventDefault(); void requestTerminalRun() }}>
                <input value={terminalRunCommand} onChange={(event) => setTerminalRunCommand(event.target.value)} placeholder="pwd" />
                <input value={terminalCwd} onChange={(event) => setTerminalCwd(event.target.value)} placeholder="workspace-relative cwd" />
                <button className="button button-secondary" type="submit">Request Run</button>
              </form>
              <div className="table-list table-list--compact">
                {terminal?.allowed_commands.map((command) => (
                  <article className="command-row" key={command.join('\u0000')}>
                    <code>{command.join(' ')}</code>
                    <button className="button button-danger" type="button" onClick={() => void runAction(() => api.removeTerminalCommand(command), `Removed: ${command.join(' ')}`)}>Remove</button>
                  </article>
                ))}
              </div>
            </div>
          </section>
        ) : null}

        {activeView === 'telegram' ? (
          <section className="telegram-layout">
            <div className="panel settings-panel">
              <div className="section-heading">
                <strong>Telegram Remote Access</strong>
                <span>{telegram?.polling ? 'polling allowlisted users' : 'token, enablement, and allowed user required'}</span>
              </div>
              <div className="status-grid">
                <span>Token env</span><strong>{telegram?.bot_token_env ?? '-'}</strong>
                <span>Token loaded</span><strong>{telegram?.bot_token_available ? 'yes' : 'no'}</strong>
                <span>Allowed users</span><strong>{telegram?.allowed_user_ids.length ?? 0}</strong>
                <span>Polling</span><strong>{telegram?.polling ? 'running' : 'stopped'}</strong>
                {telegram?.polling_error ? <><span>Error</span><strong>{telegram.polling_error}</strong></> : null}
              </div>
              <div className="row-actions">
                <button className={telegram?.enabled ? 'button button-danger' : 'button'} type="button" onClick={() => void runAction(() => (telegram?.enabled ? api.disableTelegram() : api.enableTelegram()), telegram?.enabled ? 'Telegram disabled.' : 'Telegram enabled.')}>
                  {telegram?.enabled ? 'Disable' : 'Enable'}
                </button>
                <button className="button button-secondary" type="button" onClick={() => void toggleTelegramPolling()} disabled={!telegram?.ready}>
                  {telegram?.polling ? 'Stop Polling' : 'Start Polling'}
                </button>
              </div>
            </div>
            <div className="two-column">
              <div className="panel command-panel">
                <div className="section-heading"><strong>Token</strong><span>Stored in process memory</span></div>
                <form className="inline-form" onSubmit={(event) => { event.preventDefault(); void saveTelegramTokenEnv() }}>
                  <input value={telegramTokenEnv} onChange={(event) => setTelegramTokenEnv(event.target.value)} placeholder="DMDAGENT_TELEGRAM_BOT_TOKEN" />
                  <button className="button button-secondary" type="submit">Save Env</button>
                </form>
                <form className="inline-form" onSubmit={(event) => { event.preventDefault(); void loadTelegramToken() }}>
                  <input type="password" value={telegramToken} onChange={(event) => setTelegramToken(event.target.value)} placeholder="BotFather token" />
                  <button className="button" type="submit">Load Token</button>
                </form>
              </div>
              <div className="panel command-panel">
                <div className="section-heading"><strong>Allowed Users</strong><span>Use /id in Telegram to discover the ID</span></div>
                <form className="inline-form" onSubmit={(event) => { event.preventDefault(); void allowTelegramUser() }}>
                  <input value={telegramUserId} onChange={(event) => setTelegramUserId(event.target.value)} placeholder="123456789" inputMode="numeric" />
                  <button className="button" type="submit">Allow User</button>
                </form>
                <div className="table-list table-list--compact">
                  {telegram?.allowed_user_ids.length ? null : <p className="empty">No allowed Telegram users.</p>}
                  {telegram?.allowed_user_ids.map((userId) => (
                    <article className="command-row" key={userId}>
                      <code>{userId}</code>
                      <button className="button button-danger" type="button" onClick={() => void runAction(() => api.removeTelegramUser(userId), `Removed Telegram user: ${userId}`)}>Remove</button>
                    </article>
                  ))}
                </div>
              </div>
            </div>
          </section>
        ) : null}

        {activeView === 'memory' ? (
          <section className="split-view">
            <div className="panel file-list">
              <div className="file-list-header"><strong>Memory Files</strong><span>{memoryFiles.length} files</span></div>
              {memoryFiles.map((file) => (
                <button key={file} type="button" onClick={() => void loadMemoryFile(file)} className={selectedMemory?.path === file ? 'file-list-item file-list-item--active' : 'file-list-item'}>
                  <strong>{file}</strong>
                </button>
              ))}
            </div>
            <div className="panel editor-panel">
              {selectedMemory ? (
                <>
                  <div className="editor-header">
                    <strong>{selectedMemory.path}</strong>
                    <button className="button" type="button" onClick={() => void requestMemorySave()}>Request Save</button>
                  </div>
                  <textarea value={memoryDraft} onChange={(event) => setMemoryDraft(event.target.value)} spellCheck={false} />
                </>
              ) : (
                <p className="empty">Select a memory file.</p>
              )}
            </div>
          </section>
        ) : null}

        {activeView === 'tools' ? (
          <section className="tools-console">
            <div className="panel command-panel">
              <div className="section-heading"><strong>Tools</strong><span>{enabledToolCount} enabled</span></div>
              <div className="tools-grid">
                {tools.map((tool) => (
                  <article className="tool-card" key={tool.name}>
                    <div><strong>{tool.name}</strong><span>{tool.description}</span></div>
                    <div className="tool-meta"><span>Risk {tool.risk}</span><span>{tool.approval_required ? 'Approval' : 'No approval'}</span></div>
                    <button className={tool.enabled ? 'toggle toggle-on' : 'toggle'} type="button" onClick={() => void runAction(() => api.setTool(tool.name, !tool.enabled), `${tool.name} ${tool.enabled ? 'disabled' : 'enabled'}.`)} aria-pressed={tool.enabled}>
                      {tool.enabled ? 'Enabled' : 'Disabled'}
                    </button>
                  </article>
                ))}
              </div>
            </div>
            <div className="panel command-panel">
              <div className="section-heading"><strong>Permissions</strong><span>{permissions.filter((permission) => permission.granted).length} granted</span></div>
              <div className="table-list">
                {permissions.map((permission) => (
                  <article className="permission-row" key={permission.name}>
                    <div><strong>{permission.name}</strong><span>{permission.tools.join(', ')}</span></div>
                    <button className={permission.granted ? 'toggle toggle-on' : 'toggle'} type="button" onClick={() => void runAction(() => api.setPermission(permission.name, !permission.granted), `${permission.name} ${permission.granted ? 'revoked' : 'granted'}.`)} aria-pressed={permission.granted}>
                      {permission.granted ? 'Granted' : 'Not Granted'}
                    </button>
                  </article>
                ))}
              </div>
            </div>
          </section>
        ) : null}

        {activeView === 'logs' ? (
          <section className="panel logs-console" data-tour="tour-logs">
            <div className="table-list">
              {audit.map((event, index) => (
                <article className="audit-row" key={`${event.created_at}-${index}`}>
                  <strong>{event.event_type}</strong>
                  <span>{formatDate(event.created_at)}</span>
                  <span>{event.tool ?? 'system'} - {event.result_status ?? 'recorded'}</span>
                  <code>{JSON.stringify(event.metadata)}</code>
                </article>
              ))}
            </div>
          </section>
        ) : null}

        {activeView === 'help' ? (
          <section className="help-console">
            <div className="panel command-panel">
              <div className="section-heading"><strong>User Guide</strong><span>Operational manual</span></div>
              <div className="guide-grid">
                {tourSteps.map((step) => (
                  <article className="guide-card" key={step.id}>
                    <strong>{step.title}</strong>
                    <p>{step.body}</p>
                  </article>
                ))}
                <article className="guide-card">
                  <strong>Scraping workflow</strong>
                  <p>Ask for a page to be scraped with a URL and instructions. The agent requests approval when needed, returns Markdown, and stores the file in Downloads.</p>
                </article>
                <article className="guide-card">
                  <strong>Safe approvals</strong>
                  <p>Approval cards show the tool, risk, and arguments. Approve only when the action matches your intent.</p>
                </article>
                <article className="guide-card">
                  <strong>Configuration</strong>
                  <p>Use Config to set download paths, browser limits, planner settings, terminal allowlists, and approval thresholds.</p>
                </article>
                <article className="guide-card">
                  <strong>System prompts</strong>
                  <p>Use Config to edit the planner and answer system prompts. Saving the code default clears the local override.</p>
                </article>
              </div>
              <button className="button button-secondary" type="button" onClick={() => { setTourStep(0); setActiveView('chat'); setTourOpen(true) }}>
                Replay Tour
              </button>
            </div>
          </section>
        ) : null}
      </main>

      {tourOpen ? (
        <div className="tour-layer">
          {tourRect ? (
            <div
              className="tour-highlight"
              style={{
                top: tourRect.top - 8,
                left: tourRect.left - 8,
                width: tourRect.width + 16,
                height: tourRect.height + 16,
              }}
            />
          ) : null}
          <div className="tour-card" style={tourBubbleStyle}>
            <span>Step {tourStep + 1} of {tourSteps.length}</span>
            <strong>{currentTour.title}</strong>
            <p>{currentTour.body}</p>
            <div className="tour-actions">
              <button className="button button-secondary" type="button" onClick={() => goTourStep(tourStep - 1)} disabled={tourStep === 0}>Prev</button>
              <button className="button" type="button" onClick={() => goTourStep(tourStep + 1)}>{tourStep === tourSteps.length - 1 ? 'Finish' : 'Next'}</button>
              <button className="button button-ghost" type="button" onClick={finishTour}>Skip</button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}

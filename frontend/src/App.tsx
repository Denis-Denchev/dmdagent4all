import { type KeyboardEvent, useEffect, useMemo, useRef, useState } from 'react'
import {
  type AgentTraceEvent,
  type AgentConfiguration,
  type AgentResponse,
  type AutonomyStatus,
  type Approval,
  type AuditEvent,
  api,
  type ConnectorStatus,
  type DeepSeekStatus,
  type DoctorResponse,
  type EmergencyStatus,
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
import { AppRoutes, SidebarNav, type SidebarNavItem } from './components/layout'
import {
  ConfigHub,
  ConfigSectionPage,
  DangerZone,
  SettingGroup,
  SettingRow,
  type ConfigHubEntry,
} from './components/settings'

type View = 'chat' | 'downloads' | 'approvals' | 'models' | 'config' | 'telegram' | 'memory' | 'tools' | 'logs' | 'help'
type ConfigSection = 'general' | 'models' | 'tools' | 'security' | 'workspace' | 'memory' | 'telegram' | 'emergency' | 'advanced'

type DashboardRoute = {
  view: View
  configSection: ConfigSection | null
}

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
  chatSystemPrompt: string
  plannerSystemPrompt: string
  answerSystemPrompt: string
  sendChatHistoryToCloud: boolean
  browserTimeout: string
  browserMaxResponseBytes: string
  browserMaxTextChars: string
  approvalRisk: string
  emailMaxBodyChars: string
  gmailEnabled: boolean
  gmailAuthMethod: 'app_password' | 'oauth2'
  gmailImapHost: string
  gmailImapPort: string
  gmailSmtpHost: string
  gmailSmtpPort: string
  gmailUsernameEnv: string
  gmailPasswordEnv: string
  gmailFromEnv: string
  gmailOauthClientId: string
  gmailOauthRedirectUri: string
  gmailOauthEmail: string
  gmailOauthFromAddress: string
  gmailMailbox: string
  gmailArchiveMailbox: string
  outlookEnabled: boolean
  outlookImapHost: string
  outlookImapPort: string
  outlookSmtpHost: string
  outlookSmtpPort: string
  outlookUsernameEnv: string
  outlookPasswordEnv: string
  outlookFromEnv: string
  outlookMailbox: string
  outlookArchiveMailbox: string
}

type EmailCredentialsDraft = {
  username: string
  appPassword: string
  fromAddress: string
}

const views: Array<{ key: View; label: string; short: string; description: string }> = [
  { key: 'chat', label: 'Chat', short: 'CH', description: 'Work with the agent' },
  { key: 'downloads', label: 'Downloads', short: 'DL', description: 'Files from web work' },
  { key: 'approvals', label: 'Approvals', short: 'AP', description: 'Pending risky actions' },
  { key: 'models', label: 'Models', short: 'MD', description: 'Runtime and token controls' },
  { key: 'config', label: 'Config', short: 'CF', description: 'Agent settings and paths' },
  { key: 'telegram', label: 'Telegram', short: 'TG', description: 'Remote access' },
  { key: 'memory', label: 'Memory', short: 'MY', description: 'Local long-term context' },
  { key: 'tools', label: 'Tools', short: 'TL', description: 'Permissions and connectors' },
  { key: 'logs', label: 'Logs', short: 'LG', description: 'Audit trail' },
  { key: 'help', label: 'Help', short: 'HP', description: 'Manual and guide' },
]

const configSectionLabels: Record<ConfigSection, string> = {
  general: 'General',
  models: 'Models',
  tools: 'Tools',
  security: 'Security',
  workspace: 'Workspace',
  memory: 'Memory',
  telegram: 'Telegram',
  emergency: 'Emergency',
  advanced: 'Advanced',
}

const viewPaths: Record<View, string> = {
  chat: '/chat',
  downloads: '/downloads',
  approvals: '/approvals',
  models: '/models',
  config: '/config',
  telegram: '/telegram',
  memory: '/memory',
  tools: '/tools',
  logs: '/logs',
  help: '/help',
}

const tourSteps: TourStep[] = [
  {
    id: 'tour-chat',
    view: 'chat',
    title: 'Chat workspace',
    body: 'This is the main control surface. Ask the agent to plan, scrape, use tools, remember facts, or run approved local actions.',
  },
  {
    id: 'tour-emergency',
    view: 'chat',
    title: 'Emergency Stop',
    body: 'Use this when a command or tool action must stop immediately. It terminates active terminal processes, cancels pending approvals, and blocks further tool execution until you reset emergency mode.',
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

function traceFromResponse(response?: AgentResponse): AgentTraceEvent[] {
  const trace = dataObject(response?.data).trace
  return Array.isArray(trace) ? (trace as AgentTraceEvent[]) : []
}

function visibleTrace(events: AgentTraceEvent[]): AgentTraceEvent[] {
  return events.filter((event) => dataObject(event.metadata).visibility !== 'debug')
}

function latestTraceTitle(events: AgentTraceEvent[]): string {
  const visible = visibleTrace(events)
  return visible[visible.length - 1]?.title ?? 'Preparing runtime'
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
    chatSystemPrompt: '',
    plannerSystemPrompt: '',
    answerSystemPrompt: '',
    sendChatHistoryToCloud: false,
    browserTimeout: '15',
    browserMaxResponseBytes: '1000000',
    browserMaxTextChars: '12000',
    approvalRisk: '3',
    emailMaxBodyChars: '20000',
    gmailEnabled: false,
    gmailAuthMethod: 'app_password',
    gmailImapHost: 'imap.gmail.com',
    gmailImapPort: '993',
    gmailSmtpHost: 'smtp.gmail.com',
    gmailSmtpPort: '587',
    gmailUsernameEnv: 'DMDAGENT_GMAIL_USERNAME',
    gmailPasswordEnv: 'DMDAGENT_GMAIL_APP_PASSWORD',
    gmailFromEnv: 'DMDAGENT_GMAIL_FROM',
    gmailOauthClientId: '',
    gmailOauthRedirectUri: 'http://127.0.0.1:8765/v1/email/oauth/google/callback',
    gmailOauthEmail: '',
    gmailOauthFromAddress: '',
    gmailMailbox: 'INBOX',
    gmailArchiveMailbox: '[Gmail]/All Mail',
    outlookEnabled: false,
    outlookImapHost: 'outlook.office365.com',
    outlookImapPort: '993',
    outlookSmtpHost: 'smtp.office365.com',
    outlookSmtpPort: '587',
    outlookUsernameEnv: 'DMDAGENT_OUTLOOK_USERNAME',
    outlookPasswordEnv: 'DMDAGENT_OUTLOOK_APP_PASSWORD',
    outlookFromEnv: 'DMDAGENT_OUTLOOK_FROM',
    outlookMailbox: 'INBOX',
    outlookArchiveMailbox: 'Archive',
  }
}

function routeFromPath(pathname: string): DashboardRoute {
  const clean = pathname.replace(/\/+$/, '') || '/'
  const parts = clean.split('/').filter(Boolean)
  if (parts[0] === 'config') {
    const section = parts[1] as ConfigSection | undefined
    return {
      view: 'config',
      configSection: section && section in configSectionLabels ? section : null,
    }
  }
  const view = (parts[0] || 'chat') as View
  if (views.some((item) => item.key === view)) {
    return { view, configSection: null }
  }
  return { view: 'chat', configSection: null }
}

function pathForRoute(view: View, configSection: ConfigSection | null = null) {
  if (view === 'config' && configSection) return `/config/${configSection}`
  return viewPaths[view] ?? '/chat'
}

export function App() {
  const initialRoute = routeFromPath(window.location.pathname)
  const [activeView, setActiveViewState] = useState<View>(initialRoute.view)
  const [configSection, setConfigSection] = useState<ConfigSection | null>(initialRoute.configSection)
  const [railCollapsed, setRailCollapsed] = useState(false)
  const chatLogRef = useRef<HTMLDivElement | null>(null)
  const chatEndRef = useRef<HTMLDivElement | null>(null)
  const [status, setStatus] = useState<Status | null>(null)
  const [configuration, setConfiguration] = useState<AgentConfiguration | null>(null)
  const [tools, setTools] = useState<Tool[]>([])
  const [permissions, setPermissions] = useState<PermissionItem[]>([])
  const [connectors, setConnectors] = useState<ConnectorStatus[]>([])
  const [terminal, setTerminal] = useState<TerminalStatus | null>(null)
  const [autonomy, setAutonomy] = useState<AutonomyStatus | null>(null)
  const [activeTrace, setActiveTrace] = useState<AgentTraceEvent[]>([])
  const [telegram, setTelegram] = useState<TelegramStatus | null>(null)
  const [openai, setOpenAI] = useState<OpenAIStatus | null>(null)
  const [deepseek, setDeepSeek] = useState<DeepSeekStatus | null>(null)
  const [approvals, setApprovals] = useState<Approval[]>([])
  const [audit, setAudit] = useState<AuditEvent[]>([])
  const [doctor, setDoctor] = useState<DoctorResponse | null>(null)
  const [emergency, setEmergency] = useState<EmergencyStatus | null>(null)
  const [memoryFiles, setMemoryFiles] = useState<string[]>([])
  const [selectedMemory, setSelectedMemory] = useState<MemoryFile | null>(null)
  const [memoryDraft, setMemoryDraft] = useState('')
  const [workspaceFiles, setWorkspaceFiles] = useState<WorkspaceFilesResponse | null>(null)
  const [selectedWorkspaceFile, setSelectedWorkspaceFile] = useState<WorkspaceFileContent | null>(null)
  const [models, setModels] = useState<ModelMode[]>([])
  const [customModel, setCustomModel] = useState('')
  const [configDraft, setConfigDraft] = useState<ConfigDraft>(initialConfigDraft)
  const [gmailCredentials, setGmailCredentials] = useState<EmailCredentialsDraft>({ username: '', appPassword: '', fromAddress: '' })
  const [outlookCredentials, setOutlookCredentials] = useState<EmailCredentialsDraft>({ username: '', appPassword: '', fromAddress: '' })
  const [gmailOAuthClientSecret, setGmailOAuthClientSecret] = useState('')
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

  function navigateTo(view: View, section: ConfigSection | null = null, options: { replace?: boolean } = {}) {
    const nextPath = pathForRoute(view, section)
    setActiveViewState(view)
    setConfigSection(view === 'config' ? section : null)
    if (window.location.pathname !== nextPath) {
      const method = options.replace ? 'replaceState' : 'pushState'
      window.history[method]({}, '', nextPath)
    }
  }

  const pendingCount = approvals.length
  const enabledToolCount = useMemo(() => tools.filter((tool) => tool.enabled).length, [tools])
  const autonomyEnabled = Boolean(autonomy?.enabled)
  const activeLabel = activeView === 'config' && configSection
    ? `Config / ${configSectionLabels[configSection]}`
    : views.find((view) => view.key === activeView)?.label ?? 'Dashboard'
  const navItems: SidebarNavItem[] = views.map((view) => ({
    ...view,
    badge: view.key === 'approvals' && pendingCount > 0 ? pendingCount : undefined,
    tourId:
      view.key === 'downloads'
        ? 'tour-downloads'
        : view.key === 'models'
          ? 'tour-models'
          : view.key === 'config'
            ? 'tour-config'
            : view.key === 'logs'
              ? 'tour-logs'
              : undefined,
  }))

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
      navigateTo(tourSteps[0].view, null, { replace: true })
    }
  }, [])

  useEffect(() => {
    const onPopState = () => {
      const route = routeFromPath(window.location.pathname)
      setActiveViewState(route.view)
      setConfigSection(route.configSection)
    }
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  useEffect(() => {
    if (!tourOpen) return
    const step = tourSteps[tourStep]
    if (step.view !== activeView) {
      navigateTo(step.view, null)
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
      autonomyResult,
      telegramResult,
      openaiResult,
      deepseekResult,
      approvalsResult,
      auditResult,
      doctorResult,
      emergencyResult,
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
      api.autonomy(),
      api.telegram(),
      api.openai(),
      api.deepseek(),
      api.approvals(),
      api.audit(),
      api.doctor(),
      api.emergency(),
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
    setAutonomy(autonomyResult)
    setTelegram(telegramResult)
    setOpenAI(openaiResult)
    setDeepSeek(deepseekResult)
    setApprovals(approvalsResult)
    setAudit(auditResult)
    setDoctor(doctorResult)
    setEmergency(emergencyResult)
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
      chatSystemPrompt: configurationResult.system_prompts.chat.effective,
      plannerSystemPrompt: configurationResult.system_prompts.planner.effective,
      answerSystemPrompt: configurationResult.system_prompts.answer.effective,
      sendChatHistoryToCloud: Boolean(configurationResult.privacy.send_chat_history_to_cloud),
      browserTimeout: String(configurationResult.browser.timeout_seconds ?? 15),
      browserMaxResponseBytes: String(configurationResult.browser.max_response_bytes ?? 1000000),
      browserMaxTextChars: String(configurationResult.browser.max_text_chars ?? 12000),
      approvalRisk: String(configurationResult.permissions.approval_required_at_risk ?? 3),
      emailMaxBodyChars: String(configurationResult.email.max_body_chars ?? 20000),
      gmailEnabled: Boolean(configurationResult.email.gmail.enabled),
      gmailAuthMethod: configurationResult.email.gmail.auth_method,
      gmailImapHost: configurationResult.email.gmail.imap_host,
      gmailImapPort: String(configurationResult.email.gmail.imap_port),
      gmailSmtpHost: configurationResult.email.gmail.smtp_host,
      gmailSmtpPort: String(configurationResult.email.gmail.smtp_port),
      gmailUsernameEnv: configurationResult.email.gmail.username_env,
      gmailPasswordEnv: configurationResult.email.gmail.password_env,
      gmailFromEnv: configurationResult.email.gmail.from_env,
      gmailOauthClientId: configurationResult.email.gmail.oauth_client_id,
      gmailOauthRedirectUri: configurationResult.email.gmail.oauth_redirect_uri,
      gmailOauthEmail: configurationResult.email.gmail.oauth_email,
      gmailOauthFromAddress: configurationResult.email.gmail.oauth_from_address,
      gmailMailbox: configurationResult.email.gmail.mailbox,
      gmailArchiveMailbox: configurationResult.email.gmail.archive_mailbox,
      outlookEnabled: Boolean(configurationResult.email.outlook.enabled),
      outlookImapHost: configurationResult.email.outlook.imap_host,
      outlookImapPort: String(configurationResult.email.outlook.imap_port),
      outlookSmtpHost: configurationResult.email.outlook.smtp_host,
      outlookSmtpPort: String(configurationResult.email.outlook.smtp_port),
      outlookUsernameEnv: configurationResult.email.outlook.username_env,
      outlookPasswordEnv: configurationResult.email.outlook.password_env,
      outlookFromEnv: configurationResult.email.outlook.from_env,
      outlookMailbox: configurationResult.email.outlook.mailbox,
      outlookArchiveMailbox: configurationResult.email.outlook.archive_mailbox,
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

  async function toggleAutonomyMode() {
    const nextEnabled = !autonomyEnabled
    setBusy(true)
    setNotice(null)
    try {
      const result = await api.setAutonomy(nextEnabled)
      setAutonomy(result)
      setNotice(result.enabled ? 'Full LLM-first autonomy mode enabled.' : 'Standard safe mode enabled.')
      await refreshAll()
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not update autonomy mode')
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
    if (focusChat) navigateTo('chat')
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
    setActiveTrace([])
    try {
      const response = autonomyEnabled
        ? await api.chatStream(message, 'dashboard', (event) => {
          if (event.type === 'trace') {
            setActiveTrace((current) => [...current, event.event])
          }
        })
        : await api.chat(message, 'dashboard')
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
      setActiveTrace([])
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
          chat_system_prompt: configDraft.chatSystemPrompt,
          planner_system_prompt: configDraft.plannerSystemPrompt,
          answer_system_prompt: configDraft.answerSystemPrompt,
          send_chat_history_to_cloud: configDraft.sendChatHistoryToCloud,
          browser_timeout_seconds: toNumber(configDraft.browserTimeout),
          browser_max_response_bytes: toNumber(configDraft.browserMaxResponseBytes),
          browser_max_text_chars: toNumber(configDraft.browserMaxTextChars),
          approval_required_at_risk: toNumber(configDraft.approvalRisk),
          email: {
            max_body_chars: toNumber(configDraft.emailMaxBodyChars),
            gmail: {
              enabled: configDraft.gmailEnabled,
              auth_method: configDraft.gmailAuthMethod,
              imap_host: configDraft.gmailImapHost,
              imap_port: toNumber(configDraft.gmailImapPort),
              smtp_host: configDraft.gmailSmtpHost,
              smtp_port: toNumber(configDraft.gmailSmtpPort),
              username_env: configDraft.gmailUsernameEnv,
              password_env: configDraft.gmailPasswordEnv,
              from_env: configDraft.gmailFromEnv,
              oauth_client_id: configDraft.gmailOauthClientId,
              oauth_redirect_uri: configDraft.gmailOauthRedirectUri,
              oauth_email: configDraft.gmailOauthEmail,
              oauth_from_address: configDraft.gmailOauthFromAddress,
              mailbox: configDraft.gmailMailbox,
              archive_mailbox: configDraft.gmailArchiveMailbox,
            },
            outlook: {
              enabled: configDraft.outlookEnabled,
              imap_host: configDraft.outlookImapHost,
              imap_port: toNumber(configDraft.outlookImapPort),
              smtp_host: configDraft.outlookSmtpHost,
              smtp_port: toNumber(configDraft.outlookSmtpPort),
              username_env: configDraft.outlookUsernameEnv,
              password_env: configDraft.outlookPasswordEnv,
              from_env: configDraft.outlookFromEnv,
              mailbox: configDraft.outlookMailbox,
              archive_mailbox: configDraft.outlookArchiveMailbox,
            },
          },
        }),
      'Configuration saved.',
    )
  }

  async function loadEmailCredentials(provider: 'gmail' | 'outlook') {
    const credentials = provider === 'gmail' ? gmailCredentials : outlookCredentials
    const username = credentials.username.trim()
    const appPassword = credentials.appPassword.trim()
    if (!username || !appPassword) {
      setNotice('Email username and app password are required.')
      return
    }
    await runAction(
      () => api.loadEmailCredentials(provider, username, appPassword, credentials.fromAddress.trim()),
      `${provider === 'gmail' ? 'Gmail' : 'Outlook'} credentials loaded.`,
    )
    if (provider === 'gmail') {
      setGmailCredentials((current) => ({ ...current, appPassword: '' }))
    } else {
      setOutlookCredentials((current) => ({ ...current, appPassword: '' }))
    }
  }

  async function testEmailConnection(provider: 'gmail' | 'outlook') {
    setBusy(true)
    setNotice(null)
    try {
      const response = await api.testEmailConnection(provider)
      const data = dataObject(response.data)
      const imap = dataObject(data.imap)
      const smtp = dataObject(data.smtp)
      const imapStatus = imap.ok === true ? 'IMAP ok' : 'IMAP failed'
      const smtpStatus = smtp.ok === true ? 'SMTP ok' : 'SMTP failed'
      setNotice(`${response.message} ${imapStatus}; ${smtpStatus}.`)
      await refreshAll()
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Email connection test failed.')
    } finally {
      setBusy(false)
    }
  }

  async function loadGmailOAuthClientSecret() {
    const clientSecret = gmailOAuthClientSecret.trim()
    if (!clientSecret) {
      setNotice('Google OAuth client secret is required.')
      return
    }
    await runAction(
      () =>
        api.loadGmailOAuthClientSecret({
          client_secret: clientSecret,
          client_id: configDraft.gmailOauthClientId.trim() || undefined,
          email: configDraft.gmailOauthEmail.trim() || undefined,
          from_address: configDraft.gmailOauthFromAddress.trim() || undefined,
          redirect_uri: configDraft.gmailOauthRedirectUri.trim() || undefined,
        }),
      'Gmail OAuth client secret loaded.',
    )
    setGmailOAuthClientSecret('')
  }

  async function startGmailOAuth() {
    const clientId = configDraft.gmailOauthClientId.trim()
    const email = configDraft.gmailOauthEmail.trim()
    if (!clientId || !email) {
      setNotice('Google OAuth client ID and Gmail address are required.')
      return
    }
    const authWindow = window.open('', '_blank', 'noopener,noreferrer')
    setBusy(true)
    setNotice(null)
    try {
      const response = await api.startGmailOAuth({
        client_id: clientId,
        client_secret: gmailOAuthClientSecret.trim() || undefined,
        email,
        from_address: configDraft.gmailOauthFromAddress.trim() || undefined,
        redirect_uri: configDraft.gmailOauthRedirectUri.trim() || undefined,
      })
      const data = dataObject(response.data)
      const authorizationUrl = typeof data.authorization_url === 'string' ? data.authorization_url : ''
      if (authorizationUrl) {
        if (authWindow) {
          authWindow.location.href = authorizationUrl
        } else {
          window.open(authorizationUrl, '_blank', 'noopener,noreferrer')
        }
      }
      setNotice(response.message)
      setGmailOAuthClientSecret('')
      await refreshAll()
    } catch (error) {
      authWindow?.close()
      setNotice(error instanceof Error ? error.message : 'Could not start Gmail OAuth.')
    } finally {
      setBusy(false)
    }
  }

  async function disconnectGmailOAuth() {
    await runAction(() => api.disconnectGmailOAuth(), 'Gmail OAuth disconnected.')
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

  async function emergencyStop() {
    const response = await runAgentAction(() => api.emergencyStop(), { focusChat: true })
    if (response?.status === 'ok') {
      setNotice('Emergency stop active. Tool execution is blocked until reset.')
    }
  }

  async function emergencyReset() {
    const response = await runAgentAction(() => api.emergencyReset(), { focusChat: true })
    if (response?.status === 'ok') {
      setNotice('Emergency stop reset.')
    }
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
    navigateTo(tourSteps[nextStep].view)
  }

  function renderConfigRoutes() {
    const gmailStatus = connectors.find((connector) => connector.name === 'gmail')?.status ?? 'unknown'
    const outlookStatus = connectors.find((connector) => connector.name === 'outlook')?.status ?? 'unknown'
    const entries: ConfigHubEntry[] = [
      {
        key: 'general',
        title: 'General',
        description: 'App identity, user name, language, and startup defaults',
        status: String(configuration?.setup.agent_name ?? 'DMD Agent'),
        group: 'Core setup',
      },
      {
        key: 'models',
        title: 'Models',
        description: 'Planner behavior, prompt stack, token limits, and model-facing instructions',
        status: String(status?.llm.provider ?? 'loading'),
        group: 'Core setup',
      },
      {
        key: 'tools',
        title: 'Tools',
        description: 'Browser scrape limits, email sign-in, and connector configuration',
        status: `${gmailStatus} / ${outlookStatus}`,
        group: 'Core setup',
      },
      {
        key: 'workspace',
        title: 'Workspace',
        description: 'Downloads, current workspace, terminal policy, and command allowlist',
        status: terminal?.ready ? 'ready' : 'not ready',
        tone: terminal?.ready ? 'normal' : 'warning',
        group: 'Core setup',
      },
      {
        key: 'security',
        title: 'Security',
        description: 'Approvals, cloud context, emergency stop, and guarded execution policy',
        status: `risk ${configuration?.permissions.approval_required_at_risk ?? 3}+`,
        tone: 'warning',
        group: 'Safety and system',
      },
      {
        key: 'advanced',
        title: 'System',
        description: 'Memory, Telegram, system paths, doctor summary, and connector readiness',
        status: `${doctor?.summary.warn ?? 0} warnings`,
        group: 'Safety and system',
      },
    ]

    if (configSection === null) {
      return (
        <ConfigHub
          entries={entries}
          onOpen={(key) => navigateTo('config', key as ConfigSection)}
        />
      )
    }

    if (configSection === 'general') {
      return (
        <ConfigSectionPage
          title="General"
          description="Human-facing defaults for the local assistant."
          onBack={() => navigateTo('config')}
        >
          <SettingGroup title="Identity" description="Used in normal chat and setup/status surfaces.">
            <div className="setting-rows">
              <SettingRow label="Agent name" help="The assistant identity shown in chat and prompts.">
                <input value={configDraft.agentName} onChange={(event) => setConfigDraft({ ...configDraft, agentName: event.target.value })} />
              </SettingRow>
              <SettingRow label="User name" help="Optional local profile name.">
                <input value={configDraft.userName} onChange={(event) => setConfigDraft({ ...configDraft, userName: event.target.value })} />
              </SettingRow>
              <SettingRow label="Preferred language" help="Input preference. Use auto unless you need a fixed language.">
                <input value={configDraft.preferredLanguage} onChange={(event) => setConfigDraft({ ...configDraft, preferredLanguage: event.target.value })} />
              </SettingRow>
              <SettingRow label="Response language" help="Output language for model responses.">
                <input value={configDraft.responseLanguage} onChange={(event) => setConfigDraft({ ...configDraft, responseLanguage: event.target.value })} />
              </SettingRow>
            </div>
            <div className="row-actions">
              <button className="button" type="button" onClick={() => void saveConfiguration()}>Save General</button>
            </div>
          </SettingGroup>
        </ConfigSectionPage>
      )
    }

    if (configSection === 'models') {
      return (
        <ConfigSectionPage
          title="Models"
          description="Prompt stack and planner behavior. Provider/API keys remain in the main Models screen."
          onBack={() => navigateTo('config')}
        >
          <SettingGroup title="Planner runtime" description="Small changes here affect tool selection and response shaping.">
            <div className="setting-rows">
              <SettingRow label="Planner max tokens" help="Budget for tool-selection JSON.">
                <input value={configDraft.plannerMaxTokens} onChange={(event) => setConfigDraft({ ...configDraft, plannerMaxTokens: event.target.value })} inputMode="numeric" />
              </SettingRow>
              <SettingRow label="Planner temperature" help="Keep low for predictable routing.">
                <input value={configDraft.plannerTemperature} onChange={(event) => setConfigDraft({ ...configDraft, plannerTemperature: event.target.value })} inputMode="decimal" />
              </SettingRow>
              <label className="setting-check">
                <input type="checkbox" checked={configDraft.plannerThink} onChange={(event) => setConfigDraft({ ...configDraft, plannerThink: event.target.checked })} />
                <span><strong>Planner think mode</strong><small>Passes the think flag to compatible local/cloud providers.</small></span>
              </label>
            </div>
          </SettingGroup>
          <SettingGroup title="System prompts" description="Chat controls normal conversation; planner controls tool selection; answer controls tool-result wording.">
            <div className="prompt-grid">
              <label>Chat system prompt
                <textarea value={configDraft.chatSystemPrompt} onChange={(event) => setConfigDraft({ ...configDraft, chatSystemPrompt: event.target.value })} spellCheck={false} />
                <span className="field-note">{configuration?.system_prompts.chat.customized ? 'Custom prompt is active.' : 'Currently using the code default.'}</span>
              </label>
              <label>Planner system prompt
                <textarea value={configDraft.plannerSystemPrompt} onChange={(event) => setConfigDraft({ ...configDraft, plannerSystemPrompt: event.target.value })} spellCheck={false} />
                <span className="field-note">{configuration?.system_prompts.planner.customized ? 'Custom prompt is active.' : 'Currently using the code default.'}</span>
              </label>
              <label>Answer system prompt
                <textarea value={configDraft.answerSystemPrompt} onChange={(event) => setConfigDraft({ ...configDraft, answerSystemPrompt: event.target.value })} spellCheck={false} />
                <span className="field-note">{configuration?.system_prompts.answer.customized ? 'Custom prompt is active.' : 'Currently using the code default.'}</span>
              </label>
            </div>
            <div className="row-actions">
              <button className="button" type="button" onClick={() => void saveConfiguration()}>Save Model Settings</button>
              <button
                className="button button-secondary"
                type="button"
                onClick={() =>
                  setConfigDraft({
                    ...configDraft,
                    chatSystemPrompt: configuration?.system_prompts.chat.default ?? '',
                    plannerSystemPrompt: configuration?.system_prompts.planner.default ?? '',
                    answerSystemPrompt: configuration?.system_prompts.answer.default ?? '',
                  })
                }
              >
                Load Code Defaults
              </button>
              <button className="button button-secondary" type="button" onClick={() => navigateTo('models')}>Open Provider Models</button>
            </div>
          </SettingGroup>
        </ConfigSectionPage>
      )
    }

    if (configSection === 'tools') {
      return (
        <ConfigSectionPage
          title="Tools"
          description="Connector limits and tool-adjacent settings. Tool enablement remains in the Tools screen."
          onBack={() => navigateTo('config')}
        >
          <SettingGroup title="Browser scraping" description="Controls guarded HTTP/browser extraction size and timeout.">
            <div className="setting-rows">
              <SettingRow label="Browser timeout seconds">
                <input value={configDraft.browserTimeout} onChange={(event) => setConfigDraft({ ...configDraft, browserTimeout: event.target.value })} inputMode="numeric" />
              </SettingRow>
              <SettingRow label="Max response bytes">
                <input value={configDraft.browserMaxResponseBytes} onChange={(event) => setConfigDraft({ ...configDraft, browserMaxResponseBytes: event.target.value })} inputMode="numeric" />
              </SettingRow>
              <SettingRow label="Max text chars">
                <input value={configDraft.browserMaxTextChars} onChange={(event) => setConfigDraft({ ...configDraft, browserMaxTextChars: event.target.value })} inputMode="numeric" />
              </SettingRow>
            </div>
          </SettingGroup>
          <SettingGroup title="Email connectors" description="IMAP/SMTP settings. Credentials stay in the current API process, not config.yaml.">
            <div className="setting-rows">
              <SettingRow label="Email body read limit">
                <input value={configDraft.emailMaxBodyChars} onChange={(event) => setConfigDraft({ ...configDraft, emailMaxBodyChars: event.target.value })} inputMode="numeric" />
              </SettingRow>
              <label className="setting-check">
                <input type="checkbox" checked={configDraft.gmailEnabled} onChange={(event) => setConfigDraft({ ...configDraft, gmailEnabled: event.target.checked })} />
                <span><strong>Enable Gmail</strong><small>Status: {gmailStatus}</small></span>
              </label>
              <label className="setting-check">
                <input type="checkbox" checked={configDraft.outlookEnabled} onChange={(event) => setConfigDraft({ ...configDraft, outlookEnabled: event.target.checked })} />
                <span><strong>Enable Outlook</strong><small>Status: {outlookStatus}</small></span>
              </label>
            </div>
            <div className="email-provider-grid">
              <section className="email-provider">
                <div className="section-heading">
                  <strong>Gmail</strong>
                  <span>{configuration?.email.gmail.credentials_loaded ? 'credentials loaded' : 'credentials not loaded'}</span>
                </div>
                <div className="settings-form settings-form-compact">
                  <label>Authentication
                    <select
                      value={configDraft.gmailAuthMethod}
                      onChange={(event) => setConfigDraft({ ...configDraft, gmailAuthMethod: event.target.value as 'app_password' | 'oauth2' })}
                    >
                      <option value="app_password">App password</option>
                      <option value="oauth2">Google OAuth2</option>
                    </select>
                  </label>
                </div>
                {configDraft.gmailAuthMethod === 'oauth2' ? (
                  <>
                    <div className="email-oauth-status">
                      <span>Client secret: <strong>{configuration?.email.gmail.oauth_client_secret_loaded ? 'loaded' : 'missing'}</strong></span>
                      <span>Google token: <strong>{configuration?.email.gmail.oauth_refresh_token_loaded ? 'connected' : 'not connected'}</strong></span>
                    </div>
                    <div className="email-oauth-form">
                      <label>Gmail address
                        <input value={configDraft.gmailOauthEmail} onChange={(event) => setConfigDraft({ ...configDraft, gmailOauthEmail: event.target.value })} placeholder="name@gmail.com" autoComplete="username" />
                      </label>
                      <label>From address
                        <input value={configDraft.gmailOauthFromAddress} onChange={(event) => setConfigDraft({ ...configDraft, gmailOauthFromAddress: event.target.value })} placeholder="optional" autoComplete="email" />
                      </label>
                      <label className="setting-wide">Google OAuth client ID
                        <input value={configDraft.gmailOauthClientId} onChange={(event) => setConfigDraft({ ...configDraft, gmailOauthClientId: event.target.value })} placeholder="client-id.apps.googleusercontent.com" />
                      </label>
                      <label className="setting-wide">Google OAuth client secret
                        <input type="password" value={gmailOAuthClientSecret} onChange={(event) => setGmailOAuthClientSecret(event.target.value)} placeholder={configuration?.email.gmail.oauth_client_secret_loaded ? 'Already loaded; leave empty unless replacing' : 'Paste client secret'} autoComplete="off" />
                      </label>
                      <label className="setting-wide">Redirect URI
                        <input value={configDraft.gmailOauthRedirectUri} onChange={(event) => setConfigDraft({ ...configDraft, gmailOauthRedirectUri: event.target.value })} />
                      </label>
                    </div>
                    <span className="field-note">Add this exact Redirect URI in Google Cloud OAuth Client. The app requests only the Gmail IMAP/SMTP scope and stores tokens in the local secret store.</span>
                    <div className="row-actions">
                      <button className="button button-secondary" type="button" onClick={() => void loadGmailOAuthClientSecret()} disabled={busy || !gmailOAuthClientSecret.trim()}>Load Client Secret</button>
                      <button className="button" type="button" onClick={() => void startGmailOAuth()} disabled={busy}>Start Google OAuth</button>
                      <button className="button button-secondary" type="button" onClick={() => void testEmailConnection('gmail')} disabled={busy}>Test Gmail Connection</button>
                      <button className="button button-danger" type="button" onClick={() => void disconnectGmailOAuth()} disabled={busy || !configuration?.email.gmail.oauth_refresh_token_loaded}>Disconnect OAuth</button>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="email-login-form">
                      <label>Gmail address
                        <input value={gmailCredentials.username} onChange={(event) => setGmailCredentials({ ...gmailCredentials, username: event.target.value })} placeholder="name@gmail.com" autoComplete="username" />
                      </label>
                      <label>App password
                        <input type="password" value={gmailCredentials.appPassword} onChange={(event) => setGmailCredentials({ ...gmailCredentials, appPassword: event.target.value })} placeholder="Google app password" autoComplete="current-password" />
                      </label>
                      <label>From address
                        <input value={gmailCredentials.fromAddress} onChange={(event) => setGmailCredentials({ ...gmailCredentials, fromAddress: event.target.value })} placeholder="optional" autoComplete="email" />
                      </label>
                      <button className="button" type="button" onClick={() => void loadEmailCredentials('gmail')} disabled={busy}>Load Gmail Credentials</button>
                      <button className="button button-secondary" type="button" onClick={() => void testEmailConnection('gmail')} disabled={busy}>Test Gmail Connection</button>
                    </div>
                    <span className="field-note">Use a Google app password. The secret is loaded into the running API process and is not written to config.yaml.</span>
                  </>
                )}
                <details className="advanced-mail-settings">
                  <summary>Connection and environment settings</summary>
                  <div className="settings-form settings-form-compact">
                    <label>IMAP host<input value={configDraft.gmailImapHost} onChange={(event) => setConfigDraft({ ...configDraft, gmailImapHost: event.target.value })} /></label>
                    <label>IMAP port<input value={configDraft.gmailImapPort} onChange={(event) => setConfigDraft({ ...configDraft, gmailImapPort: event.target.value })} inputMode="numeric" /></label>
                    <label>SMTP host<input value={configDraft.gmailSmtpHost} onChange={(event) => setConfigDraft({ ...configDraft, gmailSmtpHost: event.target.value })} /></label>
                    <label>SMTP port<input value={configDraft.gmailSmtpPort} onChange={(event) => setConfigDraft({ ...configDraft, gmailSmtpPort: event.target.value })} inputMode="numeric" /></label>
                    <label>Username env<input value={configDraft.gmailUsernameEnv} onChange={(event) => setConfigDraft({ ...configDraft, gmailUsernameEnv: event.target.value })} /></label>
                    <label>Password env<input value={configDraft.gmailPasswordEnv} onChange={(event) => setConfigDraft({ ...configDraft, gmailPasswordEnv: event.target.value })} /></label>
                    <label>From env<input value={configDraft.gmailFromEnv} onChange={(event) => setConfigDraft({ ...configDraft, gmailFromEnv: event.target.value })} /></label>
                    <label>Mailbox<input value={configDraft.gmailMailbox} onChange={(event) => setConfigDraft({ ...configDraft, gmailMailbox: event.target.value })} /></label>
                    <label className="setting-wide">Archive mailbox<input value={configDraft.gmailArchiveMailbox} onChange={(event) => setConfigDraft({ ...configDraft, gmailArchiveMailbox: event.target.value })} /></label>
                  </div>
                </details>
              </section>
              <section className="email-provider">
                <div className="section-heading"><strong>Outlook</strong><span>{configuration?.email.outlook.credentials_loaded ? 'credentials loaded' : 'credentials not loaded'}</span></div>
                <div className="email-login-form">
                  <label>Outlook address
                    <input value={outlookCredentials.username} onChange={(event) => setOutlookCredentials({ ...outlookCredentials, username: event.target.value })} placeholder="name@outlook.com" autoComplete="username" />
                  </label>
                  <label>App password
                    <input type="password" value={outlookCredentials.appPassword} onChange={(event) => setOutlookCredentials({ ...outlookCredentials, appPassword: event.target.value })} placeholder="Microsoft app password" autoComplete="current-password" />
                  </label>
                  <label>From address
                    <input value={outlookCredentials.fromAddress} onChange={(event) => setOutlookCredentials({ ...outlookCredentials, fromAddress: event.target.value })} placeholder="optional" autoComplete="email" />
                  </label>
                  <button className="button" type="button" onClick={() => void loadEmailCredentials('outlook')} disabled={busy}>Load Outlook Credentials</button>
                  <button className="button button-secondary" type="button" onClick={() => void testEmailConnection('outlook')} disabled={busy}>Test Outlook Connection</button>
                </div>
                <span className="field-note">Microsoft may require SMTP AUTH or an app password. The secret is kept only in the running API process.</span>
                <details className="advanced-mail-settings">
                  <summary>Connection and environment settings</summary>
                  <div className="settings-form settings-form-compact">
                    <label>IMAP host<input value={configDraft.outlookImapHost} onChange={(event) => setConfigDraft({ ...configDraft, outlookImapHost: event.target.value })} /></label>
                    <label>IMAP port<input value={configDraft.outlookImapPort} onChange={(event) => setConfigDraft({ ...configDraft, outlookImapPort: event.target.value })} inputMode="numeric" /></label>
                    <label>SMTP host<input value={configDraft.outlookSmtpHost} onChange={(event) => setConfigDraft({ ...configDraft, outlookSmtpHost: event.target.value })} /></label>
                    <label>SMTP port<input value={configDraft.outlookSmtpPort} onChange={(event) => setConfigDraft({ ...configDraft, outlookSmtpPort: event.target.value })} inputMode="numeric" /></label>
                    <label>Username env<input value={configDraft.outlookUsernameEnv} onChange={(event) => setConfigDraft({ ...configDraft, outlookUsernameEnv: event.target.value })} /></label>
                    <label>Password env<input value={configDraft.outlookPasswordEnv} onChange={(event) => setConfigDraft({ ...configDraft, outlookPasswordEnv: event.target.value })} /></label>
                    <label>From env<input value={configDraft.outlookFromEnv} onChange={(event) => setConfigDraft({ ...configDraft, outlookFromEnv: event.target.value })} /></label>
                    <label>Mailbox<input value={configDraft.outlookMailbox} onChange={(event) => setConfigDraft({ ...configDraft, outlookMailbox: event.target.value })} /></label>
                    <label className="setting-wide">Archive mailbox<input value={configDraft.outlookArchiveMailbox} onChange={(event) => setConfigDraft({ ...configDraft, outlookArchiveMailbox: event.target.value })} /></label>
                  </div>
                </details>
              </section>
            </div>
            <div className="row-actions">
              <button className="button" type="button" onClick={() => void saveConfiguration()}>Save Tool Settings</button>
              <button className="button button-secondary" type="button" onClick={() => navigateTo('tools')}>Open Tool Permissions</button>
            </div>
          </SettingGroup>
        </ConfigSectionPage>
      )
    }

    if (configSection === 'security') {
      return (
        <ConfigSectionPage title="Security" description="Approval and privacy settings enforced by backend policy." onBack={() => navigateTo('config')}>
          <SettingGroup title="Approval policy" description="Risky actions stay approval-gated. Lower threshold means more approval prompts.">
            <div className="setting-rows">
              <SettingRow label="Approval risk threshold" help="Tools at or above this risk level require approval.">
                <input value={configDraft.approvalRisk} onChange={(event) => setConfigDraft({ ...configDraft, approvalRisk: event.target.value })} inputMode="numeric" />
              </SettingRow>
              <label className="setting-check">
                <input type="checkbox" checked={configDraft.sendChatHistoryToCloud} onChange={(event) => setConfigDraft({ ...configDraft, sendChatHistoryToCloud: event.target.checked })} />
                <span><strong>Send chat and relevant memory context to cloud models</strong><small>Allows configured cloud providers to receive recent chat and relevant local memory snippets.</small></span>
              </label>
            </div>
            <DangerZone>
              <strong>Security baseline remains backend-enforced.</strong>
              <p>Secret paths, destructive SQL, file deletion approvals, and emergency stop policy are enforced outside the prompt layer.</p>
            </DangerZone>
            <div className="row-actions">
              <button className="button" type="button" onClick={() => void saveConfiguration()}>Save Security</button>
            </div>
          </SettingGroup>
        </ConfigSectionPage>
      )
    }

    if (configSection === 'workspace') {
      return (
        <ConfigSectionPage title="Workspace" description="Where local files are stored and where terminal commands run." onBack={() => navigateTo('config')}>
          <SettingGroup title="Storage" description="Downloads and scraped files stay local.">
            <div className="setting-rows">
              <SettingRow label="Downloads root" help="Absolute path, or empty for the default workspace.">
                <input value={configDraft.downloadsRoot} onChange={(event) => setConfigDraft({ ...configDraft, downloadsRoot: event.target.value })} placeholder={configuration?.paths.workspace ?? '/absolute/path'} />
              </SettingRow>
            </div>
            <div className="status-grid">
              <span>Current workspace</span><strong>{configuration?.paths.current_workspace ?? '-'}</strong>
              <span>Downloads</span><strong>{configuration?.paths.downloads_root ?? '-'}</strong>
            </div>
          </SettingGroup>
          <SettingGroup title="Terminal policy" description="Stateless dashboard terminal commands resolve inside the current workspace policy.">
            <div className="setting-rows">
              <SettingRow label="Workspace root"><input value={terminalWorkspaceRoot} onChange={(event) => setTerminalWorkspaceRoot(event.target.value)} placeholder="/path/to/project" /></SettingRow>
              <SettingRow label="Timeout seconds"><input value={terminalTimeout} onChange={(event) => setTerminalTimeout(event.target.value)} inputMode="numeric" /></SettingRow>
              <SettingRow label="Max output chars"><input value={terminalMaxOutput} onChange={(event) => setTerminalMaxOutput(event.target.value)} inputMode="numeric" /></SettingRow>
              <label className="setting-check">
                <input type="checkbox" checked={terminalAutoApprove} onChange={(event) => setTerminalAutoApprove(event.target.checked)} />
                <span><strong>Auto-approve exact allowlist</strong><small>Only exact allowlist matches can bypass approval.</small></span>
              </label>
            </div>
            <div className="row-actions">
              <button className={terminal?.enabled ? 'button button-danger' : 'button'} type="button" onClick={() => void runAction(() => (terminal?.enabled ? api.disableTerminal() : api.enableTerminal()), terminal?.enabled ? 'Terminal disabled.' : 'Terminal enabled.')}>{terminal?.enabled ? 'Disable Terminal' : 'Enable Terminal'}</button>
              <button className="button button-secondary" type="button" onClick={() => void saveTerminalSettings()}>Save Terminal</button>
              <button className="button" type="button" onClick={() => void saveConfiguration()}>Save Storage</button>
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
          </SettingGroup>
        </ConfigSectionPage>
      )
    }

    if (configSection === 'memory') {
      return (
        <ConfigSectionPage title="Memory" description="Local long-term context and Markdown memory storage." onBack={() => navigateTo('config')}>
          <SettingGroup title="Memory status">
            <div className="status-grid">
              <span>Memory root</span><strong>{configuration?.paths.memory ?? '-'}</strong>
              <span>Files</span><strong>{memoryFiles.length}</strong>
            </div>
            <div className="row-actions">
              <button className="button button-secondary" type="button" onClick={() => navigateTo('memory')}>Open Memory Files</button>
            </div>
          </SettingGroup>
        </ConfigSectionPage>
      )
    }

    if (configSection === 'telegram') {
      return (
        <ConfigSectionPage title="Telegram" description="Remote access configuration lives in the Telegram screen." onBack={() => navigateTo('config')}>
          <SettingGroup title="Telegram status">
            <div className="status-grid">
              <span>Enabled</span><strong>{telegram?.enabled ? 'yes' : 'no'}</strong>
              <span>Token loaded</span><strong>{telegram?.bot_token_available ? 'yes' : 'no'}</strong>
              <span>Allowed users</span><strong>{telegram?.allowed_user_ids.length ?? 0}</strong>
              <span>Polling</span><strong>{telegram?.polling ? 'running' : 'stopped'}</strong>
            </div>
            <div className="row-actions">
              <button className="button button-secondary" type="button" onClick={() => navigateTo('telegram')}>Open Telegram Settings</button>
            </div>
          </SettingGroup>
        </ConfigSectionPage>
      )
    }

    if (configSection === 'emergency') {
      return (
        <ConfigSectionPage title="Emergency" description="Emergency stop remains available in the top bar from every screen." onBack={() => navigateTo('config')}>
          <SettingGroup title="Emergency state">
            <div className="status-grid">
              <span>Status</span><strong>{emergency?.active ? 'active' : 'clear'}</strong>
              <span>Triggered at</span><strong>{emergency?.triggered_at || '-'}</strong>
              <span>Reason</span><strong>{emergency?.reason || '-'}</strong>
              <span>Active terminal processes</span><strong>{emergency?.active_terminal_processes.length ?? 0}</strong>
            </div>
            <DangerZone>
              <strong>Emergency Stop interrupts terminal activity and blocks tool execution.</strong>
              <p>Use it when a running command or approval chain must stop immediately. Reset only when the system is safe to continue.</p>
            </DangerZone>
            <div className="row-actions">
              <button className="button button-danger" type="button" onClick={() => void emergencyStop()}>Emergency Stop</button>
              <button className="button button-secondary" type="button" onClick={() => void emergencyReset()} disabled={!emergency?.active}>Reset Emergency</button>
            </div>
          </SettingGroup>
        </ConfigSectionPage>
      )
    }

    return (
      <ConfigSectionPage title="Advanced" description="System paths, doctor summary, and connector readiness." onBack={() => navigateTo('config')}>
        <div className="two-column">
          <SettingGroup title="System paths">
            <div className="status-grid">
              <span>Data</span><strong>{configuration?.paths.data_dir ?? '-'}</strong>
              <span>Config</span><strong>{configuration?.paths.config ?? '-'}</strong>
              <span>Workspace</span><strong>{configuration?.paths.workspace ?? '-'}</strong>
              <span>Audit DB</span><strong>{configuration?.paths.audit_db ?? '-'}</strong>
            </div>
          </SettingGroup>
          <SettingGroup title="Readiness">
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
          </SettingGroup>
        </div>
      </ConfigSectionPage>
    )
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
  const telegramEnabled = Boolean(telegram?.enabled)
  const telegramTokenLoaded = Boolean(telegram?.bot_token_available)
  const telegramAllowedCount = telegram?.allowed_user_ids.length ?? 0
  const telegramReady = Boolean(telegram?.ready)
  const telegramPolling = Boolean(telegram?.polling)
  const telegramState = telegram?.polling_error
    ? 'error'
    : telegramPolling
      ? 'live'
      : telegramReady
        ? 'ready'
        : telegramEnabled
          ? 'setup'
          : 'offline'
  const telegramStateLabel = telegram?.polling_error
    ? 'Error'
    : telegramPolling
      ? 'Live'
      : telegramReady
        ? 'Ready'
        : telegramEnabled
          ? 'Setup required'
          : 'Offline'
  const telegramMissingSteps = [
    telegramEnabled ? '' : 'Enable Telegram interface',
    telegramTokenLoaded ? '' : 'Load a BotFather token',
    telegramAllowedCount > 0 ? '' : 'Allow at least one Telegram user ID',
  ].filter(Boolean)

  return (
    <div className={[
      'app-shell',
      railCollapsed ? 'app-shell--rail-collapsed' : '',
      autonomyEnabled ? 'app-shell--autonomy' : '',
    ].filter(Boolean).join(' ')}>
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

        <SidebarNav
          items={navItems}
          activeKey={activeView}
          onNavigate={(key) => navigateTo(key as View)}
        />

        <div className="rail-status">
          <span>Runtime</span>
          <strong>{status?.llm.provider ?? 'loading'} / {status?.llm.model ?? '-'}</strong>
          <span>Enabled tools</span>
          <strong>{enabledToolCount}</strong>
          <span>Mode</span>
          <strong>{autonomyEnabled ? 'DANGER' : 'SAFE'}</strong>
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
            <button
              className={autonomyEnabled ? 'autonomy-toggle autonomy-toggle--active' : 'autonomy-toggle'}
              type="button"
              onClick={() => void toggleAutonomyMode()}
              disabled={busy || autonomy?.toggle_locked_by_env}
              title={autonomy?.toggle_locked_by_env ? 'LOCAL_DEV_AUTONOMY env var is forcing autonomy mode.' : 'Switch runtime orchestration mode'}
            >
              <span className="autonomy-dot" />
              <span>{autonomyEnabled ? 'Danger Mode' : 'Standard Safe Mode'}</span>
            </button>
            {emergency?.active ? (
              <button className="button" type="button" onClick={() => void emergencyReset()}>
                Reset Emergency
              </button>
            ) : null}
            <button className="button button-danger" type="button" onClick={() => void emergencyStop()} data-tour="tour-emergency">
              Emergency Stop
            </button>
            <button className="button button-secondary" type="button" onClick={() => setTourOpen(true)}>
              Start Tour
            </button>
            <button className="button button-secondary" type="button" onClick={() => void refreshAll()}>
              Refresh
            </button>
          </div>
        </header>

        {notice ? <div className="notice">{notice}</div> : null}
        {emergency?.active ? (
          <div className="notice notice-danger">
            Emergency stop is active. Tool execution is blocked. Active terminal processes: {emergency.active_terminal_processes.length}.
          </div>
        ) : null}

        <AppRoutes>
        {activeView === 'chat' ? (
          <section className="chat-workspace" data-tour="tour-chat">
            <div className="chat-main panel">
              <div className="chat-log" ref={chatLogRef}>
                {messages.map((message) => (
                  <article key={message.id} className={`message message--${message.role}`}>
                    <div className="message-avatar">{message.role === 'user' ? 'YOU' : message.role === 'system' ? 'SYS' : 'AI'}</div>
                    <div className="message-body">
                      <p>{message.text}</p>
                      {visibleTrace(traceFromResponse(message.response)).length > 0 ? (
                        <details className="reasoning-panel">
                          <summary>
                            <span className="reasoning-summary-main">
                              <span className="reasoning-dot" />
                              <span>Reasoning</span>
                            </span>
                            <span>{visibleTrace(traceFromResponse(message.response)).length} steps</span>
                          </summary>
                          <div className="reasoning-events">
                            {visibleTrace(traceFromResponse(message.response)).map((event, index) => (
                              <div className="reasoning-event" key={`${message.id}-trace-${index}`}>
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
                      ) : null}
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
                    </div>
                  </article>
                ))}
                {busy && visibleTrace(activeTrace).length > 0 ? (
                  <article className="message message--agent message--runtime">
                    <div className="message-avatar">RUN</div>
                    <div className="message-body">
                      <div className="reasoning-panel reasoning-panel--live">
                        <div className="reasoning-live-header">
                          <span className="reasoning-summary-main">
                            <span className="reasoning-dot reasoning-dot--live" />
                            <strong>{latestTraceTitle(activeTrace)}</strong>
                          </span>
                          <span>{visibleTrace(activeTrace).length} steps</span>
                        </div>
                        <div className="reasoning-events">
                          {visibleTrace(activeTrace).slice(-8).map((event, index) => (
                            <div className="reasoning-event" key={`active-trace-${index}-${event.title}`}>
                              <span className={`reasoning-status reasoning-status--${event.status}`}>{event.status}</span>
                              <span className="reasoning-event-copy">
                                <strong>{event.title}</strong>
                                {event.detail ? <small>{event.detail}</small> : null}
                              </span>
                              {event.tool ? <code>{event.tool}</code> : null}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </article>
                ) : null}
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

        {activeView === 'approvals' ? (
          <section className="approvals-console">
            <div className="panel command-panel">
              <div className="section-heading">
                <strong>Approvals</strong>
                <span>{pendingCount} pending action{pendingCount === 1 ? '' : 's'}</span>
              </div>
              <p className="section-copy">
                Review high-risk or permission-gated tool actions before the agent executes them.
              </p>
              {approvals.length === 0 ? <p className="empty">No pending approvals.</p> : null}
              <div className="approval-list">
                {approvals.map((approval) => (
                  <article className="approval-card approval-card--wide" key={approval.id}>
                    <div>
                      <strong>#{approval.id} {approval.tool}</strong>
                      <span>Risk {approval.risk ?? '-'} - {approval.reason ?? approval.decision_reason ?? 'Needs approval'}</span>
                    </div>
                    <code>{JSON.stringify(approval.args)}</code>
                    <div className="row-actions">
                      <button className="button" type="button" onClick={() => void approveRequest(approval.id)}>Approve</button>
                      <button className="button button-danger" type="button" onClick={() => void denyRequest(approval.id)}>Deny</button>
                    </div>
                  </article>
                ))}
              </div>
            </div>
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

        {activeView === 'config' ? renderConfigRoutes() : null}
        {activeView === 'telegram' ? (
          <section className="telegram-layout">
            <div className={`telegram-overview telegram-overview--${telegramState}`}>
              <div className="telegram-overview-main">
                <p className="eyebrow">Remote channel</p>
                <div className="telegram-title-row">
                  <h2>Telegram Remote Access</h2>
                  <span className={`telegram-state telegram-state--${telegramState}`}>{telegramStateLabel}</span>
                </div>
                <p>{telegramPolling ? 'Polling allowlisted users.' : 'Secure Telegram control path for approved local actions.'}</p>
              </div>
              <div className="telegram-overview-actions">
                <button
                  className={telegramEnabled ? 'button button-danger' : 'button'}
                  type="button"
                  onClick={() => void runAction(() => (telegramEnabled ? api.disableTelegram() : api.enableTelegram()), telegramEnabled ? 'Telegram disabled.' : 'Telegram enabled.')}
                  disabled={busy}
                >
                  {telegramEnabled ? 'Disable' : 'Enable'}
                </button>
                <button
                  className="button button-secondary"
                  type="button"
                  onClick={() => void toggleTelegramPolling()}
                  disabled={busy || !telegramReady}
                >
                  {telegramPolling ? 'Stop Polling' : 'Start Polling'}
                </button>
              </div>
            </div>

            <div className="telegram-status-grid">
              <div className={telegramEnabled ? 'telegram-status-card telegram-status-card--ok' : 'telegram-status-card'}>
                <span>Interface</span>
                <strong>{telegramEnabled ? 'Enabled' : 'Disabled'}</strong>
              </div>
              <div className={telegramTokenLoaded ? 'telegram-status-card telegram-status-card--ok' : 'telegram-status-card telegram-status-card--warn'}>
                <span>Token</span>
                <strong>{telegramTokenLoaded ? 'Loaded' : 'Missing'}</strong>
              </div>
              <div className={telegramAllowedCount > 0 ? 'telegram-status-card telegram-status-card--ok' : 'telegram-status-card telegram-status-card--warn'}>
                <span>Allowed Users</span>
                <strong>{telegramAllowedCount}</strong>
              </div>
              <div className={telegramPolling ? 'telegram-status-card telegram-status-card--ok' : 'telegram-status-card'}>
                <span>Polling</span>
                <strong>{telegramPolling ? 'Running' : 'Stopped'}</strong>
              </div>
            </div>

            {telegram?.polling_error ? (
              <div className="telegram-error">
                <strong>Polling error</strong>
                <span>{telegram.polling_error}</span>
              </div>
            ) : null}

            {telegramMissingSteps.length ? (
              <div className="telegram-checklist">
                {telegramMissingSteps.map((step, index) => (
                  <span key={step}><b>{index + 1}</b>{step}</span>
                ))}
              </div>
            ) : null}

            <div className="telegram-panels">
              <section className="panel telegram-panel">
                <div className="section-heading">
                  <strong>Bot Token</strong>
                  <span>{telegram?.bot_token_env ?? 'DMDAGENT_TELEGRAM_BOT_TOKEN'}</span>
                </div>
                <form className="inline-form" onSubmit={(event) => { event.preventDefault(); void saveTelegramTokenEnv() }}>
                  <input value={telegramTokenEnv} onChange={(event) => setTelegramTokenEnv(event.target.value)} placeholder="DMDAGENT_TELEGRAM_BOT_TOKEN" />
                  <button className="button button-secondary" type="submit" disabled={busy}>Save Env</button>
                </form>
                <form className="inline-form" onSubmit={(event) => { event.preventDefault(); void loadTelegramToken() }}>
                  <input type="password" value={telegramToken} onChange={(event) => setTelegramToken(event.target.value)} placeholder="BotFather token" />
                  <button className="button" type="submit" disabled={busy || !telegramToken.trim()}>Load Token</button>
                </form>
              </section>

              <section className="panel telegram-panel">
                <div className="section-heading">
                  <strong>Allowed Users</strong>
                  <span>{telegramAllowedCount} active</span>
                </div>
                <form className="inline-form" onSubmit={(event) => { event.preventDefault(); void allowTelegramUser() }}>
                  <input value={telegramUserId} onChange={(event) => setTelegramUserId(event.target.value)} placeholder="123456789" inputMode="numeric" />
                  <button className="button" type="submit" disabled={busy || !telegramUserId.trim()}>Allow User</button>
                </form>
                <div className="telegram-user-list">
                  {telegramAllowedCount ? null : <p className="empty">No allowed Telegram users.</p>}
                  {telegram?.allowed_user_ids.map((userId) => (
                    <article className="telegram-user-row" key={userId}>
                      <span>
                        <strong>{userId}</strong>
                        <small>Allowlisted user</small>
                      </span>
                      <button className="button button-danger" type="button" disabled={busy} onClick={() => void runAction(() => api.removeTelegramUser(userId), `Removed Telegram user: ${userId}`)}>Remove</button>
                    </article>
                  ))}
                </div>
              </section>

              <section className="panel telegram-panel telegram-panel--commands">
                <div className="section-heading">
                  <strong>Bot Commands</strong>
                  <span>Production commands</span>
                </div>
                <div className="telegram-command-list">
                  <div><code>/id</code><span>Return your Telegram user ID</span></div>
                  <div><code>/help</code><span>Show command surface</span></div>
                  <div><code>/approvals</code><span>List pending approvals</span></div>
                  <div><code>/approve 7</code><span>Approve a queued action</span></div>
                  <div><code>/deny 7</code><span>Deny a queued action</span></div>
                </div>
              </section>
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
                  <strong>Emergency Stop</strong>
                  <p>Press Emergency Stop when a running command or tool action must be interrupted immediately. The backend stops active terminal processes, cancels pending approvals, blocks new tool execution, and stays locked until Reset Emergency is pressed.</p>
                </article>
                <article className="guide-card">
                  <strong>Configuration</strong>
                  <p>Use Config to set download paths, browser limits, planner settings, terminal allowlists, and approval thresholds.</p>
                </article>
                <article className="guide-card">
                  <strong>System prompts</strong>
                  <p>Use Config to edit the chat, planner, and answer system prompts. Saving the code default clears the local override.</p>
                </article>
              </div>
              <button className="button button-secondary" type="button" onClick={() => { setTourStep(0); navigateTo('chat'); setTourOpen(true) }}>
                Replay Tour
              </button>
            </div>
          </section>
        ) : null}
        </AppRoutes>
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

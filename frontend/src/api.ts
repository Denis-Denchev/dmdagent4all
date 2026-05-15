export type AgentResponse = {
  status: string
  message: string
  data: unknown
}

export type AgentTraceEvent = {
  at?: string
  kind: string
  title: string
  status: string
  detail?: string
  tool?: string
  metadata?: Record<string, unknown>
}

export type ChatStreamEvent =
  | { type: 'trace'; event: AgentTraceEvent }
  | { type: 'final'; response: AgentResponse }
  | { type: 'error'; message: string }

export type AutonomyStatus = {
  enabled: boolean
  configured_enabled: boolean
  env_enabled: boolean
  mode: 'standard_safe' | 'full_llm_first_autonomy' | string
  toggle_locked_by_env: boolean
  max_iterations: number
  safe_invariants: string[]
}

export type Tool = {
  name: string
  description: string
  risk: number
  permissions: string[]
  approval_required: boolean
  cloud_allowed: boolean
  default_enabled: boolean
  enabled: boolean
}

export type Approval = {
  id: number
  created_at: string
  tool: string
  risk: number | null
  status: string
  reason: string | null
  args: Record<string, unknown>
  request_reason: string | null
  decision_reason: string | null
}

export type AuditEvent = {
  created_at: string
  event_type: string
  tool: string | null
  risk: number | null
  approved: boolean | null
  result_status: string | null
  metadata: Record<string, unknown>
}

export type MemoryList = {
  files: string[]
}

export type MemoryFile = {
  path: string
  content: string
}

export type WorkspaceFile = {
  path: string
  name: string
  folder: string
  label: string
  size: number
  modified_at: string
  content_type: string
  previewable: boolean
}

export type WorkspaceRoot = {
  name: string
  label: string
  path: string
  exists: boolean
  count: number
}

export type WorkspaceFilesResponse = {
  workspace: string
  roots: WorkspaceRoot[]
  files: WorkspaceFile[]
}

export type WorkspaceFileContent = WorkspaceFile & {
  content: string
  truncated: boolean
}

export type Status = {
  version: string
  data_dir: string
  config: string
  memory: string
  workspace: string
  audit_db: string
  llm: {
    provider: string
    model: string
    planner_model?: string | null
    base_url?: string
    api_key_env?: string | null
    mode: string
    response_language: string
  }
  terminal_enabled: boolean
  terminal: {
    enabled: boolean
    workspace_only: boolean
    timeout_seconds: number
    max_output_chars: number
    allowed_commands: string[][]
  }
  browser_enabled: boolean
  telegram: {
    enabled: boolean
    allowed_user_ids: number[]
    bot_token_env: string
  }
  autonomy?: AutonomyStatus
}

export type TerminalStatus = {
  enabled: boolean
  tool_enabled: boolean
  permission_granted: boolean
  ready: boolean
  workspace_only: boolean
  workspace_root: string
  timeout_seconds: number
  max_output_chars: number
  auto_approve_allowlisted: boolean
  allowed_commands: string[][]
}

export type TelegramStatus = {
  enabled: boolean
  allowed_user_ids: number[]
  bot_token_env: string
  bot_token_available: boolean
  polling_timeout_seconds: number
  ready: boolean
  polling: boolean
  polling_error?: string | null
}

export type OpenAIUsage = {
  requests: number
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  estimated_cost_usd: number
  limit_usd: number | null
  remaining_usd: number | null
  limit_reached: boolean
}

export type OpenAIStatus = {
  provider: string
  model: string
  planner_model?: string | null
  base_url: string
  api_key_env: string
  api_key_available: boolean
  usage: OpenAIUsage
}

export type OpenAIModelsResponse = {
  status: string
  models: string[]
  data: OpenAIStatus
}

export type DeepSeekStatus = {
  provider: string
  model: string
  planner_model?: string | null
  base_url: string
  api_key_env: string
  api_key_available: boolean
  default_models: string[]
}

export type DeepSeekModelsResponse = {
  status: string
  models: string[]
  data: DeepSeekStatus
}

export type ConnectorStatus = {
  name: string
  status: string
  detail: string
  tools_total: number
  enabled_tools: string[]
  permissions_required: string[]
  permissions_granted: string[]
  interface?: unknown
}

export type ModelMode = {
  key: string
  label: string
  default_model: string
  alternatives: string[]
  description: string
}

export type ModelsResponse = {
  current: Status['llm']
  modes: ModelMode[]
}

export type PermissionItem = {
  name: string
  granted: boolean
  tools: string[]
}

export type PermissionsResponse = {
  granted: string[]
  available: PermissionItem[]
}

export type DoctorCheck = {
  status: 'ok' | 'warn' | 'fail'
  area: string
  message: string
  hint: string
}

export type DoctorResponse = {
  summary: {
    ok: number
    warn: number
    fail: number
  }
  checks: DoctorCheck[]
}

export type EmergencyStatus = {
  active: boolean
  triggered_at: string
  reason: string
  active_terminal_processes: Array<{
    command_id: string
    pid: number
    running: boolean
  }>
}

export type AgentConfiguration = {
  paths: {
    data_dir: string
    config: string
    memory: string
    workspace: string
    current_workspace: string
    downloads_root: string
    downloads_root_custom: boolean
    audit_db: string
  }
  setup: Record<string, unknown>
  llm: Status['llm'] & {
    planner_max_tokens?: number
    planner_temperature?: number
    planner_think?: boolean
  }
  browser: {
    enabled?: boolean
    isolated_profile?: boolean
    downloads_to_workspace?: boolean
    approval_required_for_submit?: boolean
    timeout_seconds?: number
    max_response_bytes?: number
    max_text_chars?: number
  }
  email: {
    max_body_chars: number
    gmail: EmailProviderConfiguration
    outlook: EmailProviderConfiguration
  }
  terminal: Record<string, unknown>
  privacy: Record<string, unknown>
  permissions: {
    granted?: string[]
    approval_required_at_risk?: number
  }
  storage: {
    downloads_root?: string
  }
  runtime?: Record<string, unknown>
  autonomy?: AutonomyStatus
  system_prompts: {
    chat: {
      default: string
      custom: string
      effective: string
      customized: boolean
    }
    planner: {
      default: string
      custom: string
      effective: string
      customized: boolean
    }
    answer: {
      default: string
      custom: string
      effective: string
      customized: boolean
    }
  }
}

export type EmailProviderConfiguration = {
  enabled: boolean
  auth_method: 'app_password' | 'oauth2'
  imap_host: string
  imap_port: number
  smtp_host: string
  smtp_port: number
  username_env: string
  password_env: string
  from_env: string
  oauth_client_id: string
  oauth_redirect_uri: string
  oauth_email: string
  oauth_from_address: string
  oauth_client_secret_loaded: boolean
  oauth_refresh_token_loaded: boolean
  oauth_connected: boolean
  mailbox: string
  archive_mailbox: string
  credentials_loaded: boolean
  from_loaded: boolean
}

export type AgentConfigurationUpdate = {
  downloads_root?: string
  agent_name?: string
  user_name?: string
  preferred_language?: string
  response_language?: string
  planner_max_tokens?: number
  planner_temperature?: number
  planner_think?: boolean
  chat_system_prompt?: string
  planner_system_prompt?: string
  answer_system_prompt?: string
  send_chat_history_to_cloud?: boolean
  browser_timeout_seconds?: number
  browser_max_response_bytes?: number
  browser_max_text_chars?: number
  approval_required_at_risk?: number
  email?: {
    max_body_chars?: number
    gmail?: EmailProviderConfigurationUpdate
    outlook?: EmailProviderConfigurationUpdate
  }
}

export type EmailProviderConfigurationUpdate = {
  enabled?: boolean
  auth_method?: 'app_password' | 'oauth2'
  imap_host?: string
  imap_port?: number
  smtp_host?: string
  smtp_port?: number
  username_env?: string
  password_env?: string
  from_env?: string
  oauth_client_id?: string
  oauth_redirect_uri?: string
  oauth_email?: string
  oauth_from_address?: string
  mailbox?: string
  archive_mailbox?: string
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })
  if (!response.ok) {
    let detail = ''
    try {
      const body = await response.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body)
    } catch {
      detail = await response.text()
    }
    throw new Error(`${response.status} ${response.statusText}${detail ? `: ${detail}` : ''}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  status: () => request<Status>('/v1/status'),
  autonomy: () => request<AutonomyStatus>('/v1/autonomy'),
  setAutonomy: (enabled: boolean) =>
    request<AutonomyStatus>('/v1/autonomy', {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    }),
  chat: (message: string, session_id = 'dashboard') =>
    request<AgentResponse>('/v1/chat', {
      method: 'POST',
      body: JSON.stringify({ message, session_id }),
    }),
  chatStream: async (
    message: string,
    session_id = 'dashboard',
    onEvent: (event: ChatStreamEvent) => void,
  ) => {
    const response = await fetch('/v1/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id }),
    })
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}: ${await response.text()}`)
    }
    if (!response.body) {
      throw new Error('Streaming response did not include a body.')
    }
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let finalResponse: AgentResponse | null = null
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.trim()) continue
        const event = JSON.parse(line) as ChatStreamEvent
        onEvent(event)
        if (event.type === 'final') finalResponse = event.response
        if (event.type === 'error') throw new Error(event.message)
      }
    }
    if (buffer.trim()) {
      const event = JSON.parse(buffer) as ChatStreamEvent
      onEvent(event)
      if (event.type === 'final') finalResponse = event.response
      if (event.type === 'error') throw new Error(event.message)
    }
    if (!finalResponse) throw new Error('Agent stream ended without a final response.')
    return finalResponse
  },
  tools: () => request<Tool[]>('/v1/tools'),
  setTool: (tool: string, enabled: boolean) =>
    request<AgentResponse>(`/v1/tools/${encodeURIComponent(tool)}/${enabled ? 'enable' : 'disable'}`, {
      method: 'POST',
    }),
  permissions: () => request<PermissionsResponse>('/v1/permissions'),
  setPermission: (permission: string, granted: boolean) =>
    request<AgentResponse>(`/v1/permissions/${granted ? 'grant' : 'revoke'}`, {
      method: 'POST',
      body: JSON.stringify({ permission }),
    }),
  approvals: () => request<Approval[]>('/v1/approvals?status=pending&limit=50'),
  approve: (id: number) =>
    request<AgentResponse>(`/v1/approvals/${id}/approve`, { method: 'POST' }),
  deny: (id: number) =>
    request<AgentResponse>(`/v1/approvals/${id}/deny`, { method: 'POST' }),
  audit: () => request<AuditEvent[]>('/v1/audit?limit=30'),
  doctor: () => request<DoctorResponse>('/v1/doctor?check_network=false'),
  configuration: () => request<AgentConfiguration>('/v1/configuration'),
  updateConfiguration: (settings: AgentConfigurationUpdate) =>
    request<AgentResponse>('/v1/configuration', {
      method: 'POST',
      body: JSON.stringify(settings),
    }),
  loadEmailCredentials: (provider: 'gmail' | 'outlook', username: string, app_password: string, from_address?: string) =>
    request<AgentResponse>('/v1/email/credentials', {
      method: 'POST',
      body: JSON.stringify({ provider, username, app_password, from_address: from_address || null }),
    }),
  testEmailConnection: (provider: 'gmail' | 'outlook') =>
    request<AgentResponse>('/v1/email/test', {
      method: 'POST',
      body: JSON.stringify({ provider }),
    }),
  startGmailOAuth: (settings: {
    client_id: string
    client_secret?: string
    email: string
    from_address?: string
    redirect_uri?: string
  }) =>
    request<AgentResponse>('/v1/email/oauth/google/start', {
      method: 'POST',
      body: JSON.stringify(settings),
    }),
  loadGmailOAuthClientSecret: (settings: {
    client_secret: string
    client_id?: string
    email?: string
    from_address?: string
    redirect_uri?: string
  }) =>
    request<AgentResponse>('/v1/email/oauth/google/client-secret', {
      method: 'POST',
      body: JSON.stringify(settings),
    }),
  disconnectGmailOAuth: () => request<AgentResponse>('/v1/email/oauth/google/disconnect', { method: 'POST' }),
  memory: () => request<MemoryList>('/v1/memory'),
  memoryFile: (path: string) =>
    request<MemoryFile>(`/v1/memory/file?path=${encodeURIComponent(path)}`),
  writeMemory: (path: string, body: string) =>
    request<AgentResponse>('/v1/memory/file', {
      method: 'POST',
      body: JSON.stringify({ path, body }),
    }),
  workspaceFiles: () => request<WorkspaceFilesResponse>('/v1/workspace-files'),
  workspaceFile: (path: string) =>
    request<WorkspaceFileContent>(`/v1/workspace-files/file?path=${encodeURIComponent(path)}`),
  workspaceFileDownloadUrl: (path: string) =>
    `/v1/workspace-files/download?path=${encodeURIComponent(path)}`,
  models: () => request<ModelsResponse>('/v1/models'),
  setModelMode: (mode: string) =>
    request<AgentResponse>('/v1/models/mode', {
      method: 'POST',
      body: JSON.stringify({ mode }),
    }),
  setModel: (model: string) =>
    request<AgentResponse>('/v1/models/model', {
      method: 'POST',
      body: JSON.stringify({ model }),
    }),
  connectors: () => request<ConnectorStatus[]>('/v1/connectors'),
  terminal: () => request<TerminalStatus>('/v1/terminal'),
  enableTerminal: () => request<AgentResponse>('/v1/terminal/enable', { method: 'POST' }),
  disableTerminal: () => request<AgentResponse>('/v1/terminal/disable', { method: 'POST' }),
  updateTerminalSettings: (settings: {
    workspace_only?: boolean
    workspace_root?: string
    timeout_seconds?: number
    max_output_chars?: number
    auto_approve_allowlisted?: boolean
  }) =>
    request<AgentResponse>('/v1/terminal/settings', {
      method: 'POST',
      body: JSON.stringify(settings),
    }),
  allowTerminalCommand: (command: string) =>
    request<AgentResponse>('/v1/terminal/allow', {
      method: 'POST',
      body: JSON.stringify({ command }),
    }),
  removeTerminalCommand: (command: string[]) =>
    request<AgentResponse>('/v1/terminal/remove', {
      method: 'POST',
      body: JSON.stringify({ command }),
    }),
  runTerminalCommand: (command: string, cwd?: string) =>
    request<AgentResponse>('/v1/terminal/run', {
      method: 'POST',
      body: JSON.stringify({ command, cwd: cwd || null }),
    }),
  telegram: () => request<TelegramStatus>('/v1/telegram'),
  enableTelegram: () => request<AgentResponse>('/v1/telegram/enable', { method: 'POST' }),
  disableTelegram: () => request<AgentResponse>('/v1/telegram/disable', { method: 'POST' }),
  startTelegram: () => request<AgentResponse>('/v1/telegram/start', { method: 'POST' }),
  stopTelegram: () => request<AgentResponse>('/v1/telegram/stop', { method: 'POST' }),
  allowTelegramUser: (user_id: number) =>
    request<AgentResponse>('/v1/telegram/allow', {
      method: 'POST',
      body: JSON.stringify({ user_id }),
    }),
  removeTelegramUser: (user_id: number) =>
    request<AgentResponse>('/v1/telegram/remove', {
      method: 'POST',
      body: JSON.stringify({ user_id }),
    }),
  setTelegramTokenEnv: (bot_token_env: string) =>
    request<AgentResponse>('/v1/telegram/token-env', {
      method: 'POST',
      body: JSON.stringify({ bot_token_env }),
    }),
  loadTelegramToken: (token: string) =>
    request<AgentResponse>('/v1/telegram/token', {
      method: 'POST',
      body: JSON.stringify({ token }),
    }),
  openai: () => request<OpenAIStatus>('/v1/openai'),
  loadOpenAIKey: (api_key: string) =>
    request<AgentResponse>('/v1/openai/key', {
      method: 'POST',
      body: JSON.stringify({ api_key }),
    }),
  openAIModels: () => request<OpenAIModelsResponse>('/v1/openai/models'),
  setOpenAIModel: (model: string) =>
    request<AgentResponse>('/v1/openai/model', {
      method: 'POST',
      body: JSON.stringify({ model }),
    }),
  setOpenAILimit: (limit_usd: number | null) =>
    request<AgentResponse>('/v1/openai/limit', {
      method: 'POST',
      body: JSON.stringify({ limit_usd }),
    }),
  resetOpenAIUsage: () => request<AgentResponse>('/v1/openai/usage/reset', { method: 'POST' }),
  emergency: () => request<EmergencyStatus>('/v1/emergency'),
  emergencyStop: () => request<AgentResponse>('/v1/emergency/stop', { method: 'POST' }),
  emergencyReset: () => request<AgentResponse>('/v1/emergency/reset', { method: 'POST' }),
  deepseek: () => request<DeepSeekStatus>('/v1/deepseek'),
  loadDeepSeekKey: (api_key: string) =>
    request<AgentResponse>('/v1/deepseek/key', {
      method: 'POST',
      body: JSON.stringify({ api_key }),
    }),
  deepSeekModels: () => request<DeepSeekModelsResponse>('/v1/deepseek/models'),
  setDeepSeekModel: (model: string) =>
    request<AgentResponse>('/v1/deepseek/model', {
      method: 'POST',
      body: JSON.stringify({ model }),
    }),
}

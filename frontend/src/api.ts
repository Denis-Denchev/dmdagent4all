export type AgentResponse = {
  status: string
  message: string
  data: unknown
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
}

export type TerminalStatus = {
  enabled: boolean
  tool_enabled: boolean
  permission_granted: boolean
  ready: boolean
  workspace_only: boolean
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
    throw new Error(`${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  status: () => request<Status>('/v1/status'),
  chat: (message: string) =>
    request<AgentResponse>('/v1/chat', {
      method: 'POST',
      body: JSON.stringify({ message }),
    }),
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
  memory: () => request<MemoryList>('/v1/memory'),
  memoryFile: (path: string) =>
    request<MemoryFile>(`/v1/memory/file?path=${encodeURIComponent(path)}`),
  writeMemory: (path: string, body: string) =>
    request<AgentResponse>('/v1/memory/file', {
      method: 'POST',
      body: JSON.stringify({ path, body }),
    }),
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
}

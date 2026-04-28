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
  browser_enabled: boolean
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
  approvals: () => request<Approval[]>('/v1/approvals?status=pending&limit=50'),
  approve: (id: number) =>
    request<AgentResponse>(`/v1/approvals/${id}/approve`, { method: 'POST' }),
  deny: (id: number) =>
    request<AgentResponse>(`/v1/approvals/${id}/deny`, { method: 'POST' }),
  audit: () => request<AuditEvent[]>('/v1/audit?limit=30'),
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
}

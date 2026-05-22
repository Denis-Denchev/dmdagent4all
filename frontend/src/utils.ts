import type { AgentResponse, AgentTraceEvent } from './api'

export function dataObject(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : {}
}

export function formatDate(value: string | undefined) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

export function formatBytes(value: number) {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / 1024 / 1024).toFixed(1)} MB`
}

export function formatUsd(value: number | null | undefined, digits = 2) {
  return value === null || value === undefined ? '-' : `$${value.toFixed(digits)}`
}

export function approvalIdFromResponse(response?: AgentResponse) {
  if (response?.status !== 'approval_required') return null
  const approvalId = dataObject(response?.data).approval_id
  return typeof approvalId === 'number' ? approvalId : null
}

export function traceFromResponse(response?: AgentResponse): AgentTraceEvent[] {
  const trace = dataObject(response?.data).trace
  return Array.isArray(trace) ? (trace as AgentTraceEvent[]) : []
}

export function visibleTrace(events: AgentTraceEvent[]): AgentTraceEvent[] {
  return events.filter((event) => dataObject(event.metadata).visibility !== 'debug')
}

// Single place every backend call goes through -- replaces the scattered
// apiCall()/uploadFiles() pattern in the old index.html.

const API_BASE = '/api'

async function handle(response) {
  let body
  try {
    body = await response.json()
  } catch {
    throw new Error(`HTTP ${response.status}: response was not JSON`)
  }
  if (!response.ok) {
    throw new Error(body.error || `HTTP ${response.status}`)
  }
  return body
}

export async function uploadFiles(files) {
  const formData = new FormData()
  for (const file of files) formData.append('files', file)
  const response = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData })
  return handle(response)
}

export async function configure(payload) {
  const response = await fetch(`${API_BASE}/configure`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return handle(response)
}

export async function generate() {
  const response = await fetch(`${API_BASE}/generate`, { method: 'POST' })
  return handle(response)
}

export async function evaluate() {
  const response = await fetch(`${API_BASE}/evaluate`, { method: 'POST' })
  return handle(response)
}

export async function debugColumns() {
  const response = await fetch(`${API_BASE}/debug/columns`)
  return handle(response)
}

export async function exportData({ format, includeMetadata = true, includeSchema = true, includeEvaluation = true }) {
  const response = await fetch(`${API_BASE}/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      format,
      include_metadata: includeMetadata,
      include_schema: includeSchema,
      include_evaluation: includeEvaluation,
    }),
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.error || `HTTP ${response.status}`)
  }
  return response.blob()
}

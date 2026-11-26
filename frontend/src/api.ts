import type {
  AuthStatusResponse,
  JobCreateResponse,
  JobStatusResponse,
  QueueSnapshotResponse,
} from './types'

export class ApiError extends Error {
  status: number
  retryAfterSeconds: number | null

  constructor(message: string, status: number, retryAfterSeconds: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.retryAfterSeconds = retryAfterSeconds
  }
}

function parseRetryAfterHeaderValue(raw: string | null): number | null {
  if (!raw) return null
  const trimmed = raw.trim()
  if (!trimmed) return null

  const numeric = Number(trimmed)
  if (Number.isFinite(numeric) && numeric > 0) {
    return Math.ceil(numeric)
  }

  const parsedDate = Date.parse(trimmed)
  if (!Number.isNaN(parsedDate)) {
    const deltaSeconds = Math.ceil((parsedDate - Date.now()) / 1000)
    if (deltaSeconds > 0) return deltaSeconds
  }

  return null
}

function parseRetryAfterSeconds(response: Response): number | null {
  return parseRetryAfterHeaderValue(response.headers.get('retry-after'))
}

function withRateLimitFallback(
  status: number,
  fallback: string,
  retryAfterSeconds: number | null,
): string {
  if (status === 429) {
    return retryAfterSeconds
      ? `Too many requests. Please try again in ${retryAfterSeconds} second(s).`
      : 'Too many requests. Please wait a moment and try again.'
  }
  return fallback
}

function extractDetailFromPayload(data: unknown): string | null {
  if (data && typeof data === 'object' && 'detail' in data) {
    const detail = (data as { detail: unknown }).detail
    if (typeof detail === 'string' && detail.trim()) {
      return detail.trim()
    }
    if (Array.isArray(detail) && detail.length > 0) {
      const messages = detail
        .map((item) =>
          item && typeof item === 'object' && 'msg' in item
            ? String((item as { msg: unknown }).msg)
            : '',
        )
        .filter(Boolean)
      if (messages.length > 0) return messages.join('; ')
    }
  }
  return null
}

async function parseErrorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const contentType = response.headers.get('content-type') ?? ''
    if (contentType.includes('application/json')) {
      const data: unknown = await response.json()
      const detail = extractDetailFromPayload(data)
      if (detail) return detail
    } else {
      const text = (await response.text()).trim()
      if (text) return text
    }
  } catch {
    // Ignore parse errors and return fallback below
  }
  return fallback
}

function parseXhrErrorMessage(xhr: XMLHttpRequest, fallback: string): string {
  try {
    const contentType = xhr.getResponseHeader('content-type') ?? ''
    const rawText = typeof xhr.responseText === 'string' ? xhr.responseText.trim() : ''
    if (!rawText) return fallback
    if (contentType.includes('application/json') || rawText.startsWith('{')) {
      const data: unknown = JSON.parse(rawText)
      const detail = extractDetailFromPayload(data)
      if (detail) return detail
    }
    return rawText
  } catch {
    return fallback
  }
}

export async function getAuthStatus(): Promise<AuthStatusResponse> {
  const response = await fetch('/api/auth/status', {
    method: 'GET',
    credentials: 'same-origin',
    headers: { Accept: 'application/json' },
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response.status,
        `Failed to check authentication status (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as AuthStatusResponse
}

export async function loginWithPassword(password: string): Promise<AuthStatusResponse> {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({ password }),
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response.status,
        response.status === 401
          ? 'Invalid password.'
          : `Sign-in failed (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as AuthStatusResponse
}

export async function logout(): Promise<AuthStatusResponse> {
  const response = await fetch('/api/auth/logout', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { Accept: 'application/json' },
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response.status,
        `Sign-out failed (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as AuthStatusResponse
}

export interface CreateJobOptions {
  batchId?: string | null
  label?: string | null
}

export interface CreateJobWithProgressOptions extends CreateJobOptions {
  onProgress?: (loaded: number, total: number, pct: number) => void
  signal?: AbortSignal
}

function buildJobFormData(
  slides: File | File[],
  recording: File,
  options?: CreateJobOptions,
): FormData {
  const formData = new FormData()
  const slidesList = Array.isArray(slides) ? slides : [slides]
  for (const slideFile of slidesList) {
    formData.append('slides', slideFile)
  }
  formData.append('recording', recording)
  if (options?.batchId) {
    formData.append('batch_id', options.batchId)
  }
  if (options?.label) {
    formData.append('label', options.label)
  }
  return formData
}

export async function createJob(
  slides: File | File[],
  recording: File,
  options?: CreateJobOptions,
): Promise<JobCreateResponse> {
  const formData = buildJobFormData(slides, recording, options)

  const response = await fetch('/api/jobs', {
    method: 'POST',
    credentials: 'same-origin',
    body: formData,
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response.status,
        `Upload failed (HTTP ${response.status}). Please check your files and try again.`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as JobCreateResponse
}

export function createJobWithProgress(
  slides: File | File[],
  recording: File,
  options?: CreateJobWithProgressOptions,
): Promise<JobCreateResponse> {
  const slidesList = Array.isArray(slides) ? slides : [slides]
  const formData = buildJobFormData(slidesList, recording, options)
  const totalBytesFallback =
    slidesList.reduce((sum, file) => sum + file.size, 0) + recording.size

  return new Promise<JobCreateResponse>((resolve, reject) => {
    const signal = options?.signal
    if (signal?.aborted) {
      reject(new DOMException('Upload aborted.', 'AbortError'))
      return
    }

    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/jobs', true)
    xhr.setRequestHeader('Accept', 'application/json')

    const cleanup = () => {
      signal?.removeEventListener('abort', handleAbort)
    }

    const handleAbort = () => {
      xhr.abort()
    }

    signal?.addEventListener('abort', handleAbort, { once: true })

    if (xhr.upload && options?.onProgress) {
      xhr.upload.addEventListener('progress', (event) => {
        const total =
          event.lengthComputable && event.total > 0 ? event.total : totalBytesFallback
        const loaded = Math.min(event.loaded, total || event.loaded)
        const pct = total > 0 ? Math.min(100, Math.round((loaded / total) * 100)) : 0
        options.onProgress?.(loaded, total, pct)
      })
    }

    xhr.addEventListener('load', () => {
      cleanup()
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const parsed = JSON.parse(xhr.responseText) as JobCreateResponse
          options?.onProgress?.(totalBytesFallback, totalBytesFallback, 100)
          resolve(parsed)
        } catch {
          reject(new Error('Invalid JSON response from server.'))
        }
        return
      }

      const retryAfterSeconds = parseRetryAfterHeaderValue(xhr.getResponseHeader('retry-after'))
      const message = parseXhrErrorMessage(
        xhr,
        withRateLimitFallback(
          xhr.status,
          `Upload failed (HTTP ${xhr.status}). Please check your files and try again.`,
          retryAfterSeconds,
        ),
      )
      reject(new ApiError(message, xhr.status, retryAfterSeconds))
    })

    xhr.addEventListener('error', () => {
      cleanup()
      reject(new Error('Network error while uploading files.'))
    })

    xhr.addEventListener('abort', () => {
      cleanup()
      reject(new DOMException('Upload aborted.', 'AbortError'))
    })

    xhr.send(formData)
  })
}

export async function getJobStatus(token: string): Promise<JobStatusResponse> {
  const response = await fetch(`/api/jobs/${encodeURIComponent(token)}`, {
    method: 'GET',
    credentials: 'same-origin',
    headers: { Accept: 'application/json' },
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response.status,
        response.status === 404
          ? 'Job not found. Check that the link is complete.'
          : `Failed to fetch job status (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as JobStatusResponse
}

export async function getQueueSnapshot(options?: {
  recentLimit?: number
  batchId?: string
}): Promise<QueueSnapshotResponse> {
  const params = new URLSearchParams()
  params.set('recent_limit', String(options?.recentLimit ?? 50))
  if (options?.batchId) {
    params.set('batch_id', options.batchId)
  }

  const response = await fetch(`/api/queue?${params.toString()}`, {
    method: 'GET',
    credentials: 'same-origin',
    headers: { Accept: 'application/json' },
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response.status,
        `Failed to fetch queue status (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as QueueSnapshotResponse
}

export async function reorderQueue(tokens: string[]): Promise<QueueSnapshotResponse> {
  const response = await fetch('/api/queue/order', {
    method: 'PATCH',
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({ tokens }),
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response.status,
        `Failed to reorder queue (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as QueueSnapshotResponse
}

export async function pauseQueue(): Promise<QueueSnapshotResponse> {
  const response = await fetch('/api/queue/pause', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { Accept: 'application/json' },
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response.status,
        `Failed to pause queue (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as QueueSnapshotResponse
}

export async function resumeQueue(): Promise<QueueSnapshotResponse> {
  const response = await fetch('/api/queue/resume', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { Accept: 'application/json' },
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response.status,
        `Failed to resume queue (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as QueueSnapshotResponse
}

export async function cancelJob(
  token: string,
  options?: { deleteFiles?: boolean },
): Promise<JobStatusResponse> {
  const body =
    options?.deleteFiles !== undefined ? { delete_files: options.deleteFiles } : {}

  const response = await fetch(`/api/jobs/${encodeURIComponent(token)}/cancel`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response.status,
        `Failed to cancel job (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as JobStatusResponse
}

export async function retryJob(
  token: string,
  options?: { fromScratch?: boolean; position?: 'end' | 'front' },
): Promise<JobStatusResponse> {
  const response = await fetch(`/api/jobs/${encodeURIComponent(token)}/retry`, {
    method: 'POST',
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({
      from_scratch: options?.fromScratch ?? false,
      position: options?.position ?? 'end',
    }),
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response.status,
        `Failed to retry job (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }

  return (await response.json()) as JobStatusResponse
}

export async function deleteJob(token: string): Promise<void> {
  const response = await fetch(`/api/jobs/${encodeURIComponent(token)}`, {
    method: 'DELETE',
    credentials: 'same-origin',
    headers: { Accept: 'application/json' },
  })

  if (!response.ok) {
    const retryAfterSeconds = parseRetryAfterSeconds(response)
    const message = await parseErrorMessage(
      response,
      withRateLimitFallback(
        response.status,
        `Failed to delete job (HTTP ${response.status}).`,
        retryAfterSeconds,
      ),
    )
    throw new ApiError(message, response.status, retryAfterSeconds)
  }
}

export function getDownloadUrl(token: string): string {
  return `/api/jobs/${encodeURIComponent(token)}/download`
}
